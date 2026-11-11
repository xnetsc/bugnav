"""Daemon entry point.

Modes:
  observe - connect, sample, run the controller, but NEVER send commands
            to the robot. Everything the controller *would* do is logged.
            This is the mode for the first on-robot verification week.
  active  - commands are executed on the robot.

Usage:
  ECOVACS_PASSWORD=... python -m bugnav.runner --config config.json --mode observe
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request

from . import llm as llm_mod
from . import strategist, supervisor as sup_mod
from .chargeplan import ChargeModel, plan_resume
from .controller import BugNavController, ControllerConfig
from .deebot_adapter import AdapterConfig, DeebotAdapter
from .model import Command, CommandKind, RobotState


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def _main_async(cfg_path: str, mode: str) -> None:
    raw = _load_json(cfg_path)
    account = dict(raw.get("account", {}))
    # Credentials come from the environment (container env / .env file),
    # never from code and never hardcoded in config.json. The env wins:
    #   ECOVACS_ACCOUNT_ID  - the Ecovacs short ID
    #   ECOVACS_PASSWORD    - read by the adapter directly, env only
    if os.environ.get("ECOVACS_ACCOUNT_ID"):
        account["account_id"] = os.environ["ECOVACS_ACCOUNT_ID"]
    adapter = DeebotAdapter(AdapterConfig(**account))
    ctl_cfg = ControllerConfig()
    for k, v in raw.get("controller", {}).items():
        if k in ("detector", "perimeter"):
            for dk, dv in v.items():
                setattr(getattr(ctl_cfg, k), dk, dv)
        else:
            setattr(ctl_cfg, k, v)
    data_dir = raw.get("data_dir", "data")
    os.makedirs(data_dir, exist_ok=True)
    controller = BugNavController(
        ctl_cfg, stats_path=os.path.join(data_dir, "stats.json")
    )
    poll_s = float(raw.get("poll_interval_s", 2.5))
    webhook = raw.get("notify_webhook", "")
    log_path = os.path.join(data_dir, "events.jsonl")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "msg": msg}) + "\n")
        if webhook and msg.startswith("NOTIFY"):
            try:
                req = urllib.request.Request(
                    webhook,
                    data=json.dumps({"text": msg}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass

    # ---- LLM layers (strategist + supervisor). Both are inert unless
    # their config flag is on AND an LLM key is present in the env.
    sup_cfg = sup_mod.SupervisorConfig()
    for k, v in raw.get("supervisor", {}).items():
        setattr(sup_cfg, k, v)
    llm_ready = llm_mod.llm_configured()
    advisor_on = ctl_cfg.llm_strategy_enabled and llm_ready
    supervisor_on = sup_cfg.enabled and llm_ready
    if (ctl_cfg.llm_strategy_enabled or sup_cfg.enabled) and not llm_ready:
        log("LLM layers requested in config but BUGNAV_LLM_API_KEY is "
            "not set; running deterministic-only")
    sup_cfg.enabled = supervisor_on
    watch = sup_mod.SessionWatch(sup_cfg)

    # ---- persistent home map (all cells ever cleaned, 0.5 m grid) and
    # learned charge model.
    home_path = os.path.join(data_dir, "home_cells.json")
    home_cells: set[tuple[int, int]] = set()
    if os.path.exists(home_path):
        try:
            with open(home_path, encoding="utf-8") as f:
                home_cells = {tuple(c) for c in json.load(f)["cells"]}
        except Exception:
            home_cells = set()
    model_path = os.path.join(data_dir, "charge_model.json")
    charge_model = ChargeModel()
    if os.path.exists(model_path):
        try:
            with open(model_path, encoding="utf-8") as f:
                saved = json.load(f)
            charge_model.drain_pct_per_m2 = float(
                saved.get("drain_pct_per_m2", charge_model.drain_pct_per_m2))
            charge_model.overhead_s = float(
                saved.get("overhead_s", charge_model.overhead_s))
        except Exception:
            pass

    state = {
        "last_sample": None,
        "consult_task": None,
        "review_task": None,
        "charge_target": None,  # resume at this battery % (one-shot)
        "session_stopped": False,
        "last_home_save": 0.0,
    }

    async def dispatch(cmds) -> None:
        for cmd in cmds:
            if cmd.kind == CommandKind.NOTIFY:
                log(f"NOTIFY: {cmd.message}")
            elif mode == "active":
                try:
                    await adapter.execute(cmd, log)
                except Exception as exc:  # never crash the daemon
                    log(f"command {cmd.kind} failed: {exc}")
            else:
                log(f"[observe] would send {cmd.kind} {cmd.point or ''}")

    async def service_consult(req) -> None:
        plan = None
        try:
            text = await llm_mod.chat(strategist.build_messages(req.situation))
            plan = strategist.parse_plan(text)
            log(f"strategy advisor ({req.kind}): plan={plan}")
        except Exception as exc:
            log(f"strategy advisor call failed: {exc}; using fallback")
        sample = state["last_sample"]
        if sample is None:
            return
        if mode == "active":
            await dispatch(controller.apply_consult_result(plan, sample))
        else:
            log(f"[observe] advisor plan logged only; fixed strategy continues")
            await dispatch(controller.apply_consult_result(None, sample))

    async def service_review() -> None:
        sample = state["last_sample"]
        if sample is None:
            return
        plan = None
        remaining = None
        if home_cells:
            remaining = max(
                0, len(home_cells) - len(watch.cells)) * sup_mod.CELL_AREA_M2
        if (sample.state == RobotState.DOCKED and watch.bout_open
                and adapter.battery_pct is not None
                and remaining is not None):
            drain = watch.drain_pct_per_m2()
            if drain:
                charge_model.drain_pct_per_m2 = drain
            plan = plan_resume(adapter.battery_pct, remaining, charge_model)
        digest = watch.digest(
            time.time(), controller.phase.value,
            home_cells=len(home_cells) or None,
            robot_state=sample.state.value, charge_plan=plan,
            charge_model=charge_model)
        verdict = sup_mod.Verdict()
        try:
            text = await llm_mod.chat(sup_mod.build_messages(digest))
            verdict = sup_mod.parse_verdict(text)
        except Exception as exc:
            log(f"supervisor call failed: {exc}; continuing")
        log(f"supervisor verdict: {verdict.action} "
            f"reason={verdict.reason!r} resume_pct={verdict.resume_charge_pct}")
        if verdict.action == "notify" and verdict.message:
            log(f"NOTIFY: supervisor: {verdict.message}")
        elif verdict.action == "abandon_current":
            if mode == "active":
                await dispatch(controller.force_abandon(sample))
            else:
                log("[observe] supervisor would abandon current episode")
        elif verdict.action == "stop_session":
            log(f"NOTIFY: supervisor stopping session: "
                f"{verdict.message or verdict.reason}")
            state["session_stopped"] = True
            if mode == "active":
                await dispatch([Command(CommandKind.STOP)])
            else:
                log("[observe] supervisor would stop the session")
        # Charge decision: an explicit verdict override wins; otherwise
        # adopt the deterministic solver's proposal from a charge review.
        if watch.charge_review_pending or plan is not None:
            target = verdict.resume_charge_pct
            if target is None and plan is not None:
                target = plan["resume_soc"]
            if target is not None and verdict.action == "continue":
                state["charge_target"] = target
                log(f"charge plan: resume at {target}% "
                    f"(plan={plan}, verdict_override={verdict.resume_charge_pct})")

    await adapter.connect(log)
    log(f"runner started in mode={mode}, poll={poll_s}s, "
        f"advisor={'on' if advisor_on else 'off'}, "
        f"supervisor={'on' if supervisor_on else 'off'}")
    last_refresh = 0.0
    while True:
        now = time.monotonic()
        if now - last_refresh >= poll_s:
            last_refresh = now
            adapter.request_refresh()
            sample = adapter.sample(now)
            if sample is not None:
                state["last_sample"] = sample
                wall = time.time()
                with open(
                    os.path.join(data_dir, "samples.jsonl"), "a", encoding="utf-8"
                ) as f:
                    f.write(
                        json.dumps(
                            {
                                "t": sample.t,
                                "x": sample.pos.x,
                                "y": sample.pos.y,
                                "state": sample.state.value,
                                "heading": sample.heading_deg,
                                "error": sample.error_code,
                                "phase": controller.phase.value,
                                "battery": adapter.battery_pct,
                                "water_amount": adapter.water_amount,
                            }
                        )
                        + "\n"
                    )
                if sample.state == RobotState.CLEANING:
                    cell = (round(sample.pos.x / sup_mod.CELL_M),
                            round(sample.pos.y / sup_mod.CELL_M))
                    if cell not in home_cells:
                        home_cells.add(cell)
                        if wall - state["last_home_save"] > 300:
                            state["last_home_save"] = wall
                            with open(home_path, "w", encoding="utf-8") as f:
                                json.dump({"cells": sorted(home_cells)}, f)
                watch.feed(sample, controller.phase.value, wall,
                           battery=adapter.battery_pct,
                           water_amount=adapter.water_amount)
                if not state["session_stopped"]:
                    await dispatch(controller.tick(sample))
                # episode strategist servicing
                req = controller.pending_consult
                if (advisor_on and req is not None
                        and (state["consult_task"] is None
                             or state["consult_task"].done())):
                    state["consult_task"] = asyncio.create_task(
                        service_consult(req))
                # session supervisor servicing
                if (supervisor_on and not state["session_stopped"]
                        and watch.due()
                        and (state["review_task"] is None
                             or state["review_task"].done())):
                    watch.mark_reviewed()
                    state["review_task"] = asyncio.create_task(service_review())
                # charge-target enforcement while docked
                target = state["charge_target"]
                if target is not None:
                    if sample.state == RobotState.CLEANING:
                        state["charge_target"] = None  # firmware resumed
                    elif (sample.state == RobotState.DOCKED
                            and adapter.battery_pct is not None
                            and adapter.battery_pct >= target
                            and watch.bout_open):
                        state["charge_target"] = None
                        log(f"charge target reached ({adapter.battery_pct}% "
                            f">= {target}%); resuming clean")
                        if mode == "active":
                            await dispatch(
                                [Command(CommandKind.RESUME_AUTO)])
                        else:
                            log("[observe] would resume clean now")
        await asyncio.sleep(0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--mode", choices=["observe", "active"], default="observe")
    args = ap.parse_args()
    try:
        asyncio.run(_main_async(args.config, args.mode))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
