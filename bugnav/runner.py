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

    # ---- persistent home map baseline (0.5 m grid) and learned charge
    # model.  home_cells is the ADOPTED baseline only: bout footprints
    # are folded in at bout end when they match it; a mismatching bout
    # (new map / other home / different rooms) is stashed as a
    # candidate and adopted only when the new layout repeats, so a
    # one-off trip never poisons the denominator (user 2026-11-11).
    home_path = os.path.join(data_dir, "home_cells.json")
    candidate_path = os.path.join(data_dir, "home_candidate.json")

    def _load_cells(path):
        try:
            with open(path, encoding="utf-8") as f:
                return {tuple(c) for c in json.load(f)["cells"]}
        except Exception:
            return set()

    def _save_cells(path, cells):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"cells": sorted(cells)}, f)

    home_cells: set[tuple[int, int]] = (
        _load_cells(home_path) if os.path.exists(home_path) else set())
    candidate_cells: set[tuple[int, int]] | None = (
        _load_cells(candidate_path)
        if os.path.exists(candidate_path) else None)
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
            if saved.get("band_rates"):
                charge_model.band_rates = [
                    (int(t), float(r)) for t, r in saved["band_rates"]]
        except Exception:
            pass
    history_path = os.path.join(data_dir, "charge_history.json")

    def _save_charge_state() -> None:
        """Persist the charge model (incl. AI-calibrated band rates)
        and the raw charge history the calibration feeds on."""
        try:
            with open(model_path, "w", encoding="utf-8") as f:
                json.dump({
                    "drain_pct_per_m2": charge_model.drain_pct_per_m2,
                    "overhead_s": charge_model.overhead_s,
                    "band_rates": [list(b)
                                   for b in charge_model.band_rates],
                }, f)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump({"stays": watch.charge_history()}, f)
        except Exception:
            pass

    state = {
        "last_sample": None,
        "consult_task": None,
        "review_task": None,
        "charge_target": None,  # resume at this battery % (one-shot)
        "session_stopped": False,
        "awaiting_until": None,   # ask_user: monotonic deadline
        "awaiting_since": 0.0,
        "awaiting_from": None,    # robot state when the ask began
        "last_home_save": 0.0,
        "bout_was_open": False,
    }

    if os.path.exists(history_path):
        try:
            with open(history_path, encoding="utf-8") as f:
                watch.restore_charge_history(json.load(f)["stays"])
        except Exception:
            pass

    def _maintain_home_cache() -> None:
        """Bout-end cache maintenance (see candidate_decision)."""
        nonlocal candidate_cells
        _save_charge_state()
        bout = set(watch.cells)
        decision = sup_mod.candidate_decision(
            bout, home_cells, candidate_cells, sup_cfg)
        if decision == "merge":
            if bout - home_cells:
                home_cells.update(bout)
                _save_cells(home_path, home_cells)
            if candidate_cells is not None:
                candidate_cells = None
                if os.path.exists(candidate_path):
                    os.remove(candidate_path)
        elif decision == "stash":
            cons = sup_mod.home_consistency(bout, home_cells, sup_cfg)
            candidate_cells = bout
            _save_cells(candidate_path, candidate_cells)
            log(f"NOTIFY: 本场清扫区域与缓存地图显著不同（新面积占比 "
                f"{cons['novel_area_share']:.0%}），进度/剩余已停用缓存口径；"
                "若新布局再出现一场，将采用为新基准")
        elif decision == "adopt":
            home_cells.clear()
            home_cells.update((candidate_cells or set()) | bout)
            _save_cells(home_path, home_cells)
            candidate_cells = None
            if os.path.exists(candidate_path):
                os.remove(candidate_path)
            log("NOTIFY: 新布局已连续出现两场，全屋面积基准已按新布局重建")

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
        if home_cells and not sup_mod.home_consistency(
                watch.cells, home_cells, sup_cfg)["suspect"]:
            remaining = max(
                0, len(home_cells) - len(watch.cells)) * sup_mod.CELL_AREA_M2
        if (sample.state == RobotState.DOCKED and watch.bout_open
                and adapter.battery_pct is not None
                and remaining is not None):
            drain = watch.drain_pct_per_m2()
            if drain:
                charge_model.drain_pct_per_m2 = drain
            plan = plan_resume(adapter.battery_pct, remaining, charge_model)
        # Blocked-area estimate: baseline cells lying inside a known
        # obstacle cluster's footprint (+ margin) are area the robot
        # can never clean; expectations must not charge time for them.
        blocked_m2 = None
        clusters = controller.clusters.clusters
        if home_cells and clusters:
            blocked_cells = 0
            for cell in home_cells:
                cx, cy = cell[0] * sup_mod.CELL_M, cell[1] * sup_mod.CELL_M
                for cl in clusters:
                    dx, dy = cx - cl.centre.x, cy - cl.centre.y
                    if dx * dx + dy * dy <= (cl.radius_m + 0.25) ** 2:
                        blocked_cells += 1
                        break
            if blocked_cells:
                blocked_m2 = blocked_cells * sup_mod.CELL_AREA_M2
        digest = watch.digest(
            time.time(), controller.phase.value,
            home_cells=len(home_cells) or None,
            robot_state=sample.state.value, charge_plan=plan,
            charge_model=charge_model, home_set=home_cells,
            blocked_m2=blocked_m2,
            charge_history=watch.charge_history())
        verdict = sup_mod.Verdict()
        try:
            text = await llm_mod.chat(sup_mod.build_messages(digest))
            verdict = sup_mod.parse_verdict(text)
        except Exception as exc:
            log(f"supervisor call failed: {exc}; continuing")
        log(f"supervisor verdict: {verdict.action} "
            f"reason={verdict.reason!r} resume_pct={verdict.resume_charge_pct}")
        if verdict.charge_model_update:
            # AI calibration from the raw history: adopt its values
            # (shape already validated), persist for every future
            # plan.  Applies in observe too - learning is not control.
            upd = verdict.charge_model_update
            if "band_rates" in upd:
                charge_model.band_rates = [tuple(b)
                                           for b in upd["band_rates"]]
            if "drain_pct_per_m2" in upd:
                charge_model.drain_pct_per_m2 = upd["drain_pct_per_m2"]
            _save_charge_state()
            log(f"charge model calibrated by supervisor: {upd}")
        if verdict.action == "notify" and verdict.message:
            log(f"NOTIFY: supervisor: {verdict.message}")
        elif (verdict.action == "ask_user"
                and sample.state != RobotState.RETURNING):
            # Hand the decision to the user (user 2026-11-11): pause,
            # notify with the choices, and wait ask_timeout_s.  The
            # user's own hand resolves it - resume (button/app) means
            # continue, sending it home means end; NO feedback within
            # the window defaults to return-to-dock and end.
            mins = max(1, round(sup_cfg.ask_timeout_s / 60))
            log(f"NOTIFY: 需要您决定：{verdict.message or verdict.reason}"
                f"（{mins} 分钟内按机身键或在 App 里恢复清扫=继续；"
                f"让它回站=结束；无反馈将自动回站结束）")
            state["awaiting_until"] = (time.monotonic()
                                       + sup_cfg.ask_timeout_s)
            state["awaiting_since"] = time.monotonic()
            state["awaiting_from"] = sample.state
            state["charge_target"] = None
            if mode == "active" and sample.state == RobotState.CLEANING:
                await dispatch([Command(CommandKind.PAUSE)])
            elif mode != "active":
                log("[observe] supervisor would pause and ask the user")
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
                # Stop cleaning AND send it home: a bare STOP leaves
                # the robot sitting wherever it was; the user rule is
                # "abnormal -> back to the station, do not keep going".
                await dispatch([Command(CommandKind.STOP),
                                Command(CommandKind.RETURN_TO_DOCK)])
            else:
                log("[observe] supervisor would stop the session "
                    "and return it to the dock")
        # Charge decision: an explicit verdict override wins; otherwise
        # adopt the deterministic solver's proposal from a charge review.
        if watch.charge_review_pending or plan is not None:
            target = verdict.resume_charge_pct
            if target is None and plan is not None:
                target = plan["resume_soc"]
            if watch.short_cycle_thrash():
                # Measurement only: the pattern is in the digest
                # (short_cycle_thrashing + segments) and the verdict
                # above already weighed it.  No hardcoded floor here -
                # what to do about it is the supervisor's conclusion,
                # not code's (user 2026-11-11: 不要写死).
                log("short-cycle thrash pattern present; resume "
                    "level left to the supervisor verdict")
            if target is not None and verdict.action == "continue":
                state["charge_target"] = target
                log(f"charge plan: resume at {target}% "
                    f"(plan={plan}, verdict_override={verdict.resume_charge_pct})")

    await adapter.connect(log)
    if sup_mod.adopt_device_tz(
            sup_cfg,
            config_had_key="tz_offset_hours" in raw.get("supervisor", {}),
            device_hours=adapter.device_tz_offset_hours):
        log(f"supervisor clock follows the robot: UTC"
            f"{sup_cfg.tz_offset_hours:+g}")
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
                                # Live session progress vs the
                                # ADOPTED home baseline (only once the
                                # baseline is mature; clamped at 100 -
                                # a foreign bout can outgrow it, and the
                                # digest flags that case separately).
                                "progress_pct": (
                                    round(min(100.0, 100.0 * len(watch.cells)
                                              / len(home_cells)), 1)
                                    if len(home_cells)
                                    >= sup_cfg.home_min_cache_cells
                                    else None),
                            }
                        )
                        + "\n"
                    )
                watch.feed(sample, controller.phase.value, wall,
                           battery=adapter.battery_pct,
                           water_amount=adapter.water_amount)
                if state["bout_was_open"] and not watch.bout_open:
                    _maintain_home_cache()
                state["bout_was_open"] = watch.bout_open
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
                        and state["awaiting_until"] is None
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
                # ask_user resolution: the user's own hand on the
                # robot (or in the app) is the answer; silence for
                # ask_timeout_s defaults to return-and-end (user
                # 2026-11-11: 5 minutes, no feedback -> 回站结束).
                if state["awaiting_until"] is not None:
                    waited = time.monotonic() - state["awaiting_since"]
                    frm = state["awaiting_from"]
                    user_continue = (
                        sample.state == RobotState.CLEANING
                        and (frm != RobotState.CLEANING or waited > 8))
                    user_end = (
                        waited > 8 and frm != RobotState.DOCKED
                        and sample.state != frm
                        and sample.state in (RobotState.RETURNING,
                                             RobotState.IDLE,
                                             RobotState.DOCKED))
                    if user_continue:
                        state["awaiting_until"] = None
                        log("user choice: resumed during the ask "
                            "window -> continue cleaning")
                    elif user_end:
                        state["awaiting_until"] = None
                        state["session_stopped"] = True
                        log("user choice: sent home during the ask "
                            "window -> end session")
                    elif time.monotonic() > state["awaiting_until"]:
                        state["awaiting_until"] = None
                        state["session_stopped"] = True
                        state["charge_target"] = None
                        log("NOTIFY: 等待选择超时（5 分钟无反馈），"
                            "已按默认回站结束本场清扫")
                        if mode == "active" and frm != RobotState.DOCKED:
                            await dispatch(
                                [Command(CommandKind.STOP),
                                 Command(CommandKind.RETURN_TO_DOCK)])
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
