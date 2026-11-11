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
from datetime import datetime, timedelta, timezone

from . import llm as llm_mod
from . import strategist, supervisor as sup_mod
from .chargeplan import ChargeModel, plan_resume
from .controller import BugNavController, ControllerConfig
from .deebot_adapter import AdapterConfig, DeebotAdapter
from .model import Command, CommandKind, RobotState
from .network import is_transient_network_error, retry_forever


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _posix_tz(offset_hours: float) -> str:
    """A POSIX TZ string for a UTC offset (e.g. +8 -> 'LOC-8'):
    the POSIX value is what you ADD to local time to get UTC, the
    negation of the usual offset.  Built from the robot-reported
    offset at runtime - no zone is hardcoded anywhere (user
    2026-11-11: 容器启动时自动探测时区，不要写死)."""
    total_min = int(round(-offset_hours * 60))
    sign = "-" if total_min < 0 else "+"
    hh, mm = divmod(abs(total_min), 60)
    if mm == 0:
        return f"LOC{sign}{hh}" if total_min else "LOC0"
    return f"LOC{sign}{hh}:{mm:02d}"


def log_stamp(device_epoch: float | None,
              tz_offset_hours: float) -> str | None:
    """HH:MM:SS for a log line, rendered from the COMPUTED time -
    the robot's clock in the robot's zone - or None when no device
    clock basis has been captured yet (the caller then falls back
    to the process clock for the first startup lines only).

    Log timestamps must come from the same computed time everything
    else uses, never from the process-local clock: a container
    without TZ configured printed UTC stamps next to verdicts that
    (correctly) reasoned in local time (user 2026-11-11: 改为计算
    后的时间做时间戳打印)."""
    if device_epoch is None:
        return None
    tz = timezone(timedelta(hours=tz_offset_hours))
    return datetime.fromtimestamp(device_epoch, tz).strftime("%H:%M:%S")


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

    # Lines logged before the robot clock lands are BUFFERED with
    # their monotonic instant and stamped retroactively on the
    # robot's timeline once the basis exists (device_now_at projects
    # backwards) - so no zone needs to be preset anywhere for the
    # startup lines to come out in household time either (user
    # 2026-11-11: 自动探测、不要写死).  If no header ever arrives
    # (connect failure), the buffer flushes with the container-local
    # fallback instead of being lost.
    pending_stamps: list[tuple[float, str]] = []
    clock_settled = {"done": False}

    def _emit(stamp: str, epoch: float, msg: str) -> None:
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": epoch, "msg": msg}) + "\n")
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

    def _flush_pending_fallback() -> None:
        stamp, epoch = time.strftime("%H:%M:%S"), time.time()
        for _, old in pending_stamps:
            _emit(stamp, epoch, old)
        pending_stamps.clear()

    def log(msg: str) -> None:
        dn = adapter.device_now()
        if pending_stamps and dn is not None:
            hours = (adapter.device_tz_offset_hours
                     if adapter.device_tz_offset_hours is not None
                     else sup_cfg.tz_offset_hours)
            for t_mono, old in pending_stamps:
                e = adapter.device_now_at(t_mono)
                _emit(log_stamp(e, hours), e, old)
            pending_stamps.clear()
        if dn is not None:
            _emit(log_stamp(dn, sup_cfg.tz_offset_hours), dn, msg)
        elif not clock_settled["done"] and len(pending_stamps) < 50:
            pending_stamps.append((time.monotonic(), msg))
        else:
            # No robot clock: the fallback is the container's
            # local time (itself adopted from the robot after a
            # successful connect - see the startup block).
            if pending_stamps:
                _flush_pending_fallback()
            _emit(time.strftime("%H:%M:%S"), time.time(), msg)

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
                    if is_transient_network_error(exc):
                        log(f"command {cmd.kind} failed after retries "
                            f"(network still down); continuing")
                    else:
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
            _wall(), controller.phase.value,
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

    # Background tasks (the library's refresh/event machinery, our
    # consult/review services) must not dump a traceback for a cloud
    # hiccup: transient network failures become one log line;
    # anything else keeps the default handler with full detail.
    def _on_loop_error(loop_, context) -> None:
        exc = context.get("exception")
        if exc is not None and is_transient_network_error(exc):
            log(f"background task hit a network error "
                f"({type(exc).__name__}); requests keep cycling "
                f"and recover when the network returns")
        else:
            loop_.default_exception_handler(context)

    asyncio.get_running_loop().set_exception_handler(_on_loop_error)

    async def _connect_once() -> None:
        # A fresh adapter per attempt: a connect that died halfway
        # can leave a half-initialised device/MQTT client behind,
        # and retrying on top of it risks duplicate subscriptions.
        nonlocal adapter
        adapter = DeebotAdapter(AdapterConfig(**account))
        await adapter.connect(log)

    try:
        # A network error during the initial connect is retried at
        # a paced backoff (15s -> 30s -> 60s cap) instead of
        # crash-looping the container; auth/protocol failures still
        # raise immediately (see bugnav/network.py).
        await retry_forever(
            _connect_once, log=log,
            what="connecting to the Ecovacs cloud")
    finally:
        clock_settled["done"] = True
        if pending_stamps and adapter.device_now() is None:
            _flush_pending_fallback()

    def _wall() -> float:
        """Wall clock for everything the AI (or the user, via a
        notification) reads as 'now': the ROBOT's own time - its
        response-header timestamp projected on the monotonic clock.
        The host's wall clock is never the basis (user 2026-11-11:
        AI 的时钟以扫地机自己上报的时间为准，不可以写死).  Falls
        back to the host clock only until the first header lands."""
        dn = adapter.device_now()
        return dn if dn is not None else time.time()

    if sup_mod.adopt_device_tz(
            sup_cfg,
            config_had_key="tz_offset_hours" in raw.get("supervisor", {}),
            device_hours=adapter.device_tz_offset_hours):
        log(f"supervisor clock follows the robot: UTC"
            f"{sup_cfg.tz_offset_hours:+g}")
    # The container's own local time adopts the same zone, so no TZ
    # needs to be preset in the deployment at all: from here on,
    # even the local-time fallback speaks the robot's zone.
    if hasattr(time, "tzset"):
        os.environ["TZ"] = _posix_tz(sup_cfg.tz_offset_hours)
        time.tzset()
        log(f"container local time adopted from the robot: UTC"
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
                wall = _wall()
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
