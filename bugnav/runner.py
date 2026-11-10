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

from .controller import BugNavController, ControllerConfig
from .deebot_adapter import AdapterConfig, DeebotAdapter
from .model import CommandKind


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
        if k == "detector":
            for dk, dv in v.items():
                setattr(ctl_cfg.detector, dk, dv)
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

    await adapter.connect(log)
    log(f"runner started in mode={mode}, poll={poll_s}s")
    last_refresh = 0.0
    while True:
        now = time.monotonic()
        if now - last_refresh >= poll_s:
            last_refresh = now
            adapter.request_refresh()
            sample = adapter.sample(now)
            if sample is not None:
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
                            }
                        )
                        + "\n"
                    )
                for cmd in controller.tick(sample):
                    if cmd.kind == CommandKind.NOTIFY:
                        log(f"NOTIFY: {cmd.message}")
                    elif mode == "active":
                        try:
                            await adapter.execute(cmd, log)
                        except Exception as exc:  # never crash the daemon
                            log(f"command {cmd.kind} failed: {exc}")
                    else:
                        log(f"[observe] would send {cmd.kind} {cmd.point or ''}")
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
