#!/usr/bin/env python3
"""bugnav notification bridge - runs on the Mac, OUTSIDE Docker.

bugnav itself runs in a container and can only write log files; this
small host-side watcher tails data/events.jsonl and turns every
NOTIFY line into:
  * a macOS Notification Center banner (via osascript), and
  * optionally an iMessage to yourself (set BUGNAV_IMESSAGE_TO to your
    own iMessage handle in the launchd plist) - which is what makes
    the PHONE buzz too, since macOS notifications alone do not sync
    to iOS.

A state file remembers the newest event timestamp already delivered,
so restarting the bridge neither replays old history nor loses
notifications raised while it was down.  On the very first run it
starts from "now" (history is not replayed).

Install (once, in Terminal):
  cp notify-bridge.py ~/bugnav/
  cp com.bugnav.notify-bridge.plist ~/Library/LaunchAgents/
  launchctl load ~/Library/LaunchAgents/com.bugnav.notify-bridge.plist
"""

from __future__ import annotations

import json
import os
import subprocess
import time

EVENTS_PATH = os.path.expanduser(
    os.environ.get("BUGNAV_EVENTS", "~/bugnav/data/events.jsonl"))
STATE_PATH = os.path.expanduser(
    os.environ.get("BUGNAV_BRIDGE_STATE",
                   "~/bugnav/data/.notify-bridge-state.json"))
IMESSAGE_TO = os.environ.get("BUGNAV_IMESSAGE_TO", "").strip()
TITLE = "bugnav"


def extract_notifications(lines, after_ts: float):
    """[(ts, text)] for NOTIFY event lines strictly newer than after_ts."""
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        msg = str(rec.get("msg", ""))
        try:
            ts = float(rec.get("ts", 0) or 0)
        except (TypeError, ValueError):
            continue
        if msg.startswith("NOTIFY") and ts > after_ts:
            out.append((ts, msg))
    return out


def send_macos(text: str) -> None:
    script = ("on run argv\n"
              "display notification (item 1 of argv) "
              f"with title \"{TITLE}\"\n"
              "end run")
    subprocess.run(["osascript", "-e", script, text],
                   check=False, capture_output=True, timeout=15)


def send_imessage(text: str) -> None:
    if not IMESSAGE_TO:
        return
    script = ("on run argv\n"
              "set theText to item 1 of argv\n"
              "set theHandle to item 2 of argv\n"
              "tell application \"Messages\"\n"
              "set svc to first service whose service type is iMessage\n"
              "send theText to buddy theHandle of svc\n"
              "end tell\n"
              "end run")
    subprocess.run(["osascript", "-e", script, text, IMESSAGE_TO],
                   check=False, capture_output=True, timeout=15)


def load_state() -> float | None:
    """Newest delivered ts, or None when never run (first install)."""
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return float(json.load(f).get("last_ts", 0))
    except Exception:
        return None


def save_state(ts: float) -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"last_ts": ts}, f)
    except Exception:
        pass


def deliver(text: str) -> None:
    for fn in (send_macos, send_imessage):
        try:
            fn(text)
        except Exception:
            pass  # a failed banner must never kill the watcher


def main() -> None:
    last_ts = load_state()
    if last_ts is None:
        last_ts = time.time()  # first install: do not replay history
        save_state(last_ts)
    pos = 0
    while True:
        try:
            if os.path.getsize(EVENTS_PATH) < pos:
                pos = 0  # truncated / rotated: start over
            with open(EVENTS_PATH, encoding="utf-8") as f:
                f.seek(pos)
                lines = f.readlines()
                pos = f.tell()
        except (FileNotFoundError, OSError):
            lines, pos = [], 0
        for ts, text in extract_notifications(lines, last_ts):
            deliver(text)
            last_ts = max(last_ts, ts)
            save_state(last_ts)
        time.sleep(2.0)


if __name__ == "__main__":
    main()
