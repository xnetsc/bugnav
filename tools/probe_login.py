"""One-off READ-ONLY probe: log in, list devices, listen to positions.

Credentials come from env vars only (ECOVACS_ACCOUNT_ID / ECOVACS_PASSWORD).
No clean commands are ever executed - subscriptions and GetPos refresh
(reads) only. Usage:
  ECOVACS_ACCOUNT_ID=EXXXXXXX ECOVACS_PASSWORD=... python tools/probe_login.py [seconds]
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bugnav.deebot_adapter import AdapterConfig, DeebotAdapter


async def main(seconds: int) -> None:
    account = os.environ["ECOVACS_ACCOUNT_ID"]
    adapter = DeebotAdapter(AdapterConfig(account_id=account, device_name=""))
    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    await adapter.connect(log)
    log("login + MQTT init OK")
    t0 = time.monotonic()
    seen = 0
    last = None
    while time.monotonic() - t0 < seconds:
        adapter.request_refresh()
        s = adapter.sample(time.monotonic())
        if s is not None and (last is None or s.pos != last.pos or s.state != last.state):
            seen += 1
            last = s
            log(f"sample: pos=({s.pos.x:.3f},{s.pos.y:.3f}) raw-heading={s.heading_deg} "
                f"state={s.state.value} error={s.error_code}")
        await asyncio.sleep(2.5)
    log(f"probe done: {seen} distinct samples over {seconds}s")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 40))
