"""Transient network failure handling for the Ecovacs cloud link.

The daemon reaches the cloud over the household internet and the
Mac's proxy; that path hiccups (proxy restarts, router reboots,
cloud blips).  A dropped request must become a one-line event with
a paced retry - never an unhandled traceback, never a dead process
(user 2026-11-11: 网络错误要处理这个异常，保持合理频率的重试).

Pacing is deliberately gentle.  The cloud account must never see
a login or command storm: frequent re-logins are exactly what
triggers new-device verification on this account."""

from __future__ import annotations

import asyncio
import socket

import aiohttp

# aiohttp.ClientError covers ClientConnectorError (connect/DNS),
# ServerDisconnectedError and response errors; ConnectionError and
# TimeoutError cover the stdlib paths; gaierror is DNS at socket
# level.  Anything else - auth failures, firmware rejections like
# 20011 - is NOT transient and must surface, not be retried.
_TRANSIENT_TYPES = (
    aiohttp.ClientError,
    ConnectionError,
    TimeoutError,
    socket.gaierror,
)


def is_transient_network_error(exc: BaseException) -> bool:
    return isinstance(exc, _TRANSIENT_TYPES)


async def call_with_retry(factory, *, log, what: str,
                          delays=(2.0, 5.0, 15.0),
                          sleep=asyncio.sleep):
    """Await factory(), retrying TRANSIENT failures after each of
    `delays` in turn.  Re-raises the last error when the delays run
    out, and any non-transient error immediately.  Sized for robot
    commands: a ~22s window rides out a proxy restart; a longer
    outage is the caller's next decision cycle, not a spin here."""
    attempt = 0
    while True:
        try:
            return await factory()
        except Exception as exc:
            if (not is_transient_network_error(exc)
                    or attempt >= len(delays)):
                raise
            delay = delays[attempt]
            attempt += 1
            log(f"{what}: network error ({type(exc).__name__}); "
                f"retry {attempt}/{len(delays)} in {delay:.0f}s")
            await sleep(delay)


async def retry_forever(factory, *, log, what: str,
                        base_delay: float = 15.0,
                        max_delay: float = 60.0,
                        sleep=asyncio.sleep):
    """Await factory(), retrying TRANSIENT failures forever with
    backoff base -> x2 -> capped at max_delay.  For the initial
    cloud connect: a home daemon must survive a router reboot or a
    cloud outage with nobody restarting it, and one attempt per
    minute at the cap is a reasonable frequency that cannot trip
    the account's new-device verification.  Non-transient errors
    (bad password, unsupported device) raise at once."""
    attempt = 0
    while True:
        try:
            return await factory()
        except Exception as exc:
            if not is_transient_network_error(exc):
                raise
            attempt += 1
            delay = min(max_delay,
                        base_delay * (2 ** min(attempt - 1, 2)))
            log(f"{what}: network error ({type(exc).__name__}); "
                f"attempt {attempt} failed, next try in {delay:.0f}s")
            await sleep(delay)
