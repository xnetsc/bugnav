"""Transient network failures on the Ecovacs cloud link are
classified, logged as one line and retried at a paced backoff -
never a traceback, never a storm."""

import asyncio
import socket
import unittest

import aiohttp

from bugnav.network import (
    call_with_retry,
    is_transient_network_error,
    retry_forever,
)


class TestClassification(unittest.TestCase):
    def test_transient_family(self):
        self.assertTrue(is_transient_network_error(
            aiohttp.ClientConnectionError()))
        self.assertTrue(is_transient_network_error(
            aiohttp.ServerDisconnectedError()))
        self.assertTrue(is_transient_network_error(
            ConnectionError("reset by peer")))
        self.assertTrue(is_transient_network_error(TimeoutError()))
        self.assertTrue(is_transient_network_error(
            socket.gaierror("dns failed")))

    def test_not_transient(self):
        self.assertFalse(is_transient_network_error(
            ValueError("bad config")))
        self.assertFalse(is_transient_network_error(
            RuntimeError("firmware said 20011")))


class _Sleep:
    def __init__(self):
        self.delays = []

    async def __call__(self, delay):
        self.delays.append(delay)


class TestCallWithRetry(unittest.TestCase):
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro) \
            if False else asyncio.run(coro)

    def test_succeeds_after_two_hiccups(self):
        calls = {"n": 0}

        async def factory():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("down")
            return 42

        sleep = _Sleep()
        logs = []
        result = self._run(call_with_retry(
            factory, log=logs.append, what="command GetPos",
            sleep=sleep))
        self.assertEqual(result, 42)
        self.assertEqual(sleep.delays, [2.0, 5.0])
        self.assertEqual(len(logs), 2)

    def test_exhaustion_reraises(self):
        async def factory():
            raise ConnectionError("still down")

        sleep = _Sleep()
        with self.assertRaises(ConnectionError):
            self._run(call_with_retry(
                factory, log=lambda m: None, what="command",
                sleep=sleep))
        self.assertEqual(sleep.delays, [2.0, 5.0, 15.0])

    def test_non_transient_raises_at_once(self):
        async def factory():
            raise ValueError("nope")

        sleep = _Sleep()
        with self.assertRaises(ValueError):
            self._run(call_with_retry(
                factory, log=lambda m: None, what="command",
                sleep=sleep))
        self.assertEqual(sleep.delays, [])


class TestRetryForever(unittest.TestCase):
    def test_backoff_caps_and_eventual_success(self):
        calls = {"n": 0}

        async def factory():
            calls["n"] += 1
            if calls["n"] <= 4:
                raise aiohttp.ClientConnectionError()
            return "connected"

        sleep = _Sleep()
        result = asyncio.run(retry_forever(
            factory, log=lambda m: None, what="connecting",
            sleep=sleep))
        self.assertEqual(result, "connected")
        self.assertEqual(sleep.delays, [15.0, 30.0, 60.0, 60.0])

    def test_non_transient_raises_at_once(self):
        async def factory():
            raise RuntimeError("bad password")

        sleep = _Sleep()
        with self.assertRaises(RuntimeError):
            asyncio.run(retry_forever(
                factory, log=lambda m: None, what="connecting",
                sleep=sleep))
        self.assertEqual(sleep.delays, [])


if __name__ == "__main__":
    unittest.main()


class TestLinkWatchdog(unittest.TestCase):
    def setUp(self):
        from bugnav.network import LinkWatchdog
        self.wd = LinkWatchdog(ping_interval_s=60.0,
                               fails_to_reconnect=3,
                               contact_fresh_s=120.0)

    def test_ping_cadence(self):
        self.assertTrue(self.wd.ping_due(0.0))
        self.wd.note_ping_started(0.0)
        self.assertFalse(self.wd.ping_due(59.0))
        self.assertTrue(self.wd.ping_due(60.0))

    def test_three_failures_trigger_reconnect(self):
        stale = None
        self.assertFalse(self.wd.note_result(
            0.0, ping_ok=False, last_contact_t=stale))
        self.assertFalse(self.wd.note_result(
            60.0, ping_ok=False, last_contact_t=stale))
        self.assertTrue(self.wd.note_result(
            120.0, ping_ok=False, last_contact_t=stale))

    def test_success_resets_failure_count(self):
        self.wd.note_result(0.0, ping_ok=False, last_contact_t=None)
        self.wd.note_result(60.0, ping_ok=False, last_contact_t=None)
        self.assertFalse(self.wd.note_result(
            120.0, ping_ok=True, last_contact_t=None))
        self.assertFalse(self.wd.note_result(
            180.0, ping_ok=False, last_contact_t=None))
        self.assertFalse(self.wd.note_result(
            240.0, ping_ok=False, last_contact_t=None))

    def test_fresh_events_rescue_a_failed_ping(self):
        # The ping raced a reconnect inside the library, but the
        # robot pushed an event 10s ago: the link is proven alive.
        self.wd.note_result(0.0, ping_ok=False, last_contact_t=None)
        self.wd.note_result(60.0, ping_ok=False, last_contact_t=None)
        self.assertFalse(self.wd.note_result(
            1000.0, ping_ok=False, last_contact_t=990.0))

    def test_recovered_resets(self):
        for i in range(3):
            self.wd.note_result(60.0 * i, ping_ok=False,
                                last_contact_t=None)
        self.wd.note_recovered()
        self.assertFalse(self.wd.note_result(
            1000.0, ping_ok=False, last_contact_t=None))
        self.assertTrue(self.wd.ping_due(1000.0))
