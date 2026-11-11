"""Transient network failures on the Ecovacs cloud link are
classified, logged as one line and retried at a paced backoff -
never a traceback, never a storm (user 2026-11-11)."""

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
