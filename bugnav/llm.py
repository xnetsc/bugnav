"""LLM access for the two judgement layers (supervisor, strategist).

The real-time controller stays deterministic on purpose: cloud LLM
calls are far too slow and too failure-prone to sit in a robot's
reflex path.  The LLM is used where latency does not matter - the
supervisor's periodic whole-bout verdicts and the strategist's
per-episode consults.

Configuration is environment-only (credentials never live in code or in
config.json), against any OpenAI-compatible endpoint:

  BUGNAV_LLM_API_KEY   - required to enable the feature
  BUGNAV_LLM_BASE_URL  - default https://api.meta.ai/v1
  BUGNAV_LLM_MODEL     - default muse-spark-1.3
"""

from __future__ import annotations

import json
import os

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"


class LLMNotConfigured(RuntimeError):
    """Raised when no API key is present in the environment."""


def llm_configured() -> bool:
    return bool(os.environ.get("BUGNAV_LLM_API_KEY"))


def build_payload(model: str, messages: list[dict], *,
                  json_mode: bool = True) -> dict:
    """Chat-completions payload.  Every bugnav consumer parses the
    reply as JSON (advisor Plan, supervisor Verdict), so JSON mode is
    enforced at the API level via response_format - not merely
    requested in the prompt .  Both prompts do
    mention JSON, which json_object mode requires."""
    payload = {"model": model, "messages": messages}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


async def chat(messages: list[dict], *, timeout_s: float = 120.0) -> str:
    """One chat-completions round trip; returns the assistant text."""
    import aiohttp

    api_key = os.environ.get("BUGNAV_LLM_API_KEY")
    if not api_key:
        raise LLMNotConfigured(
            "BUGNAV_LLM_API_KEY is not set - LLM analysis is disabled")
    base = os.environ.get("BUGNAV_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("BUGNAV_LLM_MODEL", DEFAULT_MODEL)
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for json_mode in (True, False):
            payload = build_payload(model, messages, json_mode=json_mode)
            async with session.post(
                    f"{base}/chat/completions", json=payload,
                    headers=headers) as resp:
                data = await resp.json()
                if resp.status == 200:
                    return data["choices"][0]["message"]["content"]
                # An endpoint that does not know response_format
                # rejects the request outright; retry once without
                # it rather than losing the whole AI layer (the
                # prompts still demand JSON and the parsers stay
                # tolerant).
                if not (json_mode and resp.status == 400
                        and "response_format" in str(data)):
                    raise RuntimeError(
                        f"LLM endpoint returned {resp.status}: "
                        f"{str(data)[:300]}")
    raise RuntimeError("LLM request failed")  # pragma: no cover


class DecisionTrace:
    """Append-only JSONL record of LLM decisions: the exact messages
    sent, the raw reply and the parsed result, one line per call.
    This is the input set for auditing verdict quality and for
    replaying candidate models against identical inputs offline.
    Recording is best-effort: a failed write never disturbs a
    decision path."""

    def __init__(self, path: str):
        self.path = path

    def record(self, kind: str, messages: list[dict], response: str,
               parsed: dict | None, *, ts: float,
               trigger: str | None = None) -> None:
        try:
            rec = {"ts": ts, "kind": kind, "trigger": trigger,
                   "model": os.environ.get(
                       "BUGNAV_LLM_MODEL", DEFAULT_MODEL),
                   "messages": messages, "response": response,
                   "parsed": parsed}
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
