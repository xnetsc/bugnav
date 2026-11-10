"""Optional LLM assistance for OFFLINE analysis (never in the control loop).

The real-time controller stays deterministic on purpose: cloud LLM calls
are far too slow and too failure-prone to sit in a robot's reflex path.
The LLM is used where latency does not matter - post-hoc analysis of the
recorded samples/events (episode recaps, obstacle guesses, detector
threshold advice). See tools/analyze_episodes.py.

Configuration is environment-only (credentials never live in code or in
config.json), against any OpenAI-compatible endpoint:

  BUGNAV_LLM_API_KEY   - required to enable the feature
  BUGNAV_LLM_BASE_URL  - default https://api.meta.ai/v1
  BUGNAV_LLM_MODEL     - default muse-spark-1.3
"""

from __future__ import annotations

import os

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"


class LLMNotConfigured(RuntimeError):
    """Raised when no API key is present in the environment."""


def llm_configured() -> bool:
    return bool(os.environ.get("BUGNAV_LLM_API_KEY"))


async def chat(messages: list[dict], *, timeout_s: float = 120.0) -> str:
    """One chat-completions round trip; returns the assistant text."""
    import aiohttp

    api_key = os.environ.get("BUGNAV_LLM_API_KEY")
    if not api_key:
        raise LLMNotConfigured(
            "BUGNAV_LLM_API_KEY is not set - LLM analysis is disabled")
    base = os.environ.get("BUGNAV_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("BUGNAV_LLM_MODEL", DEFAULT_MODEL)
    payload = {"model": model, "messages": messages}
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
                f"{base}/chat/completions", json=payload,
                headers=headers) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(
                    f"LLM endpoint returned {resp.status}: "
                    f"{str(data)[:300]}")
    return data["choices"][0]["message"]["content"]
