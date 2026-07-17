"""Multi-provider LLM client with per-task defaults and automatic fallback.

Free-tier providers deprecate models without notice, so every call site picks
a *task* (currently just "synthesize"), not a provider directly. Each task has
a configured provider chain (primary, then fallbacks); if the primary errors
or its API key is unset, the next provider in the chain is tried.

Task default (overridable via env, see ``_CHAINS`` below):
  synthesize -> Gemini Flash (bigger context for the retrieved subgraph)
Falls back to a local Ollama call so the service still works with no cloud
API keys configured at all (e.g. in tests or offline dev).
"""

from __future__ import annotations

import os
import time

import requests

from .config import LLM_TEMPERATURE
from .llm import call_ollama

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)


# Gemini's free tier answers 429 (rate limit) and 503 ("model is currently
# experiencing high demand") often enough that a single attempt makes the
# hosted demo look broken for reasons that have nothing to do with it -- seen
# live. These are transient by definition, so retry them; anything else (401,
# 400, a bad model name) is a real error and fails immediately.
_GEMINI_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_GEMINI_MAX_ATTEMPTS = 3
_GEMINI_BACKOFF_SECONDS = 2.0


def call_gemini(prompt: str, system: str = "", temperature: float = LLM_TEMPERATURE) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    text = f"{system}\n\n{prompt}" if system else prompt
    last_error: Exception | None = None

    for attempt in range(_GEMINI_MAX_ATTEMPTS):
        if attempt:
            # Exponential: 2s, then 4s. Bounded well under the client's own
            # timeout budget so a retry can't turn into a hang.
            time.sleep(_GEMINI_BACKOFF_SECONDS * (2 ** (attempt - 1)))
        try:
            resp = requests.post(
                GEMINI_API_URL,
                # Header, not a query param: keys in URLs leak into proxy logs
                # and request traces.
                headers={"x-goog-api-key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": text}]}],
                    "generationConfig": {"temperature": temperature},
                },
                timeout=60,
            )
        except requests.RequestException as exc:  # connection reset, read timeout, ...
            last_error = RuntimeError(f"Gemini request failed: {exc}")
            continue

        if resp.status_code in _GEMINI_RETRY_STATUSES:
            last_error = RuntimeError(f"Gemini {resp.status_code}: {resp.text[:500]}")
            continue
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:500]}") from exc
        return _gemini_text(resp.json())

    raise last_error if last_error else RuntimeError("Gemini call failed")


def _gemini_text(payload: dict) -> str:
    """Pull the answer out of a Gemini response, naming what went wrong.

    A 200 doesn't guarantee text: the model can return a candidate with no
    parts when it stops for safety filtering or hits the token cap. Indexing
    blindly turns that into an opaque KeyError/IndexError in the logs, so
    surface ``finishReason`` instead -- it's the difference between a
    diagnosable failure and a mystery.
    """
    candidates = payload.get("candidates") or []
    if not candidates:
        feedback = payload.get("promptFeedback", {})
        raise RuntimeError(f"Gemini returned no candidates (promptFeedback={feedback})")
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts") or []
    if not parts:
        reason = candidate.get("finishReason", "unknown")
        raise RuntimeError(f"Gemini returned no text (finishReason={reason})")
    return parts[0]["text"]


def _call_ollama_task(prompt: str, system: str = "", temperature: float = LLM_TEMPERATURE) -> str:
    return call_ollama(prompt, system=system, temperature=temperature)


# Provider chains per task: (name, fn). Order matters -- first that succeeds wins.
_PROVIDERS = {
    "gemini": call_gemini,
    "ollama": _call_ollama_task,
}

_CHAINS = {
    "synthesize": os.environ.get("LLM_CHAIN_SYNTHESIZE", "gemini,ollama").split(","),
}


def call_llm(task: str, prompt: str, system: str = "", temperature: float = LLM_TEMPERATURE) -> str:
    """Run ``prompt`` through the provider chain configured for ``task``.

    Tries each provider in order, falling back on any exception (missing key,
    network error, rate limit) so a single provider outage or deprecation
    doesn't take the whole service down.
    """
    chain = _CHAINS.get(task, ["ollama"])
    errors = []
    for name in chain:
        fn = _PROVIDERS.get(name.strip())
        if fn is None:
            continue
        try:
            return fn(prompt, system=system, temperature=temperature)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a fallback chain
            errors.append(f"{name}: {exc}")
    raise RuntimeError(f"All providers failed for task '{task}': {'; '.join(errors)}")
