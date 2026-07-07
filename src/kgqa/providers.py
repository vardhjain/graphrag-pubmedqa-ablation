"""Multi-provider LLM client with per-task defaults and automatic fallback.

Free-tier providers deprecate models without notice, so every call site picks
a *task* ("decompose", "extract", "synthesize"), not a provider directly. Each
task has a configured provider chain (primary, then fallbacks); if the primary
errors or its API key is unset, the next provider in the chain is tried.

Task defaults (overridable via env, see ``_CHAINS`` below):
  decompose / extract  -> Groq (fast, free, several calls per question)
  synthesize            -> Gemini Flash (bigger context for the retrieved subgraph)
Both fall back to a local Ollama call so the service still works with no cloud
API keys configured at all (e.g. in tests or offline dev).
"""

from __future__ import annotations

import os

import requests

from .config import LLM_TEMPERATURE
from .llm import call_ollama

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)


def call_groq(prompt: str, system: str = "", temperature: float = LLM_TEMPERATURE) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": temperature},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_gemini(prompt: str, system: str = "", temperature: float = LLM_TEMPERATURE) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    text = f"{system}\n\n{prompt}" if system else prompt
    resp = requests.post(
        GEMINI_API_URL,
        params={"key": GEMINI_API_KEY},
        json={
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"temperature": temperature},
        },
        timeout=60,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:500]}") from exc
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_ollama_task(prompt: str, system: str = "", temperature: float = LLM_TEMPERATURE) -> str:
    return call_ollama(prompt, system=system, temperature=temperature)


# Provider chains per task: (name, fn). Order matters -- first that succeeds wins.
_PROVIDERS = {
    "groq": call_groq,
    "gemini": call_gemini,
    "ollama": _call_ollama_task,
}

_CHAINS = {
    "decompose": os.environ.get("LLM_CHAIN_DECOMPOSE", "groq,ollama").split(","),
    "extract": os.environ.get("LLM_CHAIN_EXTRACT", "groq,ollama").split(","),
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
