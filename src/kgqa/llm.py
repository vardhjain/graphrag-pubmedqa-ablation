"""Thin Ollama client — the single LLM entry point shared by all arms."""

from __future__ import annotations

import requests

from .config import (
    LLM_KEEP_ALIVE,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OLLAMA_API,
)


def _post_chat(prompt, system, temperature, model, api_url) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": LLM_KEEP_ALIVE,  # keep the model resident across the run
        "options": {
            "temperature": temperature,
            "num_ctx": LLM_NUM_CTX,
            "num_predict": LLM_NUM_PREDICT,  # cap generation so a call can't run away
        },
    }
    resp = requests.post(api_url, json=payload, timeout=LLM_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def call_ollama(
    prompt: str,
    system: str = "",
    temperature: float = LLM_TEMPERATURE,
    model: str = LLM_MODEL,
    api_url: str = OLLAMA_API,
) -> str:
    """Single synchronous chat completion against a local Ollama server."""
    return _post_chat(prompt, system, temperature, model, api_url)["message"]["content"]


def call_ollama_meta(
    prompt: str,
    system: str = "",
    temperature: float = LLM_TEMPERATURE,
    model: str = LLM_MODEL,
    api_url: str = OLLAMA_API,
) -> tuple[str, str | None]:
    """Like :func:`call_ollama` but also returns Ollama's ``done_reason``.

    ``done_reason == "length"`` means generation hit ``num_predict`` (i.e. the
    answer was truncated). ``None`` if the server didn't report one.
    """
    data = _post_chat(prompt, system, temperature, model, api_url)
    return data["message"]["content"], data.get("done_reason")
