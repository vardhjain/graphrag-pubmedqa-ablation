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


def call_ollama(
    prompt: str,
    system: str = "",
    temperature: float = LLM_TEMPERATURE,
    model: str = LLM_MODEL,
    api_url: str = OLLAMA_API,
) -> str:
    """Single synchronous chat completion against a local Ollama server."""
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
    return resp.json()["message"]["content"]
