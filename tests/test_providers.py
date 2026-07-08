"""Tests for the multi-provider LLM chain -- all HTTP calls are faked."""

from __future__ import annotations

import pytest


def test_call_gemini_builds_payload_and_returns_content(monkeypatch):
    import kgqa.providers as providers

    monkeypatch.setattr(providers, "GEMINI_API_KEY", "test-key")

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "gemini answer"}]}}]}

    def fake_post(url, params=None, json=None, timeout=None):
        assert params == {"key": "test-key"}
        return FakeResp()

    monkeypatch.setattr(providers.requests, "post", fake_post)
    out = providers.call_gemini("prompt", system="sys")
    assert out == "gemini answer"


def test_call_llm_falls_back_when_primary_provider_fails(monkeypatch):
    import kgqa.providers as providers

    def broken(*args, **kwargs):
        raise RuntimeError("gemini is down")

    def works(*args, **kwargs):
        return "fallback answer"

    monkeypatch.setitem(providers._PROVIDERS, "gemini", broken)
    monkeypatch.setitem(providers._PROVIDERS, "ollama", works)
    monkeypatch.setitem(providers._CHAINS, "synthesize", ["gemini", "ollama"])

    assert providers.call_llm("synthesize", "prompt") == "fallback answer"


def test_call_llm_raises_when_all_providers_fail(monkeypatch):
    import kgqa.providers as providers

    def broken(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setitem(providers._PROVIDERS, "gemini", broken)
    monkeypatch.setitem(providers._PROVIDERS, "ollama", broken)
    monkeypatch.setitem(providers._CHAINS, "synthesize", ["gemini", "ollama"])

    with pytest.raises(RuntimeError, match="All providers failed"):
        providers.call_llm("synthesize", "prompt")
