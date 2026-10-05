"""Tests for the multi-provider LLM chain -- all HTTP calls are faked."""

from __future__ import annotations

import pytest


class FakeResp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, payload=None, status_code=200, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


def _ok_payload(text="gemini answer"):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Keep the retry tests instant -- the backoff itself isn't under test."""
    import kgqa.providers as providers

    monkeypatch.setattr(providers.time, "sleep", lambda _s: None)


def test_call_gemini_sends_key_as_header_and_returns_content(monkeypatch):
    """The key goes in a header, not a query param -- URLs leak into logs."""
    import kgqa.providers as providers

    monkeypatch.setattr(providers, "GEMINI_API_KEY", "test-key")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["headers"] = headers
        seen["url"] = url
        return FakeResp(_ok_payload())

    monkeypatch.setattr(providers.requests, "post", fake_post)

    assert providers.call_gemini("prompt", system="sys") == "gemini answer"
    assert seen["headers"] == {"x-goog-api-key": "test-key"}
    assert "test-key" not in seen["url"]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_call_gemini_retries_transient_errors_then_succeeds(monkeypatch, status):
    """A transient 429/5xx must not surface as a hard failure -- this is the
    live-observed 'Gemini 503: high demand' case that 502'd the demo."""
    import kgqa.providers as providers

    monkeypatch.setattr(providers, "GEMINI_API_KEY", "test-key")
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            return FakeResp(status_code=status, text="high demand")
        return FakeResp(_ok_payload())

    monkeypatch.setattr(providers.requests, "post", fake_post)

    assert providers.call_gemini("prompt") == "gemini answer"
    assert len(calls) == 2  # retried once, then succeeded


def test_call_gemini_gives_up_after_max_attempts(monkeypatch):
    import kgqa.providers as providers

    monkeypatch.setattr(providers, "GEMINI_API_KEY", "test-key")
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return FakeResp(status_code=503, text="high demand")

    monkeypatch.setattr(providers.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="503"):
        providers.call_gemini("prompt")
    assert len(calls) == providers._GEMINI_MAX_ATTEMPTS


def test_call_gemini_does_not_retry_client_errors(monkeypatch):
    """A 401/400 is a real bug (bad key, bad model) -- retrying just delays
    the error and burns the request budget."""
    import kgqa.providers as providers

    monkeypatch.setattr(providers, "GEMINI_API_KEY", "bad-key")
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(1)
        return FakeResp(status_code=401, text="invalid key")

    monkeypatch.setattr(providers.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="401"):
        providers.call_gemini("prompt")
    assert len(calls) == 1


def test_call_gemini_names_finish_reason_when_response_has_no_text(monkeypatch):
    """A 200 with a safety-blocked candidate must fail with a diagnosable
    message, not an opaque IndexError."""
    import kgqa.providers as providers

    monkeypatch.setattr(providers, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        providers.requests,
        "post",
        lambda *a, **k: FakeResp({"candidates": [{"finishReason": "SAFETY"}]}),
    )

    with pytest.raises(RuntimeError, match="SAFETY"):
        providers.call_gemini("prompt")


def test_call_gemini_raises_when_no_candidates(monkeypatch):
    import kgqa.providers as providers

    monkeypatch.setattr(providers, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        providers.requests,
        "post",
        lambda *a, **k: FakeResp({"promptFeedback": {"blockReason": "OTHER"}}),
    )

    with pytest.raises(RuntimeError, match="no candidates"):
        providers.call_gemini("prompt")


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


def test_gemini_text_joins_non_thought_parts():
    import kgqa.providers as providers

    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "internal reasoning", "thought": True},
                        {"text": "Yes, "},
                        {"text": "it does."},
                    ]
                }
            }
        ]
    }
    assert providers._gemini_text(payload) == "Yes, it does."


def test_gemini_text_raises_when_only_thoughts():
    import kgqa.providers as providers

    payload = {
        "candidates": [
            {
                "finishReason": "MAX_TOKENS",
                "content": {"parts": [{"text": "thinking...", "thought": True}]},
            }
        ]
    }
    with pytest.raises(RuntimeError, match="MAX_TOKENS"):
        providers._gemini_text(payload)


def test_call_gemini_does_not_retry_read_timeout(monkeypatch):
    import requests

    import kgqa.providers as providers

    monkeypatch.setattr(providers, "GEMINI_API_KEY", "k")
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        raise requests.exceptions.ReadTimeout("stalled")

    monkeypatch.setattr(providers.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="timed out"):
        providers.call_gemini("prompt")
    assert len(calls) == 1


def test_call_gemini_respects_overall_deadline(monkeypatch):
    import kgqa.providers as providers

    monkeypatch.setattr(providers, "GEMINI_API_KEY", "k")
    clock = [0.0]
    monkeypatch.setattr(providers.time, "monotonic", lambda: clock[0])
    timeouts = []

    def fake_post(url, headers=None, json=None, timeout=None):
        timeouts.append(timeout)
        clock[0] += 80  # each attempt eats most of the budget
        return FakeResp(status_code=503, text="busy")

    monkeypatch.setattr(providers.requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="503"):
        providers.call_gemini("prompt")
    # Second attempt only gets what's left of the 90s budget; no third.
    assert timeouts == [60.0, 10.0]
    assert sum(timeouts) <= providers._GEMINI_DEADLINE_SECONDS
