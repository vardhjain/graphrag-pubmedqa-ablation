"""Tests for the Ollama client — requests.post is faked, so no server is needed."""

from __future__ import annotations


def test_call_ollama_builds_payload_and_returns_content(monkeypatch):
    import kgqa.llm as llm

    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "the answer"}}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr(llm.requests, "post", fake_post)

    out = llm.call_ollama("my prompt", system="be helpful", temperature=0.0)
    assert out == "the answer"

    payload = captured["payload"]
    assert payload["messages"][0] == {"role": "system", "content": "be helpful"}
    assert payload["messages"][-1] == {"role": "user", "content": "my prompt"}
    assert payload["stream"] is False
    assert "num_predict" in payload["options"]   # generation cap is applied
    assert "keep_alive" in payload                # model kept resident
