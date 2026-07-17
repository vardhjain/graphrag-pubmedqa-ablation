"""Tests for the FastAPI backend -- graphrag.answer is monkeypatched, no live
ArangoDB/LLM calls. Run with: pytest backend/test_main.py
"""

from __future__ import annotations

import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["demo_graph_loaded"], bool)


def test_health_reports_demo_graph_loaded_state(monkeypatch):
    """demo_graph_loaded reads the existing cache and must never trigger a
    load itself -- it's a readiness signal, not a warm-up trigger."""
    import kgqa.service as service

    monkeypatch.setitem(service._STORE_CACHE, "demo", object())
    assert client.get("/health").json()["demo_graph_loaded"] is True

    monkeypatch.delitem(service._STORE_CACHE, "demo", raising=False)
    assert client.get("/health").json()["demo_graph_loaded"] is False


def test_warm_up_in_background_is_off_by_default(monkeypatch):
    """Importing/running the app must never trigger a real model/DB load
    just because a test module happened to import it -- opt-in only."""
    import backend.main as main

    monkeypatch.delenv("KGQA_WARM_ON_STARTUP", raising=False)
    calls = []
    monkeypatch.setattr(threading, "Thread", lambda *a, **k: calls.append((a, k)))

    main._maybe_warm_up_in_background()

    assert calls == []


def test_warm_up_in_background_starts_a_thread_when_opted_in(monkeypatch):
    import backend.main as main

    monkeypatch.setenv("KGQA_WARM_ON_STARTUP", "true")
    started = {}

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            started["target"] = target
            started["daemon"] = daemon

        def start(self):
            started["started"] = True

    monkeypatch.setattr(threading, "Thread", FakeThread)

    main._maybe_warm_up_in_background()

    assert started["target"] is main._warm_up_demo_graph
    assert started["daemon"] is True
    assert started["started"] is True


def test_warm_up_demo_graph_swallows_failures(monkeypatch):
    """A failed warm-up (e.g. Neo4j unreachable at boot) must not propagate --
    the same work just happens lazily on the first real /query instead."""
    import backend.main as main
    import kgqa.service as service

    def broken(graph_id, use_concepts=False):
        raise RuntimeError("Neo4j unavailable")

    monkeypatch.setattr(service, "_get_retriever", broken)

    main._warm_up_demo_graph()  # must not raise


def test_query_returns_answer_shape(monkeypatch):
    import graphrag

    def fake_answer(question, graph_id="demo", use_concepts=False):
        return {
            "answer": f"answer to: {question}",
            "reasoning_path": [{"kind": "seed_chunk", "node_id": "Chunks/1_0", "label": "x"}],
            "sources": ["1"],
        }

    monkeypatch.setattr(graphrag, "answer", fake_answer)

    resp = client.post("/query", json={"question": "does aspirin help?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "answer to: does aspirin help?"
    assert body["sources"] == ["1"]
    assert body["reasoning_path"][0]["kind"] == "seed_chunk"


def test_query_rejects_empty_question():
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_query_rejects_unknown_graph_id(monkeypatch):
    """An unrecognized graph_id must 404 before ever reaching graphrag.answer --
    otherwise it falls through to a full local re-encode of the corpus on an
    unauthenticated endpoint (see the comment on _KNOWN_GRAPH_IDS)."""
    import graphrag

    def fail_if_called(*args, **kwargs):
        raise AssertionError("answer() must not be called for an unknown graph_id")

    monkeypatch.setattr(graphrag, "answer", fail_if_called)

    resp = client.post("/query", json={"question": "does aspirin help?", "graph_id": "../../etc"})
    assert resp.status_code == 404
    assert "../../etc" in resp.json()["detail"]


def test_query_failure_returns_502(monkeypatch):
    import graphrag

    def broken(*args, **kwargs):
        raise RuntimeError("no providers available")

    monkeypatch.setattr(graphrag, "answer", broken)

    resp = client.post("/query", json={"question": "does aspirin help?"})
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Answering failed. Please try again."
    assert "no providers available" not in resp.json()["detail"]


def test_ingest_known_dataset_returns_graph_id():
    resp = client.post("/ingest", json={"dataset_id": "demo"})
    assert resp.status_code == 200
    assert resp.json() == {"graph_id": "demo"}


def test_ingest_unknown_dataset_returns_501():
    resp = client.post("/ingest", json={"dataset_id": "my_arbitrary.pdf"})
    assert resp.status_code == 501
