"""Tests for the FastAPI backend -- graphrag.answer is monkeypatched, no live
ArangoDB/LLM calls. Run with: pytest backend/test_main.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


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


def test_query_failure_returns_502(monkeypatch):
    import graphrag

    def broken(*args, **kwargs):
        raise RuntimeError("no providers available")

    monkeypatch.setattr(graphrag, "answer", broken)

    resp = client.post("/query", json={"question": "does aspirin help?"})
    assert resp.status_code == 502
    assert "no providers available" in resp.json()["detail"]


def test_ingest_known_dataset_returns_graph_id():
    resp = client.post("/ingest", json={"dataset_id": "demo"})
    assert resp.status_code == 200
    assert resp.json() == {"graph_id": "demo"}


def test_ingest_unknown_dataset_returns_501():
    resp = client.post("/ingest", json={"dataset_id": "my_arbitrary.pdf"})
    assert resp.status_code == 501
