"""Tests for scripts/run_benchmark.py's arm -> retriever mapping -- this is
the fairness contract itself (every arm shares the same store/encoder,
differing only in reranker/graph usage), so it's worth pinning directly.
"""

from __future__ import annotations

from scripts.run_benchmark import build_retriever

from kgqa.retrieval import GraphRetriever, PlainRetriever
from tests.conftest import FakeDB


def test_plain_arm_has_no_reranker(fake_encoder, fake_reranker):
    r = build_retriever("plain", store=None, encoder=fake_encoder, reranker=fake_reranker, db=None)
    assert isinstance(r, PlainRetriever)
    assert r.reranker is None


def test_plain_rr_arm_uses_the_given_reranker(fake_encoder, fake_reranker):
    r = build_retriever(
        "plain_rr", store=None, encoder=fake_encoder, reranker=fake_reranker, db=None
    )
    assert isinstance(r, PlainRetriever)
    assert r.reranker is fake_reranker


def test_graph_arm_uses_reranker_and_no_concepts(fake_encoder, fake_reranker):
    db = FakeDB(abstracts={})
    r = build_retriever("graph", store=None, encoder=fake_encoder, reranker=fake_reranker, db=db)
    assert isinstance(r, GraphRetriever)
    assert r.reranker is fake_reranker
    assert r.use_concepts is False


def test_graph_concepts_arm_uses_reranker_and_concepts(fake_encoder, fake_reranker):
    db = FakeDB(abstracts={})
    r = build_retriever(
        "graph_concepts", store=None, encoder=fake_encoder, reranker=fake_reranker, db=db
    )
    assert isinstance(r, GraphRetriever)
    assert r.reranker is fake_reranker
    assert r.use_concepts is True


def test_graph_norr_arm_ignores_any_passed_reranker(fake_encoder, fake_reranker):
    """The whole point of this arm: parent expansion without the reranker --
    it must come out reranker=None even if a reranker is passed in, since
    that's exactly what the hosted demo's KGQA_SKIP_RERANKER=true does."""
    db = FakeDB(abstracts={})
    r = build_retriever(
        "graph_norr", store=None, encoder=fake_encoder, reranker=fake_reranker, db=db
    )
    assert isinstance(r, GraphRetriever)
    assert r.reranker is None
    assert r.use_concepts is False


def test_unknown_arm_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown arm"):
        build_retriever("not_a_real_arm", store=None, encoder=None, reranker=None, db=None)


# ── meta / resume / truncation helpers ────────────────────────────────────────
def test_build_meta_records_run_settings():
    from scripts.run_benchmark import build_meta

    from kgqa import config

    meta = build_meta("graph", 42, 200)
    assert meta["arm"] == "graph" and meta["seed"] == 42 and meta["n_requested"] == 200
    assert meta["llm_model"] == config.LLM_MODEL
    assert meta["num_ctx"] == config.LLM_NUM_CTX
    assert meta["num_predict"] == config.LLM_NUM_PREDICT
    assert meta["complete"] is False
    assert "git_sha" in meta


def test_check_resume_accepts_matching_prefix():
    from scripts.run_benchmark import check_resume

    data = {"meta": {"arm": "graph", "seed": 42, "n_requested": 3}, "ids": ["a", "b"]}
    assert check_resume(data, "graph", 42, 3, ["a", "b", "c"]) == ["a", "b"]


def test_check_resume_rejects_mismatched_meta_or_order():
    import pytest
    from scripts.run_benchmark import check_resume

    data = {"meta": {"arm": "graph", "seed": 42, "n_requested": 3}, "ids": ["a", "b"]}
    with pytest.raises(ValueError, match="seed"):
        check_resume(data, "graph", 7, 3, ["a", "b", "c"])
    with pytest.raises(ValueError, match="arm"):
        check_resume(data, "plain", 42, 3, ["a", "b", "c"])
    with pytest.raises(ValueError, match="prefix"):
        check_resume(data, "graph", 42, 3, ["b", "a", "c"])
    with pytest.raises(ValueError):
        check_resume({"ids": []}, "graph", 42, 3, ["a"])  # no meta at all


def test_is_truncated_prefers_done_reason_then_heuristic():
    from scripts.run_benchmark import is_truncated

    assert is_truncated("<think>x</think> yes", "length") is True
    assert is_truncated("<think>never closed", "stop") is False
    assert is_truncated("<think>never closed", None) is True
    assert is_truncated("<think>x</think> Final answer: yes", None) is False
    assert is_truncated(None, "length") is False


def test_done_reason_probe_wraps_base_call_ollama(monkeypatch):
    from scripts.run_benchmark import install_done_reason_probe

    import kgqa.llm as llm
    import kgqa.retrieval.base as base

    monkeypatch.setattr(base, "call_ollama", base.call_ollama)  # restored after test
    monkeypatch.setattr(llm, "call_ollama_meta", lambda *a, **k: ("text", "length"))
    last = install_done_reason_probe()
    assert base.call_ollama("prompt", system="s") == "text"
    assert last["reason"] == "length"


def test_evaluator_roundtrip_through_saved_summary(tmp_path):
    import json

    from kgqa.evaluation import Evaluator

    ev = Evaluator("graph")
    ev.meta = {"arm": "graph", "seed": 42}
    ev.record("yes", "yes", 1.5, sample_id="1", degraded=True)
    ev.record("no", "maybe", 2.5, sample_id="2", truncated=True)
    path = tmp_path / "r.json"
    ev.save(str(path))
    data = json.loads(path.read_text())
    assert data["n_degraded"] == 1 and data["n_truncated"] == 1
    assert data["meta"]["seed"] == 42

    back = Evaluator.from_saved(data)
    assert back.ids == ["1", "2"] and back.latencies == [1.5, 2.5]
    assert back.degraded == [True, False] and back.truncated == [False, True]
    assert back.summary()["accuracy"] == ev.summary()["accuracy"]


def test_build_meta_versions_parser_and_fingerprints_prompt():
    from scripts.run_benchmark import build_meta

    strict = build_meta("graph", 42, 200, parser="strict")
    legacy = build_meta("graph", 42, 200, parser="legacy")
    assert strict["protocol_version"] == 2
    assert strict["answer_parser"] == "strict-v2"
    assert legacy["answer_parser"] == "legacy-v1"
    assert strict["prompt_sha256"] == legacy["prompt_sha256"]
    assert len(strict["prompt_sha256"]) == 64
    assert strict["embedding_model"] and strict["reranker_model"]


def test_resume_refuses_parser_or_corpus_drift():
    import pytest
    from scripts.run_benchmark import build_meta, check_resume

    expected = build_meta("graph", 42, 3)
    expected["sample_sha256"] = "a" * 64
    expected["corpus_sha256"] = "b" * 64
    good = {"meta": dict(expected), "ids": ["a"]}
    assert check_resume(good, "graph", 42, 3, ["a", "b", "c"], expected_meta=expected) == [
        "a"
    ]

    for field, new_value in (
        ("answer_parser", "legacy-v1"),
        ("corpus_sha256", "other"),
        ("num_ctx", 1),
    ):
        changed = {"meta": {**expected, field: new_value}, "ids": ["a"]}
        with pytest.raises(ValueError, match=field):
            check_resume(
                changed, "graph", 42, 3, ["a", "b", "c"], expected_meta=expected
            )


def test_resume_refuses_unversioned_checkpoint_on_new_run():
    import pytest
    from scripts.run_benchmark import build_meta, check_resume

    expected = build_meta("graph", 42, 3)
    expected["sample_sha256"] = "a" * 64
    expected["corpus_sha256"] = "b" * 64
    old = {"meta": {"arm": "graph", "seed": 42, "n_requested": 3}, "ids": ["a"]}
    with pytest.raises(ValueError, match="protocol_version"):
        check_resume(old, "graph", 42, 3, ["a", "b", "c"], expected_meta=expected)
