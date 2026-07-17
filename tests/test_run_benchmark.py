"""Tests for scripts/run_benchmark.py's arm -> retriever mapping -- this is
the fairness contract itself (every arm shares the same store/encoder,
differing only in reranker/graph usage), so it's worth pinning directly.
"""

from __future__ import annotations

from scripts.run_benchmark import build_retriever

from kgqa.retrieval import GraphRetriever, PlainRetriever
from tests.conftest import FakeDB


def test_plain_arm_has_no_reranker(fake_encoder, fake_reranker):
    r = build_retriever("plain", store=None, encoder=fake_encoder,
                        reranker=fake_reranker, db=None)
    assert isinstance(r, PlainRetriever)
    assert r.reranker is None


def test_plain_rr_arm_uses_the_given_reranker(fake_encoder, fake_reranker):
    r = build_retriever("plain_rr", store=None, encoder=fake_encoder,
                        reranker=fake_reranker, db=None)
    assert isinstance(r, PlainRetriever)
    assert r.reranker is fake_reranker


def test_graph_arm_uses_reranker_and_no_concepts(fake_encoder, fake_reranker):
    db = FakeDB(abstracts={})
    r = build_retriever("graph", store=None, encoder=fake_encoder,
                        reranker=fake_reranker, db=db)
    assert isinstance(r, GraphRetriever)
    assert r.reranker is fake_reranker
    assert r.use_concepts is False


def test_graph_concepts_arm_uses_reranker_and_concepts(fake_encoder, fake_reranker):
    db = FakeDB(abstracts={})
    r = build_retriever("graph_concepts", store=None, encoder=fake_encoder,
                        reranker=fake_reranker, db=db)
    assert isinstance(r, GraphRetriever)
    assert r.reranker is fake_reranker
    assert r.use_concepts is True


def test_graph_norr_arm_ignores_any_passed_reranker(fake_encoder, fake_reranker):
    """The whole point of this arm: parent expansion without the reranker --
    it must come out reranker=None even if a reranker is passed in, since
    that's exactly what the hosted demo's KGQA_SKIP_RERANKER=true does."""
    db = FakeDB(abstracts={})
    r = build_retriever("graph_norr", store=None, encoder=fake_encoder,
                        reranker=fake_reranker, db=db)
    assert isinstance(r, GraphRetriever)
    assert r.reranker is None
    assert r.use_concepts is False


def test_unknown_arm_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown arm"):
        build_retriever("not_a_real_arm", store=None, encoder=None, reranker=None, db=None)
