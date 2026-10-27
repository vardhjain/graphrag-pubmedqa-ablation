"""CPU-only negative-control tests. No ArangoDB, LLM, or ML downloads."""

from __future__ import annotations

import numpy as np
import pytest
from scripts.check_parent_parity import check_parent_context_parity
from scripts.run_benchmark import ARMS, build_retriever, stamp_parent_comparison

from kgqa.retrieval import ChunkStore, FlatParentRetriever, GraphRetriever
from kgqa.retrieval.base import Candidate
from tests.conftest import FakeDB


def _store(fake_encoder):
    ids = ["Chunks/b_0", "Chunks/a_1", "Chunks/a_0", "Chunks/b_1"]
    papers = ["b", "a", "a", "b"]
    texts = ["b methods", "a results", "a methods", "b results"]
    vectors = fake_encoder.encode(texts, normalize_embeddings=True)
    return ChunkStore(ids, papers, texts, np.asarray(vectors))


def _graph_db():
    return FakeDB(
        abstracts={"a": "a methods a results", "b": "b methods b results"}
    )


def test_flat_parent_is_reranked_negative_control(fake_encoder, fake_reranker):
    store = _store(fake_encoder)
    graph = build_retriever("graph", store, fake_encoder, fake_reranker, _graph_db())
    flat = build_retriever("flat_parent", store, fake_encoder, fake_reranker, None)
    assert isinstance(flat, FlatParentRetriever)
    assert isinstance(graph, GraphRetriever)
    assert flat.reranker is graph.reranker is fake_reranker
    assert flat.store is graph.store is store
    assert flat.encoder is graph.encoder is fake_encoder
    assert "flat_parent" in ARMS
    assert flat.name == "flat_parent"
    assert not hasattr(flat, "db")


def test_same_chunks_produce_identical_graph_and_flat_prompts(fake_encoder):
    store = _store(fake_encoder)
    graph = GraphRetriever(store, fake_encoder, _graph_db())
    flat = FlatParentRetriever(store, fake_encoder)
    question = "Is outcome improved?"
    candidates = [
        Candidate("Chunks/b_0", "b", "b methods"),
        Candidate("Chunks/a_1", "a", "a results"),
        Candidate("Chunks/a_0", "a", "a methods"),
    ]
    assert flat.gather_studies(candidates) == graph.gather_studies(candidates)
    assert flat._build_context(question, candidates) == graph._build_context(
        question, candidates
    )
    assert flat.gather_studies(candidates) == [
        ("b", "b methods b results"),
        ("a", "a methods a results"),
    ]


def test_flat_parent_is_invariant_to_chunk_store_scan_order(fake_encoder):
    first = _store(fake_encoder)
    order = [3, 1, 0, 2]
    second = ChunkStore(
        [first.ids[i] for i in order],
        [first.paper_keys[i] for i in order],
        [first.texts[i] for i in order],
        first.embeddings[order],
    )
    ids = ["Chunks/a_1", "Chunks/b_0"]
    a = FlatParentRetriever(first, fake_encoder)._parent_abstracts(ids)
    b = FlatParentRetriever(second, fake_encoder)._parent_abstracts(ids)
    assert a == b


def test_historical_section_key_order_is_matched_not_silently_changed(fake_encoder):
    texts = [f"section {i}" for i in range(12)]
    ids = [f"Chunks/p_{i}" for i in range(12)]
    store = ChunkStore(
        ids, ["p"] * len(ids), texts, fake_encoder.encode(texts)
    )
    flat = FlatParentRetriever(store, fake_encoder)
    got = flat._parent_abstracts([ids[0]])[0][1]
    expected = " ".join(texts[i] for i in sorted(range(12), key=lambda i: ids[i]))
    assert got == expected
    assert got != " ".join(texts)  # Known graph AQL section-order defect


def test_missing_flat_parent_lookup_is_counted_as_degraded(fake_encoder):
    flat = FlatParentRetriever(_store(fake_encoder), fake_encoder)
    candidate = Candidate("Chunks/unknown_0", "unknown", "raw context")
    assert flat.gather_studies([candidate]) == [("unknown", "raw context")]
    assert flat._last_degraded is True
    assert flat._degraded_count == 1


def test_flat_parent_rejects_duplicate_chunk_ids(fake_encoder):
    store = _store(fake_encoder)
    store.ids[1] = store.ids[0]
    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        FlatParentRetriever(store, fake_encoder)


def test_parity_preflight_passes_on_clean_fixture(fake_encoder):
    db = _graph_db()
    result = check_parent_context_parity(db, _store(fake_encoder), n=4, seed=7)
    assert result == {"checked_chunks": 4, "seed": 7, "identical_contexts": True}


def test_parity_preflight_fails_on_duplicate_context_sections(fake_encoder):
    db = FakeDB(
        abstracts={"a": "a methods a results a results", "b": "b methods b results"}
    )
    with pytest.raises(ValueError, match="Context mismatch"):
        check_parent_context_parity(db, _store(fake_encoder), n=4, seed=7)


def test_stamp_parent_fingerprints_are_order_independent(fake_encoder):
    from types import SimpleNamespace

    store = _store(fake_encoder)
    order = [2, 0, 3, 1]
    reordered = ChunkStore(
        [store.ids[i] for i in order],
        [store.paper_keys[i] for i in order],
        [store.texts[i] for i in order],
        store.embeddings[order],
    )
    samples = [SimpleNamespace(pubid="x", question="Question?", final_decision="yes")]
    first = {}
    second = {}
    stamp_parent_comparison(first, samples, store)
    stamp_parent_comparison(second, samples, reordered)
    assert first == second
    assert first["parent_control_protocol"] == "flat-parent-v1"
    assert len(first["sample_sha256"]) == len(first["corpus_sha256"]) == 64

    changed = {}
    different = [SimpleNamespace(pubid="x", question="Different?", final_decision="yes")]
    stamp_parent_comparison(changed, different, store)
    assert changed["sample_sha256"] != first["sample_sha256"]


def test_contexts_use_identical_llm_prompt_template(fake_encoder):
    from kgqa.prompts import BENCHMARK_SYSTEM_PROMPT, build_prompt

    graph = GraphRetriever(_store(fake_encoder), fake_encoder, _graph_db())
    flat = FlatParentRetriever(_store(fake_encoder), fake_encoder)
    candidates = [Candidate("Chunks/a_0", "a", "a methods")]
    question = "What happened?"
    assert build_prompt(
        graph._build_context(question, candidates), question
    ) == build_prompt(flat._build_context(question, candidates), question)
    assert "Final Answer" in BENCHMARK_SYSTEM_PROMPT


def test_resume_rejects_different_parent_comparison_inputs(fake_encoder):
    from scripts.run_benchmark import check_parent_control_resume

    expected = {"llm_model": "test", "num_ctx": 8192, "num_predict": 100}
    store = _store(fake_encoder)
    stamp_parent_comparison(expected, [], store)
    expected.update(temperature=0.0, top_k=3, top_k_candidates=75, git_sha="abc")
    check_parent_control_resume({"meta": dict(expected)}, expected)

    mismatched = {"meta": {**expected, "corpus_sha256": "different"}}
    with pytest.raises(ValueError, match="corpus_sha256"):
        check_parent_control_resume(mismatched, expected)
