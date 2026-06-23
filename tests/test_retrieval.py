import numpy as np

from kgqa.retrieval import ChunkStore, GraphRetriever, PlainRetriever
from tests.conftest import FakeDB


def make_store(encoder):
    texts = [
        "aspirin reduces heart attack risk in patients",
        "statins lower cholesterol levels significantly",
        "regular exercise improves mood and sleep",
    ]
    keys = ["1", "2", "3"]
    ids = [f"Chunks/{k}_0" for k in keys]
    embs = encoder.encode(texts, normalize_embeddings=True)
    return ChunkStore(ids, keys, texts, np.asarray(embs))


def test_chunkstore_search_ranks_relevant_first(fake_encoder):
    store = make_store(fake_encoder)
    idxs = store.search(fake_encoder.encode(["aspirin heart attack"]), k=3)
    assert store.paper_keys[idxs[0]] == "1"


def test_plain_arm_naming(fake_encoder, fake_reranker):
    assert PlainRetriever(make_store(fake_encoder), fake_encoder).name == "plain"
    assert PlainRetriever(make_store(fake_encoder), fake_encoder,
                          reranker=fake_reranker).name == "plain_rr"


def test_plain_context_is_raw_chunks(fake_encoder):
    store = make_store(fake_encoder)
    r = PlainRetriever(store, fake_encoder, top_k_final=1)
    ctx = r.retrieve("aspirin heart attack")
    assert ctx.startswith("Abstract 1:")
    assert "aspirin" in ctx


def test_graph_parent_expansion_uses_full_abstract(fake_encoder, fake_reranker):
    store = make_store(fake_encoder)
    db = FakeDB(abstracts={
        "1": "FULL ABSTRACT 1: aspirin trial methods results conclusion",
        "2": "FULL ABSTRACT 2: statin trial",
        "3": "FULL ABSTRACT 3: exercise study",
    })
    r = GraphRetriever(store, fake_encoder, db, reranker=fake_reranker, top_k_final=1)
    assert r.name == "graph"
    ctx = r.retrieve("aspirin heart attack")
    assert "=== STUDY 1 ===" in ctx
    assert "FULL ABSTRACT 1" in ctx


def test_graph_concept_hop_adds_neighbour(fake_encoder, fake_reranker):
    store = make_store(fake_encoder)
    db = FakeDB(
        abstracts={"1": "FULL ABSTRACT 1: aspirin", "2": "x", "3": "y"},
        neighbours=[("99", "NEIGHBOUR ABSTRACT via shared MeSH concept")],
    )
    r = GraphRetriever(store, fake_encoder, db, reranker=fake_reranker,
                       use_concepts=True, top_k_final=1)
    assert r.name == "graph_concepts"
    ctx = r.retrieve("aspirin heart attack")
    assert "NEIGHBOUR ABSTRACT" in ctx
    assert ctx.count("=== STUDY") == 2


def test_graph_context_has_no_question_leakage(fake_encoder, fake_reranker):
    """The benchmark question/title must never appear in the graph context."""
    store = make_store(fake_encoder)
    db = FakeDB(abstracts={"1": "FULL ABSTRACT 1: aspirin", "2": "x", "3": "y"})
    r = GraphRetriever(store, fake_encoder, db, reranker=fake_reranker, top_k_final=1)
    question = "does aspirin reduce heart attack risk"
    ctx = r.retrieve(question)
    assert question not in ctx
    assert "STUDY:" not in ctx  # old leaky "=== STUDY: {title} ===" format is gone


def test_graph_degrades_to_raw_chunks_on_db_error(fake_encoder, fake_reranker):
    class BrokenDB:
        class aql:
            @staticmethod
            def execute(*a, **k):
                raise RuntimeError("no connection")
    store = make_store(fake_encoder)
    r = GraphRetriever(store, fake_encoder, BrokenDB(), reranker=fake_reranker, top_k_final=1)
    ctx = r.retrieve("aspirin heart attack")
    assert "=== STUDY 1 ===" in ctx
    assert "aspirin" in ctx


def test_chat_returns_answer_and_source_pubids(fake_encoder, fake_reranker, monkeypatch):
    import kgqa.retrieval.base as base
    monkeypatch.setattr(base, "call_ollama",
                        lambda *a, **k: "<think>reasoning</think> Yes, it does.")
    store = make_store(fake_encoder)
    db = FakeDB(abstracts={"1": "FULL ABS 1: aspirin", "2": "x", "3": "y"})
    r = GraphRetriever(store, fake_encoder, db, reranker=fake_reranker, top_k_final=1)
    out = r.chat("does aspirin reduce heart attack risk")
    assert set(out) >= {"answer", "sources", "context"}
    assert out["sources"] == ["1"]          # the retrieved paper's pubid
    assert "Yes" in out["answer"]


def test_chunkstore_from_dataset_builds_corpus(monkeypatch, fake_encoder):
    import kgqa.data as data
    from kgqa.retrieval import ChunkStore

    monkeypatch.setattr(data, "iter_chunks",
                        lambda include_unlabeled=True: iter([("1", 0, "alpha"), ("2", 0, "beta")]))
    store = ChunkStore.from_dataset(fake_encoder, include_unlabeled=False)
    assert len(store) == 2
    assert store.paper_keys == ["1", "2"]
    assert store.ids == ["Chunks/1_0", "Chunks/2_0"]
