import numpy as np

from kgqa.retrieval import ChunkStore, Neo4jGraphRetriever
from tests.conftest import FakeNeo4jDriver


def make_retriever(encoder, reranker):
    texts = [
        "aspirin reduces heart attack risk in patients",
        "statins lower cholesterol levels significantly",
    ]
    keys = ["1", "2"]
    ids = [f"Chunks/{k}_0" for k in keys]
    embs = encoder.encode(texts, normalize_embeddings=True)
    store = ChunkStore(ids, keys, texts, np.asarray(embs))
    driver = FakeNeo4jDriver(abstracts={
        "1": "FULL ABSTRACT 1: aspirin trial methods results conclusion",
        "2": "FULL ABSTRACT 2: statin trial",
    })
    return Neo4jGraphRetriever(store, encoder, driver, reranker=reranker, top_k_final=1)


def test_answer_returns_answer_reasoning_path_and_sources(fake_encoder, fake_reranker, monkeypatch):
    import kgqa.service as service

    retriever = make_retriever(fake_encoder, fake_reranker)
    monkeypatch.setattr(service, "_get_retriever", lambda graph_id, use_concepts=False: retriever)
    monkeypatch.setattr(service, "call_llm", lambda task, prompt, system="": "Yes, it does.")

    result = service.answer("does aspirin reduce heart attack risk", graph_id="demo")

    assert result["answer"] == "Yes, it does."
    assert result["sources"] == ["1"]
    assert any(step["kind"] == "seed_chunk" for step in result["reasoning_path"])
    assert any(step["kind"] == "parent_paper" for step in result["reasoning_path"])


def test_answer_reasoning_path_includes_concept_neighbours(fake_encoder, fake_reranker, monkeypatch):
    import kgqa.service as service

    texts = ["aspirin reduces heart attack risk in patients"]
    embs = fake_encoder.encode(texts, normalize_embeddings=True)
    store = ChunkStore(["Chunks/1_0"], ["1"], texts, np.asarray(embs))
    driver = FakeNeo4jDriver(
        abstracts={"1": "FULL ABSTRACT 1: aspirin"},
        neighbours=[("99", "NEIGHBOUR ABSTRACT via shared MeSH concept")],
    )
    retriever = Neo4jGraphRetriever(store, fake_encoder, driver, reranker=fake_reranker,
                                    use_concepts=True, top_k_final=1)
    monkeypatch.setattr(service, "_get_retriever", lambda graph_id, use_concepts=False: retriever)
    monkeypatch.setattr(service, "call_llm", lambda task, prompt, system="": "answer")

    result = service.answer("q", graph_id="demo", use_concepts=True)

    kinds = [step["kind"] for step in result["reasoning_path"]]
    assert "concept_neighbour" in kinds


def test_shared_db_degrades_to_none_without_arango_configured(monkeypatch):
    """Real code path: a non-demo (real-dataset) graph_id must not crash just
    because ARANGO_PASS is unset -- it should degrade, not raise."""
    import kgqa.service as service

    monkeypatch.delenv("ARANGO_PASS", raising=False)
    service._ARANGO_DB_CACHE.clear()

    db = service._shared_db("some_dataset")

    assert db is None
    assert service._ARANGO_DB_CACHE["some_dataset"] is None  # cached, so the failure isn't retried


def test_shared_neo4j_driver_degrades_to_none_without_config(monkeypatch):
    """Real code path: the demo graph must not crash just because
    NEO4J_PASSWORD is unset -- it should degrade, not raise."""
    import kgqa.service as service

    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    service._NEO4J_DRIVER_CACHE.clear()

    driver = service._shared_neo4j_driver()

    assert driver is None
    assert service._NEO4J_DRIVER_CACHE[None] is None  # cached, not retried


def test_get_retriever_builds_when_neo4j_unavailable(monkeypatch, fake_encoder, fake_reranker):
    """answer()'s retriever must still build (and later degrade to raw chunks
    inside gather_studies) rather than raising when the demo graph is
    unreachable."""
    import kgqa.service as service

    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    service._NEO4J_DRIVER_CACHE.clear()
    service._STORE_CACHE.clear()
    service._RETRIEVER_CACHE.clear()
    monkeypatch.setattr(service, "_shared_encoder", lambda: fake_encoder)
    monkeypatch.setattr(service, "_shared_reranker", lambda: fake_reranker)
    monkeypatch.setattr(
        service.ChunkStore, "from_dataset",
        classmethod(lambda cls, encoder, include_unlabeled=True: ChunkStore(
            ["Chunks/1_0"], ["1"], ["aspirin study"],
            np.asarray(encoder.encode(["aspirin study"], normalize_embeddings=True)))),
    )

    retriever = service._get_retriever("demo")

    assert retriever.driver is None
    candidates = retriever._select("aspirin")
    studies = retriever.gather_studies(candidates)  # degrades instead of raising
    assert studies == [(c.paper_key, c.text) for c in candidates]
