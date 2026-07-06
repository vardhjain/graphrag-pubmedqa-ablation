import numpy as np

from kgqa.retrieval import ChunkStore, Neo4jGraphRetriever
from tests.conftest import FakeNeo4jDriver


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


def test_neo4j_arm_naming(fake_encoder, fake_reranker):
    driver = FakeNeo4jDriver(abstracts={})
    assert Neo4jGraphRetriever(make_store(fake_encoder), fake_encoder, driver,
                               reranker=fake_reranker).name == "graph"
    assert Neo4jGraphRetriever(make_store(fake_encoder), fake_encoder, driver,
                               reranker=fake_reranker, use_concepts=True).name == "graph_concepts"


def test_neo4j_parent_expansion_uses_full_abstract(fake_encoder, fake_reranker):
    store = make_store(fake_encoder)
    driver = FakeNeo4jDriver(abstracts={
        "1": "FULL ABSTRACT 1: aspirin trial methods results conclusion",
        "2": "FULL ABSTRACT 2: statin trial",
        "3": "FULL ABSTRACT 3: exercise study",
    })
    r = Neo4jGraphRetriever(store, fake_encoder, driver, reranker=fake_reranker, top_k_final=1)
    ctx = r.retrieve("aspirin heart attack")
    assert "=== STUDY 1 ===" in ctx
    assert "FULL ABSTRACT 1" in ctx


def test_neo4j_concept_hop_adds_neighbour(fake_encoder, fake_reranker):
    store = make_store(fake_encoder)
    driver = FakeNeo4jDriver(
        abstracts={"1": "FULL ABSTRACT 1: aspirin", "2": "x", "3": "y"},
        neighbours=[("99", "NEIGHBOUR ABSTRACT via shared MeSH concept")],
    )
    r = Neo4jGraphRetriever(store, fake_encoder, driver, reranker=fake_reranker,
                            use_concepts=True, top_k_final=1)
    ctx = r.retrieve("aspirin heart attack")
    assert "NEIGHBOUR ABSTRACT" in ctx
    assert ctx.count("=== STUDY") == 2


def test_neo4j_degrades_to_raw_chunks_on_driver_error(fake_encoder, fake_reranker):
    class BrokenDriver:
        def session(self):
            raise RuntimeError("no connection")

    store = make_store(fake_encoder)
    r = Neo4jGraphRetriever(store, fake_encoder, BrokenDriver(), reranker=fake_reranker, top_k_final=1)
    ctx = r.retrieve("aspirin heart attack")
    assert "=== STUDY 1 ===" in ctx
    assert "aspirin" in ctx


def test_neo4j_chunk_id_local_key_stripping(fake_encoder, fake_reranker):
    """Candidate.chunk_id is "Chunks/1_0" (Arango-style prefix, kept for
    compatibility) -- the Neo4j retriever must strip it before querying."""
    store = make_store(fake_encoder)
    driver = FakeNeo4jDriver(abstracts={"1": "FULL ABSTRACT 1", "2": "x", "3": "y"})
    r = Neo4jGraphRetriever(store, fake_encoder, driver, reranker=fake_reranker, top_k_final=1)
    candidates = r._select("aspirin heart attack")
    studies = r.gather_studies(candidates)
    assert studies == [("1", "FULL ABSTRACT 1")]
