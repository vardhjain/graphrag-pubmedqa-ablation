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


class _BulkNeo4jSession:
    """Session fake for ChunkStore.from_neo4j's bulk vector download."""

    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        return self._rows


class _BulkNeo4jDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _BulkNeo4jSession(self._rows)


def _bulk_rows():
    return [
        {"key": "1_0", "paper": "1", "text": "aspirin trial", "emb": [1.0, 0.0]},
        {"key": "2_0", "paper": "2", "text": "statin trial", "emb": [0.0, 1.0]},
    ]


def test_from_neo4j_recovers_from_truncated_cache(tmp_path):
    """A kill mid-write leaves a truncated pickle; the load must treat it as a
    miss (delete + rebuild), not wedge every request with 'Ran out of input'."""
    cache = tmp_path / "demo_neo4j_vectors.pkl"
    cache.write_bytes(b"")  # what an OOM-killed write leaves behind

    store = ChunkStore.from_neo4j(_BulkNeo4jDriver(_bulk_rows()), cache_file=str(cache))

    assert store.ids == ["Chunks/1_0", "Chunks/2_0"]
    # The rebuilt cache must be valid: a second load comes straight from disk.
    again = ChunkStore.from_neo4j(_BulkNeo4jDriver([]), cache_file=str(cache))
    assert again.ids == ["Chunks/1_0", "Chunks/2_0"]


def test_vector_cache_write_is_atomic(tmp_path, monkeypatch):
    """The cache is written to a temp file and renamed, so the final path never
    holds a partial pickle even if the dump itself dies."""
    import kgqa.retrieval.base as base

    cache = tmp_path / "vectors.pkl"

    def exploding_dump(payload, f):
        f.write(b"partial")
        raise MemoryError("simulated OOM mid-write")

    monkeypatch.setattr(base.pickle, "dump", exploding_dump)
    try:
        ChunkStore.from_neo4j(_BulkNeo4jDriver(_bulk_rows()), cache_file=str(cache))
    except MemoryError:
        pass

    assert not cache.exists()  # partial bytes stayed in the .tmp file, never renamed


class _BulkArangoAQL:
    """AQL fake for ChunkStore.from_arango's bulk vector download.

    Records every execute() call so tests can assert exactly one query is
    issued -- the pagination bug this replaces issued N independent LIMIT
    offset,batch queries with no ordering guarantee relative to each other.
    """

    def __init__(self, rows):
        self._rows = rows
        self.execute_calls: list[dict] = []

    def execute(self, query, bind_vars=None, **kwargs):
        self.execute_calls.append({"query": query, "bind_vars": bind_vars, **kwargs})
        return iter(self._rows)


class _BulkArangoDB:
    def __init__(self, rows):
        self.aql = _BulkArangoAQL(rows)


def _bulk_arango_rows():
    return [
        {"id": "Chunks/1_0", "paper": "1", "text": "aspirin trial", "emb": [1.0, 0.0]},
        {"id": "Chunks/2_0", "paper": "2", "text": "statin trial", "emb": [0.0, 1.0]},
    ]


def test_from_arango_issues_a_single_query_not_offset_pagination():
    """Regression guard: from_arango must stream one query's cursor to
    exhaustion, not repeat independent LIMIT offset,batch queries (which have
    no ordering guarantee relative to each other and could silently skip or
    duplicate chunks across pages -- a corpus perturbation shared by every
    retrieval arm). Also confirms the collection name is bound, not
    f-string-interpolated (GAPS #11)."""
    db = _BulkArangoDB(_bulk_arango_rows())

    store = ChunkStore.from_arango(db)

    assert store.ids == ["Chunks/1_0", "Chunks/2_0"]
    assert store.paper_keys == ["1", "2"]
    assert store.texts == ["aspirin trial", "statin trial"]
    assert len(db.aql.execute_calls) == 1
    assert db.aql.execute_calls[0]["bind_vars"] == {"@collection": "Chunks"}


def test_from_arango_cache_hit_skips_the_query_entirely(tmp_path):
    """A populated cache must short-circuit before ever calling db.aql --
    the whole point of caching the bulk download."""
    cache = tmp_path / "vectors.pkl"
    ChunkStore.from_arango(_BulkArangoDB(_bulk_arango_rows()), cache_file=str(cache))

    db = _BulkArangoDB(_bulk_arango_rows())
    store = ChunkStore.from_arango(db, cache_file=str(cache))

    assert store.ids == ["Chunks/1_0", "Chunks/2_0"]
    assert db.aql.execute_calls == []


def test_answer_benchmark_returns_answer_and_retrieved_paper_keys(fake_encoder, fake_reranker, monkeypatch):
    """recall@k (src/kgqa/evaluation.py) needs the paper keys _select()
    retrieved, not just the LLM's answer text -- this is the contract that
    supplies them."""
    import kgqa.retrieval.base as base

    store = make_store(fake_encoder)
    retriever = PlainRetriever(store, fake_encoder, reranker=fake_reranker)
    monkeypatch.setattr(base, "call_ollama", lambda prompt, system="": "Final Answer: yes")

    answer, retrieved_papers = retriever.answer_benchmark("does aspirin help heart attack risk")

    assert answer == "Final Answer: yes"
    assert retrieved_papers  # non-empty
    assert all(isinstance(k, str) for k in retrieved_papers)
    assert len(retrieved_papers) == len(set(retrieved_papers))  # de-duplicated
