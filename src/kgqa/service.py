"""Service boundary: ``answer(question, graph_id)`` is the one entry point a
web backend calls. Everything upstream (ingestion, extraction, graph
construction, multi-hop retrieval) stays internal to this package; a caller
never needs to know about ``ChunkStore``, ``GraphRetriever``, ArangoDB, Neo4j,
or which of the two backs a given ``graph_id`` (see the note below).

The ``reasoning_path`` in the return value is an ordered list of graph steps
(seed chunk -> parent paper -> optional concept-hop neighbour) so a frontend
can draw the traversed subgraph without re-deriving it from the answer text.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field

from .config import ArangoConfig, Neo4jConfig
from .prompts import CHAT_SYSTEM_PROMPT, build_prompt
from .providers import call_llm
from .retrieval import ChunkStore
from .retrieval.graph import GraphRetriever, format_studies
from .retrieval.neo4j_graph import Neo4jGraphRetriever

# ``graph_id="demo"`` is served by Neo4j AuraDB (see retrieval/neo4j_graph.py
# and scripts/ingest_neo4j.py) -- a small, genuinely-free-forever graph DB
# scoped to the labeled-split corpus. Any other graph_id (e.g. one /ingest
# hands back for a real uploaded dataset) still uses ArangoDB, unchanged from
# how the benchmarked pipeline works.
_DEMO_GRAPH_ID = "demo"

_STORE_CACHE: dict[str, ChunkStore] = {}
_RETRIEVER_CACHE: dict[tuple[str, bool], GraphRetriever | Neo4jGraphRetriever] = {}
_ARANGO_DB_CACHE: dict[str, object] = {}  # keyed by graph_id -- non-demo datasets only
_NEO4J_DRIVER_CACHE: dict[None, object] = {}  # single slot -- only ever serves the demo graph
_ENCODER = None
_RERANKER = None
_CACHE_DIR = os.environ.get("KGQA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "kgqa_cache"))
os.makedirs(_CACHE_DIR, exist_ok=True)


def _shared_encoder():
    global _ENCODER
    if _ENCODER is None:
        from .models import load_encoder

        _ENCODER = load_encoder()
    return _ENCODER


def _shared_reranker():
    global _RERANKER
    if _RERANKER is None:
        from .models import load_reranker

        _RERANKER = load_reranker()
    return _RERANKER


def _shared_db(graph_id: str):
    """One ArangoDB connection per non-demo ``graph_id``, reused by the store
    and retriever. Used for real datasets ``/ingest`` has built -- the demo
    graph is served by Neo4j instead, see ``_shared_neo4j_driver``.

    Returns ``None`` (cached, so this is tried at most once per process) if
    ArangoDB isn't configured or isn't reachable -- ``GraphRetriever`` already
    degrades to raw retrieved chunks when its ``db`` calls fail, so the
    service still answers (without parent-document expansion) rather than
    hard-crashing when no graph is available.
    """
    if graph_id not in _ARANGO_DB_CACHE:
        from .models import connect_arango

        try:
            _ARANGO_DB_CACHE[graph_id] = connect_arango(ArangoConfig(db_name=graph_id), max_retries=1)
        except Exception as exc:  # noqa: BLE001 - degrade to no-graph, not a crash
            print(f"[GraphRAG] ArangoDB unavailable for graph_id={graph_id!r} ({exc}). "
                  "Falling back to ungraphed retrieval.")
            _ARANGO_DB_CACHE[graph_id] = None
    return _ARANGO_DB_CACHE[graph_id]


def _shared_neo4j_driver():
    """One Neo4j driver for the demo graph (``scripts/ingest_neo4j.py``'s
    labeled-split corpus), reused by the store and retriever. Kept in its own
    single-slot cache (not ``_ARANGO_DB_CACHE``) so there's no possibility of
    a graph_id string colliding with an internal cache key.

    Returns ``None`` (cached, so this is tried at most once per process) if
    Neo4j isn't configured or isn't reachable -- ``Neo4jGraphRetriever``
    already degrades to raw retrieved chunks when its driver calls fail.
    """
    if None not in _NEO4J_DRIVER_CACHE:
        from .models import connect_neo4j

        try:
            _NEO4J_DRIVER_CACHE[None] = connect_neo4j(Neo4jConfig(), max_retries=1)
        except Exception as exc:  # noqa: BLE001 - degrade to no-graph, not a crash
            print(f"[GraphRAG] Neo4j unavailable for the demo graph ({exc}). "
                  "Falling back to ungraphed retrieval.")
            _NEO4J_DRIVER_CACHE[None] = None
    return _NEO4J_DRIVER_CACHE[None]


@dataclass
class ReasoningStep:
    kind: str  # "seed_chunk" | "parent_paper" | "concept_neighbour"
    node_id: str
    label: str
    from_node: str | None = None
    edge: str | None = None


@dataclass
class AnswerResult:
    answer: str
    reasoning_path: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


def _get_store(graph_id: str) -> ChunkStore:
    """Resolve a ``graph_id`` to its chunk store, building/caching on first use.

    ``graph_id="demo"`` prefers ``ChunkStore.from_neo4j`` -- ``scripts/
    ingest_neo4j.py`` pre-computes and stores every chunk's embedding in
    Neo4j, so this just downloads vectors (fast, no re-encoding; a local
    pickle cache makes even that a one-time cost). Any other graph_id uses
    ``ChunkStore.from_arango`` the same way, unchanged from before.

    Only falls back to ``ChunkStore.from_dataset`` (encoding chunks locally,
    bounded to the small labeled split) when the graph DB isn't reachable --
    a degraded local-dev mode, never used for the full corpus.
    """
    if graph_id in _STORE_CACHE:
        return _STORE_CACHE[graph_id]

    encoder = _shared_encoder()
    if graph_id == _DEMO_GRAPH_ID:
        driver = _shared_neo4j_driver()
        if driver is not None:
            cache_file = os.path.join(_CACHE_DIR, "demo_neo4j_vectors.pkl")
            store = ChunkStore.from_neo4j(driver, cache_file=cache_file)
        else:
            print("[GraphRAG] No Neo4j for the demo graph; encoding the labeled "
                  "split locally as a bounded fallback (this is slow -- not for production).")
            store = ChunkStore.from_dataset(encoder, include_unlabeled=False)
    else:
        db = _shared_db(graph_id)
        if db is not None:
            cache_file = os.path.join(_CACHE_DIR, f"{graph_id}_vectors.pkl")
            store = ChunkStore.from_arango(db, cache_file=cache_file)
        else:
            print(f"[GraphRAG] No ArangoDB for graph_id={graph_id!r}; encoding the labeled "
                  "split locally as a bounded fallback (this is slow -- not for production).")
            store = ChunkStore.from_dataset(encoder, include_unlabeled=False)
    _STORE_CACHE[graph_id] = store
    return store


def _get_retriever(graph_id: str, use_concepts: bool = False) -> GraphRetriever | Neo4jGraphRetriever:
    key = (graph_id, use_concepts)
    if key in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[key]

    store = _get_store(graph_id)
    if graph_id == _DEMO_GRAPH_ID:
        retriever = Neo4jGraphRetriever(
            store, _shared_encoder(), _shared_neo4j_driver(),
            reranker=_shared_reranker(), use_concepts=use_concepts,
        )
    else:
        retriever = GraphRetriever(
            store, _shared_encoder(), _shared_db(graph_id),
            reranker=_shared_reranker(), use_concepts=use_concepts,
        )
    _RETRIEVER_CACHE[key] = retriever
    return retriever


def _build_reasoning_path(candidates, studies: list[tuple[str, str]]) -> list[dict]:
    steps: list[ReasoningStep] = []
    seed_papers = {c.paper_key for c in candidates}
    for c in candidates:
        steps.append(ReasoningStep(kind="seed_chunk", node_id=c.chunk_id, label=c.text[:80]))
    for paper_key, _abstract in studies:
        if paper_key in seed_papers:
            for c in candidates:
                if c.paper_key == paper_key:
                    steps.append(
                        ReasoningStep(
                            kind="parent_paper", node_id=f"Papers/{paper_key}",
                            label=paper_key, from_node=c.chunk_id, edge="HAS_CONTEXT",
                        )
                    )
        else:
            steps.append(
                ReasoningStep(
                    kind="concept_neighbour", node_id=f"Papers/{paper_key}",
                    label=paper_key, from_node=None, edge="MENTIONS",
                )
            )
    return [step.__dict__ for step in steps]


def answer(question: str, graph_id: str = "demo", use_concepts: bool = False) -> dict:
    """Answer ``question`` against ``graph_id``.

    Returns ``{"answer": str, "reasoning_path": list[dict], "sources": list[str]}``.
    Synthesis runs through the ``synthesize`` provider chain (Gemini Flash by
    default, falling back to local Ollama) so the service degrades gracefully
    without a cloud API key configured.
    """
    retriever = _get_retriever(graph_id, use_concepts=use_concepts)
    candidates = retriever._select(question)
    studies = retriever.gather_studies(candidates)
    context = format_studies(studies)

    response = call_llm("synthesize", build_prompt(context, question), system=CHAT_SYSTEM_PROMPT)
    reasoning_path = _build_reasoning_path(candidates, studies)
    sources = list(dict.fromkeys(c.paper_key for c in candidates))

    return AnswerResult(answer=response, reasoning_path=reasoning_path, sources=sources).__dict__
