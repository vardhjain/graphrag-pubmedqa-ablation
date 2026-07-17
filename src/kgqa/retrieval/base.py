"""Shared retrieval scaffolding.

``ChunkStore`` is the single document pool every arm searches over, so the
corpus, chunking, and embeddings are provably identical across arms.
``BaseRetriever`` owns the encode -> (optional) rerank -> select pipeline; each
subclass only customises how the selected chunks become an LLM context string.
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..config import TOP_K_CANDIDATES, TOP_K_FINAL
from ..llm import call_ollama
from ..prompts import BENCHMARK_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT, build_prompt


@dataclass
class Candidate:
    """A retrieved chunk plus its provenance."""

    chunk_id: str  # ArangoDB _id or local id, e.g. "Chunks/12345_0"
    paper_key: str  # owning paper, e.g. "12345"
    text: str
    score: float = 0.0


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _load_vector_cache(cache_file: str) -> dict | None:
    """Read a pickle vector cache, treating a corrupt file as a miss.

    A process kill mid-write (OOM on a small host, platform restart) can leave
    a truncated pickle at the cache path; ``pickle.load`` then raises
    ``EOFError("Ran out of input")`` on every subsequent request, wedging the
    service until the file is removed. Delete the bad file and rebuild instead.
    """
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            os.remove(cache_file)
        except OSError:
            pass
        return None


def _dump_vector_cache(cache_file: str, payload: dict) -> None:
    """Write the pickle cache atomically so a mid-write kill can't corrupt it."""
    tmp = f"{cache_file}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, cache_file)


class ChunkStore:
    """In-memory, L2-normalised chunk embeddings with cosine search."""

    def __init__(self, ids: list[str], paper_keys: list[str],
                 texts: list[str], embeddings: np.ndarray):
        self.ids = ids
        self.paper_keys = paper_keys
        self.texts = texts
        self.embeddings = _normalize(np.asarray(embeddings, dtype=np.float32)) \
            if len(embeddings) else np.zeros((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query_emb: np.ndarray, k: int) -> list[int]:
        """Return indices of the top-k chunks by cosine similarity."""
        if len(self) == 0:
            return []
        q = _normalize(np.atleast_2d(np.asarray(query_emb, dtype=np.float32)))
        sims = (self.embeddings @ q[0])
        k = min(k, len(self))
        top = np.argpartition(sims, -k)[-k:]
        return list(top[np.argsort(sims[top])[::-1]])

    def candidate(self, idx: int, score: float = 0.0) -> Candidate:
        return Candidate(self.ids[idx], self.paper_keys[idx], self.texts[idx], score)

    # ── builders ───────────────────────────────────────────────────────────────
    @classmethod
    def from_dataset(cls, encoder, include_unlabeled: bool = True,
                     batch_size: int = 128) -> ChunkStore:
        """Build the corpus locally from PubMedQA (no ArangoDB needed)."""
        from ..data import iter_chunks

        ids, paper_keys, texts = [], [], []
        for paper_key, chunk_idx, text in iter_chunks(include_unlabeled):
            ids.append(f"Chunks/{paper_key}_{chunk_idx}")
            paper_keys.append(paper_key)
            texts.append(text)
        embeddings = encoder.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        )
        return cls(ids, paper_keys, texts, embeddings)

    @classmethod
    def from_arango(cls, db, collection: str = "Chunks", batch: int = 5000,
                    cache_file: str | None = None) -> ChunkStore:
        """Download chunk vectors from ArangoDB (with optional pickle cache).

        Uses a single query, paged server-side via ``batch_size`` and
        iterated to exhaustion -- not N independent ``LIMIT offset, batch``
        queries. Repeated unsorted queries have no ordering guarantee
        relative to each other (ArangoDB doesn't promise stable order for an
        unsorted scan), so paging that way could silently skip or duplicate
        chunks across pages -- a corpus perturbation shared by every
        retrieval arm. One query execution, streamed via its cursor, is a
        single consistent snapshot and has no such gap.
        """
        if cache_file:
            data = _load_vector_cache(cache_file)
            if data is not None and len(data["embeddings"]):
                return cls(data["ids"], data["paper_keys"],
                           data["texts"], np.asarray(data["embeddings"]))

        ids, paper_keys, texts, embeddings = [], [], [], []
        aql = """
            FOR c IN @@collection
                FILTER c.embedding != null
                RETURN { id: c._id, paper: c.paper_key,
                         text: c.text, emb: c.embedding }
        """
        cursor = db.aql.execute(aql, bind_vars={"@collection": collection}, batch_size=batch, ttl=3600)
        for doc in cursor:
            ids.append(doc["id"])
            paper_keys.append(doc.get("paper") or doc["id"].split("/")[-1].rsplit("_", 1)[0])
            texts.append(doc["text"])
            embeddings.append(doc["emb"])

        embeddings_np = np.asarray(embeddings, dtype=np.float32)
        if cache_file and ids:
            _dump_vector_cache(cache_file, {"ids": ids, "paper_keys": paper_keys,
                                            "texts": texts, "embeddings": embeddings_np})
        return cls(ids, paper_keys, texts, embeddings_np)

    @classmethod
    def from_neo4j(cls, driver, cache_file: str | None = None) -> ChunkStore:
        """Download chunk vectors from Neo4j (with optional pickle cache).

        Mirrors ``from_arango`` -- chunk embeddings are pre-computed once at
        ingest time (``scripts/ingest_neo4j.py``) and stored as a node
        property, so this is a bulk read, not a live re-encode.
        """
        if cache_file:
            data = _load_vector_cache(cache_file)
            if data is not None and len(data["embeddings"]):
                return cls(data["ids"], data["paper_keys"],
                           data["texts"], np.asarray(data["embeddings"]))

        ids, paper_keys, texts, embeddings = [], [], [], []
        with driver.session() as session:
            rows = session.run(
                "MATCH (p:Paper)-[:HAS_CONTEXT]->(c:Chunk) "
                "WHERE c.embedding IS NOT NULL "
                "RETURN c.key AS key, p.key AS paper, c.text AS text, c.embedding AS emb"
            )
            for row in rows:
                ids.append(f"Chunks/{row['key']}")
                paper_keys.append(row["paper"])
                texts.append(row["text"])
                embeddings.append(row["emb"])

        embeddings_np = np.asarray(embeddings, dtype=np.float32)
        if cache_file and ids:
            _dump_vector_cache(cache_file, {"ids": ids, "paper_keys": paper_keys,
                                            "texts": texts, "embeddings": embeddings_np})
        return cls(ids, paper_keys, texts, embeddings_np)


class BaseRetriever(ABC):
    """encode -> (optional) rerank -> select -> build context -> answer."""

    name: str = "base"

    def __init__(self, store: ChunkStore, encoder, reranker=None,
                 top_k_final: int = TOP_K_FINAL,
                 top_k_candidates: int = TOP_K_CANDIDATES):
        self.store = store
        self.encoder = encoder
        self.reranker = reranker
        self.top_k_final = top_k_final
        self.top_k_candidates = top_k_candidates

    def _select(self, query: str) -> list[Candidate]:
        """Top-k chunks, optionally cross-encoder reranked from a wide pool."""
        query_emb = self.encoder.encode([query], normalize_embeddings=True)
        pool_k = self.top_k_candidates if self.reranker else self.top_k_final
        idxs = self.store.search(query_emb, pool_k)
        candidates = [self.store.candidate(i) for i in idxs]

        if self.reranker and candidates:
            scores = self.reranker.predict([[query, c.text] for c in candidates])
            order = np.argsort(scores)[::-1][:self.top_k_final]
            return [
                Candidate(candidates[i].chunk_id, candidates[i].paper_key,
                          candidates[i].text, float(scores[i]))
                for i in order
            ]
        return candidates[:self.top_k_final]

    @abstractmethod
    def _build_context(self, query: str, candidates: list[Candidate]) -> str:
        """Turn selected chunks into the LLM context string."""

    def retrieve(self, query: str) -> str:
        return self._build_context(query, self._select(query))

    def answer_benchmark(self, question: str) -> str:
        context = self.retrieve(question)
        return call_ollama(build_prompt(context, question),
                           system=BENCHMARK_SYSTEM_PROMPT)

    def chat(self, question: str, temperature: float = 0.3) -> dict:
        """Conversational answer plus the source paper pubids it retrieved.

        Runs retrieval once and returns the cited papers (their PubMedQA pubids,
        which are real PubMed IDs) so a UI can link back to the sources.
        """
        candidates = self._select(question)
        context = self._build_context(question, candidates)
        answer = call_ollama(build_prompt(context, question),
                             system=CHAT_SYSTEM_PROMPT, temperature=temperature)
        sources = list(dict.fromkeys(c.paper_key for c in candidates))
        return {"answer": answer, "sources": sources, "context": context}
