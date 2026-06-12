"""Shared fakes so the suite runs on CPU with no Ollama, ArangoDB, or ML deps."""

from __future__ import annotations

import numpy as np
import pytest


class FakeEncoder:
    """Deterministic hashing encoder — stable vectors without downloading a model."""

    dim = 16

    def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True,
               batch_size=32, show_progress_bar=False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = np.zeros((len(items), self.dim), dtype=np.float32)
        for i, t in enumerate(items):
            for token in str(t).lower().split():
                vecs[i, hash(token) % self.dim] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms
        return vecs[0] if single else vecs


class FakeReranker:
    """Scores by lexical overlap between query and candidate text."""

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            q = set(str(query).lower().split())
            d = set(str(text).lower().split())
            scores.append(float(len(q & d)))
        return np.array(scores)


class FakeAQL:
    def __init__(self, db):
        self.db = db

    def execute(self, query, bind_vars=None, **kwargs):
        bind_vars = bind_vars or {}
        # Parent expansion: map chunk ids -> parent paper full abstracts.
        if "INBOUND chunk" in query:
            seen, out = set(), []
            for cid in bind_vars["ids"]:
                pkey = cid.split("/")[-1].rsplit("_", 1)[0]
                if pkey in seen:
                    continue
                seen.add(pkey)
                out.append({"paper": pkey, "abstract": self.db.abstracts[pkey]})
            return out
        # Concept hop: return configured neighbours for the seed papers.
        if "@mentions" in query or "mentions" in query.lower():
            seeds = set(bind_vars["paper_keys"])
            out = []
            for nkey, abstract in self.db.neighbours:
                if nkey not in seeds:
                    out.append({"paper": nkey, "abstract": abstract, "shared": 1})
            return out[: bind_vars.get("limit", 3)]
        return []


class FakeDB:
    """Minimal ArangoDB stand-in for graph-expansion tests."""

    def __init__(self, abstracts, neighbours=()):
        self.abstracts = abstracts  # {paper_key: full abstract}
        self.neighbours = list(neighbours)  # [(paper_key, abstract), ...]
        self.aql = FakeAQL(self)


@pytest.fixture
def fake_encoder():
    return FakeEncoder()


@pytest.fixture
def fake_reranker():
    return FakeReranker()
