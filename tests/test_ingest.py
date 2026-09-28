"""Tests for scripts/ingest.py's idempotency: deterministic edge keys mean a
re-run can't append duplicate edges. Fake DB/model -- no ArangoDB needed."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from scripts.ingest import check_duplicates, has_context_key, ingest_split, mentions_key


class _Col:
    def __init__(self):
        self.docs: dict = {}
        self.n_unkeyed = 0

    def import_bulk(self, docs, on_duplicate="ignore"):
        for d in docs:
            if "_key" not in d:
                self.n_unkeyed += 1
                self.docs[f"auto{self.n_unkeyed}"] = d
            elif d["_key"] not in self.docs or on_duplicate == "update":
                self.docs[d["_key"]] = d

    def count(self):
        return len(self.docs)


class _DB:
    def __init__(self):
        self.cols = {c: _Col() for c in ("Papers", "Chunks", "Concepts", "HAS_CONTEXT", "MENTIONS")}

    def collection(self, name):
        return self.cols[name]


class _Model:
    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.zeros((len(texts), 4))


ROWS = [
    {"pubid": 1, "context": {"contexts": ["a", "b"], "labels": ["BG"], "meshes": ["Humans", "X-1"]}},
    {"pubid": 2, "context": {"contexts": ["c"], "labels": [], "meshes": ["Humans"]}},
]


@pytest.fixture(autouse=True)
def _fake_tqdm(monkeypatch):
    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=lambda it: it))


def test_edge_keys_are_deterministic():
    assert has_context_key("123_0") == "123_0"
    assert mentions_key("123", "Humans") == "123-Humans"


def test_reingest_is_idempotent():
    db, model = _DB(), _Model()
    ingest_split(db, ROWS, model)
    ingest_split(db, ROWS, model, on_duplicate_paper="update")  # the notebook's second pass
    assert db.cols["Chunks"].count() == 3
    assert db.cols["HAS_CONTEXT"].count() == 3
    assert db.cols["MENTIONS"].count() == 3
    assert db.cols["HAS_CONTEXT"].n_unkeyed == 0 and db.cols["MENTIONS"].n_unkeyed == 0
    assert db.cols["HAS_CONTEXT"].docs["1_1"]["_to"] == "Chunks/1_1"
    assert "embedding" in db.cols["Chunks"].docs["2_0"]
    assert model.calls == 2  # one batched encode per flush, not per paper


def test_check_duplicates_reports_surplus(capsys):
    db = _DB()
    db.cols["Chunks"].docs = {"a": {}, "b": {}}
    db.cols["HAS_CONTEXT"].docs = {"x": {}, "y": {}, "z": {}}
    assert check_duplicates(db) == 1
    out = capsys.readouterr().out
    assert "REMOVE" in out and "surplus: +1" in out
