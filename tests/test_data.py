"""Tests for dataset sampling and chunking — the `datasets` dependency is faked
so these run without it installed and without any download."""

from __future__ import annotations

import sys
import types


def _fake_datasets(monkeypatch, rows):
    mod = types.ModuleType("datasets")
    mod.load_dataset = lambda *a, **k: rows
    monkeypatch.setitem(sys.modules, "datasets", mod)


def test_load_benchmark_samples_seeded_and_filtered(monkeypatch):
    from kgqa import data

    rows = [{"pubid": i, "question": f"q{i}", "final_decision": ["yes", "no", "maybe"][i % 3]}
            for i in range(30)]
    rows.append({"pubid": 900, "question": "", "final_decision": "yes"})    # dropped: no question
    rows.append({"pubid": 901, "question": "x", "final_decision": None})    # dropped: no label
    _fake_datasets(monkeypatch, rows)

    a = data.load_benchmark_samples(n=10, seed=42)
    b = data.load_benchmark_samples(n=10, seed=42)
    assert len(a) == 10
    assert [s.pubid for s in a] == [s.pubid for s in b]          # deterministic
    assert all(s.question and s.final_decision for s in a)       # filtered
    assert all(isinstance(s.pubid, str) for s in a)              # pubid stringified
    assert data.load_benchmark_samples(n=10, seed=7) != a        # seed changes order


def test_iter_chunks_skips_empty_and_yields_indices(monkeypatch):
    from kgqa import data

    rows = [{"pubid": 5, "context": {"contexts": ["alpha", "beta", "  "]}}]
    _fake_datasets(monkeypatch, rows)

    chunks = list(data.iter_chunks(include_unlabeled=False))
    assert ("5", 0, "alpha") in chunks
    assert ("5", 1, "beta") in chunks
    assert len(chunks) == 2     # blank section dropped
