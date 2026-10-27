"""Content fingerprints make comparability explicit without a running database."""

from types import SimpleNamespace

import numpy as np
import pytest

from kgqa.provenance import corpus_fingerprint, sample_fingerprint


def _store(order=(0, 1), change=False):
    items = [
        ("a", "p1", "First section", [0.5, 0.5]),
        ("b", "p2", "Second section" if not change else "Changed", [0.2, 0.8]),
    ]
    selected = [items[i] for i in order]
    return SimpleNamespace(
        ids=[item[0] for item in selected],
        paper_keys=[item[1] for item in selected],
        texts=[item[2] for item in selected],
        embeddings=np.array([item[3] for item in selected], dtype=np.float32),
    )


def test_sample_hash_depends_on_question_and_label():
    one = [SimpleNamespace(pubid="123", question="Is it effective?", final_decision="yes")]
    two = [SimpleNamespace(pubid="123", question="Is it effective?", final_decision="no")]
    assert sample_fingerprint(one) == sample_fingerprint(one)
    assert sample_fingerprint(one) != sample_fingerprint(two)


def test_corpus_hash_is_independent_of_db_scan_order():
    assert corpus_fingerprint(_store()) == corpus_fingerprint(_store((1, 0)))
    assert corpus_fingerprint(_store()) != corpus_fingerprint(_store(change=True))


def test_corpus_hash_rejects_duplicate_ids():
    store = _store()
    store.ids[1] = store.ids[0]
    with pytest.raises(ValueError, match="Duplicate"):
        corpus_fingerprint(store)
