"""Reproducible fingerprints for benchmark input samples and stored chunks.

Hash content rather than file paths: caches and graph exports can move between
machines. A chunk-store fingerprint does NOT attest to graph edge integrity;
deduplicate/audit the graph independently before making accuracy claims.
"""

from __future__ import annotations

import hashlib
import json


def sample_fingerprint(samples) -> str:
    """Hash the ordered evaluation question, id, and ground truth triples."""
    values = [
        [str(s.pubid), s.question, s.final_decision]
        for s in samples
    ]
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def corpus_fingerprint(store) -> str:
    """Hash IDs, associated papers, text, and float32 embeddings in key order.

    Does not allocate an additional corpus-sized copy. Sorting makes the hash
    independent of ArangoDB's iteration order, without hiding content changes.
    """
    if len(store.ids) != len(set(store.ids)):
        raise ValueError("Duplicate chunk IDs in retrieval corpus")
    if not (
        len(store.ids) == len(store.paper_keys) == len(store.texts) == len(store.embeddings)
    ):
        raise ValueError("Inconsistent retrieval corpus array lengths")
    digest = hashlib.sha256()
    for index in sorted(range(len(store.ids)), key=store.ids.__getitem__):
        fields = (store.ids[index], store.paper_keys[index], store.texts[index])
        for field in fields:
            encoded = str(field).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        vector = store.embeddings[index].tobytes()
        digest.update(len(vector).to_bytes(8, "big"))
        digest.update(vector)
    return digest.hexdigest()
