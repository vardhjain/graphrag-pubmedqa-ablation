"""No-LLM preflight: ensure graph and flat parent contexts are byte-identical.

This requires your existing ArangoDB corpus but does NOT load embedding
models, call Ollama, or spend inference compute. It fails on duplicate graph
edges, orphaned sections, ordering disagreement, or missing parents.
Run before a matched fresh graph/flat_parent benchmark.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kgqa.config import ArangoConfig  # noqa: E402
from kgqa.models import connect_arango  # noqa: E402
from kgqa.retrieval import ChunkStore, FlatParentRetriever, GraphRetriever  # noqa: E402


def check_parent_context_parity(db, store, *, n: int = 100, seed: int = 42) -> dict:
    """Check that the exact graph and flat prompt contexts agree on sampled chunks."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not len(store):
        raise ValueError("empty chunk store")

    graph = GraphRetriever(store, encoder=None, db=db)
    flat = FlatParentRetriever(store, encoder=None)
    indices = random.Random(seed).sample(range(len(store)), min(n, len(store)))
    for pos in indices:
        candidate = store.candidate(pos)
        a = graph._build_context("", [candidate])
        b = flat._build_context("", [candidate])
        if getattr(graph, "_last_degraded", False) or getattr(flat, "_last_degraded", False):
            raise ValueError(
                f"Graph or flat parent expansion degraded for {candidate.chunk_id}; "
                "repair the corpus before any paired evaluation"
            )
        if a != b:
            raise ValueError(
                f"Context mismatch for {candidate.chunk_id}. This may indicate "
                "duplicated HAS_CONTEXT edges, orphaned nodes or section-order "
                "differences. Do not compare LLM predictions until resolved."
            )
    return {"checked_chunks": len(indices), "seed": seed, "identical_contexts": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", default=None, help="Existing chunk vector cache, if any")
    args = parser.parse_args()
    db = connect_arango(ArangoConfig())
    store = ChunkStore.from_arango(db, cache_file=args.cache)
    report = check_parent_context_parity(db, store, n=args.n, seed=args.seed)
    print(
        f"Context parity OK: {report['checked_chunks']} sampled chunks, "
        f"seed={report['seed']}. No LLM calls."
    )


if __name__ == "__main__":
    main()
