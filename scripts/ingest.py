"""Build the ArangoDB knowledge graph from PubMedQA — leakage-free schema.

Differences from the original ingestion (the fairness fixes):
  * Papers store NO question-derived title and NO final_decision, so the
    benchmark question/answer can never leak into a retrieved context.
  * Chunks carry an explicit ``paper_key`` for fast, unambiguous corpus loading.
  * Edges get deterministic ``_key``s (HAS_CONTEXT: the chunk key; MENTIONS:
    ``{paper_key}-{mesh_key}``), so re-running ingest is idempotent. Before
    this, edges had no _key and every re-run appended duplicates (the published
    DB had 209,971 HAS_CONTEXT edges for 206,613 Chunks -- the 3,358 surplus is
    exactly the labeled split, which the Colab notebook ingests twice).

Audit an existing DB for duplicate edges without ingesting anything:
    python scripts/ingest.py --check-duplicates

Run ONCE before benchmarking:
    export ARANGO_PASS=...   # or set in PowerShell / Colab Secrets
    python scripts/ingest.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kgqa.config import (  # noqa: E402
    DATASET_NAME,
    EDGE_COLLECTIONS,
    EMBEDDING_MODEL,
    LABELED_CONFIG,
    NODE_COLLECTIONS,
    UNLABELED_CONFIG,
    ArangoConfig,
)


def setup_schema(db):
    for col in NODE_COLLECTIONS:
        if not db.has_collection(col):
            db.create_collection(col)
            print(f"  created node collection: {col}")
    for col in EDGE_COLLECTIONS:
        if not db.has_collection(col):
            db.create_collection(col, edge=True)
            print(f"  created edge collection: {col}")


DEDUPE_AQL = {
    "HAS_CONTEXT": (
        "FOR e IN HAS_CONTEXT COLLECT f = e._from, t = e._to INTO g "
        "FILTER LENGTH(g) > 1 FOR x IN SLICE(g, 1) REMOVE x.e IN HAS_CONTEXT"
    ),
    "MENTIONS": (
        "FOR e IN MENTIONS COLLECT f = e._from, t = e._to INTO g "
        "FILTER LENGTH(g) > 1 FOR x IN SLICE(g, 1) REMOVE x.e IN MENTIONS"
    ),
}


def has_context_key(chunk_key: str) -> str:
    """Deterministic HAS_CONTEXT edge key: one edge per chunk."""
    return chunk_key


def mentions_key(paper_key: str, mesh_key: str) -> str:
    """Deterministic MENTIONS edge key: one edge per (paper, concept)."""
    return f"{paper_key}-{mesh_key}"


def check_duplicates(db) -> int:
    """Report HAS_CONTEXT vs Chunks counts (should be equal: one edge per
    chunk). Returns the surplus edge count and prints the dedupe AQL."""
    n_edges = db.collection("HAS_CONTEXT").count()
    n_chunks = db.collection("Chunks").count()
    surplus = n_edges - n_chunks
    print(f"\nHAS_CONTEXT: {n_edges:,}   Chunks: {n_chunks:,}   surplus: {surplus:+,}")
    if surplus > 0:
        print("  Duplicate HAS_CONTEXT edges detected (legacy ingest without edge keys).")
        print("  Dedupe with (keeps one edge per _from/_to pair):")
        print(f"    {DEDUPE_AQL['HAS_CONTEXT']}")
        print("  MENTIONS may have the same problem:")
        print(f"    {DEDUPE_AQL['MENTIONS']}")
        print(
            "  Note: legacy edges keep their random _keys, so a later (keyed) re-ingest\n"
            "  would add keyed copies. To migrate fully, TRUNCATE HAS_CONTEXT and\n"
            "  MENTIONS and re-run this script instead (nodes are kept; edges rebuilt)."
        )
    elif surplus < 0:
        print("  Fewer edges than chunks -- some chunks are orphaned (ingest incomplete?).")
    else:
        print("  OK: exactly one HAS_CONTEXT edge per chunk.")
    return surplus


def ingest_split(db, dataset, model, on_duplicate_paper="ignore", batch_size=50):
    from tqdm import tqdm

    papers, chunks, concepts, has_ctx, mentions = [], [], [], [], []
    count = 0

    def flush():
        if chunks:
            # One encode call per flush (not per paper) -- same model, same
            # vectors, far fewer tiny batches.
            embeddings = model.encode([c["text"] for c in chunks])
            for c, emb in zip(chunks, embeddings, strict=True):
                c["embedding"] = emb.tolist()
        if papers:
            db.collection("Papers").import_bulk(papers, on_duplicate=on_duplicate_paper)
        if concepts:
            db.collection("Concepts").import_bulk(concepts, on_duplicate="ignore")
        if chunks:
            db.collection("Chunks").import_bulk(chunks, on_duplicate="ignore")
        if has_ctx:
            db.collection("HAS_CONTEXT").import_bulk(has_ctx, on_duplicate="ignore")
        if mentions:
            db.collection("MENTIONS").import_bulk(mentions, on_duplicate="ignore")
        for buf in (papers, chunks, concepts, has_ctx, mentions):
            buf.clear()

    for row in tqdm(dataset):
        paper_key = str(row["pubid"])
        # Leakage-free Paper node: no title, no final_decision.
        papers.append({"_key": paper_key})

        for mesh in row.get("context", {}).get("meshes", []):
            mesh_key = "".join(c for c in mesh if c.isalnum())
            if not mesh_key:
                continue
            concepts.append({"_key": mesh_key, "name": mesh})
            mentions.append(
                {
                    "_key": mentions_key(paper_key, mesh_key),
                    "_from": f"Papers/{paper_key}",
                    "_to": f"Concepts/{mesh_key}",
                }
            )

        ctx_texts = row.get("context", {}).get("contexts", [])
        ctx_labels = row.get("context", {}).get("labels", [])
        for idx, text in enumerate(ctx_texts):
            chunk_key = f"{paper_key}_{idx}"
            chunks.append(
                {
                    "_key": chunk_key,
                    "paper_key": paper_key,
                    "text": text,
                    "label": ctx_labels[idx] if idx < len(ctx_labels) else "context",
                }
            )
            has_ctx.append(
                {
                    "_key": has_context_key(chunk_key),
                    "_from": f"Papers/{paper_key}",
                    "_to": f"Chunks/{chunk_key}",
                }
            )

        count += 1
        if count % batch_size == 0:
            flush()
    flush()
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest PubMedQA into ArangoDB.")
    parser.add_argument(
        "--no-unlabeled",
        action="store_true",
        help="Ingest only the labeled split (faster, for testing).",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Only audit HAS_CONTEXT vs Chunks counts and print dedupe AQL.",
    )
    args = parser.parse_args()

    # Heavy imports are local so the helpers above stay importable in CI.
    from arango import ArangoClient
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    cfg = ArangoConfig()
    cfg.require_password()
    client = ArangoClient(hosts=cfg.host)
    sys_db = client.db("_system", username=cfg.user, password=cfg.password)
    if not sys_db.has_database(cfg.db_name):
        sys_db.create_database(cfg.db_name)
        print(f"created database: {cfg.db_name}")
    db = client.db(cfg.db_name, username=cfg.user, password=cfg.password)

    if args.check_duplicates:
        check_duplicates(db)
        return

    setup_schema(db)
    model = SentenceTransformer(EMBEDDING_MODEL)

    if not args.no_unlabeled:
        print("Ingesting pqa_unlabeled...")
        ds = load_dataset(DATASET_NAME, UNLABELED_CONFIG, split="train")
        t0 = time.time()
        n = ingest_split(db, ds, model, on_duplicate_paper="ignore")
        print(f"  {n:,} papers in {time.time() - t0:.1f}s")

    print("Ingesting pqa_labeled...")
    ds = load_dataset(DATASET_NAME, LABELED_CONFIG, split="train")
    t0 = time.time()
    n = ingest_split(db, ds, model, on_duplicate_paper="update")
    print(f"  {n:,} papers in {time.time() - t0:.1f}s")

    print("\nCollection counts:")
    for col in (*NODE_COLLECTIONS, *EDGE_COLLECTIONS):
        print(f"  {col:<15}: {db.collection(col).count():>8,}")
    check_duplicates(db)


if __name__ == "__main__":
    main()
