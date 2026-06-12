"""Build the ArangoDB knowledge graph from PubMedQA — leakage-free schema.

Differences from the original ingestion (the fairness fixes):
  * Papers store NO question-derived title and NO final_decision, so the
    benchmark question/answer can never leak into a retrieved context.
  * Chunks carry an explicit ``paper_key`` for fast, unambiguous corpus loading.

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

from arango import ArangoClient  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from tqdm import tqdm  # noqa: E402

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


def ingest_split(db, dataset, model, on_duplicate_paper="ignore", batch_size=50):
    papers, chunks, concepts, has_ctx, mentions = [], [], [], [], []
    count = 0

    def flush():
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
            mentions.append({"_from": f"Papers/{paper_key}", "_to": f"Concepts/{mesh_key}"})

        ctx_texts = row.get("context", {}).get("contexts", [])
        ctx_labels = row.get("context", {}).get("labels", [])
        if ctx_texts:
            embeddings = model.encode(ctx_texts)
            for idx, (text, emb) in enumerate(zip(ctx_texts, embeddings, strict=False)):
                chunk_key = f"{paper_key}_{idx}"
                chunks.append({
                    "_key": chunk_key,
                    "paper_key": paper_key,
                    "text": text,
                    "label": ctx_labels[idx] if idx < len(ctx_labels) else "context",
                    "embedding": emb.tolist(),
                })
                has_ctx.append({"_from": f"Papers/{paper_key}", "_to": f"Chunks/{chunk_key}"})

        count += 1
        if count % batch_size == 0:
            flush()
    flush()
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest PubMedQA into ArangoDB.")
    parser.add_argument("--no-unlabeled", action="store_true",
                        help="Ingest only the labeled split (faster, for testing).")
    args = parser.parse_args()

    cfg = ArangoConfig()
    cfg.require_password()
    client = ArangoClient(hosts=cfg.host)
    sys_db = client.db("_system", username=cfg.user, password=cfg.password)
    if not sys_db.has_database(cfg.db_name):
        sys_db.create_database(cfg.db_name)
        print(f"created database: {cfg.db_name}")
    db = client.db(cfg.db_name, username=cfg.user, password=cfg.password)

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


if __name__ == "__main__":
    main()
