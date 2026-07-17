"""Build the Neo4j graph for the hosted agent's demo -- labeled split ONLY.

This is deliberately scoped to ``pqa_labeled`` (1,000 papers, ~3,358 chunks,
~3,408 MeSH concepts -- see RESULTS.md for how this compares to the full
62k-paper corpus used in the benchmark ablation). That's small enough to
comfortably fit Neo4j AuraDB Free's limits and keeps the hosted demo cheap to
rebuild from scratch at any time.

This does NOT touch the benchmarked pipeline -- ``scripts/ingest.py`` (Arango)
is unchanged and still what regenerates RESULTS.md's numbers. This script
only feeds the live hosted agent (``kgqa/service.py``'s ``graph_id="demo"``
path via ``kgqa/retrieval/neo4j_graph.py``).

Run ONCE before the hosted demo can answer with real graph expansion:
    export NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=...
    python scripts/ingest_neo4j.py
"""

from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sentence_transformers import SentenceTransformer  # noqa: E402
from tqdm import tqdm  # noqa: E402

from kgqa.config import (  # noqa: E402
    DATASET_NAME,
    EMBEDDING_MODEL,
    LABELED_CONFIG,
    Neo4jConfig,
)
from kgqa.models import connect_neo4j  # noqa: E402


def setup_constraints(driver):
    with driver.session() as session:
        for label, prop in [("Paper", "key"), ("Chunk", "key"), ("Concept", "key")]:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
            )
    print("  constraints ready")


_MERGE_PAPERS = "UNWIND $rows AS row MERGE (p:Paper {key: row.key})"
_MERGE_CONCEPTS = """
UNWIND $rows AS row
MERGE (c:Concept {key: row.key})
SET c.name = row.name
"""
_MERGE_MENTIONS = """
UNWIND $rows AS row
MATCH (p:Paper {key: row.paper}), (c:Concept {key: row.concept})
MERGE (p)-[:MENTIONS]->(c)
"""
_MERGE_CHUNKS = """
UNWIND $rows AS row
MATCH (p:Paper {key: row.paper})
MERGE (c:Chunk {key: row.key})
SET c.text = row.text, c.embedding = row.embedding
MERGE (p)-[:HAS_CONTEXT]->(c)
"""


def ingest_labeled_split(driver, model, batch_size: int = 50) -> int:
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, LABELED_CONFIG, split="train")

    papers, concepts, mentions, chunk_rows = [], [], [], []
    count = 0

    def flush(session):
        if papers:
            session.run(_MERGE_PAPERS, rows=papers)
        if concepts:
            session.run(_MERGE_CONCEPTS, rows=concepts)
        if mentions:
            session.run(_MERGE_MENTIONS, rows=mentions)
        if chunk_rows:
            session.run(_MERGE_CHUNKS, rows=chunk_rows)
        for buf in (papers, concepts, mentions, chunk_rows):
            buf.clear()

    with driver.session() as session:
        for row in tqdm(ds):
            paper_key = str(row["pubid"])
            papers.append({"key": paper_key})

            for mesh in row.get("context", {}).get("meshes", []):
                mesh_key = "".join(c for c in mesh if c.isalnum())
                if not mesh_key:
                    continue
                concepts.append({"key": mesh_key, "name": mesh})
                mentions.append({"paper": paper_key, "concept": mesh_key})

            ctx_texts = row.get("context", {}).get("contexts", [])
            if ctx_texts:
                embeddings = model.encode(ctx_texts)
                for idx, (text, emb) in enumerate(zip(ctx_texts, embeddings, strict=False)):
                    chunk_rows.append({
                        "key": f"{paper_key}_{idx}",
                        "paper": paper_key,
                        "text": text,
                        "embedding": emb.tolist(),
                    })

            count += 1
            if count % batch_size == 0:
                flush(session)
        flush(session)
    return count


def main():
    cfg = Neo4jConfig()
    driver = connect_neo4j(cfg)
    setup_constraints(driver)

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Ingesting pqa_labeled (demo scope only)...")
    t0 = time.time()
    n = ingest_labeled_split(driver, model)
    print(f"  {n:,} papers in {time.time() - t0:.1f}s")

    with driver.session() as session:
        counts = session.run(
            "MATCH (p:Paper) WITH count(p) AS papers "
            "MATCH (c:Chunk) WITH papers, count(c) AS chunks "
            "MATCH (k:Concept) RETURN papers, chunks, count(k) AS concepts"
        ).single()
    print("\nNode counts:")
    print(f"  Paper   : {counts['papers']:>8,}")
    print(f"  Chunk   : {counts['chunks']:>8,}")
    print(f"  Concept : {counts['concepts']:>8,}")

    driver.close()


if __name__ == "__main__":
    main()
