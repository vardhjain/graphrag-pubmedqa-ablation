"""Run the actual AQL and Cypher against disposable, localhost-only DBs.

Opt in with KGQA_RUN_DB_INTEGRATION=1. GitHub Actions starts two dedicated
Docker services; ordinary pytest needs neither servers nor DB drivers.
No production or hosted database URL is accepted by these tests.
"""

from __future__ import annotations

import os
import time
import uuid
from urllib.parse import urlparse

import numpy as np
import pytest

from scripts.ingest import ingest_split, setup_schema
from scripts.ingest_neo4j import (
    _MERGE_CHUNKS,
    _MERGE_CONCEPTS,
    _MERGE_MENTIONS,
    _MERGE_PAPERS,
    setup_constraints,
)

from kgqa.retrieval.graph import GraphRetriever
from kgqa.retrieval.neo4j_graph import Neo4jGraphRetriever

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("KGQA_RUN_DB_INTEGRATION") != "1",
        reason="Requires local databases; opt in with KGQA_RUN_DB_INTEGRATION=1",
    ),
]


def _localhost_url(value: str) -> str:
    host = urlparse(value).hostname
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Database integration tests must use a localhost endpoint")
    return value


def _retry(operation, attempts=40):
    """Allow clean Neo4j and Arango containers to finish initial startup."""
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2)
    raise AssertionError("unreachable")


@pytest.fixture
def arango_db():
    arango = pytest.importorskip("arango")
    url = _localhost_url(os.environ.get("KGQA_TEST_ARANGO_URL", "http://127.0.0.1:8529"))
    user = "root"
    password = os.environ.get("KGQA_TEST_ARANGO_PASSWORD", "integration_only_password")
    client = arango.ArangoClient(hosts=url)
    system = client.db("_system", username=user, password=password)
    _retry(system.version)
    name = "kgqa_ci_" + uuid.uuid4().hex[:16]
    system.create_database(name)
    try:
        db = client.db(name, username=user, password=password)
        setup_schema(db)
        yield db
    finally:
        system.delete_database(name)


@pytest.fixture
def neo4j_graph():
    neo4j = pytest.importorskip("neo4j")
    url = _localhost_url(os.environ.get("KGQA_TEST_NEO4J_URI", "bolt://127.0.0.1:7687"))
    password = os.environ.get("KGQA_TEST_NEO4J_PASSWORD", "integration_only_password")
    driver = neo4j.GraphDatabase.driver(url, auth=("neo4j", password))
    _retry(driver.verify_connectivity)
    # Every graph node in this fixture has a unique prefix; teardown never
    # deletes nodes belonging to other test runs or the rest of the DB.
    prefix = "ci_" + uuid.uuid4().hex[:12] + "_"
    setup_constraints(driver)
    try:
        yield driver, prefix
    finally:
        with driver.session() as session:
            session.run(
                "MATCH (n) WHERE n.key STARTS WITH $prefix DETACH DELETE n",
                prefix=prefix,
            ).consume()
        driver.close()


class TinyEncoder:
    def encode(self, texts):
        return np.asarray([[float(i + 1), 0.5] for i in range(len(texts))])


def _record(paper, texts, concepts=()):
    return {
        "pubid": paper,
        "context": {
            "contexts": texts,
            "labels": ["section"] * len(texts),
            "meshes": list(concepts),
        },
        # The fixture intentionally includes a forbidden answer-bearing field:
        # ingestion must never store it on its Paper nodes.
        "final_decision": "yes",
    }


def _neo4j_merge(driver, papers, chunks, mentions, concepts):
    """Exercise the exact Cypher used by the real demo ingestion script."""
    with driver.session() as session:
        session.run(_MERGE_PAPERS, rows=[{"key": p} for p in papers]).consume()
        session.run(
            _MERGE_CONCEPTS,
            rows=[{"key": c, "name": c} for c in concepts],
        ).consume()
        session.run(
            _MERGE_MENTIONS,
            rows=[{"paper": p, "concept": c} for p, c in mentions],
        ).consume()
        session.run(
            _MERGE_CHUNKS,
            rows=[
                {"key": key, "paper": paper, "text": text, "embedding": [0.1, 0.2]}
                for key, paper, text in chunks
            ],
        ).consume()


def test_arango_ingestion_is_idempotent_and_does_not_leak_labels(arango_db):
    rows = [_record("p1", ["methods", "outcome"], ["ConceptA", "ConceptB"])]
    for _ in range(2):
        assert ingest_split(arango_db, rows, TinyEncoder(), batch_size=1) == 1

    assert arango_db.collection("Papers").count() == 1
    assert arango_db.collection("Chunks").count() == 2
    assert arango_db.collection("Concepts").count() == 2
    assert arango_db.collection("HAS_CONTEXT").count() == 2
    assert arango_db.collection("MENTIONS").count() == 2
    assert "final_decision" not in arango_db.collection("Papers").get("p1")

    retriever = GraphRetriever(store=None, encoder=None, db=arango_db)
    assert retriever._parent_abstracts(["Chunks/p1_0", "Chunks/p1_1"]) == [
        ("p1", "methods outcome")
    ]


@pytest.mark.xfail(
    strict=True,
    reason="Known AQL bug: c._key sorts section _10 before _2 (GAPS #18)",
)
def test_arango_parent_abstract_has_numerical_section_order(arango_db):
    texts = [f"section{i}" for i in range(12)]
    ingest_split(arango_db, [_record("order", texts)], TinyEncoder(), batch_size=1)
    retriever = GraphRetriever(store=None, encoder=None, db=arango_db)
    assert retriever._parent_abstracts(["Chunks/order_0"]) == [("order", " ".join(texts))]


@pytest.mark.xfail(
    strict=True,
    reason="Known AQL bug: counts seed paths, not DISTINCT shared concepts (GAPS #9)",
)
def test_arango_concept_neighbours_rank_distinct_shared_mesh(arango_db):
    # All three seeds share A. Neighbour n1 shares only A (1 DISTINCT concept,
    # but 3 paths); n2 shares B and C with seed s1 (2 DISTINCT concepts).
    rows = [
        _record("s1", ["seed 1"], ["A", "B", "C"]),
        _record("s2", ["seed 2"], ["A"]),
        _record("s3", ["seed 3"], ["A"]),
        _record("n1", ["neighbour 1"], ["A"]),
        _record("n2", ["neighbour 2"], ["B", "C"]),
    ]
    ingest_split(arango_db, rows, TinyEncoder(), batch_size=2)
    retriever = GraphRetriever(
        store=None, encoder=None, db=arango_db, use_concepts=True, concept_hop_papers=2
    )
    assert [key for key, _ in retriever._concept_neighbours(["s1", "s2", "s3"])] == [
        "n2",
        "n1",
    ]


def test_neo4j_merge_is_idempotent_and_parent_deduplicates(neo4j_graph):
    driver, prefix = neo4j_graph
    paper = prefix + "paper"
    chunks = [(paper + "_0", paper, "first"), (paper + "_1", paper, "second")]
    concept = prefix + "concept"
    for _ in range(2):
        _neo4j_merge(driver, [paper], chunks, [(paper, concept)], [concept])

    with driver.session() as session:
        row = session.run(
            "MATCH (p:Paper {key: $key}) "
            "OPTIONAL MATCH (p)-[h:HAS_CONTEXT]->(c:Chunk) "
            "WITH p, count(DISTINCT c) AS chunks, count(DISTINCT h) AS context_edges "
            "OPTIONAL MATCH (p)-[m:MENTIONS]->(:Concept) "
            "RETURN chunks, context_edges, count(m) AS mentions",
            key=paper,
        ).single()
    assert row["chunks"] == 2
    assert row["context_edges"] == 2
    assert row["mentions"] == 1

    retriever = Neo4jGraphRetriever(store=None, encoder=None, driver=driver)
    assert retriever._parent_abstracts([f"Chunks/{paper}_0", f"Chunks/{paper}_1"]) == [
        (paper, "first second")
    ]


@pytest.mark.xfail(
    strict=True,
    reason="Known Cypher bug: ORDER BY c.key sorts section _10 before _2",
)
def test_neo4j_parent_abstract_has_numerical_section_order(neo4j_graph):
    driver, prefix = neo4j_graph
    paper = prefix + "order"
    texts = [f"section{i}" for i in range(12)]
    chunks = [(f"{paper}_{i}", paper, text) for i, text in enumerate(texts)]
    _neo4j_merge(driver, [paper], chunks, [], [])

    retriever = Neo4jGraphRetriever(store=None, encoder=None, driver=driver)
    assert retriever._parent_abstracts([f"Chunks/{paper}_0"]) == [(paper, " ".join(texts))]


def test_neo4j_concept_neighbours_count_distinct_mesh(neo4j_graph):
    driver, prefix = neo4j_graph
    s1, s2, s3 = (prefix + k for k in ("s1", "s2", "s3"))
    n1, n2 = (prefix + k for k in ("n1", "n2"))
    a, b, c = (prefix + k for k in ("A", "B", "C"))
    papers = [s1, s2, s3, n1, n2]
    chunks = [(p + "_0", p, p) for p in papers]
    mentions = [
        (s1, a),
        (s1, b),
        (s1, c),
        (s2, a),
        (s3, a),
        (n1, a),
        (n2, b),
        (n2, c),
    ]
    _neo4j_merge(driver, papers, chunks, mentions, [a, b, c])
    retriever = Neo4jGraphRetriever(
        store=None, encoder=None, driver=driver, use_concepts=True, concept_hop_papers=2
    )
    assert [key for key, _ in retriever._concept_neighbours([s1, s2, s3])] == [
        n2,
        n1,
    ]
