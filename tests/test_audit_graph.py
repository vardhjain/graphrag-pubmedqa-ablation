"""Contract tests for the read-only ArangoDB graph audit.

A malformed graph can have the same number of chunks and context edges:
duplicates cancel orphaned chunks. Never mark that graph clean.
"""

from __future__ import annotations

import pytest

from scripts.audit_graph import CHECK_QUERIES, COLLECTIONS, audit_graph


class FakeAQL:
    def __init__(self, issues):
        self.issues = issues
        self.scans = []

    def execute(self, query, **kwargs):
        self.scans.append((query, kwargs))
        for key, statement in CHECK_QUERIES.items():
            if query == statement:
                return list(self.issues.get(key, []))
        raise AssertionError("Unrecognized audit query")


class FakeCollection:
    def __init__(self, count):
        self.value = count

    def count(self):
        return self.value


class FakeAuditDB:
    def __init__(self, issues=None, missing=(), counts=None):
        self.counts = counts or {
            "Papers": 2,
            "Chunks": 3,
            "Concepts": 2,
            "HAS_CONTEXT": 3,
            "MENTIONS": 2,
        }
        self.missing = set(missing)
        self.aql = FakeAQL(issues or {})

    def has_collection(self, name):
        return name in self.counts and name not in self.missing

    def collection(self, name):
        return FakeCollection(self.counts[name])


def test_graph_audit_good_structure_only_returns_clean():
    db = FakeAuditDB()
    report = audit_graph(db)
    assert report["schema_version"] == "graph-audit-v1"
    assert report["ready_for_benchmark"] is True
    assert report["collections"]["Chunks"] == 3
    assert set(report["checks"]) == set(CHECK_QUERIES)
    assert all(v == 0 for v in report["checks"].values())
    assert len(db.aql.scans) == len(CHECK_QUERIES)


def test_equal_edge_chunk_counts_cannot_hide_duplicates_and_orphans():
    # Three chunks / three edges in aggregate, but one paper has a duplicate
    # link while another chunk has no context edge. Old check_duplicates()
    # would say "OK" based only on 3 == 3.
    db = FakeAuditDB(
        issues={
            "duplicate_context_relationships": [1],
            "chunks_without_context_edges": [1],
            "chunks_with_multiple_context_edges": [1],
        }
    )
    report = audit_graph(db)
    assert report["collections"]["HAS_CONTEXT"] == report["collections"]["Chunks"]
    assert report["ready_for_benchmark"] is False
    assert report["checks"]["duplicate_context_relationships"] == 1
    assert report["checks"]["chunks_without_context_edges"] == 1


@pytest.mark.parametrize(
    "issue",
    [
        "context_edges_without_valid_endpoints",
        "context_edges_to_wrong_paper",
        "chunks_without_embedding",
        "chunks_without_parent_key",
        "papers_with_leaked_answers_or_questions",
        "duplicate_mentions_relationships",
        "mentions_without_valid_endpoints",
    ],
)
def test_every_integrity_failure_blocks_a_clean_graph_claim(issue):
    assert audit_graph(FakeAuditDB(issues={issue: [1]}))["ready_for_benchmark"] is False


def test_duplicate_excess_sums_every_repeat_group():
    db = FakeAuditDB(issues={"duplicate_context_relationships": [2, 4]})
    assert audit_graph(db)["checks"]["duplicate_context_relationships"] == 6


def test_missing_collection_fails_closed():
    db = FakeAuditDB(missing=("HAS_CONTEXT",))
    with pytest.raises(ValueError, match="HAS_CONTEXT"):
        audit_graph(db)
    assert db.aql.scans == []


def test_empty_graph_is_not_ready():
    db = FakeAuditDB(counts={name: 0 for name in COLLECTIONS})
    assert audit_graph(db)["ready_for_benchmark"] is False


def test_audit_queries_never_mutate_database():
    banned = (" REMOVE ", " UPDATE ", " UPSERT ", " REPLACE ", " INSERT ", " TRUNCATE ")
    for query in CHECK_QUERIES.values():
        padded = " " + " ".join(query.upper().split()) + " "
        assert not any(word in padded for word in banned)
    assert "FILTER e._to == c._id" in CHECK_QUERIES["chunks_without_context_edges"]
