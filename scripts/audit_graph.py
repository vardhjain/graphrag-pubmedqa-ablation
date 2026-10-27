"""Read-only, fail-closed structural audit of the actual ArangoDB benchmark graph.

    python scripts/audit_graph.py --report results/graph_audit.json
    python scripts/audit_graph.py --allow-issues  # inspect a known dirty graph

No model, dataset download, embedding generation, ingestion, database creation,
or AQL writes. In contrast to comparing edge and chunk COUNTS, this checks
duplicate relationships, orphans, endpoint integrity, stored answer leakage,
and missing vectors. A passing audit is necessary but not sufficient to
replicate the historical benchmark's accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kgqa.config import ArangoConfig  # noqa: E402
from kgqa.models import connect_arango  # noqa: E402

COLLECTIONS = ("Papers", "Chunks", "Concepts", "HAS_CONTEXT", "MENTIONS")

# Only read operations; each query returns a count or a sequence of counts.
# The edge collection's built-in _to index makes the chunk orphan test cheap.
# Do not replace this with n_edges - n_chunks: duplicates and missing edges
# may cancel, producing an apparently clean count on a malformed graph.
CHECK_QUERIES = {
    "duplicate_context_relationships": """
        FOR e IN HAS_CONTEXT
            COLLECT source = e._from, target = e._to WITH COUNT INTO copies
            FILTER copies > 1
            RETURN copies - 1
    """,
    "context_edges_without_valid_endpoints": """
        FOR e IN HAS_CONTEXT
            FILTER !STARTS_WITH(e._from, "Papers/")
                OR !STARTS_WITH(e._to, "Chunks/")
                OR DOCUMENT(e._from) == null
                OR DOCUMENT(e._to) == null
            RETURN 1
    """,
    "context_edges_to_wrong_paper": """
        FOR e IN HAS_CONTEXT
            LET chunk = DOCUMENT(e._to)
            FILTER chunk != null
                AND CONCAT("Papers/", chunk.paper_key) != e._from
            RETURN 1
    """,
    "chunks_without_context_edges": """
        FOR c IN Chunks
            LET any_edge = (
                FOR e IN HAS_CONTEXT
                    FILTER e._to == c._id
                    LIMIT 1
                    RETURN 1
            )
            FILTER LENGTH(any_edge) == 0
            RETURN 1
    """,
    "chunks_with_multiple_context_edges": """
        FOR e IN HAS_CONTEXT
            COLLECT target = e._to WITH COUNT INTO copies
            FILTER copies > 1
            RETURN 1
    """,
    "chunks_without_embedding": """
        FOR c IN Chunks
            FILTER c.embedding == null
            RETURN 1
    """,
    "chunks_without_parent_key": """
        FOR c IN Chunks
            FILTER c.paper_key == null OR c.paper_key == ""
            RETURN 1
    """,
    "papers_with_leaked_answers_or_questions": """
        FOR p IN Papers
            FILTER HAS(p, "final_decision") OR HAS(p, "question") OR HAS(p, "title")
            RETURN 1
    """,
    "duplicate_mentions_relationships": """
        FOR e IN MENTIONS
            COLLECT source = e._from, target = e._to WITH COUNT INTO copies
            FILTER copies > 1
            RETURN copies - 1
    """,
    "mentions_without_valid_endpoints": """
        FOR e IN MENTIONS
            FILTER !STARTS_WITH(e._from, "Papers/")
                OR !STARTS_WITH(e._to, "Concepts/")
                OR DOCUMENT(e._from) == null
                OR DOCUMENT(e._to) == null
            RETURN 1
    """,
}


def audit_graph(db) -> dict:
    """Audit real graph relationships, not merely collection cardinality.

    Fail closed when a required collection is absent or an AQL query fails.
    Never return a green report on partial scan success.
    """
    missing = [name for name in COLLECTIONS if not db.has_collection(name)]
    if missing:
        raise ValueError(f"Missing required collections: {', '.join(missing)}")

    counts = {name: int(db.collection(name).count()) for name in COLLECTIONS}
    checks = {}
    for name, query in CHECK_QUERIES.items():
        checks[name] = sum(int(value) for value in db.aql.execute(query, batch_size=2000))
    ready = counts["Papers"] > 0 and counts["Chunks"] > 0 and all(
        value == 0 for value in checks.values()
    )
    return {
        "schema_version": "graph-audit-v1",
        "collections": counts,
        "checks": checks,
        "ready_for_benchmark": ready,
        "limitations": (
            "Structural audit only: does not verify PubMedQA dataset versions, "
            "embedding-model identity, corpus completeness or prompt/context parity. "
            "The historical accuracy result must be rerun after repairs."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=None, help="Write machine-readable audit report"
    )
    parser.add_argument(
        "--allow-issues",
        action="store_true",
        help="Return success for inspecting a historical/dirty graph; report remains flagged",
    )
    args = parser.parse_args()

    # connect_arango verifies the database exists. This audit never calls
    # create_database, setup_schema, ingest_split or REMOVE/TRUNCATE AQL.
    db = connect_arango(ArangoConfig())
    report = audit_graph(db)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    if not report["ready_for_benchmark"] and not args.allow_issues:
        print("Graph audit FAILED; do not run or publish benchmark comparisons.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
