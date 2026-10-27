"""Compare fresh graph vs flat-parent predictions on *identical* test cases.

Deliberately separate from scripts/compare.py: that script's published
four-arm Holm family and existing results/summary.json must not be changed
until a fully rerun, preregistered experiment is available.

Example:
    python scripts/compare_parent_baseline.py --flat results/flat_parent_results.json --graph results/graph_results.json

Fails closed if a historical graph run is supplied (missing the new protocol
marker), if model/inputs/settings differ, or if any call or graph expansion
failed. Both result files must have been generated afresh under one code
revision after a clean graph and the parity preflight.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kgqa.evaluation import mcnemar_test  # noqa: E402

# All of these are present in run_benchmark's current raw result metadata.
# Optional protocol-v2 fields, when present after PR #3, must also match.
REQUIRED_META = (
    "parent_control_protocol",
    "seed",
    "n_requested",
    "llm_model",
    "num_ctx",
    "num_predict",
    "temperature",
    "top_k",
    "top_k_candidates",
    "git_sha",
    "sample_sha256",
    "corpus_sha256",
)
OPTIONAL_META = (
    "protocol_version",
    "answer_parser",
    "prompt_sha256",
    "embedding_model",
    "reranker_model",
    "dataset_name",
)


def compare_parent_results(flat: dict, graph: dict) -> dict:
    """Compute an exploratory paired contrast, or reject invalid evidence."""
    expected = (("flat_parent", flat), ("graph", graph))
    for name, data in expected:
        meta = data.get("meta") or {}
        if data.get("model") != name or meta.get("arm") != name:
            raise ValueError(f"Expected arm {name!r} in result data and metadata")
        if meta.get("complete") is not True:
            raise ValueError(f"{name}: incomplete run")
        missing = [key for key in REQUIRED_META if meta.get(key) is None]
        if missing:
            raise ValueError(f"{name}: missing provenance {', '.join(missing)}")
        if meta["parent_control_protocol"] != "flat-parent-v1":
            raise ValueError(f"{name}: unknown parent-control protocol")
        if meta.get("n_done") != meta["n_requested"]:
            raise ValueError(f"{name}: not all requested samples were evaluated")
        n = meta["n_requested"]
        for field in ("ids", "y_true", "y_pred", "failed", "degraded", "truncated"):
            if not isinstance(data.get(field), list) or len(data[field]) != n:
                raise ValueError(f"{name}: {field} must contain all {n} sample records")
        if not all(isinstance(sid, str) and sid for sid in data["ids"]):
            raise ValueError(f"{name}: sample IDs must be nonempty strings")
        if len(set(data["ids"])) != n:
            raise ValueError(f"{name}: duplicate question IDs")
        for key in ("failed", "degraded", "truncated"):
            if any(data[key]):
                raise ValueError(f"{name}: {key} calls are not valid for this contrast")

    a, b = flat["meta"], graph["meta"]
    for key in REQUIRED_META + OPTIONAL_META:
        if a.get(key) != b.get(key):
            raise ValueError(f"Incompatible benchmark protocol: {key}")

    ids = flat["ids"]
    if graph["ids"] != ids:
        raise ValueError("Results are not aligned to the identical ordered question IDs")
    if graph["y_true"] != flat["y_true"]:
        raise ValueError("Ground-truth mismatch between arms")
    if not ids:
        raise ValueError("No paired observations")

    truth, pa, pb = flat["y_true"], flat["y_pred"], graph["y_pred"]
    stats = mcnemar_test(truth, pa, pb)
    n = len(ids)
    flat_correct = sum(p == g for p, g in zip(pa, truth, strict=True))
    graph_correct = sum(p == g for p, g in zip(pb, truth, strict=True))

    return {
        "status": "exploratory_requires_clean_graph_and_context_parity",
        "comparison": "flat_parent -> graph",
        "paired_n": n,
        "flat_accuracy_pct": round(flat_correct * 100 / n, 4),
        "graph_accuracy_pct": round(graph_correct * 100 / n, 4),
        "graph_minus_flat_pp": round((graph_correct - flat_correct) * 100 / n, 4),
        "gains": stats["b_gains"],
        "losses": stats["c_losses"],
        "mcnemar_exact_p": stats["p_value"],
        "multiplicity_adjustment": "none; standalone exploratory contrast",
        "provenance": {key: a[key] for key in REQUIRED_META},
        "limitations": [
            "Run the graph deduplication audit and no-LLM context-parity preflight first.",
            "This contrast isolates graph traversal vs flat parent lookup at equal context.",
            "Not part of the historical four-arm Holm-adjusted significance family.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat", type=Path, default=Path("results/flat_parent_results.json"))
    parser.add_argument("--graph", type=Path, default=Path("results/graph_results.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON; omitted by default to protect published artifacts",
    )
    args = parser.parse_args()
    with args.flat.open(encoding="utf-8") as f:
        flat = json.load(f)
    with args.graph.open(encoding="utf-8") as f:
        graph = json.load(f)
    report = compare_parent_results(flat, graph)
    encoded = json.dumps(report, indent=2) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
