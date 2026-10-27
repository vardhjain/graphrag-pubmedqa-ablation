"""Safe paired comparison refuses historical and mismatched results."""

from __future__ import annotations

import copy

import pytest

from scripts.compare_parent_baseline import compare_parent_results


def _result(arm, predictions):
    n = len(predictions)
    return {
        "model": arm,
        "meta": {
            "arm": arm,
            "parent_control_protocol": "flat-parent-v1",
            "seed": 42,
            "n_requested": n,
            "n_done": n,
            "complete": True,
            "llm_model": "test-llm",
            "num_ctx": 8192,
            "num_predict": 1024,
            "temperature": 0.0,
            "top_k": 3,
            "top_k_candidates": 75,
            "git_sha": "shared_sha",
            "sample_sha256": "a" * 64,
            "corpus_sha256": "b" * 64,
        },
        "ids": [f"pubid{i}" for i in range(n)],
        "y_true": ["yes", "no", "yes", "no"][:n],
        "y_pred": predictions,
        "failed": [False] * n,
        "degraded": [False] * n,
        "truncated": [False] * n,
    }


def test_valid_negative_result_is_not_filtered():
    flat = _result("flat_parent", ["yes", "no", "yes", "no"])
    graph = _result("graph", ["no", "no", "no", "no"])
    out = compare_parent_results(flat, graph)
    assert out["paired_n"] == 4
    assert out["flat_accuracy_pct"] == 100.0
    assert out["graph_accuracy_pct"] == 50.0
    assert out["graph_minus_flat_pp"] == -50.0
    assert out["gains"] == 0 and out["losses"] == 2
    assert out["multiplicity_adjustment"].startswith("none")


def test_nonsignificant_results_are_valid():
    flat = _result("flat_parent", ["yes", "no", "yes", "no"])
    graph = _result("graph", ["yes", "no", "yes", "no"])
    out = compare_parent_results(flat, graph)
    assert out["graph_minus_flat_pp"] == 0
    assert out["mcnemar_exact_p"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 9),
        ("llm_model", "different"),
        ("num_ctx", 4096),
        ("temperature", 0.9),
        ("sample_sha256", "z" * 64),
        ("corpus_sha256", "z" * 64),
        ("git_sha", "other"),
    ],
)
def test_rejects_protocol_drift(field, value):
    flat = _result("flat_parent", ["yes", "no", "yes", "no"])
    graph = _result("graph", ["yes", "no", "yes", "no"])
    graph["meta"][field] = value
    with pytest.raises(ValueError, match=field):
        compare_parent_results(flat, graph)


def test_rejects_old_graph_results_without_protocol():
    flat = _result("flat_parent", ["yes", "no"])
    graph = _result("graph", ["yes", "no"])
    graph["meta"].pop("parent_control_protocol")
    with pytest.raises(ValueError, match="missing provenance"):
        compare_parent_results(flat, graph)


def test_rejects_missing_protocol_or_parser_metadata_when_one_arm_has_it():
    flat = _result("flat_parent", ["yes", "no"])
    graph = _result("graph", ["yes", "no"])
    graph["meta"]["answer_parser"] = "strict-v2"
    with pytest.raises(ValueError, match="answer_parser"):
        compare_parent_results(flat, graph)


@pytest.mark.parametrize("flag", ["failed", "degraded", "truncated"])
def test_rejects_unreliable_or_incomplete_calls(flag):
    flat = _result("flat_parent", ["yes", "no", "yes"])
    graph = _result("graph", ["yes", "no", "yes"])
    graph[flag][1] = True
    with pytest.raises(ValueError, match=flag):
        compare_parent_results(flat, graph)


def test_rejects_incomplete_runs():
    flat = _result("flat_parent", ["yes", "no"])
    graph = _result("graph", ["yes", "no"])
    graph["meta"]["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        compare_parent_results(flat, graph)


def test_rejects_id_duplicates_and_label_mismatch():
    flat = _result("flat_parent", ["yes", "no"])
    graph = _result("graph", ["yes", "no"])
    graph["ids"] = ["pubid0", "pubid0"]
    with pytest.raises(ValueError, match="duplicate question"):
        compare_parent_results(flat, graph)
    graph["ids"] = ["pubid0", "pubid1"]
    graph["y_true"][1] = "yes"
    with pytest.raises(ValueError, match="Ground-truth"):
        compare_parent_results(flat, graph)


def test_rejects_reordered_ids_instead_of_silently_mispairing():
    flat = _result("flat_parent", ["yes", "no"])
    graph = copy.deepcopy(_result("graph", ["yes", "no"]))
    graph["ids"].reverse()
    with pytest.raises(ValueError, match="identical ordered"):
        compare_parent_results(flat, graph)
