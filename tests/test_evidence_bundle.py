"""Public audit-evidence export tests: synthetic, CPU-only and outcome-neutral."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evidence_bundle import export_bundle, verify_bundle


def raw_arm(arm, predictions, *, seed=42):
    gold = ["yes", "no", "maybe", "yes"]
    n = len(predictions)
    return {
        "model": arm,
        "accuracy": sum(p == g for p, g in zip(predictions, gold, strict=False)) / n,
        "macro_f1": macro_f1(gold[:n], predictions),
        "samples": n,
        "ids": [f"pmid{i}" for i in range(n)],
        "y_true": gold[:n],
        "y_pred": predictions,
        "latencies": [1.0, 0.5, 2.0, 1.5][:n],
        "retrieved_papers": [[f"pmid{i}"] for i in range(n)],
        "failed": [False] * n,
        "degraded": [False] * n,
        "truncated": [False] * n,
        "meta": {
            "arm": arm,
            "complete": True,
            "seed": seed,
            "n_requested": n,
            "n_done": n,
            "llm_model": "dummy-local",
            "git_sha": "a" * 40,
            "sample_sha256": "b" * 64,
            "corpus_sha256": "c" * 64,
            "answer_parser": "strict-v2",
        },
    }


def macro_f1(gold, predictions):
    scores = []
    for label in ("yes", "no", "maybe"):
        tp = sum(a == label and b == label for a, b in zip(gold, predictions, strict=True))
        fp = sum(a != label and b == label for a, b in zip(gold, predictions, strict=True))
        fn = sum(a == label and b != label for a, b in zip(gold, predictions, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(scores) / 3


def save_arm(path: Path, arm: str, data: dict):
    (path / f"{arm}_results.json").write_text(json.dumps(data), encoding="utf-8")


def test_exported_evidence_round_trip_and_negative_results(tmp_path):
    inputs = tmp_path / "inputs"
    bundle = tmp_path / "evidence"
    inputs.mkdir()
    save_arm(inputs, "plain_rr", raw_arm("plain_rr", ["yes", "no", "maybe", "yes"]))
    save_arm(inputs, "graph", raw_arm("graph", ["no", "no", "maybe", "no"]))

    manifest = export_bundle(inputs, bundle)
    assert manifest["comparable"] is True
    assert manifest["exploratory_pairs"]["plain_rr->graph"]["delta_accuracy_pp"] == -50.0
    assert manifest["exploratory_pairs"]["plain_rr->graph"]["losses"] == 2
    assert "graph" in manifest["arms"]
    assert manifest["arms"]["graph"]["records"] == 4
    assert verify_bundle(bundle)["verified"] is True
    rows = (bundle / "graph.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4
    assert json.loads(rows[0])["id"] == "pmid0"


def test_bundle_is_deterministic_for_same_raw_input(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    save_arm(inputs, "graph", raw_arm("graph", ["yes", "no", "maybe", "yes"]))
    first, second = tmp_path / "first", tmp_path / "second"
    export_bundle(inputs, first)
    export_bundle(inputs, second)
    assert (first / "graph.jsonl").read_bytes() == (second / "graph.jsonl").read_bytes()
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_verifier_catches_tampered_jsonl_or_manifest(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    save_arm(inputs, "graph", raw_arm("graph", ["yes", "no", "maybe", "yes"]))
    bundle = tmp_path / "bundle"
    export_bundle(inputs, bundle)
    evidence = bundle / "graph.jsonl"
    evidence.write_bytes(evidence.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(bundle)


def test_verifier_recomputes_metrics_even_if_hash_is_updated(tmp_path):
    import hashlib

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    save_arm(inputs, "graph", raw_arm("graph", ["yes", "no", "maybe", "yes"]))
    bundle = tmp_path / "bundle"
    export_bundle(inputs, bundle)
    path = bundle / "graph.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["prediction"] = "no"
    new_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in records
    )
    path.write_bytes(new_bytes)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["arms"]["graph"]["sha256"] = hashlib.sha256(new_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="published accuracy differs"):
        verify_bundle(bundle)


def test_export_blocks_changed_seed_from_paired_claim(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    save_arm(inputs, "plain_rr", raw_arm("plain_rr", ["yes", "no", "maybe", "yes"]))
    save_arm(inputs, "graph", raw_arm("graph", ["yes", "no", "maybe", "yes"], seed=7))
    bundle = tmp_path / "bundle"
    manifest = export_bundle(inputs, bundle)
    assert manifest["comparable"] is False
    assert "Provenance mismatch: seed" in manifest["warnings"]
    assert manifest["exploratory_pairs"] == {}
    assert verify_bundle(bundle)["comparable"] is False


def test_export_rejects_incomplete_and_fake_accuracy(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bad = raw_arm("graph", ["yes", "no", "maybe", "yes"])
    bad["meta"]["complete"] = False
    save_arm(inputs, "graph", bad)
    with pytest.raises(ValueError, match="partial"):
        export_bundle(inputs, tmp_path / "out")
    bad["meta"]["complete"] = True
    bad["accuracy"] = 0.5
    save_arm(inputs, "graph", bad)
    with pytest.raises(ValueError, match="accuracy"):
        export_bundle(inputs, tmp_path / "out")


def test_export_rejects_duplicate_ids_and_invalid_flags(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bad = raw_arm("graph", ["yes", "no", "maybe", "yes"])
    bad["ids"][0] = bad["ids"][1]
    save_arm(inputs, "graph", bad)
    with pytest.raises(ValueError, match="duplicate"):
        export_bundle(inputs, tmp_path / "out")
    bad["ids"][0] = "pmid0"
    bad["failed"] = [False]
    save_arm(inputs, "graph", bad)
    with pytest.raises(ValueError, match="failed"):
        export_bundle(inputs, tmp_path / "out")


def test_export_rejects_missing_raw_outputs(tmp_path):
    with pytest.raises(ValueError, match="No .*_results.json"):
        export_bundle(tmp_path, tmp_path / "evidence")


def test_export_only_includes_allowlisted_provenance_fields(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    raw = raw_arm("graph", ["yes", "no", "maybe", "yes"])
    raw["meta"]["ARANGO_PASS"] = "test-secret-placeholder"
    raw["meta"]["custom_prompt_text"] = "private question text"
    save_arm(inputs, "graph", raw)
    bundle = tmp_path / "evidence"
    export_bundle(inputs, bundle)
    assert "test-secret-placeholder" not in (bundle / "manifest.json").read_text()
    assert "private question text" not in (bundle / "manifest.json").read_text()
