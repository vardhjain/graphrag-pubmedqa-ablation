"""Export and independently verify per-question benchmark evidence with stdlib.

Usage:
  python scripts/evidence_bundle.py export --results-dir results/ --out results/evidence/
  python scripts/evidence_bundle.py verify --bundle results/evidence/

Requires the original per-arm *_results.json outputs from a *fresh* run.
The historical repository commits only summary.json, so do not imply its
unavailable predictions can be recovered or independently verified.
No databases, model downloads, inference, or external dependencies are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

LABELS = ("yes", "no", "maybe")
FIELDS = ("ids", "y_true", "y_pred")
OPTIONAL = ("failed", "degraded", "truncated", "invalid", "latencies", "retrieved_papers")
PROVENANCE = (
    "protocol_version",
    "answer_parser",
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
    "prompt_sha256",
    "embedding_model",
    "reranker_model",
    "dataset_name",
)
PAIRS = (
    ("plain", "plain_rr"),
    ("plain_rr", "graph"),
    ("graph", "graph_concepts"),
    ("plain", "graph_norr"),
    ("flat_parent", "graph"),
)


def _encoded(data: dict) -> bytes:
    return (json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("An empty arm cannot be audited")
    accuracy = sum(row["prediction"] == row["gold"] for row in rows) / len(rows)
    f1_scores = []
    for label in LABELS:
        tp = sum(r["gold"] == label and r["prediction"] == label for r in rows)
        fp = sum(r["gold"] != label and r["prediction"] == label for r in rows)
        fn = sum(r["gold"] == label and r["prediction"] != label for r in rows)
        f1_scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return {"samples": len(rows), "accuracy": accuracy, "macro_f1": sum(f1_scores) / 3}


def _rows(data: dict, arm: str) -> tuple[list[dict], list[str]]:
    n = len(data.get("y_pred") or [])
    if n == 0 or not all(isinstance(data.get(k), list) and len(data[k]) == n for k in FIELDS):
        raise ValueError(f"{arm}: ids/y_true/y_pred must be nonempty aligned lists")
    ids = data["ids"]
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != n:
        raise ValueError(f"{arm}: empty, non-string or duplicate sample ids")
    if not all(label in LABELS for label in data["y_true"]):
        raise ValueError(f"{arm}: invalid gold labels")
    if not all(label in LABELS + ("invalid",) for label in data["y_pred"]):
        raise ValueError(f"{arm}: invalid prediction labels")
    if (data.get("meta") or {}).get("complete") is not True:
        raise ValueError(f"{arm}: partial or unversioned run; export completed runs only")
    meta = data["meta"]
    if meta.get("arm") != arm or meta.get("n_requested") != n or meta.get("n_done") != n:
        raise ValueError(f"{arm}: inconsistent arm or requested/done sample count")
    available = [field for field in OPTIONAL if field in data]
    for field in available:
        if not isinstance(data[field], list) or len(data[field]) != n:
            raise ValueError(f"{arm}: {field} must have exactly {n} elements")
    rows = []
    for i, sample_id in enumerate(ids):
        row = {"id": sample_id, "gold": data["y_true"][i], "prediction": data["y_pred"][i]}
        for field in ("failed", "degraded", "truncated", "invalid"):
            value = data[field][i] if field in available else None
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{arm}: {field}[{i}] is not boolean")
            row[field] = value
        value = data["latencies"][i] if "latencies" in available else None
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"{arm}: invalid latency for {sample_id}")
        row["latency_seconds"] = value
        retrieved = data["retrieved_papers"][i] if "retrieved_papers" in available else None
        if retrieved is not None and (
            not isinstance(retrieved, list)
            or any(not isinstance(p, str) for p in retrieved)
        ):
            raise ValueError(f"{arm}: invalid retrieved_papers for {sample_id}")
        row["retrieved_papers"] = retrieved
        rows.append(row)
    expected = metrics(rows)
    for key in ("accuracy", "macro_f1"):
        if not isinstance(data.get(key), (int, float)) or not math.isclose(
            data[key], expected[key], abs_tol=1e-8
        ):
            raise ValueError(f"{arm}: stored {key} does not match per-question predictions")
    if data.get("samples") != n:
        raise ValueError(f"{arm}: summary sample count differs from question records")
    return rows, available


def _mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict:
    # A rejected/failed LLM call cannot be credited to a retrieval strategy.
    a_map = {r["id"]: r for r in rows_a}
    b_map = {r["id"]: r for r in rows_b}
    if set(a_map) != set(b_map):
        raise ValueError("Paired contrast has mismatched question IDs")
    gains = losses = paired = 0
    for sid, a in a_map.items():
        b = b_map[sid]
        if a["gold"] != b["gold"]:
            raise ValueError("Paired contrast has mismatched gold labels")
        if a["failed"] or b["failed"]:
            continue
        paired += 1
        a_ok = a["prediction"] == a["gold"]
        b_ok = b["prediction"] == b["gold"]
        gains += bool(b_ok and not a_ok)
        losses += bool(a_ok and not b_ok)
    if not paired:
        raise ValueError("No valid paired observations")
    discordant = gains + losses
    p = (
        min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(gains, losses) + 1))
            / 2**discordant)
        if discordant
        else 1.0
    )
    return {
        "paired_n": paired,
        "gains": gains,
        "losses": losses,
        "delta_accuracy_pp": (gains - losses) * 100 / paired,
        "mcnemar_exact_p": p,
        "adjustment": "none; exploratory only",
    }


def _comparability(arms: dict[str, dict]) -> tuple[bool, list[str]]:
    warnings = []
    if len(arms) < 2:
        return False, ["Only one arm supplied; no cross-arm comparison"]
    metas = [entry["meta"] for entry in arms.values()]
    for field in PROVENANCE:
        vals = [m.get(field) for m in metas]
        if any(v != vals[0] for v in vals[1:]):
            warnings.append(f"Provenance mismatch: {field}")
    for field in ("seed", "llm_model", "n_requested", "git_sha", "sample_sha256", "corpus_sha256"):
        if any(m.get(field) is None for m in metas):
            warnings.append(f"Missing cross-arm provenance: {field}")
    rows = [entry["rows"] for entry in arms.values()]
    id_gold = [[(r["id"], r["gold"]) for r in rr] for rr in rows]
    if any(value != id_gold[0] for value in id_gold[1:]):
        warnings.append("Question order or gold labels differ across arms")
    return not warnings, warnings


def export_bundle(source_dir: Path, destination: Path) -> dict:
    paths = sorted(source_dir.glob("*_results.json"))
    if not paths:
        raise ValueError("No *_results.json found. Published summary.json has no per-question data.")
    if destination.resolve() == source_dir.resolve():
        raise ValueError("Output directory must differ from input directory")
    arms: dict[str, dict] = {}
    for path in paths:
        arm = path.name.removesuffix("_results.json")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", arm):
            raise ValueError(f"Unsafe arm name: {arm!r}")
        source_bytes = path.read_bytes()
        raw = json.loads(source_bytes)
        rows, fields = _rows(raw, arm)
        # Export only a fixed safe subset of metadata, not arbitrary env vars,
        # free-form model text, confidential path strings or API credentials.
        meta = raw.get("meta") or {}
        arms[arm] = {
            "rows": rows,
            "meta": {name: meta.get(name) for name in PROVENANCE if name in meta},
            "metrics": metrics(rows),
            "fields_present": fields,
            "source_sha256": _sha(source_bytes),
        }
    compatible, warnings = _comparability(arms)
    entries = {}
    output_bytes = {}
    for arm, entry in arms.items():
        filename = f"{arm}.jsonl"
        payload = b"".join(_encoded(row) for row in entry["rows"])
        output_bytes[filename] = payload
        entries[arm] = {
            "file": filename,
            "sha256": _sha(payload),
            "source_results_sha256": entry["source_sha256"],
            "records": len(entry["rows"]),
            "metrics": entry["metrics"],
            "fields_present": entry["fields_present"],
            "provenance": entry["meta"],
        }
    pairs = {}
    if compatible:
        for a, b in PAIRS:
            if a in arms and b in arms:
                pairs[f"{a}->{b}"] = _mcnemar(arms[a]["rows"], arms[b]["rows"])
    manifest = {
        "schema_version": "pubmedqa-evidence-v1",
        "source": "per-question *_results.json (not the historical summary.json)",
        "comparable": compatible,
        "warnings": warnings,
        "arms": entries,
        "exploratory_pairs": pairs,
        "limitations": (
            "Checks internal consistency of saved predictions, not authenticity of model "
            "outputs, PubMedQA dataset version or the state of the knowledge graph. "
            "Run the read-only graph audit and provenance checks separately."
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    for filename, payload in output_bytes.items():
        (destination / filename).write_bytes(payload)
    (destination / "manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    )
    return manifest


def verify_bundle(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "pubmedqa-evidence-v1":
        raise ValueError("Unsupported evidence manifest schema")
    arms = manifest.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ValueError("Evidence manifest has no arms")
    verified = {}
    for name, entry in arms.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
            raise ValueError("Invalid manifest arm name")
        if entry.get("file") != f"{name}.jsonl":
            raise ValueError(f"{name}: invalid evidence filename")
        payload = (directory / entry["file"]).read_bytes()
        if _sha(payload) != entry["sha256"]:
            raise ValueError(f"{name}: evidence file hash mismatch")
        rows = [json.loads(line) for line in payload.splitlines()]
        if len(rows) != entry["records"]:
            raise ValueError(f"{name}: evidence row count changed")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"{name}: duplicate question ID")
        if not all(r["gold"] in LABELS and r["prediction"] in LABELS + ("invalid",) for r in rows):
            raise ValueError(f"{name}: invalid gold or prediction label")
        result = metrics(rows)
        for key, value in result.items():
            if not math.isclose(value, entry["metrics"][key], abs_tol=1e-8):
                raise ValueError(f"{name}: published {key} differs from evidence")
        verified[name] = {"rows": rows, "meta": entry["provenance"]}
    comparable, warnings = _comparability(verified)
    if comparable != manifest["comparable"] or warnings != manifest["warnings"]:
        raise ValueError("Manifest comparability is inconsistent with evidence files")
    expected_pairs = {}
    if comparable:
        for a, b in PAIRS:
            if a in verified and b in verified:
                expected_pairs[f"{a}->{b}"] = _mcnemar(verified[a]["rows"], verified[b]["rows"])
    if expected_pairs != manifest["exploratory_pairs"]:
        raise ValueError("Manifest paired statistics are inconsistent with evidence")
    return {
        "verified": True,
        "arms": sorted(verified),
        "comparable": comparable,
        "pairs_verified": sorted(expected_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    export = subs.add_parser("export", help="Export completed raw runs into auditable JSONL files")
    export.add_argument("--results-dir", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    verify = subs.add_parser("verify", help="Recalculate metrics, hashes and paired tests")
    verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        report = export_bundle(args.results_dir, args.out)
        # Verify the just-created evidence rather than trust the exporter.
        verified = verify_bundle(args.out)
        print(json.dumps({"exported_arms": sorted(report["arms"]), **verified}, indent=2))
    else:
        print(json.dumps(verify_bundle(args.bundle), indent=2))


if __name__ == "__main__":
    main()
