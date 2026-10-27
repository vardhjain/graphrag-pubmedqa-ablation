"""Tests for scripts/compare.py's arm-pairing logic (feeds every RESULTS.md
number). No files touched -- aligned() is pure.
"""

from __future__ import annotations

from scripts.compare import CONTRASTS, aligned, format_p, holm_adjust, wilson_ci


def test_aligned_pairs_by_id_when_ids_match():
    a = {"ids": [1, 2, 3], "y_true": ["yes", "no", "maybe"], "y_pred": ["yes", "yes", "maybe"]}
    b = {"ids": [1, 2, 3], "y_true": ["yes", "no", "maybe"], "y_pred": ["yes", "no", "no"]}

    gt, pa, pb = aligned(a, b)

    assert gt == ["yes", "no", "maybe"]
    assert pa == ["yes", "yes", "maybe"]
    assert pb == ["yes", "no", "no"]


def test_aligned_reorders_when_ids_are_shuffled():
    a = {"ids": [1, 2, 3], "y_true": ["yes", "no", "maybe"], "y_pred": ["yes", "no", "maybe"]}
    b = {"ids": [3, 1, 2], "y_true": ["maybe", "yes", "no"], "y_pred": ["no", "yes", "yes"]}

    gt, pa, pb = aligned(a, b)

    # b's prediction for id=1 is "yes", for id=2 is "yes", for id=3 is "no" --
    # must land against a's rows in a's order (1, 2, 3), not b's file order.
    assert gt == ["yes", "no", "maybe"]
    assert pa == ["yes", "no", "maybe"]
    assert pb == ["yes", "yes", "no"]


def test_aligned_drops_ids_not_present_in_both_arms():
    a = {"ids": [1, 2, 3], "y_true": ["yes", "no", "maybe"], "y_pred": ["yes", "no", "maybe"]}
    b = {"ids": [2, 3], "y_true": ["no", "maybe"], "y_pred": ["yes", "maybe"]}

    gt, pa, pb = aligned(a, b)

    assert gt == ["no", "maybe"]
    assert pa == ["no", "maybe"]
    assert pb == ["yes", "maybe"]


def test_aligned_falls_back_to_positional_ids_when_missing():
    a = {"y_true": ["yes", "no"], "y_pred": ["yes", "no"]}
    b = {"y_true": ["yes", "no"], "y_pred": ["no", "no"]}

    gt, pa, pb = aligned(a, b)

    assert gt == ["yes", "no"]
    assert pa == ["yes", "no"]
    assert pb == ["no", "no"]


def test_aligned_excludes_ids_where_either_arm_failed():
    """A sample where the LLM call exhausted retries records a placeholder
    'maybe', not a genuine judgment -- pairing it into McNemar would
    attribute an infra timeout to the retrieval strategy being compared."""
    a = {
        "ids": [1, 2, 3],
        "y_true": ["yes", "no", "maybe"],
        "y_pred": ["yes", "no", "maybe"],
        "failed": [False, True, False],  # id=2 failed in arm A
    }
    b = {
        "ids": [1, 2, 3],
        "y_true": ["yes", "no", "maybe"],
        "y_pred": ["yes", "yes", "no"],
        "failed": [False, False, True],  # id=3 failed in arm B
    }

    gt, pa, pb = aligned(a, b)

    assert gt == ["yes"]  # only id=1 survives -- 2 and 3 each failed in one arm
    assert pa == ["yes"]
    assert pb == ["yes"]


def test_aligned_treats_missing_failed_field_as_no_failures():
    """Old-format result JSONs (generated before the failed field existed)
    must keep working exactly as before."""
    a = {"ids": [1, 2], "y_true": ["yes", "no"], "y_pred": ["yes", "no"]}
    b = {"ids": [1, 2], "y_true": ["yes", "no"], "y_pred": ["yes", "yes"]}

    gt, pa, pb = aligned(a, b)

    assert gt == ["yes", "no"]
    assert pb == ["yes", "yes"]


def test_format_p_does_not_collapse_a_tiny_real_effect_to_zero():
    """round(p, 4) turned real McNemar p-values (~5.5e-6 on the parent-
    expansion contrast) into the literal string "0.0000" -- this is the
    regression guard for that bug."""
    assert format_p(5.4e-11) == "<0.0001"
    assert format_p(0.00009999) == "<0.0001"


def test_format_p_shows_four_decimals_above_the_threshold():
    assert format_p(0.0814) == "0.0814"
    assert format_p(0.6889) == "0.6889"
    assert format_p(0.05) == "0.0500"


def test_wilson_ci_matches_known_reference_values():
    """59.5% on n=200 (the actual 'graph' arm result) has a textbook Wilson
    interval of roughly [52.6%, 66.1%] -- checked against a reference
    calculator rather than re-deriving the formula in the test itself."""
    lo, hi = wilson_ci(0.595, 200)
    assert abs(lo - 0.5257) < 0.001
    assert abs(hi - 0.6606) < 0.001


def test_wilson_ci_is_narrower_at_larger_n():
    lo_small, hi_small = wilson_ci(0.6, 20)
    lo_large, hi_large = wilson_ci(0.6, 2000)
    assert (hi_large - lo_large) < (hi_small - lo_small)


def test_wilson_ci_stays_within_bounds_at_extreme_proportions():
    lo, hi = wilson_ci(1.0, 10)
    assert 0.0 <= lo <= hi <= 1.0
    lo, hi = wilson_ci(0.0, 10)
    assert 0.0 <= lo <= hi <= 1.0


def test_holm_adjust_matches_hand_computed_example():
    """Textbook Holm example: raw p = [0.01, 0.02, 0.03, 0.04], m=4.
    Multipliers (largest p-value first, i.e. smallest multiplier): rank0 x4,
    rank1 x3, rank2 x2, rank3 x1, monotonized via running max."""
    adjusted = holm_adjust([0.01, 0.02, 0.03, 0.04])
    assert [round(p, 4) for p in adjusted] == [0.04, 0.06, 0.06, 0.06]


def test_holm_adjust_never_makes_a_borderline_effect_look_more_significant():
    """A real, tiny effect (p~5.5e-6, the actual parent-expansion contrast)
    must survive correction alongside two much larger, non-significant
    p-values -- Holm should not touch it."""
    adjusted = holm_adjust([5.46e-6, 0.08, 0.69])
    assert adjusted[0] < 0.0001
    assert adjusted[0] < 0.05  # still significant after correction


def test_holm_adjust_preserves_input_order():
    # Input order is [largest, smallest, middle] -- output must match that
    # order, not the internal sorted-by-p-value order used to compute it.
    # (0.01*3=0.03; 0.03*2=0.06; 0.04*1=0.04, monotonized to 0.06 -- worked
    # out by hand and cross-checked by running the function before writing
    # this assertion, not assumed.)
    adjusted = holm_adjust([0.04, 0.01, 0.03])
    assert [round(p, 4) for p in adjusted] == [0.06, 0.03, 0.06]


# ── end-to-end: compare.main() on a temp RESULTS_DIR ──────────────────────────
def _write_arm(d, arm, preds, truth, meta=None, **extra):
    import json

    data = {
        "model": arm,
        "accuracy": sum(p == t for p, t in zip(preds, truth, strict=True)) / len(truth),
        "macro_f1": 0.5,
        "samples": len(truth),
        "avg_latency": 1.0,
        "y_true": truth,
        "y_pred": preds,
        "ids": list(range(len(truth))),
        **extra,
    }
    if meta is not None:
        data["meta"] = meta
    (d / f"{arm}_results.json").write_text(json.dumps(data))


def _meta(arm, seed=42, model="m", n=4, complete=True):
    return {
        "arm": arm,
        "seed": seed,
        "llm_model": model,
        "n_requested": n,
        "n_done": n,
        "complete": complete,
    }


def _run_main(tmp_path, monkeypatch):
    import json

    import scripts.compare as compare

    monkeypatch.setattr(compare, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(compare, "_plot", lambda results: None)
    compare.main()
    return json.loads((tmp_path / "summary.json").read_text())


TRUTH = ["yes", "no", "yes", "no"]


def test_compare_main_pins_summary_json_key_contract(tmp_path, monkeypatch):
    _write_arm(tmp_path, "plain", ["no", "no", "no", "no"], TRUTH, _meta("plain", seed=7))
    _write_arm(
        tmp_path,
        "plain_rr",
        ["yes", "no", "no", "no"],
        TRUTH,
        _meta("plain_rr", seed=7),
        n_degraded=0,
        n_truncated=1,
    )
    _write_arm(tmp_path, "graph", ["yes", "no", "yes", "no"], TRUTH, _meta("graph", seed=7))
    out = _run_main(tmp_path, monkeypatch)

    # Keys the README / dashboard / Next.js /benchmark page rely on.
    assert {"n", "seed", "model", "dataset", "arms", "contrasts"} <= set(out)
    assert out["seed"] == 7 and out["model"] == "m"  # stamped from meta, not env
    assert out["holm_family"] == [f"{a}->{b}" for a, b, _ in CONTRASTS]
    arm_keys = {
        "arm",
        "accuracy",
        "accuracy_ci95",
        "macro_f1",
        "recall_at_k",
        "avg_latency",
        "samples",
        "n_failed",
        "adds",
    }
    for a in out["arms"]:
        assert arm_keys <= set(a)
    rr = next(a for a in out["arms"] if a["arm"] == "plain_rr")
    assert rr["n_truncated"] == 1 and rr["n_degraded"] == 0
    contrast_keys = {
        "from",
        "to",
        "effect",
        "delta_acc",
        "gains",
        "losses",
        "p_value",
        "p_display",
        "p_holm",
        "p_holm_display",
        "significant",
    }
    for c in out["contrasts"]:
        assert contrast_keys <= set(c)
    c = next(c for c in out["contrasts"] if c["to"] == "graph")
    assert c["delta_acc"] == 25.0 and c["gains"] == 1 and c["losses"] == 0
    # Holm is over the full pre-declared family (missing contrasts at p=1).
    assert c["p_holm"] == min(1.0, c["p_value"] * len(CONTRASTS))


def test_compare_main_excludes_incomplete_and_mismatched_arms(tmp_path, monkeypatch, capsys):
    _write_arm(tmp_path, "plain", ["no"] * 4, TRUTH, _meta("plain"))
    _write_arm(tmp_path, "plain_rr", ["yes", "no", "no", "no"], TRUTH, _meta("plain_rr"))
    _write_arm(tmp_path, "graph", ["yes"] * 4, TRUTH, _meta("graph", complete=False))
    _write_arm(tmp_path, "graph_norr", ["yes"] * 4, TRUTH, _meta("graph_norr", seed=1))
    out = _run_main(tmp_path, monkeypatch)

    tested = {(c["from"], c["to"]) for c in out["contrasts"]}
    assert tested == {("plain", "plain_rr")}
    assert len(out["warnings"]) == 2
    assert "INCOMPLETE" in capsys.readouterr().out


def test_compare_main_accepts_legacy_files_without_meta(tmp_path, monkeypatch):
    from kgqa.config import RANDOM_SEED

    _write_arm(tmp_path, "plain", ["no"] * 4, TRUTH)
    _write_arm(tmp_path, "plain_rr", ["yes", "no", "no", "no"], TRUTH)
    out = _run_main(tmp_path, monkeypatch)
    assert out["seed"] == RANDOM_SEED
    assert "warnings" not in out
    assert len(out["contrasts"]) == 1


def _v2_meta(arm):
    from scripts.run_benchmark import build_meta

    meta = build_meta(arm, 42, 4, parser="strict")
    meta.update(
        sample_sha256="a" * 64,
        corpus_sha256="b" * 64,
        git_sha="example-commit",
        complete=True,
        n_done=4,
    )
    return meta


def test_compare_rejects_mixed_parser_and_corpus_protocols():
    from scripts.compare import check_runs

    base = _v2_meta("plain")
    changed_parser = {**_v2_meta("plain_rr"), "answer_parser": "legacy-v1"}
    changed_corpus = {**_v2_meta("graph"), "corpus_sha256": "z" * 64}
    arms = {
        "plain": {"meta": base},
        "plain_rr": {"meta": changed_parser},
        "graph": {"meta": changed_corpus},
        "graph_concepts": {"meta": {"seed": 42, "llm_model": "m", "n_requested": 4}},
    }
    excluded, warnings = check_runs(arms)
    assert excluded == {"plain_rr", "graph", "graph_concepts"}
    assert any("legacy metadata" in w for w in warnings)


def test_compare_rejects_missing_v2_provenance():
    from scripts.compare import check_runs

    broken = _v2_meta("graph")
    broken.pop("prompt_sha256")
    excluded, warnings = check_runs({"graph": {"meta": broken}})
    assert excluded == {"graph"}
    assert any("prompt_sha256" in w for w in warnings)


def test_compare_accepts_matching_v2_provenance():
    from scripts.compare import check_runs

    excluded, warnings = check_runs(
        {
            "plain": {"meta": _v2_meta("plain")},
            "graph": {"meta": _v2_meta("graph")},
        }
    )
    assert excluded == set()
    assert warnings == []


def test_pairing_rejects_inconsistent_labels_and_duplicate_ids():
    import pytest

    a = {"ids": [1], "y_true": ["yes"], "y_pred": ["yes"]}
    bad_gold = {"ids": [1], "y_true": ["no"], "y_pred": ["no"]}
    with pytest.raises(ValueError, match="Ground-truth"):
        aligned(a, bad_gold)
    duplicate = {"ids": [1, 1], "y_true": ["yes", "yes"], "y_pred": ["yes", "no"]}
    with pytest.raises(ValueError, match="Duplicate"):
        aligned(duplicate, bad_gold)


def test_compare_records_invalid_outputs_separately(tmp_path, monkeypatch):
    _write_arm(
        tmp_path,
        "plain",
        ["invalid", "no", "yes", "no"],
        TRUTH,
        _meta("plain"),
        n_invalid=1,
    )
    out = _run_main(tmp_path, monkeypatch)
    assert out["arms"][0]["n_invalid"] == 1
