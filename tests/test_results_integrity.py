"""Check published evaluation artifacts for integrity, not favorable outcomes.

CI cannot re-run the GPU/ArangoDB benchmark. It can, however, detect malformed
summaries and impossible statistics. A valid new run is allowed to be worse,
slower, or non-significant: those are findings, not software bugs.

Legacy summary.json stores p-values to four decimal places and has no paired_n
or Holm metadata. The next generated report uses full-precision p-values and
can carry paired_n when failed calls reduce a paired comparison's sample.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from scipy.stats import binomtest

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "summary.json"


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _percent(value: object) -> bool:
    return _finite_number(value) and 0 <= value <= 100


def _probability(value: object) -> bool:
    return _finite_number(value) and 0 <= value <= 1


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Independently recalculate Holm step-down adjustment over a fixed family."""
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, p_values[idx] * (len(p_values) - rank))
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def validate_summary(data: dict) -> None:
    """Validate only the claims that can be checked without raw predictions."""
    assert isinstance(data, dict), "summary must be an object"
    n = data["n"]
    assert _positive_int(n), "n must be a positive integer"
    assert _nonnegative_int(data["seed"]), "seed must be a nonnegative integer"
    assert isinstance(data["model"], str) and data["model"].strip()
    assert isinstance(data["dataset"], str) and data["dataset"].strip()

    arms = data["arms"]
    assert isinstance(arms, list) and arms, "need at least one evaluated arm"
    by_name: dict[str, dict] = {}
    for arm in arms:
        name = arm["arm"]
        assert isinstance(name, str) and name.strip(), "arm needs a name"
        assert name not in by_name, f"duplicate arm: {name}"
        by_name[name] = arm

        samples = arm["samples"]
        assert _positive_int(samples) and samples <= n, f"{name}: invalid samples"
        assert _percent(arm["accuracy"]), f"{name}: accuracy must be 0–100%"
        assert _percent(arm["macro_f1"]), f"{name}: macro F1 must be 0–100%"
        assert _finite_number(arm["avg_latency"]) and arm["avg_latency"] >= 0

        for field in ("n_failed", "n_degraded", "n_truncated"):
            if field in arm:
                assert _nonnegative_int(arm[field]) and arm[field] <= samples, (
                    f"{name}: invalid {field}"
                )
        if "recall_at_k" in arm and arm["recall_at_k"] is not None:
            assert _probability(arm["recall_at_k"]), f"{name}: invalid recall@k"
        if "accuracy_ci95" in arm:
            ci = arm["accuracy_ci95"]
            assert isinstance(ci, list) and len(ci) == 2
            lo, hi = ci
            assert _percent(lo) and _percent(hi) and lo <= hi
            # Reported CIs and accuracies can be rounded to 2 decimal places.
            assert lo <= arm["accuracy"] + 0.02 <= hi + 0.02, (
                f"{name}: accuracy outside reported CI"
            )
        for flag in ("complete", "excluded_from_tests"):
            if flag in arm:
                assert type(arm[flag]) is bool, f"{name}: invalid {flag}"

    contrasts = data["contrasts"]
    assert isinstance(contrasts, list), "contrasts must be a list"
    by_pair: dict[str, dict] = {}
    for contrast in contrasts:
        source, target = contrast["from"], contrast["to"]
        pair = f"{source}->{target}"
        assert source in by_name and target in by_name and source != target, (
            f"{pair}: unknown or identical arms"
        )
        assert pair not in by_pair, f"duplicate contrast: {pair}"
        by_pair[pair] = contrast

        gains, losses = contrast["gains"], contrast["losses"]
        assert _nonnegative_int(gains) and _nonnegative_int(losses), (
            f"{pair}: invalid discordant counts"
        )
        max_pairs = min(by_name[source]["samples"], by_name[target]["samples"])
        assert gains + losses <= max_pairs, f"{pair}: impossible discordant counts"
        assert _finite_number(contrast["delta_acc"])
        assert -100 <= contrast["delta_acc"] <= 100

        expected_p = float(binomtest(gains, gains + losses, 0.5).pvalue) if gains + losses else 1.0
        assert _probability(contrast["p_value"]), f"{pair}: invalid p-value"
        # The checked-in legacy JSON rounded 5.46e-6 to 0.0000; newer
        # compare.py output retains full precision.
        tolerance = 5.00001e-5 if "p_holm" not in contrast else 1e-10
        assert abs(contrast["p_value"] - expected_p) <= tolerance, (
            f"{pair}: p-value disagrees with exact McNemar"
        )

        paired_n = contrast.get("paired_n")
        if paired_n is not None:
            assert _positive_int(paired_n) and paired_n <= max_pairs, (
                f"{pair}: invalid paired_n"
            )
        else:
            # Legacy summaries omit paired_n. Only infer it when neither
            # arm failed and both cover the summary's full sample.
            both_complete = all(
                by_name[name]["samples"] == n and by_name[name].get("n_failed", 0) == 0
                for name in (source, target)
            )
            if both_complete:
                paired_n = n
        if paired_n is not None:
            assert gains + losses <= paired_n, f"{pair}: discordance exceeds paired n"
            expected_delta = (gains - losses) * 100 / paired_n
            assert abs(contrast["delta_acc"] - expected_delta) <= 0.015, (
                f"{pair}: effect size disagrees with discordant counts"
            )
            if paired_n == n and all(by_name[name]["samples"] == n for name in (source, target)):
                arm_delta = by_name[target]["accuracy"] - by_name[source]["accuracy"]
                assert abs(contrast["delta_acc"] - arm_delta) <= 0.025, (
                    f"{pair}: effect disagrees with per-arm accuracy"
                )

        if "p_holm" in contrast:
            assert _probability(contrast["p_holm"])
            assert contrast["p_holm"] + 1e-10 >= contrast["p_value"], (
                f"{pair}: adjusted p-value below unadjusted value"
            )
        significance_p = contrast.get("p_holm", expected_p)
        assert type(contrast["significant"]) is bool
        assert contrast["significant"] == (significance_p < 0.05), (
            f"{pair}: significance flag disagrees with p-value"
        )

    # When Holm metadata is published, verify *the declared family*, including
    # comparisons omitted due to absent or invalid runs (assigned p=1).
    if "holm_family" in data:
        family = data["holm_family"]
        assert isinstance(family, list) and family
        assert all(isinstance(pair, str) and pair for pair in family)
        assert len(set(family)) == len(family)
        assert set(by_pair) <= set(family), "contrast missing from declared Holm family"
        raw = [by_pair[pair]["p_value"] if pair in by_pair else 1.0 for pair in family]
        adjusted = _holm_adjust(raw)
        for pair, expected in zip(family, adjusted, strict=True):
            if pair in by_pair:
                assert "p_holm" in by_pair[pair], f"{pair}: missing adjusted p-value"
                assert abs(by_pair[pair]["p_holm"] - expected) <= 1e-9, (
                    f"{pair}: incorrect Holm correction"
                )


def _synthetic_summary(gains: int, losses: int) -> dict:
    """Small coherent result with either a positive or negative effect."""
    n = 100
    p = float(binomtest(gains, gains + losses, 0.5).pvalue)
    delta = float(gains - losses)
    return {
        "n": n,
        "seed": 42,
        "model": "synthetic-test-model",
        "dataset": "synthetic",
        "arms": [
            {"arm": "baseline", "samples": n, "accuracy": 50.0, "macro_f1": 40.0, "avg_latency": 1.0},
            {"arm": "candidate", "samples": n, "accuracy": 50.0 + delta, "macro_f1": 38.0, "avg_latency": 2.0},
        ],
        "contrasts": [{
            "from": "baseline",
            "to": "candidate",
            "effect": "synthetic change",
            "gains": gains,
            "losses": losses,
            "delta_acc": delta,
            "p_value": p,
            "significant": p < 0.05,
        }],
    }


def test_published_summary_is_internally_consistent():
    with RESULTS_PATH.open(encoding="utf-8") as f:
        validate_summary(json.load(f))


def test_negative_significant_effect_is_allowed():
    # A disappointing but sound scientific finding must pass CI.
    validate_summary(_synthetic_summary(gains=4, losses=14))


def test_nonsignificant_effect_is_allowed():
    validate_summary(_synthetic_summary(gains=12, losses=8))


def test_incorrect_effect_size_is_rejected():
    data = _synthetic_summary(4, 14)
    data["contrasts"][0]["delta_acc"] = 10.0
    with pytest.raises(AssertionError, match="effect size"):
        validate_summary(data)


def test_incorrect_p_value_is_rejected():
    data = _synthetic_summary(4, 14)
    data["contrasts"][0]["p_value"] = 0.9
    with pytest.raises(AssertionError, match="p-value disagrees"):
        validate_summary(data)


def test_incorrect_significance_flag_is_rejected():
    data = _synthetic_summary(4, 14)
    data["contrasts"][0]["significant"] = False
    with pytest.raises(AssertionError, match="significance flag"):
        validate_summary(data)


def test_duplicate_arm_is_rejected():
    data = _synthetic_summary(4, 14)
    data["arms"].append(dict(data["arms"][0]))
    with pytest.raises(AssertionError, match="duplicate arm"):
        validate_summary(data)


def test_bad_holm_adjustment_is_rejected():
    data = _synthetic_summary(4, 14)
    data["holm_family"] = ["baseline->candidate", "absent->arm"]
    data["contrasts"][0]["p_holm"] = data["contrasts"][0]["p_value"]
    with pytest.raises(AssertionError, match="Holm correction"):
        validate_summary(data)
