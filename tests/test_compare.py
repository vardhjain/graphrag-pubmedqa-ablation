"""Tests for scripts/compare.py's arm-pairing logic (feeds every RESULTS.md
number). No files touched -- aligned() is pure.
"""

from __future__ import annotations

from scripts.compare import aligned, format_p, holm_adjust, wilson_ci


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
    """round(p, 4) turned real McNemar p-values (~1e-11 on the parent-
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
    """A real, tiny effect (p~1e-11, the actual parent-expansion contrast)
    must survive correction alongside two much larger, non-significant
    p-values -- Holm should not touch it."""
    adjusted = holm_adjust([5.4e-11, 0.08, 0.69])
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
