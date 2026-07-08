"""Tests for scripts/compare.py's arm-pairing logic (feeds every RESULTS.md
number). No files touched -- aligned() is pure.
"""

from __future__ import annotations

from scripts.compare import aligned


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
