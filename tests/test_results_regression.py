"""CI eval gate: fails the build if the checked-in benchmark results regress.

This does NOT re-run the LLM benchmark in CI -- that needs a GPU and a live
ArangoDB (see project notes: this repo's benchmark runs on Colab), neither of
which CI has. Instead it guards the artifact everything else (README,
RESULTS.md, the /benchmark dashboard, resume claims) points to: if
``results/summary.json`` is ever edited down, or a re-run regresses, this
test catches it instead of a human noticing it went stale.

Thresholds are set with headroom below the actual recorded numbers (see
RESULTS.md) so normal run-to-run noise doesn't false-positive, while still
catching a real regression or accidental edit.
"""

from __future__ import annotations

import json
import os

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "summary.json"
)


def _load():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _arm(data, name):
    for a in data["arms"]:
        if a["arm"] == name:
            return a
    raise AssertionError(f"arm {name!r} not found in results/summary.json -- was it renamed or removed?")


def _contrast(data, from_arm, to_arm):
    for c in data["contrasts"]:
        if c["from"] == from_arm and c["to"] == to_arm:
            return c
    raise AssertionError(
        f"contrast {from_arm!r} -> {to_arm!r} not found in results/summary.json -- was it renamed or removed?"
    )


def test_results_file_has_minimum_sample_size():
    data = _load()
    assert data["n"] >= 100, "benchmark sample size dropped below a trustworthy floor"


def test_graph_arm_accuracy_has_not_regressed():
    graph = _arm(_load(), "graph")
    assert graph["accuracy"] >= 55.0, "graph arm accuracy regressed below floor"
    assert graph["macro_f1"] >= 45.0, "graph arm macro-F1 regressed below floor"


def test_graph_beats_reranked_baseline():
    data = _load()
    graph, plain_rr = _arm(data, "graph"), _arm(data, "plain_rr")
    assert graph["accuracy"] > plain_rr["accuracy"], (
        "graph arm no longer beats the reranked baseline -- the headline claim is broken"
    )


def test_parent_expansion_effect_is_still_significant():
    """The repo's whole pitch is +22.5pp from parent-document expansion,
    McNemar p<0.0001 -- this is the one number that must never quietly break."""
    contrast = _contrast(_load(), "plain_rr", "graph")
    assert contrast["significant"] is True
    assert contrast["p_value"] < 0.05
    assert contrast["delta_acc"] >= 15.0, "parent-expansion lift shrank well below the claimed +22.5pp"


def test_graph_concepts_latency_within_bounds():
    """Not a regression gate on accuracy (graph_concepts isn't the shipped arm),
    just a sanity check that the recorded latency multiplier hasn't exploded
    further, since that number is quoted in RESULTS.md too."""
    data = _load()
    graph, concepts = _arm(data, "graph"), _arm(data, "graph_concepts")
    assert concepts["avg_latency"] < graph["avg_latency"] * 10
