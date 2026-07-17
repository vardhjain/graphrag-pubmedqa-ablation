from kgqa.evaluation import Evaluator, FuzzyEvaluator, mcnemar_test


def test_extract_final_answer_tag():
    fz = FuzzyEvaluator()
    assert fz.extract_answer("blah blah Final Answer: yes") == "yes"
    assert fz.extract_answer("FINAL ANSWER : No") == "no"


def test_extract_strips_think_block():
    fz = FuzzyEvaluator()
    text = "<think>maybe yes no</think> The study shows ... Final Answer: maybe"
    assert fz.extract_answer(text) == "maybe"


def test_extract_falls_back_to_last_mention():
    fz = FuzzyEvaluator()
    assert fz.extract_answer("I think the answer is no") == "no"
    assert fz.extract_answer("nothing useful here") == "maybe"


def test_evaluator_metrics_and_normalisation():
    ev = Evaluator("plain")
    ev.record("yes", "yes", 1.0, sample_id="1")
    ev.record("no", "garbage", 2.0, sample_id="2")  # invalid -> maybe
    ev.record("maybe", "maybe", 3.0, sample_id="3")
    s = ev.summary()
    assert s["samples"] == 3
    assert s["y_pred"][1] == "maybe"
    assert abs(s["accuracy"] - 2 / 3) < 1e-9
    assert abs(s["avg_latency"] - 2.0) < 1e-9
    assert s["ids"] == ["1", "2", "3"]


def test_evaluator_tracks_failed_calls_separately_from_predictions():
    """A failed LLM call is still counted toward accuracy (it really did
    fail end-to-end -- silently dropping it would compare arms on different
    sub-samples), but must be flagged so paired significance tests
    (scripts/compare.py's aligned()) can exclude it."""
    ev = Evaluator("plain")
    ev.record("yes", "yes", 1.0, sample_id="1")  # genuine success
    ev.record("no", "maybe", 2.0, sample_id="2", failed=True)  # exhausted retries
    s = ev.summary()
    assert s["failed"] == [False, True]
    assert s["n_failed"] == 1
    assert s["samples"] == 2  # failures still counted in samples/accuracy


def test_recall_at_k_scores_gold_paper_membership():
    """The gold paper is the question's own source paper -- sample_id is set
    to that same pubid by convention (run_benchmark.py), so recall@k is just
    'is sample_id in retrieved_papers for that sample'."""
    ev = Evaluator("graph")
    ev.record("yes", "yes", 1.0, sample_id="1", retrieved_papers=["1", "9"])  # hit
    ev.record("no", "no", 1.0, sample_id="2", retrieved_papers=["8", "9"])  # miss
    ev.record("yes", "yes", 1.0, sample_id="3", retrieved_papers=["3"])  # hit

    assert abs(ev.recall_at_k() - 2 / 3) < 1e-9


def test_recall_at_k_excludes_failed_samples_not_just_scores_them_as_misses():
    """A failed LLM call means retrieved_papers reflects whatever partial
    attempt happened (or nothing) -- not a genuine retrieval-quality signal,
    so it must be excluded from the denominator, not counted as a miss."""
    ev = Evaluator("graph")
    ev.record("yes", "yes", 1.0, sample_id="1", retrieved_papers=["1"])  # hit
    ev.record("no", "maybe", 1.0, sample_id="2", retrieved_papers=[], failed=True)  # excluded

    assert ev.recall_at_k() == 1.0  # only the 1 valid sample counts, and it's a hit


def test_recall_at_k_is_zero_with_no_scorable_samples():
    ev = Evaluator("graph")
    assert ev.recall_at_k() == 0.0


def test_mcnemar_detects_one_sided_gain():
    gt = ["yes"] * 10
    a = ["no"] * 10           # arm A always wrong
    b = ["yes"] * 10          # arm B always right
    res = mcnemar_test(gt, a, b)
    assert res["b_gains"] == 10
    assert res["c_losses"] == 0
    assert res["significant_at_0.05"] is True


def test_mcnemar_no_difference():
    gt = ["yes", "no", "maybe"]
    res = mcnemar_test(gt, gt, gt)
    assert res["discordant"] == 0
    assert res["p_value"] == 1.0


def test_mcnemar_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        mcnemar_test(["yes"], ["yes"], ["yes", "no"])


def test_report_and_save_roundtrip(tmp_path):
    import json
    ev = Evaluator("graph")
    ev.record("yes", "yes", 1.0, "1")
    ev.record("no", "yes", 2.0, "2")
    summary = ev.report()
    assert summary["model"] == "graph" and summary["samples"] == 2
    assert "macro_f1" in summary

    path = tmp_path / "results.json"
    ev.save(str(path))
    loaded = json.loads(path.read_text())
    assert loaded["samples"] == 2
    assert loaded["ids"] == ["1", "2"]
