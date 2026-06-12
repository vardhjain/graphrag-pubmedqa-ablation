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
