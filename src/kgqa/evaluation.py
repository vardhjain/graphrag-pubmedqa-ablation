"""Answer extraction, metrics, and significance testing.

Kept free of any plotting import at module load so it is importable in headless
CI. Figure generation lives in ``scripts/compare.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from sklearn.metrics import accuracy_score, classification_report, f1_score

LABELS = ("yes", "no", "maybe")


class FuzzyEvaluator:
    """Extracts a normalised yes/no/maybe from verbose model output."""

    def extract_answer(self, text: str) -> str:
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).lower()
        match = re.search(r"final answer\s*:\s*(yes|no|maybe)", clean)
        if match:
            return match.group(1)
        matches = re.findall(r"\b(yes|no|maybe)\b", clean)
        return matches[-1] if matches else "maybe"


@dataclass
class Evaluator:
    """Accumulates predictions and computes plot-free metrics.

    ``ids`` records the dataset pubid of each sample so a paired significance
    test (McNemar) can be run across arms on exactly the same questions.
    """

    model_name: str
    y_true: list = field(default_factory=list)
    y_pred: list = field(default_factory=list)
    latencies: list = field(default_factory=list)
    ids: list = field(default_factory=list)
    # True where the LLM call exhausted MAX_TRIES and pred is a placeholder
    # "maybe", not a genuine model judgment -- kept in accuracy/macro_f1 (the
    # sample really did fail end-to-end, and silently dropping it would make
    # arms comparable on different sub-samples), but scripts/compare.py's
    # paired McNemar test excludes any id where either arm failed, since
    # attributing a timeout to "the retrieval strategy" would contaminate
    # the one number this project's whole ablation exists to get right.
    failed: list = field(default_factory=list)

    def record(self, ground_truth: str, prediction: str,
               latency: float = 0.0, sample_id: str | None = None,
               failed: bool = False) -> None:
        pred = prediction.lower().strip()
        if pred not in LABELS:
            pred = "maybe"
        self.y_true.append(ground_truth.lower().strip())
        self.y_pred.append(pred)
        self.latencies.append(latency)
        self.ids.append(sample_id)
        self.failed.append(failed)

    def n_failed(self) -> int:
        return sum(self.failed)

    # ── metrics ───────────────────────────────────────────────────────────────
    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred) if self.y_true else 0.0

    def macro_f1(self) -> float:
        if not self.y_true:
            return 0.0
        return f1_score(self.y_true, self.y_pred, labels=list(LABELS),
                        average="macro", zero_division=0)

    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "accuracy": self.accuracy(),
            "macro_f1": self.macro_f1(),
            "samples": len(self.y_true),
            "n_failed": self.n_failed(),
            "total_time": sum(self.latencies),
            "avg_latency": self.avg_latency(),
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "ids": self.ids,
            "failed": self.failed,
        }

    def report(self) -> dict:
        if not self.y_true:
            print("No data recorded.")
            return {}
        print(f"\n{'=' * 52}")
        print(f"  {self.model_name} — Evaluation Report")
        print(f"{'=' * 52}")
        print(f"  Samples   : {len(self.y_true)}")
        if self.n_failed():
            print(f"  Failed    : {self.n_failed()} (recorded as 'maybe'; "
                  "excluded from paired significance tests, see compare.py)")
        print(f"  Accuracy  : {self.accuracy():.2%}")
        print(f"  Macro F1  : {self.macro_f1():.2%}")
        print(f"  Avg/query : {self.avg_latency():.1f}s")
        print(f"{'-' * 52}")
        print(classification_report(self.y_true, self.y_pred,
                                    labels=list(LABELS), zero_division=0))
        return self.summary()

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        print(f"Results saved to {path}")


def mcnemar_test(y_true: list, pred_a: list, pred_b: list) -> dict:
    """Paired McNemar test: is arm B's accuracy change over arm A significant?

    Compares the two arms only on the samples where exactly one is correct
    (the discordant pairs). Uses the exact binomial test, which is valid for
    the small discordant counts typical of n~200 benchmarks.
    """
    from scipy.stats import binomtest

    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true, pred_a, pred_b must be the same length")

    # b: A wrong, B right (B's gains). c: A right, B wrong (B's losses).
    b = c = 0
    for gt, a, bb in zip(y_true, pred_a, pred_b, strict=False):
        a_ok, b_ok = (a == gt), (bb == gt)
        if a_ok and not b_ok:
            c += 1
        elif b_ok and not a_ok:
            b += 1

    n = b + c
    p_value = float(binomtest(b, n, 0.5).pvalue) if n > 0 else 1.0
    return {
        "b_gains": b,        # B right, A wrong
        "c_losses": c,       # A right, B wrong
        "discordant": n,
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
    }
