"""Answer extraction, metrics, and significance testing.

Kept free of any plotting import at module load so it is importable in headless
CI. Figure generation lives in ``scripts/compare.py``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from sklearn.metrics import accuracy_score, classification_report, f1_score

LABELS = ("yes", "no", "maybe")


class FuzzyEvaluator:
    """Extracts a normalised yes/no/maybe from verbose model output."""

    def extract_answer(self, text: str) -> str:
        """Map raw LLM output to yes/no/maybe.

        Known failure modes (documented, deliberately NOT fixed -- changing
        this would change every published number in results/summary.json):
          * ``final answer: X`` takes the FIRST match, so a model that restates
            the instruction ("...end with Final answer: yes/no/maybe") before its
            real verdict can be scored on the echoed template.
          * That pattern has no leading ``\\b``, so e.g. "semifinal answer: no"
            also matches.
          * ``<think>`` stripping needs both tags: an unterminated ``<think>``
            (truncated by ``num_predict``) is left in place, so the fallback
            last-yes/no/maybe scan runs over the reasoning chain itself.
        """
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
    # Paper keys _select() retrieved for this question, pre any graph
    # expansion. Retrieval is identical *within a reranker setting*:
    # {plain, graph_norr} share the cosine top-k and {plain_rr, graph,
    # graph_concepts} share the reranked top-k, so within each group recall@k over this set tests whether accuracy differences
    # come from retrieving different papers or from presenting the same
    # retrieved paper differently. See BaseRetriever.answer_benchmark.
    retrieved_papers: list = field(default_factory=list)
    # Optional per-sample flags (run_benchmark.py): graph expansion fell back
    # to raw chunks / LLM output hit the generation cap. Absent in old files.
    degraded: list = field(default_factory=list)
    truncated: list = field(default_factory=list)
    # Optional run metadata (arm, seed, model, ...). Omitted from the saved
    # JSON when empty so old-format consumers see exactly the old keys.
    meta: dict = field(default_factory=dict)

    def record(
        self,
        ground_truth: str,
        prediction: str,
        latency: float = 0.0,
        sample_id: str | None = None,
        failed: bool = False,
        retrieved_papers: list | None = None,
        degraded: bool = False,
        truncated: bool = False,
    ) -> None:
        pred = prediction.lower().strip()
        if pred not in LABELS:
            pred = "maybe"
        self.y_true.append(ground_truth.lower().strip())
        self.y_pred.append(pred)
        self.latencies.append(latency)
        self.ids.append(sample_id)
        self.failed.append(failed)
        self.retrieved_papers.append(retrieved_papers or [])
        self.degraded.append(bool(degraded))
        self.truncated.append(bool(truncated))

    def n_failed(self) -> int:
        return sum(self.failed)

    def recall_at_k(self) -> float:
        """Fraction of (non-failed) samples where the gold paper -- the
        question's own source paper, i.e. PubMedQA's pubid -- appears among
        the papers _select() retrieved. Isolates retrieval quality from the
        LLM synthesis step. Only scored where sample_id was set to the gold
        pubid (run_benchmark.py's convention); a genuinely empty retrieval
        counts as a miss, not an exclusion."""
        scored = [
            (sid, papers)
            for sid, papers, failed in zip(
                self.ids, self.retrieved_papers, self.failed, strict=False
            )
            if sid is not None and not failed
        ]
        if not scored:
            return 0.0
        return sum(1 for sid, papers in scored if sid in papers) / len(scored)

    # ── metrics ───────────────────────────────────────────────────────────────
    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred) if self.y_true else 0.0

    def macro_f1(self) -> float:
        if not self.y_true:
            return 0.0
        return f1_score(
            self.y_true, self.y_pred, labels=list(LABELS), average="macro", zero_division=0
        )

    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "accuracy": self.accuracy(),
            "macro_f1": self.macro_f1(),
            "samples": len(self.y_true),
            "n_failed": self.n_failed(),
            "recall_at_k": self.recall_at_k(),
            "total_time": sum(self.latencies),
            "avg_latency": self.avg_latency(),
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "ids": self.ids,
            "failed": self.failed,
            "latencies": self.latencies,
            "retrieved_papers": self.retrieved_papers,
            "degraded": self.degraded,
            "truncated": self.truncated,
            "n_degraded": sum(self.degraded),
            "n_truncated": sum(self.truncated),
            **({"meta": self.meta} if self.meta else {}),
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
            print(
                f"  Failed    : {self.n_failed()} (recorded as 'maybe'; "
                "excluded from paired significance tests, see compare.py)"
            )
        print(f"  Accuracy  : {self.accuracy():.2%}")
        print(f"  Macro F1  : {self.macro_f1():.2%}")
        print(
            f"  Recall@k  : {self.recall_at_k():.2%}  (gold paper among retrieved, pre-expansion)"
        )
        print(f"  Avg/query : {self.avg_latency():.1f}s")
        print(f"{'-' * 52}")
        print(classification_report(self.y_true, self.y_pred, labels=list(LABELS), zero_division=0))
        return self.summary()

    def save(self, path: str, quiet: bool = False) -> None:
        """Write summary() atomically (tmp file + os.replace), so a crash
        mid-write can never leave a half-written JSON behind."""
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.summary(), f, indent=2)
        os.replace(tmp, path)
        if not quiet:
            print(f"Results saved to {path}")

    @classmethod
    def from_saved(cls, data: dict) -> Evaluator:
        """Rebuild an Evaluator from a saved summary() dict (for --resume)."""
        ev = cls(data.get("model", ""))
        n = len(data.get("y_true", []))
        ev.y_true = list(data.get("y_true", []))
        ev.y_pred = list(data.get("y_pred", []))
        ev.ids = list(data.get("ids") or [None] * n)
        ev.failed = list(data.get("failed") or [False] * n)
        ev.retrieved_papers = list(data.get("retrieved_papers") or [[] for _ in range(n)])
        ev.degraded = list(data.get("degraded") or [False] * n)
        ev.truncated = list(data.get("truncated") or [False] * n)
        # Per-sample latencies aren't saved; spread the total so avg is preserved.
        avg = data.get("avg_latency", 0.0)
        ev.latencies = list(data.get("latencies") or [avg] * n)
        ev.meta = dict(data.get("meta") or {})
        return ev


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
        "b_gains": b,  # B right, A wrong
        "c_losses": c,  # A right, B wrong
        "discordant": n,
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
    }
