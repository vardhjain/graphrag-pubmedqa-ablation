"""Aggregate arm results: summary table, McNemar tests, and figures.

    python scripts/compare.py

Reads results/{arm}_results.json for whichever arms are present and writes
figures + a markdown snippet to results/. The McNemar tests are paired on pubid,
so they only run for arms evaluated on the same seeded sample.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from kgqa.config import DATASET_NAME, LLM_MODEL, RANDOM_SEED  # noqa: E402
from kgqa.evaluation import mcnemar_test  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "results")
# The first 4 are the primary cumulative ladder (each adds exactly one thing
# over the last); graph_norr is a fifth, non-cumulative arm appended after --
# it isolates parent-expansion *without* the reranker, i.e. the exact
# configuration the hosted demo runs (render.yaml: KGQA_SKIP_RERANKER=true,
# a 512MB-tier memory concession) that the main ladder never measures alone.
ARM_ORDER = ["plain", "plain_rr", "graph", "graph_concepts", "graph_norr"]
ARM_ADDS = {
    "plain": "baseline chunk RAG",
    "plain_rr": "+ cross-encoder reranker",
    "graph": "+ parent-paper expansion",
    "graph_concepts": "+ MeSH concept hop",
    "graph_norr": "+ parent-paper expansion, no reranker (hosted demo's config)",
}
# Adjacent-arm contrasts that isolate each component's contribution.
CONTRASTS = [
    ("plain", "plain_rr", "reranker"),
    ("plain_rr", "graph", "parent expansion"),
    ("graph", "graph_concepts", "concept hop"),
    ("plain", "graph_norr", "parent expansion without reranker"),
]


def load_results():
    out = {}
    for arm in ARM_ORDER:
        path = os.path.join(RESULTS_DIR, f"{arm}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                out[arm] = json.load(f)
    return out


def format_p(p: float) -> str:
    """Human-readable p-value that can't misrepresent a real effect as zero.

    McNemar p-values on a clear effect (e.g. the parent-expansion contrast)
    are often ~1e-11 -- round(p, 4) collapses that to the literal string
    "0.0000", which reads as a data error on a project whose whole pitch is
    statistical rigor (the prose elsewhere correctly says "p < 0.0001", so
    the generated artifacts contradicting it is the bug, not the number).
    """
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def aligned(a, b):
    """Align two arms' predictions on shared pubids (same seed -> same order),
    excluding any id where either arm's LLM call failed (recorded as a
    placeholder "maybe", not a genuine model judgment -- see
    Evaluator.record's failed param). Attributing an infra timeout to "the
    retrieval strategy" would contaminate the one contrast this project's
    ablation exists to get right. Old-format result JSONs with no "failed"
    field are treated as zero failures.
    """
    ids_a = a.get("ids") or list(range(len(a["y_pred"])))
    ids_b = b.get("ids") or list(range(len(b["y_pred"])))
    failed_a = a.get("failed") or [False] * len(ids_a)
    failed_b = b.get("failed") or [False] * len(ids_b)
    idx_b = {sid: i for i, sid in enumerate(ids_b)}
    gt, pa, pb = [], [], []
    for i, sid in enumerate(ids_a):
        if failed_a[i]:
            continue
        j = idx_b.get(sid)
        if j is None or failed_b[j]:
            continue
        gt.append(a["y_true"][i])
        pa.append(a["y_pred"][i])
        pb.append(b["y_pred"][j])
    return gt, pa, pb


def main():
    results = load_results()
    if not results:
        print(f"No results found in {RESULTS_DIR}. Run scripts/run_benchmark.py first.")
        sys.exit(1)

    lines = ["| Arm | Accuracy | Macro F1 | Recall@k | Avg latency (s) | n |",
             "| --- | --- | --- | --- | --- | --- |"]
    arms_json, contrasts_json, max_n = [], [], 0
    print("\n" + "=" * 64)
    print("  RESULTS SUMMARY")
    print("=" * 64)
    for arm in ARM_ORDER:
        if arm not in results:
            continue
        r = results[arm]
        acc, f1 = r["accuracy"] * 100, r.get("macro_f1", 0) * 100
        lat, n = r["avg_latency"], r["samples"]
        n_failed = r.get("n_failed", 0)
        recall_k = r.get("recall_at_k")  # None for result JSONs predating this metric
        recall_display = f"{recall_k * 100:.1f}%" if recall_k is not None else "n/a"
        max_n = max(max_n, n)
        failed_note = f"  failed={n_failed}" if n_failed else ""
        print(f"  {arm:<16} acc={acc:6.2f}%  f1={f1:6.2f}%  recall@k={recall_display:>6}  "
              f"lat={lat:5.1f}s  n={n}{failed_note}")
        lines.append(f"| {arm} | {acc:.2f}% | {f1:.2f}% | {recall_display} | {lat:.1f} | {n} |")
        arms_json.append({"arm": arm, "accuracy": round(acc, 2), "macro_f1": round(f1, 2),
                          "recall_at_k": round(recall_k, 4) if recall_k is not None else None,
                          "avg_latency": round(lat, 1), "samples": n, "n_failed": n_failed,
                          "adds": ARM_ADDS.get(arm, "")})

    print("\n" + "=" * 64)
    print("  PAIRED McNEMAR TESTS (adjacent ablation contrasts)")
    print("=" * 64)
    lines += ["", "### Significance (paired McNemar)", "",
              "| Contrast | Δacc (pp) | gains | losses | p | sig? |",
              "| --- | --- | --- | --- | --- | --- |"]
    for a_name, b_name, desc in CONTRASTS:
        if a_name not in results or b_name not in results:
            continue
        gt, pa, pb = aligned(results[a_name], results[b_name])
        if not gt:
            continue
        test = mcnemar_test(gt, pa, pb)
        acc_a = sum(p == g for p, g in zip(pa, gt, strict=False)) / len(gt)
        acc_b = sum(p == g for p, g in zip(pb, gt, strict=False)) / len(gt)
        d = (acc_b - acc_a) * 100
        sig = "yes" if test["significant_at_0.05"] else "no"
        p_display = format_p(test["p_value"])
        print(f"  {a_name} -> {b_name}  ({desc})")
        print(f"      Δacc={d:+.2f}pp  gains={test['b_gains']}  losses={test['c_losses']}"
              f"  p={p_display}  sig={sig}")
        lines.append(f"| {a_name} → {b_name} ({desc}) | {d:+.2f} | {test['b_gains']} "
                     f"| {test['c_losses']} | {p_display} | {sig} |")
        contrasts_json.append({"from": a_name, "to": b_name, "effect": desc,
                               "delta_acc": round(d, 2), "gains": test["b_gains"],
                               "losses": test["c_losses"],
                               # Full precision, not rounded to 4dp -- a real
                               # p~1e-11 must not collapse to the literal 0.0.
                               "p_value": test["p_value"],
                               "p_display": p_display,
                               "significant": test["significant_at_0.05"]})

    md_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {md_path}")

    json_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"n": max_n, "seed": RANDOM_SEED, "model": LLM_MODEL,
                   "dataset": "PubMedQA (pqa_labeled)" if "PubMedQA" in DATASET_NAME
                   else DATASET_NAME,
                   "arms": arms_json, "contrasts": contrasts_json}, f, indent=2)
    print(f"Wrote {json_path}")

    _plot(results)


def _plot(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping figures: {exc})")
        return

    arms = [a for a in ARM_ORDER if a in results]
    accs = [results[a]["accuracy"] * 100 for a in arms]
    f1s = [results[a].get("macro_f1", 0) * 100 for a in arms]

    fig, ax = plt.subplots(figsize=(9, 5))
    import numpy as np
    x = np.arange(len(arms))
    w = 0.38
    ax.bar(x - w / 2, accs, w, label="Accuracy", color="#2196F3")
    ax.bar(x + w / 2, f1s, w, label="Macro F1", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=15)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title("4-arm ablation — PubMedQA")
    ax.legend()
    for i, (a, f) in enumerate(zip(accs, f1s, strict=False)):
        ax.text(i - w / 2, a + 1, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, f + 1, f"{f:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
