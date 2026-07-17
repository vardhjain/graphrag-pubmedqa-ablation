"""Streamlit dashboard for the GraphRAG vs PlainRAG ablation results.

    pip install streamlit          # see requirements-app.txt
    streamlit run app/dashboard.py

Reads results/summary.json (always) for the headline metrics and significance
tests, and results/{arm}_results.json (if present) for confusion matrices and
per-class F1. No LLM or database needed — it just visualizes the saved results,
so it deploys cleanly to Streamlit Cloud.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS_DIR = os.path.join(ROOT, "results")
ARM_ORDER = ["plain", "plain_rr", "graph", "graph_concepts"]
LABELS = ["yes", "no", "maybe"]


@st.cache_data
def load_summary():
    with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
        return json.load(f)


@st.cache_data
def load_raw():
    raw = {}
    for arm in ARM_ORDER:
        path = os.path.join(RESULTS_DIR, f"{arm}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                raw[arm] = json.load(f)
    return raw


def main():
    repo = "https://github.com/vardhjain/graphrag-pubmedqa-ablation"
    st.set_page_config(
        page_title="GraphRAG vs PlainRAG — PubMedQA Ablation",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": repo,
            "Report a bug": f"{repo}/issues",
            "About": (
                "### GraphRAG vs PlainRAG — a fair 4-arm ablation on PubMedQA\n"
                "Every layer held constant; only the retrieval strategy changes.\n\n"
                f"Source: [{repo}]({repo})"
            ),
        },
    )
    st.title("GraphRAG vs PlainRAG — a fair 4-arm ablation on PubMedQA")

    try:
        summary = load_summary()
    except FileNotFoundError:
        st.error("results/summary.json not found. Run `python scripts/compare.py` first.")
        st.stop()

    st.caption(
        f"n = {summary['n']} questions · seed {summary['seed']} · "
        f"{summary['model']} · {summary['dataset']}. "
        "Every layer held constant; only the retrieval strategy changes."
    )

    arms = summary["arms"]
    best = max(arms, key=lambda a: a["accuracy"])

    # ── headline metrics ──────────────────────────────────────────────────────
    cols = st.columns(len(arms))
    for col, arm in zip(cols, arms, strict=False):
        delta = f"{arm['accuracy'] - arms[0]['accuracy']:+.1f} pp vs plain" \
            if arm["arm"] != "plain" else None
        col.metric(arm["arm"], f"{arm['accuracy']:.1f}%", delta)

    st.success(
        f"**Winner: `{best['arm']}` at {best['accuracy']:.1f}%.** The decisive, "
        "statistically significant gain comes from parent-document expansion "
        "(`plain_rr → graph`: +22.5 pp, McNemar p < 0.0001). The reranker helps "
        "but isn't significant; the concept hop doesn't help and costs ~5× latency."
    )

    with st.expander("How this is measured (fairness)"):
        st.markdown(
            "All four arms share the same corpus, chunking, embedder, reranker, "
            "prompt, LLM, seed, and top-k — **only the retrieval strategy changes**, "
            "so each adjacent contrast isolates one component. Significance is a "
            "paired **McNemar** test on the same questions. The graph context is "
            "leakage-free: no question-derived titles or gold labels ever reach the "
            "prompt."
        )

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Accuracy & macro-F1 by arm")
        df = pd.DataFrame(arms).set_index("arm")
        st.bar_chart(df[["accuracy", "macro_f1"]], stack=False, color=["#2196F3", "#FF9800"])
        st.dataframe(
            df[["adds", "accuracy", "macro_f1", "avg_latency", "samples"]],
            use_container_width=True,
        )

    with right:
        st.subheader("Significance (paired McNemar)")
        cdf = pd.DataFrame(summary["contrasts"])
        cdf["contrast"] = cdf["from"] + " → " + cdf["to"] + "  (" + cdf["effect"] + ")"
        cdf["significant"] = cdf["significant"].map({True: "yes", False: "no"})
        # p_display (e.g. "<0.0001") avoids rendering a real, significant
        # p-value (~1e-11 on the parent-expansion contrast) as the literal
        # "0.000000" -- fall back to formatting p_value for older summary.json
        # files generated before p_display existed.
        cdf["p"] = cdf.get("p_display", cdf["p_value"].map(lambda p: f"{p:.4f}"))
        st.dataframe(
            cdf[["contrast", "delta_acc", "gains", "losses", "p", "significant"]],
            use_container_width=True, hide_index=True,
        )
        st.caption("Latency by arm (seconds / query)")
        st.bar_chart(df["avg_latency"], color="#26A69A", horizontal=True)

    # ── optional: per-class detail from raw per-sample results ────────────────
    raw = load_raw()
    if raw:
        st.subheader("Per-class detail")
        from sklearn.metrics import confusion_matrix, f1_score
        tabs = st.tabs([a for a in ARM_ORDER if a in raw])
        for tab, arm in zip(tabs, [a for a in ARM_ORDER if a in raw], strict=False):
            with tab:
                r = raw[arm]
                cm = confusion_matrix(r["y_true"], r["y_pred"], labels=LABELS)
                st.write("Confusion matrix (rows = actual, cols = predicted)")
                st.dataframe(pd.DataFrame(cm, index=LABELS, columns=LABELS))
                f1s = f1_score(r["y_true"], r["y_pred"], labels=LABELS,
                               average=None, zero_division=0)
                st.write("Per-class F1")
                st.bar_chart(pd.Series(f1s, index=LABELS))

    st.caption(f"Source: {repo}")


if __name__ == "__main__":
    main()
