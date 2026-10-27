# Results

Canonical write-up of the published benchmark numbers — n=200, seed=42, model
`deepseek-r1:8b`, dataset PubMedQA (`pqa_labeled`), `LLM_NUM_CTX=8192` (what
`make benchmark` exports; the config default is 4096). This file is
**hand-maintained** to match `results/summary.json` (which `scripts/compare.py`
writes); the Streamlit dashboard and the frontend `/benchmark` page read the
JSON directly.

> [!WARNING]
> **The published graph-arm numbers were likely produced on a graph with
> duplicated `HAS_CONTEXT` edges.** The recorded ingest counts were **209,971
> `HAS_CONTEXT` edges vs 206,613 `Chunks`** — a surplus of 3,358, which is
> *exactly* the labeled split's chunk count. `scripts/ingest.py` created edges
> without deterministic keys, and notebook 01 ran a labeled-only smoke ingest
> before the full ingest, so every labeled paper's edges were inserted twice.
> Consequence: for labeled papers (i.e. the benchmark's own gold papers),
> `_PARENT_AQL` reconstructed each abstract with every section repeated
> (S0 S0 S1 S1 …), and concept-hop shared-concept counts were inflated.
> Information content is identical and this causes **no label leakage**, but
> the `graph`/`graph_concepts` contexts differ from the method as described.
> Confirm against the live DB by comparing
> `RETURN LENGTH(HAS_CONTEXT)` with `RETURN LENGTH(Chunks)` (they should be
> equal). `scripts/ingest.py` now uses deterministic edge keys, so a re-ingest
> is idempotent — but **the +22.5 pp result needs a re-run on a deduplicated
> graph to be confirmed.** See GAPS.md.

## Per-arm

| Arm | Accuracy | Macro F1 | Avg latency (s) | n | Adds |
| --- | --- | --- | --- | --- | --- |
| plain | 30.00% | 29.69% | 6.4 | 200 | baseline chunk RAG |
| plain_rr | 37.00% | 35.21% | 6.6 | 200 | + cross-encoder reranker |
| **graph** | **59.50%** | **50.51%** | 7.5 | 200 | + parent-paper expansion |
| graph_concepts | 57.50% | 49.97% | 40.8 | 200 | + MeSH concept hop |

`graph_norr` (parent expansion without the reranker — the hosted demo's
configuration) exists as an arm (`--arm graph_norr`) but **has not been run**;
there are no results for it.

## Significance (paired McNemar)

| Contrast | Δacc (pp) | gains | losses | p | significant? |
| --- | --- | --- | --- | --- | --- |
| plain → plain_rr (reranker effect) | +7.00 | 35 | 21 | 0.0814 | no |
| **plain_rr → graph (parent-expansion effect)** | **+22.50** | 71 | 26 | **<0.0001** | **yes** |
| graph → graph_concepts (concept-hop effect) | -2.00 | 26 | 30 | 0.6889 | no |

**Schema note:** the committed `summary.json` predates the fields
`compare.py` now emits (`n_failed`, Wilson 95% CIs, recall@k, Holm-adjusted
p). Holm-adjusting the three committed contrasts gives p = 1.6e-5 / 0.163 /
0.689 — the conclusions are unchanged. Those fields will appear on the next
benchmark run; the per-arm `results/{arm}_results.json` files needed to
regenerate the summary were never committed.

**Verification status:** the raw per-question outputs from the historical
experiment are not committed, and the historical graph may have duplicated
HAS_CONTEXT relationships. A new [audit and evidence publication protocol](EVIDENCE.md)
can verify **future**, genuinely rerun model outputs. No synthetic results
are substituted for missing historical predictions.

**Majority-class baseline:** always answering "yes" scores **56.5%** on this
seeded sample (113 yes / 68 no / 19 maybe). `plain`/`plain_rr` fall well below
it; `graph` clears it by only +3.0 pp (Wilson 95% CI 52.6–66.1%, which
overlaps it). The meaningful result is the paired contrast against `plain_rr`,
not the absolute accuracy.

## Honest summary

The graph arm's win is not "adding a knowledge graph" in the abstract — it's
one specific mechanism: **parent-document expansion**. Once a chunk is
retrieved, walking `HAS_CONTEXT` back to the parent paper and handing the LLM
the full abstract instead of an isolated chunk lifts accuracy by **+22.5
percentage points** over the reranked baseline, and a paired McNemar test on
the same 200 questions confirms this is not noise (p < 0.0001). The reranker
alone (+7.0pp) does not clear significance at this sample size. The further
MeSH concept-hop (`graph_concepts`) does not help — it's not significantly
different from `graph` (p = 0.69) and costs ~5x the latency (40.8s vs 7.5s),
because that latency is dominated by the extra graph traversal and abstract
reconstruction per hop, not by the LLM call itself. So: the graph helps
because it recovers context that flat chunking throws away, not because more
graph traversal is inherently better. Ship `graph`, not `graph_concepts`.

**Correct resume framing:** "+22.5pp accuracy lift from parent-document graph
expansion over a reranked RAG baseline, McNemar p<0.0001" — not "GraphRAG gets
65% accuracy" and not "graph retrieval is 2x faster" (it is ~14% *slower*,
7.5s vs 6.6s, and that's an honest, disclosed tradeoff for the accuracy gain).

**A methodology note on unparseable output:** `FuzzyEvaluator` (`src/kgqa/
evaluation.py`) defaults any answer it can't extract a yes/no/maybe from to
`"maybe"` — a deliberate, documented choice (one label has to absorb "the
model didn't answer clearly"), but it's not a neutral one: it biases those
errors toward the rare `maybe` class rather than splitting them across all
three, and can flatter or penalize an arm depending on how often *that arm
specifically* produces unparseable output. Worth knowing when comparing arms
with meaningfully different failure/degradation rates (see the per-arm
`n_failed` count `scripts/compare.py` now reports). The extractor also has
known mis-scoring modes (takes the first match, no word boundaries, relies on
an opening `<think>` tag); they are deliberately left unchanged so these
numbers stay reproducible, and will be fixed on the next re-run (GAPS.md).

## Caveat: the hosted demo runs an unbenchmarked configuration

The numbers above come from the untouched ArangoDB benchmark pipeline. The
[live hosted demo](https://graphrag-pubmedqa-ablation.vercel.app) is a
*different* deployment (`graph_id="demo"`, Neo4j-backed, scoped to the
1,000-paper labeled split — see the README's Hosted agent section) that
additionally runs **without the cross-encoder reranker**
(`KGQA_SKIP_RERANKER=true`), a memory-saving concession for Render's free
tier. That's `graph`'s parent-expansion mechanism on top of `plain`, not
`plain_rr` — a configuration this table has never measured in isolation. The
reranker's own effect (+7.0pp, not significant at n=200) suggests the hosted
demo likely performs somewhere between `plain` and `graph`, but that's an
inference, not a measurement. The fifth `graph_norr` arm that closes this
gap now exists but has not been run; until it is, treat the hosted demo as a *qualitative* illustration of parent
expansion, not a live instance of the `graph` row above.

Full methodology and the 4-arm ablation diagram: [README.md](README.md#-how-the-experiment-works).
