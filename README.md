# Knowledge Graph Question Answering — GraphRAG vs PlainRAG, done fairly

A controlled study of **what a knowledge graph actually contributes** to
retrieval-augmented question answering on biomedical literature
([PubMedQA](https://pubmedqa.github.io/)).

Most "GraphRAG beats RAG" demos are confounded: the graph pipeline quietly also
gets a reranker, a different corpus, or — worst of all — leaks the answer into
the prompt. This repo throws those out and runs a **4-arm ablation** where every
layer is held constant and the *only* thing that changes is how much graph
structure the retriever uses.

```
plain ─► plain_rr ─► graph ─► graph_concepts
 (RAG)   (+rerank)   (+parent  (+MeSH concept
                      expansion) hop)
```

Same corpus, same chunking, same embedder, same reranker, same prompt, same LLM,
same seeded sample, same top-k. The accuracy delta between adjacent arms is
attributable to exactly one component, and we report a **paired McNemar test** so
you can tell a real effect from noise.

---

## Why the original comparison was unfair (and what changed)

This started from a working but confounded notebook comparison. The audit found
six issues; all are fixed in this revamp:

| # | Flaw (before) | Fix (now) |
| --- | --- | --- |
| 1 | GraphRAG had a cross-encoder reranker; PlainRAG was raw FAISS top-3 | The reranker is its **own arm** (`plain_rr`). The graph arms build *on top of* `plain_rr`, so the rerank is controlled for, not a hidden advantage |
| 2 | The two pipelines indexed **different corpora** | All arms search one shared `ChunkStore` (labeled + unlabeled, identical chunks) |
| 3 | Different granularity (whole abstracts vs per-section chunks) | Identical per-section chunking for every arm |
| 4 | **Label leakage**: papers stored `title = question` and `final_decision`, injected into the prompt as `=== STUDY: {title} ===` | Ingestion stores **no** question-derived title and **no** `final_decision`; graph context uses generic `=== STUDY n ===` labels with abstracts only. A unit test asserts the question never appears in the context |
| 5 | `Concepts` (MeSH) and `MENTIONS` edges were built but **never used** | The `graph_concepts` arm hops across shared MeSH concepts to pull in related papers |
| 6 | `NameError` in the graph fallback; first-100 samples, no seed, no significance test | Fixed fallback; seeded random sample (default n=200); paired McNemar test |

**What we expected vs. what we found.** Going in, we expected concept-hop
expansion to be where the graph shines and a plain parent-expansion gain to be
modest. The data said the opposite: the decisive, statistically significant win
came from **parent-document expansion**, while concept-hop did not help on this
single-abstract dataset. We report that honestly rather than bury it — see
[Results](#results).

---

## Repository layout

```
src/kgqa/                 importable package — single source of truth
  config.py               all shared constants (models, top-k, seed, n)
  prompts.py              benchmark/chat prompts (identical across arms)
  llm.py                  Ollama client
  data.py                 seeded sampling + canonical chunking
  evaluation.py           answer extraction, metrics, McNemar test
  models.py               encoder / reranker / ArangoDB loaders
  retrieval/
    base.py               ChunkStore + BaseRetriever (encode→rerank→select)
    plain.py              plain, plain_rr arms
    graph.py              graph, graph_concepts arms
scripts/
  ingest.py               build the leakage-free graph in ArangoDB (run once)
  run_benchmark.py        run one arm: --arm {plain,plain_rr,graph,graph_concepts}
  compare.py              summary table + McNemar + ablation figure
notebooks/
  01_ingest.ipynb         thin Colab wrapper for ingestion
  02_benchmark.ipynb      thin Colab wrapper for all arms + comparison
tests/                    pytest suite (runs on CPU, no Ollama/ArangoDB needed)
docs/                     project report (PDF) and slides (PPTX)
```

## Stack

- **Dataset:** PubMedQA (`pqa_labeled` for evaluation, `pqa_unlabeled` for corpus)
- **Embeddings:** `all-MiniLM-L6-v2` (384-dim)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Graph DB:** ArangoDB Oasis (Papers / Chunks / Concepts; HAS_CONTEXT / MENTIONS)
- **LLM:** `deepseek-r1:8b` via [Ollama](https://ollama.com)

---

## Setup

```bash
pip install -r requirements.txt        # add -r requirements-dev.txt for tests
cp .env.example .env                    # then fill in ARANGO_PASS
```

Secrets are read from the environment (or a local `.env`, or Colab Secrets):
`ARANGO_HOST`, `ARANGO_USER`, `ARANGO_PASS`, `ARANGO_DB`. Nothing is hardcoded.

## Running the benchmark

The graph lives in ArangoDB Oasis and the LLM runs on a GPU, so the benchmark is
designed to run on **Google Colab (T4)** — open the notebooks below. To run
locally you need a reachable ArangoDB and a running Ollama.

```bash
# 1. Build the graph once
python scripts/ingest.py

# 2. Run each arm (downloads + caches the shared chunk corpus on first run)
python scripts/run_benchmark.py --arm plain          --n 200
python scripts/run_benchmark.py --arm plain_rr       --n 200
python scripts/run_benchmark.py --arm graph          --n 200
python scripts/run_benchmark.py --arm graph_concepts --n 200

# 3. Summary table, McNemar tests, ablation figure -> results/
python scripts/compare.py
```

On Colab, run [`notebooks/01_ingest.ipynb`](notebooks/01_ingest.ipynb) once, then
[`notebooks/02_benchmark.ipynb`](notebooks/02_benchmark.ipynb).

---

## Results

Seeded random sample of **n = 200** PubMedQA `pqa_labeled` questions (seed 42,
identical across arms), `deepseek-r1:8b` via Ollama on an A100. Regenerate with
`scripts/compare.py` (writes `results/summary.md` and `results/ablation.png`).

| Arm | Accuracy | Macro F1 | Avg latency | Adds |
| --- | --- | --- | --- | --- |
| `plain` | 30.0% | 29.7% | 6.4 s | baseline chunk RAG |
| `plain_rr` | 37.0% | 35.2% | 6.6 s | + cross-encoder reranker |
| **`graph`** | **59.5%** | **50.5%** | 7.5 s | + parent-paper expansion |
| `graph_concepts` | 57.5% | 50.0% | 40.8 s | + MeSH concept hop |

**Paired McNemar tests** — each contrast isolates one component on the same 200 questions:

| Contrast | Δ accuracy | gains / losses | p | significant? |
| --- | --- | --- | --- | --- |
| `plain → plain_rr` (reranker) | +7.0 pp | 35 / 21 | 0.081 | no |
| `plain_rr → graph` (parent expansion) | **+22.5 pp** | 71 / 26 | **<0.0001** | **yes** |
| `graph → graph_concepts` (concept hop) | −2.0 pp | 26 / 30 | 0.69 | no |

![4-arm ablation on PubMedQA](results/ablation.png)

### What the ablation shows

1. **The graph's decisive win is parent-document expansion** (+22.5 pp,
   p < 0.0001). Retrieving at the fine-grained chunk level but feeding the LLM the
   *full reconstructed abstract* (chunk → paper → all sections, via `HAS_CONTEXT`)
   is what moves the needle — for only ~1 s over `plain_rr`. With the label
   leakage fixed, this is a clean, legitimate graph advantage.
2. **Single-fragment retrieval is not enough for PubMedQA.** `plain` and
   `plain_rr` land *below* the majority-class baseline (PubMedQA is ≈55% "yes"); a
   lone ~250-character section rarely contains enough to judge the question.
   Context sufficiency — which the graph supplies — is the dominant factor, and
   `graph` is the only arm that clears the trivial baseline.
3. **The reranker helps modestly but not significantly** at this sample size
   (+7 pp, p = 0.08).
4. **Concept-hop expansion does not help here** (−2 pp, p = 0.69) and costs ~5×
   the latency. An honest — and expected — negative result: on single-abstract QA,
   papers pulled in via shared MeSH terms act mostly as distractors. The graph
   helps by *deepening* context (the full document), not by *broadening* it
   (related documents).

The macro-F1 / accuracy gap on the graph arms reflects weak recall on the rare
`maybe` class (~11% of the data) — a dataset property, not a retrieval one.

---

## Development

```bash
pytest                 # 17 tests, all CPU, no external services
ruff check src scripts tests
```

CI runs ruff + pytest on every push/PR (Python 3.10 and 3.11). Unit tests inject
fakes for the encoder, reranker, and ArangoDB, so the heavy ML dependencies are
never needed just to verify the logic.

## License

[MIT](LICENSE).
