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

**Honest expectation.** PubMedQA is mostly single-abstract QA, so a fair
parent-expansion gain may be modest. The interesting signal is in the
`graph_concepts` arm and in multi-evidence questions — that is where a graph can
legitimately beat plain retrieval. The point of this repo is to measure that
honestly, not to manufacture a win.

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

> ⏳ **Pending the Colab benchmark run.** `scripts/compare.py` regenerates the
> table below into `results/summary.md` and `results/ablation.png`.

| Arm | Accuracy | Macro F1 | What it isolates |
| --- | --- | --- | --- |
| `plain` | — | — | baseline RAG |
| `plain_rr` | — | — | + reranker |
| `graph` | — | — | + parent-paper expansion |
| `graph_concepts` | — | — | + MeSH concept hop |

Paired McNemar tests (reranker effect, parent-expansion effect, concept-hop
effect) are written alongside the table.

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
