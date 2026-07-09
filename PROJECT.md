# PROJECT.md — Knowledge Graph Question Answering (GraphRAG vs PlainRAG on PubMedQA)

> Orientation doc for an engineer or AI agent seeing this repo for the first time.
> For operational rules and commands see [CLAUDE.md](CLAUDE.md); for known weaknesses
> see [GAPS.md](GAPS.md).

---

## 1. What this is and who it's for

This project answers one research question honestly: **what does a knowledge
graph actually add to retrieval-augmented question answering, once you stop
cheating?**

It's a **controlled 4-arm ablation** on [PubMedQA](https://pubmedqa.github.io/)
(biomedical yes/no/maybe questions over paper abstracts). The four arms build on
each other, each adding exactly one component:

```
plain ──► plain_rr ──► graph ──► graph_concepts
(vector    (+ cross-    (+ parent-   (+ MeSH concept
 top-k)     encoder      document     hop to related
            reranker)    expansion)   papers)
```

Everything else — corpus, chunking, embedder, reranker, prompt, LLM, random
seed, sample size, top-k — is held **identical** across arms, so the accuracy
delta between two adjacent arms is attributable to the one component that
changed, and a **paired McNemar test** tells a real effect from noise.

The audience is twofold:
- **Researchers / reviewers / recruiters** evaluating whether "GraphRAG beats
  RAG" claims hold up. The headline finding is deliberately un-hyped: the graph's
  decisive, statistically-significant win (**+22.5 pp, p<0.0001**) comes entirely
  from **parent-document expansion** (rebuild the full abstract from a matched
  chunk), *not* from concept-graph traversal, which actually slightly hurts here.
- **Users of the hosted demo**: a live chat agent that answers biomedical
  questions from the graph, cites PubMed IDs, and visualizes its reasoning path.

This started life as a *confounded* notebook comparison (GraphRAG secretly also
got a reranker, a different corpus, and — worst — the answer leaked into its
prompt). The whole repo is the fixed, fair version. See the "Why the original was
unfair" table in [README.md](README.md) and the audit in
[KGQA_session_export.md](KGQA_session_export.md).

---

## 2. Tech stack and why

| Layer | Choice | Why |
| --- | --- | --- |
| **Dataset** | PubMedQA (`pqa_labeled` for eval, `pqa_unlabeled` for corpus) | Standard biomedical QA benchmark with a clean yes/no/maybe label and per-section context, which makes chunking natural. |
| **Embeddings** | `all-MiniLM-L6-v2` (384-dim, sentence-transformers) | Small, fast, CPU-friendly; quality is not the variable under test, fairness is. |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Its contribution is isolated as its own arm (`plain_rr`) rather than being a hidden GraphRAG advantage. |
| **Benchmark graph DB** | **ArangoDB** (multi-model, AQL) | Papers/Chunks/Concepts nodes + HAS_CONTEXT/MENTIONS edges. All RESULTS.md numbers come from here. Runs local (Docker) or ArangoDB Oasis. |
| **Hosted-demo graph DB** | **Neo4j AuraDB Free** | A *second, parallel* graph backend used **only** by the live hosted agent's `graph_id="demo"` path, scoped to the 1,000-paper labeled split so it fits a genuinely-free-forever tier. The benchmark pipeline never touches it. |
| **Benchmark LLM** | `deepseek-r1:8b` via **Ollama** (local) | Reasoning model, deterministic (`temperature=0`), GPU-friendly on Colab. Emits `<think>…</think>` which the answer extractor strips. |
| **Hosted-demo LLM** | **Groq / Gemini Flash**, falling back to Ollama | Free cloud inference so the hosted agent needs no GPU. A provider *chain* (see `providers.py`) survives a single provider's outage or model deprecation. |
| **Backend** | **FastAPI** + Uvicorn | Thin JSON API (`/query`, `/ingest`, `/health`) plus an MCP tool at `/mcp` (see `backend/mcp_server.py`), all over the `graphrag.answer()` service. Deployed on Render free tier. |
| **Frontend** | **Next.js 16** (App Router, React 19) + Tailwind v4 + reactflow | Chat UI + a `/benchmark` page + an interactive reasoning-path graph. Static-generated benchmark page reads `results/summary.json` at build time. Deploys on Vercel. |
| **Dashboard** | **Streamlit** | Zero-backend results dashboard (reads committed `results/` JSON). This is the always-on hosted demo — no LLM/DB/GPU needed. |
| **Chat demo (alt)** | **Gradio** (`app/chat_app.py`) | A *live* local chat over the ArangoDB `graph` arm; needs Ollama + ArangoDB, so it's run-it-yourself, not hosted. |
| **Tooling** | ruff, pytest, pre-commit, GitHub Actions, Makefile | Standard hygiene. Tests run on CPU with fakes — no ML deps, DB, or LLM needed in CI. |

**Key stack tension to understand up front:** there are **two graph backends and
two LLM paths**. The *research/benchmark* half uses ArangoDB + Ollama directly.
The *hosted-agent* half uses Neo4j + a cloud provider chain. They deliberately do
not share a backend — rewriting the benchmarked ArangoDB pipeline would require a
fresh GPU benchmark run to re-validate every number in RESULTS.md, which the dev
environment can't do. This split is the single most important thing to internalize.

---

## 3. Architecture: how it fits together

### The package (`src/kgqa/`) is the single source of truth

```
src/kgqa/
  config.py        ALL shared constants (models, top-k, seed, n, env-read DB configs).
                   Every arm imports from here so nothing can silently diverge.
  prompts.py       Benchmark + chat system prompts. Identical string across arms.
  llm.py           call_ollama() — the one direct Ollama entry point.
  providers.py     Multi-provider chain (groq/gemini/ollama) with per-task fallback.
                   Used by the hosted service, NOT the benchmark.
  data.py          Seeded PubMedQA sampling + canonical per-section chunking.
  evaluation.py    FuzzyEvaluator (extract yes/no/maybe), Evaluator (metrics),
                   mcnemar_test (paired significance). No plotting import.
  models.py        Lazy loaders: encoder, reranker, connect_arango, connect_neo4j.
  retrieval/
    base.py        ChunkStore (in-memory cosine search) + BaseRetriever
                   (encode → optional rerank → select → build context → answer).
    plain.py       PlainRetriever → arms "plain" / "plain_rr".
    graph.py       GraphRetriever (ArangoDB/AQL) → arms "graph" / "graph_concepts".
    neo4j_graph.py Neo4jGraphRetriever (parallel impl, same interface) — demo only.
  service.py       answer(question, graph_id) — the ONE entry point a web backend
                   calls. Caches stores/retrievers/connections; builds reasoning_path.
src/graphrag/
  __init__.py      Thin re-export: `from graphrag import answer`. Stable import
                   surface for the backend, decoupled from the kgqa package.
```

### Two flows

**A. Research / benchmark flow (ArangoDB + Ollama):**

```
scripts/ingest.py ──► ArangoDB (Papers/Chunks/Concepts, HAS_CONTEXT/MENTIONS)
                          │  leakage-free: no titles, no final_decision stored
                          ▼
scripts/run_benchmark.py --arm X
     data.load_benchmark_samples (seed 42, n=200)
     ChunkStore.from_arango (cached to pubmed_vectors_cache.pkl)
     build_retriever(X) → retriever.answer_benchmark(question)
         encode → rerank → select top-k → build context → call_ollama
     FuzzyEvaluator.extract_answer → Evaluator.record
     writes results/{arm}_results.json   (checkpoints every 25 Qs)
                          ▼
scripts/compare.py ──► results/summary.json + summary.md + ablation.png
     (per-arm table + adjacent-arm McNemar contrasts)
                          ▼
results/summary.json  ← the canonical artifact. Consumed by:
     • RESULTS.md (hand-written to match)
     • app/dashboard.py (Streamlit)
     • frontend/app/benchmark (Next.js, at build time)
     • tests/test_results_regression.py (CI gate)
```

**B. Hosted-agent flow (Neo4j + cloud LLM):**

```
frontend (Next.js, Vercel)
   └─ lib/api.ts  POST /query {question, graph_id:"demo"}
        ▼
backend/main.py (FastAPI, Render)
   └─ from graphrag import answer
        ▼
kgqa/service.py  answer(question, graph_id="demo")
   ├─ _get_retriever("demo") → Neo4jGraphRetriever (cached)
   │     store = ChunkStore.from_neo4j (vectors pre-computed at ingest; cached pickle)
   ├─ retriever._select(question)          # encode → rerank → top-k chunks
   ├─ retriever.gather_studies(candidates) # chunk → parent paper full abstract
   │                                       # (+ concept hop if use_concepts)
   ├─ call_llm("synthesize", …)            # gemini → ollama fallback chain
   └─ returns {answer, reasoning_path[], sources[]}
        ▼
frontend renders answer + PMID links + ReasoningGraph (reactflow)
```

Any `graph_id` other than `"demo"` routes to ArangoDB instead of Neo4j (for a
real dataset a future `/ingest` might build). `/ingest` currently only resolves
the preloaded `"demo"` id and returns **501** for anything else — arbitrary
document upload is explicitly out of scope for v1.

### Data model (both graph backends, same shape)

- **Paper** node — keyed by PubMed id (`pubid`). Stores **nothing** derived from
  the question and **no** gold label (the leakage fix).
- **Chunk** node — one per abstract section; holds `text` + pre-computed
  `embedding`. `HAS_CONTEXT`: Paper → Chunk.
- **Concept** node — a MeSH term (alphanumeric-normalized key). `MENTIONS`:
  Paper → Concept.

Retrieval always matches at the **chunk** level (fine-grained), then the graph
arms walk `HAS_CONTEXT` back to the parent Paper and hand the LLM the **full
reconstructed abstract**. That reconstruction — not concept traversal — is the
win.

---

## 4. Key design decisions (and the reasoning)

1. **One `config.py`, imported everywhere.** Fairness is the product. If two arms
   could ever use a different embedder/prompt/seed, the whole result is
   worthless. Centralizing every constant makes divergence structurally
   impossible, not just discouraged.

2. **Leakage-free ingestion, enforced by a test.** The original stored
   `title = question` and `final_decision` on Paper nodes and injected
   `=== STUDY: {title} ===` into the prompt — the answer literally appeared in
   the context. Now papers store only their key; graph context uses generic
   `=== STUDY n ===` labels. `test_graph_context_has_no_question_leakage` asserts
   the question string never appears in the context and the old leaky format is
   gone. This is a regression guard on the project's core integrity claim.

3. **The reranker is its own arm.** Isolating it (`plain → plain_rr`) means the
   graph arms build *on top of* rerank, so rerank is controlled for rather than
   being a hidden GraphRAG advantage — the single biggest flaw in the original.

4. **Shared `ChunkStore`.** All arms cosine-search one identical in-memory,
   L2-normalized embedding matrix (numpy, not FAISS — the old FAISS dep was
   removed as dead weight). Provably identical corpus and chunking.

5. **Paired McNemar via exact binomial.** With n≈200, discordant-pair counts are
   small; the exact binomial test is valid where a chi-square approximation is
   shaky. Pairing on `pubid` (same seed → same questions) is what makes the test
   legitimate.

6. **Graceful degradation everywhere in the service.** Every external dependency
   (ArangoDB, Neo4j, cloud LLM) can be missing and the service still answers:
   graph unreachable → fall back to raw retrieved chunks; no cloud key → fall
   back to local Ollama; no reranker (memory-constrained host) → raw top-k. The
   hosted demo must never hard-crash on a free-tier hiccup.

7. **Two graph backends on purpose.** ArangoDB for the immutable benchmark; Neo4j
   AuraDB Free for the hosted demo (small enough to be free-forever). See §2.
   `neo4j_graph.py` is a deliberate parallel implementation with the same
   `gather_studies` / `_build_context` interface, avoiding APOC so it works on any
   AuraDB tier.

8. **Per-task LLM provider chains.** Free cloud LLMs deprecate models without
   notice, so call sites pick a *task* (currently just `synthesize`) not a
   provider; the chain tries providers in order and falls back on any error.

9. **`results/summary.json` is the one artifact everything points at.** README,
   RESULTS.md, the Streamlit dashboard, the Next.js benchmark page, and the CI
   regression gate all read the same file, so published numbers can't drift out
   of sync. CI does **not** re-run the LLM benchmark (no GPU/DB); it guards the
   artifact instead.

10. **Ingestion runs once, offline; serving reads vectors.** Embeddings are
    computed at ingest time and stored as node properties. `ChunkStore.from_*`
    is a bulk read (further cached to a local pickle), not a live re-encode.

---

## 5. Critical paths — what's load-bearing vs. safe to change

**Load-bearing (change with great care, re-run tests, and re-benchmark if the
numbers could move):**

- `src/kgqa/config.py` — a change here silently affects *every* arm and can
  invalidate RESULTS.md. Treat constants as frozen unless you intend to
  re-benchmark.
- `src/kgqa/retrieval/base.py` — `ChunkStore` search and `BaseRetriever._select`
  are the shared retrieval spine of all four arms and both graph backends.
- `src/kgqa/retrieval/graph.py` — the AQL for parent expansion (`_PARENT_AQL`) is
  the +22.5 pp mechanism. The generic `=== STUDY n ===` formatting in
  `format_studies` is the anti-leakage contract.
- `scripts/ingest.py` — defines the leakage-free schema. Any regression here can
  reintroduce leakage; the ingestion and retrieval sides must agree on chunk keys
  (`{paper_key}_{idx}`) and `paper_key` fields.
- `src/kgqa/evaluation.py` — `FuzzyEvaluator.extract_answer` and `mcnemar_test`
  determine every reported metric.
- `results/summary.json` — the published-numbers artifact; edited only by
  `scripts/compare.py`, guarded by `tests/test_results_regression.py`.

**Safe-ish to change (UI/plumbing, well-covered or non-scientific):**

- `frontend/` — pure presentation; can't affect benchmark validity.
- `app/dashboard.py`, `app/chat_app.py` — read-only over results / a live wrapper.
- `backend/main.py` request/response shapes — covered by `backend/test_main.py`.
- `service.py` caching internals — behavior covered by `tests/test_service.py`.

---

## 6. Surprising / non-obvious things that will trip you up

- **Two graph databases, two LLM paths.** If you "simplify" by unifying them, you
  will either break the hosted free tier or invalidate the benchmark. This is the
  #1 gotcha. (§2, §4.7)
- **`graph_id="demo"` → Neo4j; everything else → ArangoDB.** The routing lives in
  `service.py` and is easy to miss.
- **The benchmark never runs in CI.** CI runs ruff + pytest with *fakes*; the real
  LLM/DB benchmark runs on Colab (see `notebooks/`). "Tests pass" says nothing
  about the accuracy numbers — those are guarded by the regression gate on the
  committed JSON, not re-computed.
- **`Candidate.chunk_id` always carries an Arango-style `Chunks/…` prefix**, even
  in the Neo4j path — `neo4j_graph.py` strips it with `_local_key`. Keep the
  prefix; other call sites depend on it.
- **`BaseRetriever.chat()` calls Ollama directly**, but `service.answer()` uses
  the provider chain. So the Gradio chat (`app/chat_app.py`) needs local Ollama,
  while the hosted FastAPI agent uses Gemini/Groq. Same "chat" word, different LLM
  path.
- **The reasoning model emits `<think>…</think>`.** `FuzzyEvaluator`, the Gradio
  app, and the frontend all assume this and strip it. A non-reasoning model won't
  break extraction but changes output shape.
- **`num_predict` caps generation** so a runaway reasoning chain can't stall the
  Ollama server; the extractor tolerates a truncated chain. Don't remove the cap.
- **`.env` on disk holds a real Neo4j credential.** It's gitignored (untracked),
  but it's a *live* secret — see GAPS.md before touching or sharing it.
- **`run_benchmark.py`'s Ollama auto-restart uses `pkill`**, which doesn't exist
  on Windows (it degrades to a no-op there). The benchmark is really meant for
  Linux/Colab.
