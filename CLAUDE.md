# CLAUDE.md — operational guide for this repo

Read this first. For architecture narrative see **[PROJECT.md](PROJECT.md)**; for
known weaknesses see **[GAPS.md](GAPS.md)**.

**One-line summary:** a fair 4-arm GraphRAG-vs-PlainRAG ablation on PubMedQA
(`plain → plain_rr → graph → graph_concepts`), plus a hosted chat agent
(FastAPI + Next.js) that serves the winning `graph` arm. Everything held constant
across arms except retrieval strategy; the headline result is **+22.5 pp accuracy
from parent-document expansion, McNemar p<0.0001**.

---

## The one thing to internalize: two backends, two LLM paths

- **Benchmark / research half** → **ArangoDB** + **Ollama** (`deepseek-r1:8b`),
  called directly. Produces every number in RESULTS.md. Entry points:
  `scripts/ingest.py`, `scripts/run_benchmark.py`, `scripts/compare.py`.
- **Hosted-agent half** → **Neo4j AuraDB** (demo graph, 1,000-paper labeled split
  only) + a **cloud provider chain** (Gemini/Groq → Ollama fallback). Entry
  point: `graphrag.answer()` in `src/kgqa/service.py`, exposed via FastAPI.
- Routing: `graph_id="demo"` → Neo4j; **any other** `graph_id` → ArangoDB.

**Never unify these to "simplify."** Rewriting the ArangoDB benchmark pipeline
would require a fresh GPU benchmark run to re-validate RESULTS.md, which the dev
environment cannot do. The split is deliberate (see PROJECT.md §2).

---

## Commands

```bash
# Setup
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + pytest, ruff, pre-commit
pip install -r requirements-app.txt      # + gradio, streamlit (UIs)
cp .env.example .env                      # then set ARANGO_PASS / NEO4J_PASSWORD etc.

# Quality (what CI runs — CPU only, no DB/LLM/GPU needed)
make test        # pytest  (50 tests; testpaths = tests/ + backend/)
make lint        # ruff check src scripts tests app
make format      # ruff check --fix ...
pytest tests/test_results_regression.py -v   # the eval regression gate specifically

# Benchmark (needs ARANGO_PASS + a running Ollama; really Linux/Colab)
python scripts/ingest.py                  # build the ArangoDB graph ONCE
make benchmark                            # all 4 arms, n=200
python scripts/run_benchmark.py --arm graph --n 200   # one arm
python scripts/compare.py                 # → results/summary.{json,md} + ablation.png

# UIs
make dashboard   # streamlit run app/dashboard.py  (reads results/, no DB/LLM)
make chat        # python app/chat_app.py  (LIVE: needs ArangoDB + Ollama)

# Hosted agent
uvicorn backend.main:app --reload         # FastAPI at :8000, /docs for OpenAPI
python scripts/ingest_neo4j.py            # seed the demo Neo4j graph ONCE

# Frontend
cd frontend && npm install && npm run dev # Next.js at :3000 (talks to :8000)
npm run build                             # static-generates /benchmark from results/summary.json
```

There is **no** `make deploy`. Deploys are: backend → Render (`render.yaml`
blueprint), frontend → Vercel, dashboard → Streamlit Community Cloud. See
`backend/README.md`, `app/README.md`.

---

## Conventions this codebase actually follows

- **All shared constants live in `src/kgqa/config.py`.** Models, top-k, seed,
  sample size, DB configs. Never hardcode these elsewhere — importing from config
  is what guarantees the arms stay comparable. Changing a value here can
  invalidate RESULTS.md.
- **Heavy imports are lazy / local to functions** (`sentence_transformers`,
  `datasets`, `arango`, `neo4j`, `matplotlib`). This is why CI needs only numpy +
  sklearn + scipy + requests + fastapi. Keep it that way: don't add a top-level
  `import torch` to a module that tests import.
- **Config is read from the environment** (or a local `.env` via python-dotenv, or
  Colab Secrets). Nothing is hardcoded. Password getters raise a helpful
  `OSError` if unset (`ArangoConfig.require_password`).
- **Retrieval is a template-method pattern:** `BaseRetriever` owns
  encode→rerank→select→answer; subclasses only implement `_build_context`. Add a
  new arm by subclassing, not by branching inside the base.
- **Graph context is leakage-free by contract:** generic `=== STUDY n ===`
  labels, abstracts only, no titles, no `final_decision`. A test enforces it.
- **Chunk ids are `Chunks/{paper_key}_{idx}`** everywhere (Arango-style prefix),
  even on the Neo4j path — `neo4j_graph._local_key` strips the prefix at query
  time. Keep the prefix on `Candidate.chunk_id` and `ChunkStore.ids`.
- **Error handling in the service degrades, never crashes:** graph down → raw
  chunks; no cloud key → Ollama; no reranker → raw top-k. Preserve this for
  anything on the hosted path.
- **Naming:** snake_case Python, `test_*.py` mirroring module names, arms named
  by the exact strings `plain`/`plain_rr`/`graph`/`graph_concepts`. Frontend:
  camelCase TS, PascalCase components, Tailwind utility classes inline.
- **ruff** with `E,F,I,W,UP,B` selected, line-length 100, `E501` ignored;
  `E402` ignored in `scripts/`, `app/`, `backend/` (intentional `sys.path` insert
  before imports). Match this — don't "fix" the sys.path ordering.

---

## Gotchas (looks-like-X-but-isn't)

- **"Tests pass" ≠ "the numbers are right."** CI never runs the LLM benchmark
  (no GPU/DB). It runs unit tests with fakes + a regression gate on the *committed*
  `results/summary.json`. The real benchmark runs on Colab (`notebooks/`).
- **`results/summary.json` is generated, not hand-edited.** It's written by
  `scripts/compare.py` and guarded by `tests/test_results_regression.py`. RESULTS.md,
  the README table, the Streamlit dashboard, and the Next.js `/benchmark` page all
  read from it. Don't edit numbers by hand — regenerate.
- **`BaseRetriever.chat()` uses Ollama directly; `service.answer()` uses the
  provider chain.** So the Gradio app needs local Ollama, the FastAPI agent uses
  Gemini/Groq. Same word "chat," different LLM path.
- **The LLM emits `<think>…</think>`.** `FuzzyEvaluator.extract_answer`, the Gradio
  app, and the frontend all strip it. Don't remove the stripping.
- **`num_predict` caps generation on purpose** (config.py) so a runaway reasoning
  chain can't stall Ollama; the extractor tolerates truncation. Don't uncap it.
- **`decompose`/`extract` provider tasks are defined but never called** — only
  `synthesize` is wired. Don't assume a decomposition pipeline exists (GAPS #4).
- **`run_benchmark.py`'s Ollama auto-restart uses `pkill`** — a no-op on Windows.
  Benchmark is effectively Linux/Colab-only.
- **The hosted tier and the benchmark use different encoder *runtimes*, same
  weights.** Render sets `KGQA_ENCODER=onnx` so `service._shared_encoder()`
  loads `models.OnnxEncoder` (onnxruntime, no torch — the torch stack alone
  OOMs 512MB); the benchmark calls `models.load_encoder` (sentence-transformers)
  directly and never reads that gate. Same `all-MiniLM-L6-v2` weights and 384-dim
  space either way, so vectors stay interchangeable — `scripts/verify_onnx_parity.py`
  is the gate that proves it. Don't "unify" these either.
- **The old repo still exists and still clones** — `vardhjain/Knowledge_Graph_Question_Answering`
  is live but frozen ~25 commits back; it does *not* redirect to the canonical
  `vardhjain/graphrag-pubmedqa-ablation`. So a stale clone URL fails **silently**
  (you get old code that runs fine) rather than 404-ing. This already bit both
  Colab notebooks — see GAPS #6. Never copy a clone URL from memory; check
  `git remote -v`. The local dir is `graphrag-pubmedqa-ablation`, so any
  `clone`/`cd`/`rm -rf` trio must agree on that name.

---

## Rules — do not change without care

- **`src/kgqa/config.py`** — load-bearing for benchmark fairness. Changing a
  constant silently affects all arms and can invalidate published numbers.
- **`scripts/ingest.py` schema** (no title / no `final_decision`; chunk keys
  `{paper_key}_{idx}`) — regressing this reintroduces label leakage. The ingestion
  and retrieval sides must agree on key format.
- **`retrieval/graph.py` `_PARENT_AQL` + `format_studies`** — the +22.5 pp
  mechanism and the anti-leakage formatting. Both graph backends must stay
  behavior-compatible.
- **`results/summary.json`** — only `scripts/compare.py` writes it. Never hand-edit.
- **`.env`** — contains a **real live Neo4j credential** (gitignored, untracked).
  Treat as a secret; do not print, commit, or paste it. It should be rotated
  (GAPS #1).

**Generated / not-source (don't edit by hand):** `results/summary.json`,
`results/summary.md`, `results/ablation.png` (regenerate via `compare.py`);
`frontend/.next/`, `frontend/tsconfig.tsbuildinfo`, `frontend/next-env.d.ts`;
`.coverage`, `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`.

---

## Where to look

- Architecture, data flow, design rationale, critical paths → **[PROJECT.md](PROJECT.md)**
- Known bugs, tech debt, test gaps, security items (severity-ordered) → **[GAPS.md](GAPS.md)**
- Canonical results + honest write-up → **[RESULTS.md](RESULTS.md)**
- Hosted-agent / Neo4j demo specifics → **[backend/README.md](backend/README.md)**
- The original fairness audit that motivated the revamp → **[KGQA_session_export.md](KGQA_session_export.md)**
