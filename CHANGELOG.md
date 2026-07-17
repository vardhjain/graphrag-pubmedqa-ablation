# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] — 2026-07-16

The hosted-agent release: the winning `graph` arm becomes a live, deployed chat
agent — reachable from a browser, from any MCP client, and (as of this
release) from a genuinely free-forever host.

### Added
- **Interactive UIs** in `app/`: a Gradio chat demo (`chat_app.py`) over the
  winning `graph` arm that cites source PubMed IDs, and a Streamlit results
  dashboard (`dashboard.py`) that visualizes the ablation, McNemar tests, and
  per-class breakdown. `requirements-app.txt`, `make chat` / `make dashboard`,
  one-click **Streamlit Community Cloud** deploy.
- **Hosted chat agent** — [graphrag-pubmedqa-ablation.vercel.app](https://graphrag-pubmedqa-ablation.vercel.app):
  a FastAPI backend (`backend/`) on Render and a Next.js frontend
  (`frontend/`) on Vercel, built behind a stable `graphrag.answer(question,
  graph_id) -> {answer, reasoning_path, sources}` service boundary
  (`src/graphrag/`, `src/kgqa/service.py`) so the backend never has to know
  about `ChunkStore`, retrievers, or which graph DB backs a given `graph_id`.
  The frontend renders the graph traversal behind each answer as an
  interactive reasoning-path visualization (react-flow), plus a
  `/benchmark` page reading `results/summary.json` directly and a
  `/case-study` page telling the confounded-original → fair-ablation story.
- **MCP tool endpoint** (`/mcp`, `backend/mcp_server.py`) — any MCP client
  (Claude Code, Claude Desktop, Cursor) can use the hosted agent as a tool by
  adding one URL, no cloning or API key required. See
  [backend/README.md](backend/README.md) for the one-line setup.
- **Neo4j AuraDB demo graph** (`src/kgqa/retrieval/neo4j_graph.py`,
  `scripts/ingest_neo4j.py`) — a second, parallel graph backend scoped to the
  PubMedQA labeled split (1,000 papers) so the hosted demo can run on a
  genuinely free-forever tier without touching the benchmarked ArangoDB
  pipeline or its numbers. `graph_id="demo"` routes here; any other
  `graph_id` still uses ArangoDB, unchanged.
- **Torch-free ONNX encoder** (`KGQA_ENCODER=onnx`, `src/kgqa/models.py`) for
  memory-constrained hosting tiers — same `all-MiniLM-L6-v2` weights via that
  model's official ONNX export, no torch import, same vector space as the
  existing encoder (`scripts/verify_onnx_parity.py` is the parity gate).
- Multi-provider LLM chain (`src/kgqa/providers.py`) — Gemini Flash for the
  hosted agent with an Ollama fallback for local dev, with retry/backoff on
  transient 429/5xx responses.
- GitHub Pages project site, coverage reporting via Codecov (82%), and a
  `keep-warm` GitHub Actions cron that limits the hosted backend's free-tier
  cold starts.

### Fixed
Every one of these was caught by actually exercising the deployed agent, not
by inspection — the running system kept surfacing bugs that only show up
under real load, real hosts, or a real network:
- **OOM crash on Render's free tier.** The torch + sentence-transformers
  stack alone measured ~670MB resident against a 512MB cap — it could never
  have fit. Fixed by the ONNX encoder above (~290MB with the full demo store
  loaded). Two contributing bugs fixed on the way: a downed graph DB used to
  silently fall back to a slow local re-encode that itself crash-looped the
  same tier (`KGQA_DISABLE_DATASET_FALLBACK` now fails fast instead), and
  every cold start used to re-download the encoder from HuggingFace over
  ~20 sequential requests (now prewarmed at build time, served offline).
- A single transient `Gemini 503`/`429` used to 502 the whole demo, since the
  configured Ollama fallback doesn't exist on Render — `call_gemini` now
  retries transient errors with backoff and reports `finishReason` instead of
  throwing an opaque `IndexError` on an empty response.
- The MCP tool call ran synchronously on FastMCP's event loop, wedging
  `/health` and every concurrent request for the full duration of a slow
  `answer()` call — now runs the sync call in a thread. A related fix
  merges `/mcp`'s routes directly into the app instead of mounting behind a
  redirect, which was hanging at least one real MCP client's streaming GET.
- A race in the lazy encoder/reranker/graph-connection singletons let two
  concurrent cold-start requests both load a torch model at once — doubling
  memory right when it was tightest. Fixed with double-checked locking.
- A process killed mid-write while writing the vector pickle cache left a
  truncated file that then wedged every subsequent request with
  `EOFError: Ran out of input` — cache reads now tolerate a corrupt file and
  writes are atomic.
- **Stale repo references**: both Colab notebooks, `CITATION.cff`, and
  `CONTRIBUTING.md` still cloned/cited the pre-migration repo slug. The old
  repo still exists and still clones (it doesn't 404 or redirect), so this
  failed silently — a stranger following the reproduction path would
  benchmark 25-commits-stale code without any error telling them so.
- GAPS.md audit cleanup: `/query`'s error response no longer echoes the raw
  exception text; dead `decompose`/`extract` provider scaffolding removed;
  `.env.example`'s Gemini model pin updated past a deprecated alias.

### Changed
- README restructured around the hosted demo (live-demo badge/links, a
  "Hosted agent" section, screenshots) alongside the original ablation
  story.
- `scripts/compare.py` now also writes `results/summary.json` (structured
  metrics + contrasts) for the dashboard and the frontend's `/benchmark` page.

## [1.0.0] — 2026-06-12

The "fair comparison" revamp: turned a confounded notebook demo into a
controlled, reproducible 4-arm ablation with an industry-standard repo layout.

### Added
- Importable `src/kgqa/` package: `config`, `prompts`, `llm`, `data`,
  `evaluation`, `models`, and a `retrieval/` sub-package (`base`, `plain`, `graph`).
- Four retrieval arms isolating each component:
  `plain → plain_rr → graph → graph_concepts`.
- A shared `ChunkStore` so every arm searches an identical corpus.
- MeSH concept-hop expansion (`graph_concepts`) — the previously unused
  `Concepts`/`MENTIONS` graph is now exercised.
- Seeded random sampling and a paired **McNemar** significance test.
- `scripts/`: `ingest.py` (leakage-free graph build), `run_benchmark.py`
  (`--arm`, retry + Ollama auto-restart + checkpointing), `compare.py`.
- Test suite (CPU-only via fakes), GitHub Actions CI, `ruff` + `pre-commit`.
- Docs and meta: README with results, `CONTRIBUTING`, `CODE_OF_CONDUCT`,
  `SECURITY`, `CITATION.cff`, issue/PR templates, `Makefile`, architecture diagram.
- Benchmark results (n=200) and ablation figure under `results/`.

### Fixed
- **Label leakage:** ingestion no longer stores a question-derived `title` or
  `final_decision`; graph contexts use generic `=== STUDY n ===` labels, so the
  benchmark question/answer can never appear in a retrieved context.
- **Confounded comparison:** the cross-encoder reranker is now its own arm
  instead of a hidden advantage for GraphRAG.
- **Inconsistent corpus/chunking** across arms — now identical.
- `NameError` in the graph-expansion fallback path.

### Changed
- Generation is bounded (`num_predict`) and the model kept resident
  (`keep_alive`); `LLM_NUM_CTX` / `LLM_NUM_PREDICT` are environment-tunable.
- Removed the dead `faiss` dependency (PlainRAG uses the shared numpy-cosine store).

[Unreleased]: https://github.com/vardhjain/graphrag-pubmedqa-ablation/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/vardhjain/graphrag-pubmedqa-ablation/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/vardhjain/graphrag-pubmedqa-ablation/releases/tag/v1.0.0
