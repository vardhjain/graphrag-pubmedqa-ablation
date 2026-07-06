# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Interactive UIs** in `app/`: a Gradio chat demo (`chat_app.py`) over the
  winning `graph` arm that cites source PubMed IDs, and a Streamlit results
  dashboard (`dashboard.py`) that visualizes the ablation, McNemar tests, and
  per-class breakdown. `requirements-app.txt`, `make chat` / `make dashboard`.
- `BaseRetriever.chat()` — conversational answer plus the retrieved source pubids.
- `scripts/compare.py` now also writes `results/summary.json` (structured metrics
  + contrasts) for the dashboard.
- One-click **Streamlit Community Cloud** deploy for the dashboard: a light
  `app/requirements.txt` (picked up before the heavy root file), a themed
  `.streamlit/config.toml`, a richer page config, and a README live-demo badge.

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

[Unreleased]: https://github.com/vardhjain/graphrag-pubmedqa-ablation/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/vardhjain/graphrag-pubmedqa-ablation/releases/tag/v1.0.0
