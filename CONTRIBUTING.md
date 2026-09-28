# Contributing

Thanks for your interest in this project! It's a research codebase for a *fair*
GraphRAG vs PlainRAG comparison on PubMedQA, so contributions that improve
rigor, reproducibility, or clarity are especially welcome.

## Development setup

```bash
git clone https://github.com/vardhjain/graphrag-pubmedqa-ablation.git
cd graphrag-pubmedqa-ablation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install-dev          # or: pip install -r requirements-dev.txt
pre-commit install        # optional: run ruff automatically on commit
```

The unit tests inject fakes for the encoder, reranker, ArangoDB, and Neo4j, so you can
run the whole suite on CPU with **no GPU, Ollama, or database** required:

```bash
make test     # pytest
make lint     # ruff
```

See the [Makefile](Makefile) (`make help`) for all shortcuts.

## Where things live

| Path | What |
| --- | --- |
| `src/kgqa/` | the importable package (single source of truth) |
| `src/kgqa/config.py` | shared benchmark constants + env overrides |
| `src/kgqa/retrieval/` | retrieval arms: `base`, `plain`, `graph` (ArangoDB), `neo4j_graph` (hosted demo) |
| `src/kgqa/service.py`, `providers.py` | hosted-agent entry point (`graphrag.answer`) + LLM provider chain |
| `scripts/` | `ingest.py`, `ingest_neo4j.py`, `run_benchmark.py`, `compare.py`, `verify_onnx_parity.py` |
| `backend/` | FastAPI hosted agent + MCP endpoint (tests in `backend/test_*.py`) |
| `frontend/` | Next.js chat UI + `/benchmark` page (`npm run lint`, `npx tsc --noEmit`) |
| `app/` | Gradio chat + Streamlit results dashboard |
| `notebooks/` | thin Colab wrappers (kept output-free) |
| `tests/` | pytest suite (CPU-only via fakes) |

> **Why no `configs/` directory?** Every knob that could affect the benchmark
> (models, top-k, seed, sample size, DB configs) lives in `src/kgqa/config.py`
> with environment-variable overrides — please keep new benchmark knobs there.
> Hosted-only settings (`KGQA_*`, `GEMINI_*`, `CORS_ORIGINS`) are read next to
> the code that uses them (`service.py`, `models.py`, `providers.py`,
> `backend/main.py`). All are listed in `.env.example`.

## Ground rules for changes

This repo's whole point is a **fair** comparison. Before changing retrieval or
evaluation, please make sure:

- Anything that could confound the arms (embedder, reranker, prompt, LLM, top-k,
  seed, sample size) stays in `config.py` and identical across arms.
- No benchmark answer or question text can leak into a retrieved context
  (there's a regression test for this — keep it green).
- New behavior has a test; `make test` and `make lint` both pass.

## Pull requests

1. Branch from `main`, make focused commits.
2. Run `make test && make lint`.
3. Open a PR using the template; describe what changed and why, and update
   `CHANGELOG.md` under "Unreleased".

## Commit messages

Short imperative subject line, a blank line, then a body explaining the *why*
when it isn't obvious.
