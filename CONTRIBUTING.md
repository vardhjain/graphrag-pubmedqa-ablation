# Contributing

Thanks for your interest in this project! It's a research codebase for a *fair*
GraphRAG vs PlainRAG comparison on PubMedQA, so contributions that improve
rigor, reproducibility, or clarity are especially welcome.

## Development setup

```bash
git clone https://github.com/vardhjain/graphrag-pubmedqa-ablation.git
cd Knowledge_Graph_Question_Answering
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install-dev          # or: pip install -r requirements-dev.txt
pre-commit install        # optional: run ruff automatically on commit
```

The unit tests inject fakes for the encoder, reranker, and ArangoDB, so you can
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
| `src/kgqa/config.py` | **all** shared constants + env overrides |
| `src/kgqa/retrieval/` | the four retrieval arms (`base`, `plain`, `graph`) |
| `scripts/` | `ingest.py`, `run_benchmark.py`, `compare.py` |
| `notebooks/` | thin Colab wrappers (kept output-free) |
| `tests/` | pytest suite (CPU-only via fakes) |

> **Why no `configs/` directory?** Configuration is centralized in
> `src/kgqa/config.py` as a typed dataclass with environment-variable overrides
> (and an `.env.example` template). For this project that's safer and less
> error-prone than scattering YAML/JSON config files; please keep new knobs there.

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
