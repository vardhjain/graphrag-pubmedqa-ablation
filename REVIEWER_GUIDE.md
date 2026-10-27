# GraphRAG PubMedQA — Reviewer guide

**Three minutes, no setup:** [Interactive walkthrough](https://graphrag-pubmedqa-ablation.vercel.app/walkthrough) · [Case study](https://graphrag-pubmedqa-ablation.vercel.app/case-study) · [Benchmark](https://graphrag-pubmedqa-ablation.vercel.app/benchmark) · [Live chat](https://graphrag-pubmedqa-ablation.vercel.app/)

> The walkthrough uses clearly labeled synthetic text and runs entirely in the browser. The live chat is a separate, smaller configuration that may take minutes to wake up on free hosting.

## The question

Retrieval-augmented generation (RAG) often returns a short passage. The project asks whether following a knowledge-graph link from that passage to its parent **full paper abstract** helps biomedical yes/no/maybe question answering, after removing confounding differences between baselines.

**Historical finding:** parent-document expansion improved the reranked chunk-RAG arm from **37.0% to 59.5% accuracy**, a **+22.5 percentage-point** paired difference across 200 seeded PubMedQA questions. An additional hop to related papers reduced accuracy to 57.5% and raised average latency from 7.5 to 40.8 seconds. The graph result is modest compared with the 56.5% majority-yes baseline and needs independent confirmation. See [RESULTS.md](RESULTS.md) for statistics and caveats.

## Five-minute code review

| Time | Start here | What to look for |
| --- | --- | --- |
| 0–1 min | [Interactive walkthrough](https://graphrag-pubmedqa-ablation.vercel.app/walkthrough) | The concrete chunk → parent-paper → sibling-sections mechanism, without waiting on API startup |
| 1–2 min | [Shared retrieval](src/kgqa/retrieval/base.py) and [ArangoDB retrieval](src/kgqa/retrieval/graph.py) | Common retrieval pipeline, controlled reranking, AQL parent expansion, graceful degradation |
| 2–3 min | [Benchmark runner](scripts/run_benchmark.py) and [statistics](scripts/compare.py) | Seeded evaluation, checkpoint/restart support, per-arm alignment, exact McNemar tests |
| 3–4 min | [Backend](backend/main.py) and [MCP tool](backend/mcp_server.py) | REST and MCP sharing one answer service; validated inputs and resource limits |
| 4–5 min | [Real-world tradeoffs](frontend/app/case-study/page.tsx), [CI](.github/workflows/ci.yml), [known gaps](GAPS.md) | Memory footprint, deployment tradeoffs, limitations, tests, remaining scientific work |

## The engineering decisions

- **Fix the evaluation before tuning the model.** The first version contained unfair differences: reranking, corpus selection, and answer leakage. The current benchmark implements incremental ablation arms with a shared embedding/search spine.
- **Keep experimental and hosted systems distinct.** The benchmark ran on ArangoDB + a local Ollama model over approximately 62,000 papers; the free hosted demonstration uses Neo4j AuraDB, cloud LLM synthesis, no reranker, and only the 1,000-paper labeled split. **The published percentages do not describe hosted-demo performance.**
- **Build for the resource budget.** The hosted deployment uses an ONNX encoder and skips the heavyweight cross-encoder to fit the free-tier memory limit; deployment recovery, caching, rate limiting, and input validation are tracked in code.
- **Make tradeoffs visible.** The repository documents the historical edge-duplication exposure, uncommitted raw per-question predictions, untested flat parent-document baseline, and statistical limitations rather than claiming a proven graph advantage.

## How to explore without spending money

The [static walkthrough](https://graphrag-pubmedqa-ablation.vercel.app/walkthrough), [benchmark page](https://graphrag-pubmedqa-ablation.vercel.app/benchmark), and [case study](https://graphrag-pubmedqa-ablation.vercel.app/case-study) work without an active backend. The synthetic walkthrough does **not** generate an answer or simulate a measured prediction.

To review the Python tests without GPU or graph databases:

```bash
pip install -r requirements-dev.txt
make test
```

The optional [real database integration tests](tests/integration/test_graph_db_contract.py) can run on local Docker services once PR #4 is merged; no paid compute is required. These currently expose known graph-query issues as expected failures, and do not substitute for a benchmark rerun.

## Research limitations and next experiments

1. **Clean re-ingestion and rerun:** the published graph likely contained duplicate `HAS_CONTEXT` edges. The current checked-in benchmark summary is historical; do not present it as a validated clean rerun.
2. **Save per-question outputs:** enable independent verification of paired gains/losses and error analysis.
3. **Non-graph parent-document baseline:** compare against a flat lookup with the same parent abstracts and token budget. The current measured improvement could be due to **more context**, not graph structure by itself.
4. **Uncertainty and class imbalance:** compare against the 56.5% majority-yes baseline and report class metrics and confidence intervals.
5. **Hosted-demo gap:** benchmark the deployed no-reranker configuration separately before attaching performance numbers to the web demo.

See [GAPS.md](GAPS.md) for details. Project contributions are shared with [Akash Raghavendra](https://github.com/Akash-Raghavendra); this guide describes the repository rather than assigning unverified individual credit.

## Interview discussion prompts

- Why does a paired McNemar test better reflect differences between these arms than comparing two independent accuracy intervals?
- What would convince you the **graph** is beneficial, rather than merely the full abstract?
- Why can a single missing or duplicated edge change a paper-reconstruction experiment?
- What would you keep or remove to deploy this system reliably under a strict memory limit?

**One-sentence summary:** A reproducibility-focused GraphRAG/RAG ablation on PubMedQA, with real-world deployment constraints made visible and a benchmark that still needs a clean replication.
