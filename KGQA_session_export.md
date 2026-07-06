# Session Export — Knowledge Graph QA Repo Revamp

**Date:** 2026-06-12
**Tool:** Claude Cowork session
**Repo:** https://github.com/Akash-Raghavendra/Knowledge_Graph_Question_Answering
**GitHub token:** [REDACTED — provided separately; fine-grained PAT, Contents read/write, scoped to this repo]

---

## Context

Vardh wants to work on the repo without downloading it to his PC (low storage). Chose to work in Claude's sandbox: Claude clones the repo, makes changes, pushes back via PAT. Repo is cloned at `~/work/Knowledge_Graph_Question_Answering` inside the sandbox (note: sandbox is ephemeral — re-clone in new sessions).

## Goal

Perfect the project: a fair GraphRAG vs PlainRAG comparison (identical layers, only difference = graph), plus an industry-standard GitHub repo (README, requirements, tests, CI, clean code). Desired narrative: show where GraphRAG legitimately beats PlainRAG — via fair design and a stronger graph implementation, not a rigged comparison.

## Repo contents (as audited)

- Notebooks: `GraphRAG.ipynb`, `Data_Ingestion_KG.ipynb`, `Comparison.ipynb`, `Plain_RAG/Plain_RAG.ipynb`
- Scripts: `run_graphrag.py`, `run_plainrag.py`, `run_comparison.py`, `shared_utils.py`
- Docs: `Project Report.pdf`, `Graph_RAG_PPT.pptx`
- Stack: PubMedQA dataset, all-MiniLM-L6-v2 embeddings, deepseek-r1:8b via Ollama, ArangoDB Oasis cloud (`pubmed_graph` db), FAISS for PlainRAG, cross-encoder ms-marco-MiniLM-L-6-v2 reranker (GraphRAG only)

## Audit findings — fairness flaws

1. **Confounded comparison:** GraphRAG has a cross-encoder reranker (75 candidates → top 3); PlainRAG is raw FAISS top-3. Any GraphRAG win is attributable to the reranker, not the graph.
2. **Different corpora:** PlainRAG indexes labeled + unlabeled + artificial PubMedQA splits; the graph ingests only labeled + unlabeled.
3. **Different granularity:** PlainRAG embeds whole abstracts; GraphRAG embeds per-section chunks.
4. **Label leakage:** Paper nodes store `title = row['question']` and `final_decision`; graph expansion injects `=== STUDY: {title} ===` into the prompt, so the benchmark question can appear verbatim in GraphRAG's context. PlainRAG gets no titles.
5. **Graph underused:** `Concepts` (MeSH) nodes and `MENTIONS` edges are built but never used in retrieval. Retrieval is only chunk→paper→chunks (parent-document lookup).
6. **Bugs/rigor:** `_expand_via_graph` exception fallback references out-of-scope `candidates` (NameError); benchmark uses first 100 samples (no seed, no random sampling, no significance test); unparseable LLM answers default to "maybe".

## Audit findings — repo gaps

No README, no requirements.txt, no .gitignore (.DS_Store committed), no LICENSE, no .env.example (Arango cloud host hardcoded), no tests, no CI, constants duplicated across notebooks instead of imported, print statements instead of logging, no package structure.

## Agreed plan

### Architecture (where things run)

- **Claude's sandbox:** all code work — refactor, bug fixes, tests (Ollama + ArangoDB mocked; embedding model runs on CPU for smoke tests), lint, CI config, README. Pushes to GitHub. Sandbox CANNOT reach ArangoDB cloud (port 8529 blocked, verified) and has no GPU.
- **Colab (Vardh):** benchmark execution on free T4 GPU. Connects to ArangoDB Oasis over HTTPS (already proven working). Credentials via Colab Secrets (`ARANGO_PASS`).
- **GitHub:** the hub. Branch `revamp`, review, results JSONs committed back.
- **DB:** keep ArangoDB Oasis (verify trial still active). One re-ingestion script to fix the leaky schema. Fallback if expired: docker-compose for local Arango.

### Steps

1. **Restructure (Claude):** branch `revamp`; layout: `src/kgqa/` (config.py, llm.py, evaluation.py, retrieval/{base,plain,graph}.py with shared base class), `scripts/` (ingest.py, run_benchmark.py --arm {plain,plain_rr,graph,graph_concepts}), `notebooks/` (thin Colab wrappers), `tests/`, `docs/` (PDF+PPTX moved), `.github/workflows/ci.yml`, README, requirements.txt, .gitignore, LICENSE (MIT), .env.example.
2. **Fix the science (Claude):** same corpus/chunking/embedder/prompt/LLM/top-k both arms; reranker for both; leakage stripped (no question-derived titles or final_decision in prompts); seeded random sample n=200+; McNemar's test; fix NameError; implement concept-hop expansion (chunk → paper → shared MeSH concepts → related papers) as 4th arm.
3. **Verify (Claude):** pytest green, ruff clean, mocked end-to-end dry run; push.
4. **Re-ingest + run (Vardh, Colab):** `notebooks/01_ingest.ipynb` once, then `notebooks/02_benchmark.ipynb` (all 4 arms, ~2–4 hrs).
5. **Results + polish (Claude):** comparison figures, README results section with honest ablation story, merge revamp → main.

### Key principle

4-arm ablation (PlainRAG → PlainRAG+reranker → GraphRAG parent-expansion → GraphRAG+concept-expansion) isolates exactly what the graph contributes. Honest expectation: PubMedQA is mostly single-abstract QA, so fair gains may be modest; the concept-expansion arm and multi-evidence questions are where a real gap shows.

## Status

Plan agreed; implementation not yet started. Next action: Step 1 (restructure on branch `revamp`).
