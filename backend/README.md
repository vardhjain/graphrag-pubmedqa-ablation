# Backend

FastAPI service exposing the hosted GraphRAG agent (`graphrag.answer`, see
[`../src/graphrag`](../src/graphrag)).

## Local dev

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
# -> http://localhost:8000/docs
```

## Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | liveness probe; also pinged by the keep-warm cron |
| `/query` | POST | `{question, graph_id="demo", use_concepts=false}` -> `{answer, reasoning_path, sources}` |
| `/ingest` | POST | `{dataset_id}` -> `{graph_id}`. Only preloaded dataset ids (currently `demo`) resolve; arbitrary document upload is out of scope for v1 (see the execution plan's scope warning) and returns `501`. |

## The demo graph: Neo4j, scoped to 1,000 papers

`graph_id="demo"` (what the chat UI and `/query`'s default use) is served by
a **Neo4j AuraDB Free** instance, not the ArangoDB the benchmark ablation
uses. It's seeded by [`scripts/ingest_neo4j.py`](../scripts/ingest_neo4j.py)
with **only the PubMedQA labeled split -- 1,000 papers, ~3,358 chunks,
~3,408 MeSH concepts** -- not the full ~62k-paper corpus RESULTS.md's
numbers were benchmarked over. This is a deliberate scope limit (see
RESULTS.md / the execution plan) so the hosted demo stays small enough for a
genuinely-free-forever graph DB tier, not a cost/reproducibility issue with
the benchmark itself, which is untouched.

Any other `graph_id` (e.g. a real dataset a future `/ingest` builds) still
uses ArangoDB, same as the benchmark pipeline.

Run once before the demo can answer with real graph expansion:
```bash
export NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=...
python scripts/ingest_neo4j.py
```
Without it, the backend still answers (falls back to encoding the labeled
split locally and skipping parent-document expansion) -- just without the
graph's accuracy lift.

## Deploy (Render, free tier)

[`render.yaml`](../render.yaml) at the repo root is a Render Blueprint --
connect the repo on Render and it's picked up automatically. Set the secret
env vars (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GEMINI_API_KEY`,
`CORS_ORIGINS`) in the Render dashboard; `ARANGO_*` vars
are only needed if you also want `/ingest` to build real ArangoDB-backed
datasets. Nothing else to configure.

Free tier sleeps after ~15 min idle (30-50s cold start on the next request).
[`.github/workflows/keep-warm.yml`](../.github/workflows/keep-warm.yml) pings
`/health` every 10 minutes during daytime hours to keep it warm without
burning the whole free-hours budget. Set the `BACKEND_URL` repo secret to the
deployed Render URL once it exists.
