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
| `/health` | GET | liveness probe; returns `{status, demo_graph_loaded}` (whether the demo store is warm) |
| `/query` | POST | `{question, graph_id="demo", use_concepts=false}` -> `{answer, reasoning_path, sources}`. Unknown `graph_id` -> `404`. |
| `/ingest` | POST | `{dataset_id}` -> `{graph_id}`. Only preloaded dataset ids (currently `demo`) resolve; arbitrary document upload is out of scope for v1 and returns `501`. |
| `/mcp` | MCP (streamable HTTP) | Same `graphrag.answer()` as `/query`, exposed as an MCP tool -- see below. |

## Use it as an agent (MCP)

Any MCP client can use this agent as a tool by pointing at the deployed
`/mcp` URL -- no cloning, no API key. It exposes one tool,
`ask_pubmed_graphrag(question, use_concepts=False)`, defined in
[`mcp_server.py`](mcp_server.py).

**Claude Code:**
```bash
claude mcp add --transport http graphrag-pubmedqa https://graphrag-agent-api.onrender.com/mcp
```

**Claude Desktop / Cursor** (`mcpServers` config, streamable HTTP transport):
```json
{
  "mcpServers": {
    "graphrag-pubmedqa": {
      "url": "https://graphrag-agent-api.onrender.com/mcp"
    }
  }
}
```

Two things worth knowing before you rely on it:
- **Cold start.** Render's free tier sleeps after ~15 min idle; the first tool
  call after a gap can take 60-120s while the instance wakes up and loads the
  vector store. Follow-up calls are ~15-20s.
- **Unauthenticated, demo-scoped.** No auth; a per-client rate limit (10 requests/min
  on `/query`, tunable via `KGQA_RATE_LIMIT_PER_MIN`), a body-size cap, and a
  global concurrency limit guard the instance -- this is a
  portfolio demo over the same 1,000-paper labeled-split graph described
  below, not a production API.

## The demo graph: Neo4j, scoped to 1,000 papers

`graph_id="demo"` (what the chat UI and `/query`'s default use) is served by
a **Neo4j AuraDB Free** instance, not the ArangoDB the benchmark ablation
uses. It's seeded by [`scripts/ingest_neo4j.py`](../scripts/ingest_neo4j.py)
with **only the PubMedQA labeled split -- 1,000 papers, ~3,358 chunks,
~3,408 MeSH concepts** -- not the full ~62k-paper corpus RESULTS.md's
numbers were benchmarked over. It also runs without the cross-encoder
reranker (`KGQA_SKIP_RERANKER=true`) -- the unbenchmarked `graph_norr`
configuration. This is a deliberate scope limit (see RESULTS.md) so the hosted demo stays small enough for a
genuinely-free-forever graph DB tier, not a cost/reproducibility issue with
the benchmark itself, which is untouched.

Inside `graphrag.answer()`, any other `graph_id` routes to ArangoDB, same as
the benchmark pipeline -- but `/query` 404s non-`demo` ids, so that path is
only reachable by calling `graphrag.answer()` directly.

Run once before the demo can answer with real graph expansion:
```bash
export NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=...
python scripts/ingest_neo4j.py
```
Without it, a local backend still answers (falls back to encoding the labeled
split locally and skipping parent-document expansion) -- just without the
graph's accuracy lift. On Render, `KGQA_DISABLE_DATASET_FALLBACK=true` disables
that fallback, so `/query` returns a fast `502` instead.

## Deploy (Render, free tier)

[`render.yaml`](../render.yaml) at the repo root is a Render Blueprint --
connect the repo on Render and it's picked up automatically. Set the secrets
(`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GEMINI_API_KEY`) in the Render
dashboard. `ARANGO_*` vars are not used by the deployed API. The plain values in
`render.yaml` (`CORS_ORIGINS`, `KGQA_*`, ...) apply on first creation only --
see the auto-sync note below; mirror changes into the dashboard by hand.

Free tier sleeps after ~15 min idle (60-120s cold start on the next request).
[`.github/workflows/keep-warm.yml`](../.github/workflows/keep-warm.yml) is meant
to ping `/health` every 10 minutes during daytime hours (needs the `BACKEND_URL`
repo secret), but GitHub has disabled it for repository inactivity and its
schedule was never reliable -- re-enable it in the Actions tab or use an
external uptime pinger (GAPS #19).

### Fitting in 512MB: `KGQA_ENCODER=onnx`

The free tier's 512MB is not enough for the torch + sentence-transformers
stack, which OOM-killed the instance mid-`/query`. Measured locally, loading
the encoder plus the demo store costs **~700MB on the torch path vs ~290MB on
the ONNX one** -- torch was never going to fit in 512MB.
`KGQA_ENCODER=onnx` (set in `render.yaml`) swaps in
[`OnnxEncoder`](../src/kgqa/models.py) instead: the *same* model weights via
that repo's official ONNX export, run on `onnxruntime` with no torch import,
in the same 384-dim vector space -- so the chunk embeddings already in Neo4j
work untouched, no re-ingestion. `KGQA_SKIP_RERANKER=true` (also memory) and
`KGQA_DISABLE_DATASET_FALLBACK=true` (fail fast instead of crash-looping when
the graph DB is down) are set for the same tier.

To roll back, unset `KGQA_ENCODER` -- the torch path is unchanged, and
`render.yaml`'s build step still prewarms its model files for exactly that
reason. [`scripts/verify_onnx_parity.py`](../scripts/verify_onnx_parity.py)
is the pre-deploy gate: it asserts both encoders agree (cosine > 0.999) and
retrieve the same top-3 chunks in the same order. Run it before changing
anything about the encoder.

Render does **not** auto-sync `render.yaml`'s build command or plain `value:`
env vars onto an existing service -- mirror those into the dashboard by hand,
and use "Clear build cache & deploy" when the build command changes what it
downloads.
