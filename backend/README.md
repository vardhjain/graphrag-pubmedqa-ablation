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

## Deploy (Render, free tier)

[`render.yaml`](../render.yaml) at the repo root is a Render Blueprint --
connect the repo on Render and it's picked up automatically. Set the secret
env vars (`ARANGO_HOST`, `ARANGO_PASS`, `GROQ_API_KEY`, `GEMINI_API_KEY`,
`CORS_ORIGINS`) in the Render dashboard; nothing else to configure.

Free tier sleeps after ~15 min idle (30-50s cold start on the next request).
[`.github/workflows/keep-warm.yml`](../.github/workflows/keep-warm.yml) pings
`/health` every 10 minutes during daytime hours to keep it warm without
burning the whole free-hours budget. Set the `BACKEND_URL` repo secret to the
deployed Render URL once it exists.
