"""FastAPI backend for the hosted GraphRAG agent.

    uvicorn backend.main:app --reload          # local dev, http://localhost:8000
    uvicorn backend.main:app --host 0.0.0.0 --port $PORT   # Render start command

Endpoints:
    GET  /health         -- liveness/readiness probe, also used to keep the
                            Render free-tier instance warm (see
                            .github/workflows/keep-warm.yml).
    POST /query          -- ask a question against a graph.
    POST /ingest         -- resolve a preloaded dataset id to a graph_id.
                            Arbitrary PDF/document upload is intentionally out
                            of scope for v1 (see the execution plan's scope
                            warning: ingestion on messy real-world input is
                            where this balloons) and returns 501.
    /mcp                 -- MCP (Model Context Protocol) endpoint, streamable
                            HTTP transport. Lets any MCP client (Claude Code,
                            Claude Desktop, Cursor, ...) use this agent as a
                            tool by adding this one URL. See mcp_server.py.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend.mcp_server import mcp  # noqa: E402

logger = logging.getLogger(__name__)

_mcp_app = mcp.streamable_http_app()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # The mounted MCP sub-app has its own lifespan (starts its session
    # manager); Starlette does not run a mounted app's lifespan automatically,
    # so it has to be entered explicitly alongside the parent app's.
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(_mcp_app.router.lifespan_context(_mcp_app))
        yield


app = FastAPI(
    title="GraphRAG hosted agent",
    description="GraphRAG vs PlainRAG on PubMedQA -- see /docs and RESULTS.md",
    version="1.0.0",
    lifespan=_lifespan,
)
app.mount("/mcp", _mcp_app)

# Frontend runs on a different origin (Vercel); allow it in explicitly via env
# so this isn't wide open by default in production.
_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Dataset ids /ingest is allowed to resolve without a live upload pipeline.
_KNOWN_DATASETS = {"demo": "demo"}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    graph_id: str = "demo"
    use_concepts: bool = False


class QueryResponse(BaseModel):
    answer: str
    reasoning_path: list[dict]
    sources: list[str]


class IngestRequest(BaseModel):
    dataset_id: str


class IngestResponse(BaseModel):
    graph_id: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> dict:
    from graphrag import answer

    try:
        return answer(req.question, graph_id=req.graph_id, use_concepts=req.use_concepts)
    except Exception as exc:
        logger.exception("Answering failed for graph_id=%r", req.graph_id)
        raise HTTPException(status_code=502, detail="Answering failed. Please try again.") from exc


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> dict:
    graph_id = _KNOWN_DATASETS.get(req.dataset_id)
    if graph_id is None:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Unknown dataset_id {req.dataset_id!r}. Only preloaded datasets "
                f"are supported: {sorted(_KNOWN_DATASETS)}. Arbitrary document "
                "upload is not implemented in v1."
            ),
        )
    return {"graph_id": graph_id}
