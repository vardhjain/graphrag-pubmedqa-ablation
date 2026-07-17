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
import threading
from contextlib import AsyncExitStack, asynccontextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend.mcp_server import mcp  # noqa: E402

logger = logging.getLogger(__name__)

# FastMCP's default streamable_http_path is "/mcp", so the returned Starlette
# app already has a Route("/mcp", ...) -- its routes are merged directly into
# this app's router (not app.mount("/mcp", ...)) so the final path is exactly
# "/mcp" with no prefix concatenation. Mounting would make the real path
# "/mcp/" (mount prefix + the sub-app's own "/"), forcing every bare "/mcp"
# request through a 307 redirect first; that redirect broke at least one real
# MCP client's streaming GET (it hung indefinitely rather than following it),
# even though curl followed it fine -- so avoid the redirect altogether.
_mcp_app = mcp.streamable_http_app()


def _warm_up_demo_graph() -> None:
    """Load the encoder + demo graph's store/retriever once, proactively.

    /health returns a static dict and touches nothing on the answer path, so
    the keep-warm cron's periodic pings (.github/workflows/keep-warm.yml)
    only ever kept the *process* alive -- after any restart (a new deploy,
    or the free tier spinning down and back up), the first real /query still
    paid the full cold-model-load cost inline. This runs that load eagerly
    at startup instead, in parallel with whatever request/routing latency
    gets a user to their first real question. Never lets a failure here take
    the app down: the same work just happens lazily on first /query instead,
    exactly as it did before this existed.
    """
    from kgqa.service import _get_retriever

    try:
        _get_retriever("demo")
        logger.info("Startup warm-up: demo graph loaded.")
    except Exception:
        logger.exception("Startup warm-up failed; will load lazily on first /query instead.")


def _maybe_warm_up_in_background() -> None:
    """KGQA_WARM_ON_STARTUP=true (set in render.yaml) opts into this --
    default off so importing this module never triggers a real model/DB load
    in local dev, tests, or CI."""
    if os.environ.get("KGQA_WARM_ON_STARTUP", "").lower() != "true":
        return
    threading.Thread(target=_warm_up_demo_graph, daemon=True).start()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # The MCP app's own lifespan starts its session manager; merging routes
    # (rather than app.mount()) means it's no longer run automatically by a
    # parent Mount either, so it's entered explicitly here.
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(_mcp_app.router.lifespan_context(_mcp_app))
        _maybe_warm_up_in_background()
        yield


app = FastAPI(
    title="GraphRAG hosted agent",
    description="GraphRAG vs PlainRAG on PubMedQA -- see /docs and RESULTS.md",
    version="1.0.0",
    lifespan=_lifespan,
)
app.router.routes.extend(_mcp_app.routes)

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
# graph_ids /query is allowed to answer against -- the values of the mapping
# above, i.e. only what /ingest can actually hand back. Without this, any
# unrecognized graph_id reaches service._get_store() and falls through to
# ChunkStore.from_dataset(): a full local re-encode of the corpus, cached
# under a filename built directly from the caller-supplied string. On an
# unauthenticated public endpoint that's both a cheap way to force repeated
# expensive work and a path-traversal-shaped footgun, not just wasted CPU.
_KNOWN_GRAPH_IDS = set(_KNOWN_DATASETS.values())


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
    # A cheap readiness signal alongside liveness: reads the already-built
    # cache, never triggers a load. The first real observability into
    # whether the demo graph is actually warm, not just whether the process
    # is up.
    from kgqa import service

    return {"status": "ok", "demo_graph_loaded": "demo" in service._STORE_CACHE}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> dict:
    from graphrag import answer

    if req.graph_id not in _KNOWN_GRAPH_IDS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown graph_id {req.graph_id!r}. Known graphs: {sorted(_KNOWN_GRAPH_IDS)}.",
        )

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
