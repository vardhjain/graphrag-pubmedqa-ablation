"""Tests for the MCP tool surface mounted at /mcp -- graphrag.answer is
monkeypatched, no live LLM/DB calls. Run with: pytest backend/test_mcp.py

The client-side round-trip test drives a real MCP ClientSession over the
mounted ASGI app (via httpx.ASGITransport, no live server needed) so a broken
mount/lifespan wiring fails here instead of only in production.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from backend.main import app
from backend.mcp_server import ask_pubmed_graphrag


@pytest.mark.anyio
async def test_tool_calls_graphrag_answer_directly(monkeypatch):
    """Unit-level: the tool function is a thin pass-through to graphrag.answer."""
    import graphrag

    captured = {}

    def fake_answer(question, graph_id="demo", use_concepts=False):
        captured["args"] = (question, graph_id, use_concepts)
        return {"answer": "yes", "reasoning_path": [], "sources": ["123"]}

    monkeypatch.setattr(graphrag, "answer", fake_answer)

    out = await ask_pubmed_graphrag("does aspirin help?", use_concepts=True)

    assert out == {"answer": "yes", "reasoning_path": [], "sources": ["123"]}
    assert captured["args"] == ("does aspirin help?", "demo", True)


@pytest.mark.anyio
async def test_tool_does_not_block_the_event_loop(monkeypatch):
    """Regression test for a real bug caught live: graphrag.answer() is a
    blocking call (encoder load, network I/O). FastMCP invokes sync tool
    functions inline on the event loop with no thread offload (unlike
    FastAPI's route handling), so a naive `def` tool wedges the whole process
    -- including unrelated /health -- for the call's full duration. The tool
    must run graphrag.answer() in a worker thread so the event loop stays free.
    """
    import graphrag

    monkeypatch.setattr(
        graphrag,
        "answer",
        lambda question, graph_id="demo", use_concepts=False: (
            time.sleep(0.3) or {"answer": "slow", "reasoning_path": [], "sources": []}
        ),
    )

    ticks = 0

    async def tick_counter():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    counter_task = asyncio.create_task(tick_counter())
    try:
        result = await ask_pubmed_graphrag("does aspirin help?")
    finally:
        counter_task.cancel()

    assert result["answer"] == "slow"
    # If the tool blocked the loop for the full 0.3s sleep, the counter task
    # would never have gotten a chance to run in between.
    assert ticks >= 5


@pytest.mark.anyio
async def test_mcp_round_trip_over_asgi(monkeypatch):
    """Protocol-level: initialize a real MCP session, list tools, call one --
    all over the mounted ASGI app, catching a broken mount/lifespan wire-up.
    """
    import graphrag

    monkeypatch.setattr(
        graphrag,
        "answer",
        lambda question, graph_id="demo", use_concepts=False: {
            "answer": "yes, per the retrieved studies",
            "reasoning_path": [{"kind": "seed_chunk", "node_id": "Chunks/1_0", "label": "x"}],
            "sources": ["1"],
        },
    )

    def http_client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",  # mcp's DNS-rebinding guard allows this Host
            headers=headers,
            timeout=timeout or 30,
            auth=auth,
        )

    # ASGITransport doesn't run ASGI lifespan events on its own, so the MCP
    # session manager our app.lifespan starts (backend/main.py's _lifespan)
    # never gets going unless we enter the app's lifespan context ourselves --
    # this is the exact wiring a broken mount would fail at in production.
    async with app.router.lifespan_context(app):
        async with streamablehttp_client(
            "http://localhost/mcp", httpx_client_factory=http_client_factory
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                assert "ask_pubmed_graphrag" in {t.name for t in tools.tools}

                result = await session.call_tool(
                    "ask_pubmed_graphrag", {"question": "does aspirin reduce heart attack risk?"}
                )
                assert not result.isError
                assert "yes" in result.content[0].text


@pytest.fixture
def anyio_backend():
    return "asyncio"
