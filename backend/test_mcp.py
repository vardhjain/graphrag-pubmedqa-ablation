"""Tests for the MCP tool surface mounted at /mcp -- graphrag.answer is
monkeypatched, no live LLM/DB calls. Run with: pytest backend/test_mcp.py

The client-side round-trip test drives a real MCP ClientSession over the
mounted ASGI app (via httpx.ASGITransport, no live server needed) so a broken
mount/lifespan wiring fails here instead of only in production.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from backend.main import app
from backend.mcp_server import ask_pubmed_graphrag


def test_tool_calls_graphrag_answer_directly(monkeypatch):
    """Unit-level: the tool function is a thin pass-through to graphrag.answer."""
    import graphrag

    captured = {}

    def fake_answer(question, graph_id="demo", use_concepts=False):
        captured["args"] = (question, graph_id, use_concepts)
        return {"answer": "yes", "reasoning_path": [], "sources": ["123"]}

    monkeypatch.setattr(graphrag, "answer", fake_answer)

    out = ask_pubmed_graphrag("does aspirin help?", use_concepts=True)

    assert out == {"answer": "yes", "reasoning_path": [], "sources": ["123"]}
    assert captured["args"] == ("does aspirin help?", "demo", True)


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
        # Mounting the MCP sub-app's own "/" route at "/mcp" makes a bare
        # "/mcp" request 307-redirect to "/mcp/" -- real MCP clients always
        # follow redirects (see create_mcp_http_client), so match that here.
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://localhost",  # mcp's DNS-rebinding guard allows this Host
            headers=headers,
            timeout=timeout or 30,
            auth=auth,
            follow_redirects=True,
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
