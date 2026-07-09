"""MCP tool surface for the hosted GraphRAG agent.

Exposes the same ``graphrag.answer()`` the REST ``/query`` route calls, as an
MCP tool -- so any MCP client (Claude Code, Claude Desktop, Cursor, ...) can
use this agent by adding one URL, no code required. Mounted at ``/mcp`` in
``backend/main.py``.

Stateless HTTP mode: this is a public, unauthenticated demo with no per-user
state to track, so every request is handled independently with no session
affinity required between calls.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

mcp = FastMCP(
    "graphrag-pubmedqa",
    instructions=(
        "Answers biomedical yes/no/maybe questions grounded in a knowledge-graph "
        "retrieval pipeline over a 1,000-paper PubMedQA labeled-split demo corpus "
        "(not the full 62k-paper research corpus). Cites PubMed IDs for every "
        "answer. The backend runs on a free-tier host that sleeps after ~15"
        "minutes idle -- the first call after a gap can take 60-120 seconds "
        "while it wakes up; subsequent calls take ~15-20 seconds."
    ),
    stateless_http=True,
    streamable_http_path="/",
    # The SDK's default DNS-rebinding Host-header allowlist targets servers
    # meant to be bound to localhost and reached only from same-origin browser
    # code. This server is the opposite: a public HTTPS API meant to be hit
    # directly by remote MCP clients over whatever Host/CDN edge routes to it
    # (Render's own hostname, any future custom domain), so an allowlist would
    # need constant upkeep for no real protection here.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
def ask_pubmed_graphrag(question: str, use_concepts: bool = False) -> dict:
    """Ask a biomedical question against the PubMedQA demo knowledge graph.

    Args:
        question: A yes/no/maybe biomedical question, e.g. "Does aspirin
            reduce the risk of colorectal cancer?"
        use_concepts: If true, also expand retrieval via shared MeSH concept
            neighbours (slower, not shown to improve accuracy over the
            default parent-document expansion -- see the project's RESULTS.md).

    Returns:
        A dict with ``answer`` (str), ``sources`` (list of PubMed IDs cited),
        and ``reasoning_path`` (the graph traversal: seed chunk -> parent
        paper -> optional concept neighbour).
    """
    from graphrag import answer

    return answer(question, graph_id="demo", use_concepts=use_concepts)
