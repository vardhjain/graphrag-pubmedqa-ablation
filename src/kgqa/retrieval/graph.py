"""GraphRAG arms: ``graph`` (parent expansion) and ``graph_concepts``.

Both reuse the identical encode + rerank + select pipeline from ``BaseRetriever``
(so the reranker is *controlled for*, not a confound). The graph then adds:

  graph           parent-paper expansion — reconstruct each selected chunk's
                  full abstract via HAS_CONTEXT traversal.
  graph_concepts  the above, plus a MeSH concept hop — pull in a few related
                  papers that share concepts with the selected papers.

Leakage is stripped: studies are labelled generically ("=== STUDY n ===") and
no question-derived title or ``final_decision`` ever reaches the prompt.
"""

from __future__ import annotations

from ..config import CONCEPT_HOP_PAPERS, HAS_CONTEXT, MENTIONS
from .base import BaseRetriever, Candidate, GraphExpansionMixin

# Reconstruct the full abstract of each selected chunk's parent paper.
_PARENT_AQL = """
    WITH Papers, Chunks
    FOR cid IN @ids
        LET chunk = DOCUMENT(cid)
        FOR paper IN 1..1 INBOUND chunk @@has_context
            LET sections = (
                FOR c IN 1..1 OUTBOUND paper @@has_context
                    SORT c._key
                    RETURN c.text
            )
            RETURN DISTINCT {
                paper: paper._key,
                abstract: CONCAT_SEPARATOR(" ", sections)
            }
"""

# From the seed papers, hop across shared MeSH concepts to related papers.
# Two-stage: rank neighbours by how many concepts they share with the seeds
# (cheap), then reconstruct abstracts only for the top-N (avoids building an
# abstract for every candidate on every query).
#
# NOTE -- known semantic divergence from neo4j_graph.py's _CONCEPT_CYPHER
# (GAPS #9): the COLLECT below counts every (seed, concept) -> neighbour
# edge reaching a neighbour, not the number of *distinct* concepts shared
# with it -- two different seed papers sharing one concept with the same
# neighbour count as 2, not 1. _CONCEPT_CYPHER's `count(DISTINCT concept)`
# is the intended semantics ("how many concepts they share with the seeds",
# per the comment above). Left undisturbed rather than fixed blind: AQL
# can't be exercised in this dev environment (tests fake db.aql.execute
# entirely, see tests/conftest.py's FakeAQL), and CLAUDE.md's own rule is
# that retrieval/graph.py's queries must stay behavior-compatible without a
# fresh benchmark run to re-validate against. Fix this AQL to explicitly
# count unique concept keys per neighbour (e.g. COLLECT ... INTO groups,
# then COUNT(UNIQUE(...))) the next time this file is touched with a live
# ArangoDB instance to verify against.
_CONCEPT_AQL = """
    WITH Papers, Chunks, Concepts
    LET seeds = @paper_keys
    LET ranked = (
        FOR pkey IN seeds
            LET paper = DOCUMENT(CONCAT("Papers/", pkey))
            FILTER paper != null
            FOR concept IN 1..1 OUTBOUND paper @@mentions
                FOR neighbour IN 1..1 INBOUND concept @@mentions
                    FILTER neighbour._key NOT IN seeds
                    COLLECT nkey = neighbour._key WITH COUNT INTO shared
                    SORT shared DESC
                    LIMIT @limit
                    RETURN { nkey: nkey, shared: shared }
    )
    FOR n IN ranked
        LET sections = (
            FOR c IN 1..1 OUTBOUND DOCUMENT(CONCAT("Papers/", n.nkey)) @@has_context
                SORT c._key
                RETURN c.text
        )
        RETURN { paper: n.nkey, abstract: CONCAT_SEPARATOR(" ", sections), shared: n.shared }
"""


class GraphRetriever(GraphExpansionMixin, BaseRetriever):
    name = "graph"

    def __init__(self, store, encoder, db, reranker=None,
                 use_concepts: bool = False,
                 concept_hop_papers: int = CONCEPT_HOP_PAPERS, **kwargs):
        super().__init__(store, encoder, reranker=reranker, **kwargs)
        self.db = db
        self.use_concepts = use_concepts
        self.concept_hop_papers = concept_hop_papers
        if use_concepts:
            self.name = "graph_concepts"

    def _parent_abstracts(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        rows = self.db.aql.execute(
            _PARENT_AQL,
            bind_vars={"ids": chunk_ids, "@has_context": HAS_CONTEXT},
        )
        out, seen = [], set()
        for row in rows:
            key = row["paper"]
            if key in seen:
                continue
            seen.add(key)
            out.append((key, row.get("abstract", "")))
        return out

    def _concept_neighbours(self, paper_keys: list[str]) -> list[tuple[str, str]]:
        rows = self.db.aql.execute(
            _CONCEPT_AQL,
            bind_vars={
                "paper_keys": paper_keys,
                "@mentions": MENTIONS,
                "@has_context": HAS_CONTEXT,
                "limit": self.concept_hop_papers,
            },
        )
        return [(row["paper"], row.get("abstract", "")) for row in rows]

    # gather_studies() is inherited from GraphExpansionMixin (GAPS #9 --
    # this class used to carry its own byte-identical copy of that
    # orchestration; only _parent_abstracts/_concept_neighbours differ
    # per backend now).

    def _build_context(self, query: str, candidates: list[Candidate]) -> str:
        return format_studies(self.gather_studies(candidates))


def format_studies(studies: list[tuple[str, str]]) -> str:
    """Render (paper_key, abstract) pairs into the shared ``=== STUDY n ===`` context."""
    parts = [
        f"=== STUDY {i + 1} ===\n{abstract}"
        for i, (_key, abstract) in enumerate(studies)
        if abstract
    ]
    return "\n\n".join(parts) if parts else "No context found."
