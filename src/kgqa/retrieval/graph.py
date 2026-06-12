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
from .base import BaseRetriever, Candidate

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


class GraphRetriever(BaseRetriever):
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

    def _build_context(self, query: str, candidates: list[Candidate]) -> str:
        chunk_ids = [c.chunk_id for c in candidates]
        try:
            studies = self._parent_abstracts(chunk_ids)
            seed_keys = [k for k, _ in studies]
            if self.use_concepts and seed_keys:
                for key, abstract in self._concept_neighbours(seed_keys):
                    if key not in seed_keys and abstract:
                        studies.append((key, abstract))
        except Exception as exc:  # graph unreachable -> degrade to raw chunks
            print(f"[GraphRAG] Graph expansion failed ({exc}). Using raw chunks.")
            studies = [(c.paper_key, c.text) for c in candidates]

        parts = [
            f"=== STUDY {i + 1} ===\n{abstract}"
            for i, (_key, abstract) in enumerate(studies)
            if abstract
        ]
        return "\n\n".join(parts) if parts else "No context found."
