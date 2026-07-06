"""Neo4j-backed graph retriever for the hosted agent's ``graph_id="demo"`` path.

This is a parallel implementation to ``retrieval/graph.py``'s ``GraphRetriever``
(same interface: ``gather_studies`` / ``_build_context``), not a replacement.
The benchmarked ablation (``scripts/ingest.py``, ``scripts/run_benchmark.py``,
and every number in RESULTS.md) is unchanged and keeps using ArangoDB/AQL --
rewriting that pipeline would need a fresh GPU benchmark run to re-validate,
which this project's dev environment can't do. This module exists only for
the live hosted-agent service (``kgqa/service.py``), scoped to the small
labeled-split demo corpus (see ``scripts/ingest_neo4j.py``).

Same schema shape as the Arango version, expressed as Neo4j labels/relationship
types instead of collections: ``(:Paper)-[:HAS_CONTEXT]->(:Chunk)``,
``(:Paper)-[:MENTIONS]->(:Concept)``. Deliberately avoids APOC procedures
(text aggregation happens in Python) since APOC availability isn't guaranteed
on every AuraDB tier.
"""

from __future__ import annotations

from ..config import CONCEPT_HOP_PAPERS
from .base import BaseRetriever, Candidate

_PARENT_CYPHER = """
UNWIND $keys AS ckey
MATCH (chunk:Chunk {key: ckey})<-[:HAS_CONTEXT]-(paper:Paper)
MATCH (paper)-[:HAS_CONTEXT]->(c:Chunk)
WITH paper, c
ORDER BY c.key
WITH paper, collect(c.text) AS texts
RETURN DISTINCT paper.key AS paper, texts
"""

_CONCEPT_CYPHER = """
MATCH (seed:Paper) WHERE seed.key IN $paper_keys
MATCH (seed)-[:MENTIONS]->(concept:Concept)<-[:MENTIONS]-(neighbour:Paper)
WHERE NOT neighbour.key IN $paper_keys
WITH neighbour, count(DISTINCT concept) AS shared
ORDER BY shared DESC
LIMIT $limit
MATCH (neighbour)-[:HAS_CONTEXT]->(c:Chunk)
WITH neighbour, shared, c
ORDER BY c.key
WITH neighbour, shared, collect(c.text) AS texts
RETURN neighbour.key AS paper, texts, shared
"""


def _local_key(chunk_id: str) -> str:
    """"Chunks/12345_0" -> "12345_0" -- strip the Arango-style collection prefix
    kept elsewhere in the codebase (Candidate.chunk_id, ChunkStore ids) so this
    retriever can plug into the same call sites without changing those IDs."""
    return chunk_id.split("/", 1)[-1] if "/" in chunk_id else chunk_id


class Neo4jGraphRetriever(BaseRetriever):
    name = "graph"

    def __init__(self, store, encoder, driver, reranker=None,
                 use_concepts: bool = False,
                 concept_hop_papers: int = CONCEPT_HOP_PAPERS, **kwargs):
        super().__init__(store, encoder, reranker=reranker, **kwargs)
        self.driver = driver
        self.use_concepts = use_concepts
        self.concept_hop_papers = concept_hop_papers
        if use_concepts:
            self.name = "graph_concepts"

    def _parent_abstracts(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        keys = [_local_key(cid) for cid in chunk_ids]
        with self.driver.session() as session:
            rows = list(session.run(_PARENT_CYPHER, keys=keys))
        return [(row["paper"], " ".join(row["texts"])) for row in rows]

    def _concept_neighbours(self, paper_keys: list[str]) -> list[tuple[str, str]]:
        with self.driver.session() as session:
            rows = list(session.run(
                _CONCEPT_CYPHER, paper_keys=paper_keys, limit=self.concept_hop_papers,
            ))
        return [(row["paper"], " ".join(row["texts"])) for row in rows]

    def gather_studies(self, candidates: list[Candidate]) -> list[tuple[str, str]]:
        """Expand ``candidates`` to (paper_key, full_abstract) pairs via Neo4j.

        Degrades to the raw retrieved chunks if the graph is unreachable --
        same fallback contract as ``GraphRetriever.gather_studies``.
        """
        chunk_ids = [c.chunk_id for c in candidates]
        try:
            studies = self._parent_abstracts(chunk_ids)
            seed_keys = [k for k, _ in studies]
            if self.use_concepts and seed_keys:
                for key, abstract in self._concept_neighbours(seed_keys):
                    if key not in seed_keys and abstract:
                        studies.append((key, abstract))
        except Exception as exc:  # graph unreachable -> degrade to raw chunks
            print(f"[GraphRAG] Neo4j graph expansion failed ({exc}). Using raw chunks.")
            studies = [(c.paper_key, c.text) for c in candidates]
        return studies

    def _build_context(self, query: str, candidates: list[Candidate]) -> str:
        from .graph import format_studies

        return format_studies(self.gather_studies(candidates))
