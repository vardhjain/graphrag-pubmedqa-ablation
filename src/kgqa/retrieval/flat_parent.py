"""Non-graph, full-parent-abstract retrieval control for the PubMedQA ablation.

This is *not* another vector search. It reuses precisely the same selected
chunks and reranker as the graph arm, but maps chunk IDs to their parent paper
using the chunk store's paper_key field, not HAS_CONTEXT edge traversal.

On a clean graph, with one HAS_CONTEXT edge per chunk, it should assemble the
same contexts and therefore give the LLM the exact same token budget. This
is the critical negative control: does graph traversal do more than a flat
parent-document lookup?
"""

from __future__ import annotations

from .base import BaseRetriever, Candidate, GraphExpansionMixin
from .graph import format_studies


class FlatParentRetriever(GraphExpansionMixin, BaseRetriever):
    name = "flat_parent"

    def __init__(self, store, encoder, reranker=None, **kwargs):
        super().__init__(store, encoder, reranker=reranker, **kwargs)
        # Reuse the graph arm's failure reporting and context formatting;
        # no graph connection or graph query is ever used by this class.
        self.use_concepts = False

        if not (len(store.ids) == len(store.paper_keys) == len(store.texts)):
            raise ValueError("Incomplete chunk-store columns")
        self._chunk_to_paper: dict[str, str] = {}
        self._paper_to_indices: dict[str, list[int]] = {}
        for index, (chunk_id, paper_key) in enumerate(
            zip(store.ids, store.paper_keys, strict=True)
        ):
            if not isinstance(chunk_id, str) or not isinstance(paper_key, str):
                raise ValueError("Chunk ID and paper key must be strings")
            if not chunk_id.startswith("Chunks/") or not paper_key:
                raise ValueError(f"Invalid chunk ID or paper key: {chunk_id!r}")
            if chunk_id in self._chunk_to_paper:
                raise ValueError(f"Duplicate chunk ID: {chunk_id}")
            self._chunk_to_paper[chunk_id] = paper_key
            self._paper_to_indices.setdefault(paper_key, []).append(index)

        # Match the historical Arango AQL's SORT c._key exactly. In particular
        # _10 sorts ahead of _2, a *known* defect tracked separately. Changing
        # just this baseline to numeric ordering would break input parity.
        for indices in self._paper_to_indices.values():
            indices.sort(key=lambda index: store.ids[index].split("/", 1)[1])

    def _parent_abstracts(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        studies: list[tuple[str, str]] = []
        seen: set[str] = set()
        for chunk_id in chunk_ids:
            # A missing chunk should trigger the same raw-chunk fallback and
            # degraded counter as the graph path; never quietly omit it.
            paper = self._chunk_to_paper[chunk_id]
            if paper in seen:
                continue
            seen.add(paper)
            indices = self._paper_to_indices[paper]
            studies.append((paper, " ".join(self.store.texts[i] for i in indices)))
        return studies

    def _build_context(self, query: str, candidates: list[Candidate]) -> str:
        return format_studies(self.gather_studies(candidates))
