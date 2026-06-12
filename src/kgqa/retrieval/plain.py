"""PlainRAG arms: ``plain`` (no rerank) and ``plain_rr`` (with rerank).

Context is the raw retrieved chunk text — no graph structure is used. With
``reranker=None`` this is the baseline; pass a CrossEncoder for the ``plain_rr``
arm that isolates the reranker's contribution.
"""

from __future__ import annotations

from .base import BaseRetriever, Candidate


class PlainRetriever(BaseRetriever):
    name = "plain"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.reranker is not None:
            self.name = "plain_rr"

    def _build_context(self, query: str, candidates: list[Candidate]) -> str:
        if not candidates:
            return "No context available."
        return "\n\n".join(
            f"Abstract {i + 1}: {c.text}"
            for i, c in enumerate(candidates)
        )
