"""Public package for the hosted GraphRAG agent.

Thin re-export over ``kgqa.service`` -- the research/benchmark code lives in
``kgqa``, this is the stable import surface a web backend (FastAPI, etc.)
depends on: ``from graphrag import answer``.
"""

from __future__ import annotations

from kgqa.service import answer

__all__ = ["answer"]
