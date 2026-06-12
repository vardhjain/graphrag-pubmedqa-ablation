"""Central configuration — the single source of truth for every constant.

Every arm of the comparison reads from here, so the *only* differences between
PlainRAG and GraphRAG are the retrieval strategy and context assembly. Anything
that could confound the comparison (embedder, reranker, prompt, LLM, top-k,
sample size, seed) lives in this file and nowhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


# ── Shared models (identical across all arms) ─────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-r1:8b")

# ── Retrieval hyper-parameters (identical across all arms) ────────────────────
TOP_K_FINAL = 3  # documents handed to the LLM
TOP_K_CANDIDATES = 75  # wide pool fed to the reranker (rerank arms only)
CONCEPT_HOP_PAPERS = 3  # extra related papers pulled in by the concept arm

# ── Benchmark protocol (identical across all arms) ────────────────────────────
BENCHMARK_N = int(os.environ.get("BENCHMARK_N", "200"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
DATASET_NAME = "qiaojin/PubMedQA"
LABELED_CONFIG = "pqa_labeled"
UNLABELED_CONFIG = "pqa_unlabeled"

# ── LLM serving ───────────────────────────────────────────────────────────────
OLLAMA_API = os.environ.get("OLLAMA_API", "http://localhost:11434/api/chat")
LLM_TEMPERATURE = 0.0  # deterministic for benchmarking
# Env-tunable so the run can be sized to the GPU without code changes. num_predict
# caps generation so a runaway reasoning chain can't stall (or crash) the server;
# the answer extractor tolerates a truncated chain. Lower NUM_CTX to 4096 on a
# small-VRAM GPU (e.g. T4) if you hit out-of-memory 500s.
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "4096"))
LLM_NUM_PREDICT = int(os.environ.get("LLM_NUM_PREDICT", "1024"))
LLM_KEEP_ALIVE = os.environ.get("LLM_KEEP_ALIVE", "30m")
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "180"))

# ── Graph schema (must match scripts/ingest.py) ───────────────────────────────
NODE_COLLECTIONS = ("Papers", "Chunks", "Concepts")
EDGE_COLLECTIONS = ("HAS_CONTEXT", "MENTIONS")
HAS_CONTEXT = "HAS_CONTEXT"  # Paper -> Chunk
MENTIONS = "MENTIONS"  # Paper -> Concept


@dataclass
class ArangoConfig:
    """ArangoDB Oasis connection settings, read from the environment."""

    host: str = field(default_factory=lambda: os.environ.get(
        "ARANGO_HOST", "https://581c546a8d66.arangodb.cloud:8529"))
    user: str = field(default_factory=lambda: os.environ.get("ARANGO_USER", "root"))
    password: str = field(default_factory=lambda: os.environ.get("ARANGO_PASS", ""))
    db_name: str = field(default_factory=lambda: os.environ.get("ARANGO_DB", "pubmed_graph"))

    def require_password(self) -> None:
        if not self.password:
            raise OSError(
                "ARANGO_PASS is not set. Set it before connecting:\n"
                '    PowerShell : $env:ARANGO_PASS = "your_password"\n'
                "    bash       : export ARANGO_PASS=your_password\n"
                "    Colab      : add ARANGO_PASS in the Secrets panel"
            )
