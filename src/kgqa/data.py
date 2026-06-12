"""Dataset loading, seeded sampling, and chunk-corpus construction.

The chunk corpus is built the same way the graph is ingested (per-section
chunks from the labeled + unlabeled splits), so every arm retrieves over an
identical pool of documents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .config import (
    BENCHMARK_N,
    DATASET_NAME,
    LABELED_CONFIG,
    RANDOM_SEED,
    UNLABELED_CONFIG,
)


@dataclass
class BenchmarkSample:
    pubid: str
    question: str
    final_decision: str


def load_benchmark_samples(n: int = BENCHMARK_N, seed: int = RANDOM_SEED) -> list[BenchmarkSample]:
    """Return a deterministic random sample of labeled PubMedQA questions.

    Uses a seeded shuffle so the same questions are evaluated across every arm
    and across re-runs — a prerequisite for the paired McNemar test.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, LABELED_CONFIG, split="train")
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    samples: list[BenchmarkSample] = []
    for idx in indices:
        item = ds[idx]
        decision = item.get("final_decision")
        if not item.get("question") or not decision:
            continue
        samples.append(BenchmarkSample(
            pubid=str(item["pubid"]),
            question=item["question"],
            final_decision=decision,
        ))
        if len(samples) >= n:
            break
    return samples


def iter_chunks(include_unlabeled: bool = True):
    """Yield ``(paper_key, chunk_index, text)`` for every abstract section.

    This is the canonical chunking used both at ingestion time and when
    building the in-memory PlainRAG corpus, guaranteeing an identical document
    pool across arms.
    """
    from datasets import load_dataset

    configs = [LABELED_CONFIG]
    if include_unlabeled:
        configs.append(UNLABELED_CONFIG)

    for config in configs:
        ds = load_dataset(DATASET_NAME, config, split="train")
        for item in ds:
            paper_key = str(item["pubid"])
            contexts = item.get("context", {}).get("contexts", [])
            for idx, text in enumerate(contexts):
                if text and text.strip():
                    yield paper_key, idx, text
