"""Run one arm of the GraphRAG vs PlainRAG ablation on PubMedQA.

    python scripts/run_benchmark.py --arm plain_rr --n 200

Arms:
    plain           vector top-k chunks (baseline)
    plain_rr        + cross-encoder rerank
    graph           + parent-paper expansion (full abstracts)
    graph_concepts  + MeSH concept-hop expansion

All arms share one ArangoDB-backed chunk corpus (cached locally), the same
encoder, reranker, prompt, LLM, seed and sample — so results are comparable and
the only moving part is the retrieval strategy named by --arm.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

ARMS = ("plain", "plain_rr", "graph", "graph_concepts")


def build_retriever(arm, store, encoder, reranker, db):
    from kgqa.retrieval import GraphRetriever, PlainRetriever

    if arm == "plain":
        return PlainRetriever(store, encoder, reranker=None)
    if arm == "plain_rr":
        return PlainRetriever(store, encoder, reranker=reranker)
    if arm == "graph":
        return GraphRetriever(store, encoder, db, reranker=reranker, use_concepts=False)
    if arm == "graph_concepts":
        return GraphRetriever(store, encoder, db, reranker=reranker, use_concepts=True)
    raise ValueError(f"unknown arm: {arm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--n", type=int, default=None, help="sample size (default: config BENCHMARK_N)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: config RANDOM_SEED)")
    parser.add_argument("--output", default=None, help="results JSON path")
    parser.add_argument("--no-ollama-start", action="store_true",
                        help="don't auto-start the Ollama server")
    args = parser.parse_args()

    if not args.no_ollama_start:
        print("[Ollama] Starting server (idempotent)...")
        try:
            subprocess.Popen(["ollama", "serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
        except FileNotFoundError:
            print("[Ollama] 'ollama' not found on PATH; assuming it is already running.")

    from kgqa.config import BENCHMARK_N, RANDOM_SEED, ArangoConfig
    from kgqa.data import load_benchmark_samples
    from kgqa.evaluation import Evaluator, FuzzyEvaluator
    from kgqa.models import connect_arango, load_encoder, load_reranker
    from kgqa.retrieval import ChunkStore

    n = args.n or BENCHMARK_N
    seed = args.seed if args.seed is not None else RANDOM_SEED
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.output or os.path.join(results_dir, f"{args.arm}_results.json")
    cache_file = os.path.join(ROOT, "pubmed_vectors_cache.pkl")

    db = connect_arango(ArangoConfig())
    print("[Corpus] Loading chunk store from ArangoDB (cached after first run)...")
    store = ChunkStore.from_arango(db, cache_file=cache_file)
    print(f"[Corpus] {len(store):,} chunks loaded.")

    encoder = load_encoder()
    needs_rerank = args.arm != "plain"
    reranker = load_reranker() if needs_rerank else None

    retriever = build_retriever(args.arm, store, encoder, reranker, db)
    samples = load_benchmark_samples(n=n, seed=seed)

    fuzzy = FuzzyEvaluator()
    evaluator = Evaluator(args.arm)
    print(f"\n=== Benchmark: {args.arm}  (n={len(samples)}, seed={seed}) ===")
    for i, s in enumerate(samples):
        t0 = time.time()
        raw = retriever.answer_benchmark(s.question)
        latency = time.time() - t0
        pred = fuzzy.extract_answer(raw)
        evaluator.record(s.final_decision, pred, latency, sample_id=s.pubid)
        icon = "v" if pred == s.final_decision.lower().strip() else "x"
        print(f"[{i + 1:3d}]  GT={s.final_decision:<5}  Pred={pred:<5}  {icon}  ({latency:.1f}s)")

    evaluator.report()
    evaluator.save(out_path)


if __name__ == "__main__":
    main()
