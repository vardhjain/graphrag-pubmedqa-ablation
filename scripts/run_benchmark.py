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

Resilience: each question is retried, and a wedged/crashed Ollama is restarted
between attempts, so a single 500/timeout cannot abort the whole arm. Partial
results are checkpointed every 25 questions.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

ARMS = ("plain", "plain_rr", "graph", "graph_concepts", "graph_norr")
MAX_TRIES = 3
CHECKPOINT_EVERY = 25


def _ollama_base(api_url: str) -> str:
    return api_url.split("/api/")[0]


def _ollama_healthy(api_url: str, timeout: int = 5) -> bool:
    try:
        return requests.get(_ollama_base(api_url) + "/api/tags", timeout=timeout).ok
    except Exception:
        return False


def ensure_ollama(api_url: str, model: str, restart: bool = False, wait: int = 90) -> bool:
    """Make sure a healthy Ollama is serving; (re)start it if not."""
    import shutil

    if restart:
        try:
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True)
            time.sleep(3)
        except FileNotFoundError:
            pass  # no pkill (e.g. Windows) — fall through and try to start

    if not restart and _ollama_healthy(api_url):
        return True

    ollama = shutil.which("ollama") or "/usr/local/bin/ollama"
    try:
        subprocess.Popen([ollama, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[Ollama] binary not found; assuming a server is reachable elsewhere.")

    deadline = time.time() + wait
    while time.time() < deadline:
        if _ollama_healthy(api_url):
            try:  # warm the model so the next real call isn't a cold load
                requests.post(
                    _ollama_base(api_url) + "/api/generate",
                    json={"model": model, "prompt": "ok", "stream": False,
                          "keep_alive": "30m", "options": {"num_predict": 1}},
                    timeout=180,
                )
            except Exception:
                pass
            return True
        time.sleep(2)
    print("[Ollama] WARNING: server did not become healthy in time.")
    return False


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
    if arm == "graph_norr":
        # Parent-expansion without the reranker -- the exact configuration
        # the hosted demo actually runs (render.yaml: KGQA_SKIP_RERANKER=true,
        # a 512MB-tier memory concession), which the plain_rr/graph/
        # graph_concepts ladder alone never measures in isolation.
        return GraphRetriever(store, encoder, db, reranker=None, use_concepts=False)
    raise ValueError(f"unknown arm: {arm}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--n", type=int, default=None, help="sample size (default: config BENCHMARK_N)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: config RANDOM_SEED)")
    parser.add_argument("--output", default=None, help="results JSON path")
    parser.add_argument("--no-ollama-start", action="store_true",
                        help="don't auto-start/health-check the Ollama server")
    args = parser.parse_args()

    from kgqa.config import BENCHMARK_N, LLM_MODEL, OLLAMA_API, RANDOM_SEED, ArangoConfig
    from kgqa.data import load_benchmark_samples
    from kgqa.evaluation import Evaluator, FuzzyEvaluator
    from kgqa.models import connect_arango, load_encoder, load_reranker
    from kgqa.retrieval import ChunkStore

    if not args.no_ollama_start:
        print("[Ollama] Ensuring server is healthy...")
        ensure_ollama(OLLAMA_API, LLM_MODEL)

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
    reranker = load_reranker() if args.arm not in ("plain", "graph_norr") else None

    retriever = build_retriever(args.arm, store, encoder, reranker, db)
    samples = load_benchmark_samples(n=n, seed=seed)

    fuzzy = FuzzyEvaluator()
    evaluator = Evaluator(args.arm)
    print(f"\n=== Benchmark: {args.arm}  (n={len(samples)}, seed={seed}) ===")
    for i, s in enumerate(samples):
        t0 = time.time()
        raw = None
        retrieved_papers: list[str] = []
        for attempt in range(1, MAX_TRIES + 1):
            try:
                raw, retrieved_papers = retriever.answer_benchmark(s.question)
                break
            except Exception as exc:
                print(f"      [warn] q{i + 1} attempt {attempt}/{MAX_TRIES} failed: "
                      f"{type(exc).__name__}: {exc}")
                if attempt < MAX_TRIES and not args.no_ollama_start:
                    ensure_ollama(OLLAMA_API, LLM_MODEL, restart=True)
        latency = time.time() - t0

        if raw is None:
            pred = "maybe"  # last resort so one bad call doesn't abort the arm
            print(f"[{i + 1:3d}]  GT={s.final_decision:<5}  Pred={pred:<5}  !  "
                  f"(skipped after {MAX_TRIES} tries)")
        else:
            pred = fuzzy.extract_answer(raw)
            icon = "v" if pred == s.final_decision.lower().strip() else "x"
            print(f"[{i + 1:3d}]  GT={s.final_decision:<5}  Pred={pred:<5}  {icon}  ({latency:.1f}s)")

        evaluator.record(s.final_decision, pred, latency, sample_id=s.pubid,
                        failed=(raw is None), retrieved_papers=retrieved_papers)
        if (i + 1) % CHECKPOINT_EVERY == 0:
            evaluator.save(out_path)  # checkpoint partial progress

    evaluator.report()
    evaluator.save(out_path)


if __name__ == "__main__":
    main()
