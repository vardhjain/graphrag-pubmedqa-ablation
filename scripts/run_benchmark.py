"""Run one arm of the GraphRAG vs PlainRAG ablation on PubMedQA.

    python scripts/run_benchmark.py --arm plain_rr --n 200

Arms:
    plain           vector top-k chunks (baseline)
    plain_rr        + cross-encoder rerank
    graph           + parent-paper expansion (full abstracts)
    graph_concepts  + MeSH concept-hop expansion
    graph_norr      parent-paper expansion WITHOUT the reranker (hosted demo's config)
    flat_parent     non-graph full-abstract lookup; negative control for graph

All arms share one ArangoDB-backed chunk corpus (cached locally), the same
encoder, reranker, prompt, LLM, seed and sample — so results are comparable and
the only moving part is the retrieval strategy named by --arm.

Resilience: each question is retried, and a wedged/crashed Ollama is restarted
between attempts, so a single 500/timeout cannot abort the whole arm. Partial
results are checkpointed every 25 questions to ``<output>.partial`` (atomic
tmp + os.replace) and only promoted to ``<output>`` when the arm completes;
``--resume`` continues from that checkpoint after a crash / Colab disconnect.

Every result file carries a ``meta`` block (arm, seed, n, model, generation
settings, git SHA, complete) so scripts/compare.py can refuse to pair runs
that don't match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

ARMS = ("plain", "plain_rr", "graph", "graph_concepts", "graph_norr", "flat_parent")
MAX_TRIES = 3
CHECKPOINT_EVERY = 25


def _ollama_base(api_url: str) -> str:
    return api_url.split("/api/")[0]


def _ollama_healthy(api_url: str, timeout: int = 5) -> bool:
    try:
        return requests.get(_ollama_base(api_url) + "/api/tags", timeout=timeout).ok
    except Exception:
        return False


def warm_model(api_url: str, model: str) -> None:
    """Load the model into memory so question 1's latency isn't a cold load."""
    try:
        requests.post(
            _ollama_base(api_url) + "/api/generate",
            json={
                "model": model,
                "prompt": "ok",
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 1},
            },
            timeout=180,
        )
    except Exception:
        pass


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
            warm_model(api_url, model)
            return True
        time.sleep(2)
    print("[Ollama] WARNING: server did not become healthy in time.")
    return False


def build_retriever(arm, store, encoder, reranker, db):
    from kgqa.retrieval import FlatParentRetriever, GraphRetriever, PlainRetriever

    if arm == "plain":
        return PlainRetriever(store, encoder, reranker=None)
    if arm == "plain_rr":
        return PlainRetriever(store, encoder, reranker=reranker)
    if arm == "graph":
        return GraphRetriever(store, encoder, db, reranker=reranker, use_concepts=False)
    if arm == "flat_parent":
        return FlatParentRetriever(store, encoder, reranker=reranker)
    if arm == "graph_concepts":
        return GraphRetriever(store, encoder, db, reranker=reranker, use_concepts=True)
    if arm == "graph_norr":
        # Parent-expansion without the reranker -- the exact configuration
        # the hosted demo actually runs (render.yaml: KGQA_SKIP_RERANKER=true,
        # a 512MB-tier memory concession), which the plain_rr/graph/
        # graph_concepts ladder alone never measures in isolation.
        return GraphRetriever(store, encoder, db, reranker=None, use_concepts=False)
    raise ValueError(f"unknown arm: {arm}")


# ── run metadata / checkpointing ──────────────────────────────────────────────
def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def build_meta(arm: str, seed: int, n_requested: int) -> dict:
    """Run metadata stamped into every result file (read by compare.py)."""
    from kgqa import config

    return {
        "arm": arm,
        "seed": seed,
        "n_requested": n_requested,
        "n_done": 0,
        "llm_model": config.LLM_MODEL,
        "num_ctx": config.LLM_NUM_CTX,
        "num_predict": config.LLM_NUM_PREDICT,
        "temperature": config.LLM_TEMPERATURE,
        "top_k": config.TOP_K_FINAL,
        "top_k_candidates": config.TOP_K_CANDIDATES,
        "git_sha": _git_sha(),
        "complete": False,
    }


def stamp_parent_comparison(meta: dict, samples: list, store) -> None:
    """Stamp both arms with identical required inputs for an honest contrast.

    The flat baseline needs a newly rerun graph arm -- no historical graph
    artifact contains this protocol marker. Hash all evaluated sample inputs
    and *the entire* retrieved corpus rather than assuming the same DB name
    guarantees identical bytes.
    """
    meta["parent_control_protocol"] = "flat-parent-v1"
    sample_input = [[str(s.pubid), s.question, s.final_decision] for s in samples]
    meta["sample_sha256"] = hashlib.sha256(
        json.dumps(sample_input, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    digest = hashlib.sha256()
    for index in sorted(range(len(store.ids)), key=store.ids.__getitem__):
        for value in (store.ids[index], store.paper_keys[index], store.texts[index]):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        vector = store.embeddings[index].tobytes()
        digest.update(len(vector).to_bytes(8, "big"))
        digest.update(vector)
    meta["corpus_sha256"] = digest.hexdigest()


def check_parent_control_resume(data: dict, expected: dict) -> None:
    """Never resume a graph/flat contrast with changed experimental inputs."""
    actual = data.get("meta") or {}
    fields = (
        "parent_control_protocol",
        "sample_sha256",
        "corpus_sha256",
        "llm_model",
        "num_ctx",
        "num_predict",
        "temperature",
        "top_k",
        "top_k_candidates",
        "git_sha",
    )
    for key in fields:
        if key not in actual or actual[key] != expected.get(key):
            raise ValueError(f"Checkpoint {key} differs from the paired experiment")


def check_resume(data: dict, arm: str, seed: int, n_requested: int, sample_ids: list) -> list:
    """Validate a .partial checkpoint against this run; return its done ids.

    Raises ValueError if the checkpoint came from a different arm/seed/n, or
    its ids aren't a prefix of this run's sample order.
    """
    meta = data.get("meta") or {}
    for key, want in (("arm", arm), ("seed", seed), ("n_requested", n_requested)):
        if meta.get(key) != want:
            raise ValueError(
                f"checkpoint {key}={meta.get(key)!r} does not match this run's {want!r}"
            )
    done = list(data.get("ids") or [])
    if done != list(sample_ids[: len(done)]):
        raise ValueError("checkpoint ids are not a prefix of this run's sample order")
    return done


def is_truncated(raw: str | None, done_reason: str | None) -> bool:
    """Ollama reports done_reason="length" when num_predict cut generation off.
    Without a done_reason, fall back to a heuristic: an unterminated <think>."""
    if raw is None:
        return False
    if done_reason is not None:
        return done_reason == "length"
    return raw.count("<think>") > raw.count("</think>")


def install_done_reason_probe() -> dict:
    """Route base.py's module-level ``call_ollama`` through ``call_ollama_meta``
    so the runner sees Ollama's done_reason without editing base.py or
    changing call_ollama's signature. Returns a dict whose ``"reason"`` holds
    the most recent call's done_reason."""
    import kgqa.retrieval.base as base
    from kgqa.llm import call_ollama_meta

    last: dict = {"reason": None}

    def _probe(*args, **kwargs):
        text, reason = call_ollama_meta(*args, **kwargs)
        last["reason"] = reason
        return text

    base.call_ollama = _probe
    return last


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument(
        "--n", type=int, default=None, help="sample size (default: config BENCHMARK_N)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed (default: config RANDOM_SEED)"
    )
    parser.add_argument("--output", default=None, help="results JSON path")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from <output>.partial if it matches arm/seed/n",
    )
    parser.add_argument(
        "--no-ollama-start",
        action="store_true",
        help="don't auto-start/health-check the Ollama server",
    )
    args = parser.parse_args()

    from kgqa.config import BENCHMARK_N, LLM_MODEL, OLLAMA_API, RANDOM_SEED, ArangoConfig
    from kgqa.data import load_benchmark_samples
    from kgqa.evaluation import Evaluator, FuzzyEvaluator
    from kgqa.models import connect_arango, load_encoder, load_reranker
    from kgqa.retrieval import ChunkStore

    if not args.no_ollama_start:
        print("[Ollama] Ensuring server is healthy...")
        ensure_ollama(OLLAMA_API, LLM_MODEL)
        # Warm unconditionally: ensure_ollama only warms after it (re)starts
        # the server, so an already-healthy server would otherwise charge
        # question 1 with a cold model load.
        warm_model(OLLAMA_API, LLM_MODEL)

    n = args.n or BENCHMARK_N
    seed = args.seed if args.seed is not None else RANDOM_SEED
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.output or os.path.join(results_dir, f"{args.arm}_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    partial_path = out_path + ".partial"
    cache_file = os.path.join(ROOT, "pubmed_vectors_cache.pkl")

    samples = load_benchmark_samples(n=n, seed=seed)
    meta = build_meta(args.arm, seed, n)

    evaluator = Evaluator(args.arm)
    done_ids: set = set()
    if args.resume and os.path.exists(partial_path):
        with open(partial_path) as f:
            data = json.load(f)
        done = check_resume(data, args.arm, seed, n, [s.pubid for s in samples])
        evaluator = Evaluator.from_saved(data)
        done_ids = set(done)
        print(f"[Resume] {len(done)}/{len(samples)} already done in {partial_path}")
    elif args.resume:
        print(f"[Resume] no checkpoint at {partial_path}; starting fresh.")
    evaluator.meta = meta

    db = connect_arango(ArangoConfig())
    print("[Corpus] Loading chunk store from ArangoDB (cached after first run)...")
    store = ChunkStore.from_arango(db, cache_file=cache_file)
    print(f"[Corpus] {len(store):,} chunks loaded.")
    if args.arm in ("graph", "flat_parent"):
        stamp_parent_comparison(meta, samples, store)
        if args.resume and os.path.exists(partial_path):
            check_parent_control_resume(data, meta)

    encoder = load_encoder()
    reranker = load_reranker() if args.arm not in ("plain", "graph_norr") else None

    retriever = build_retriever(args.arm, store, encoder, reranker, db)
    done_reason = install_done_reason_probe()

    fuzzy = FuzzyEvaluator()
    print(f"\n=== Benchmark: {args.arm}  (n={len(samples)}, seed={seed}) ===")
    for i, s in enumerate(samples):
        if s.pubid in done_ids:
            continue
        raw = None
        latency = 0.0
        retrieved_papers: list[str] = []
        # GAPS #10: gather_studies() (graph arms only) silently degrades to
        # raw chunks on any graph-expansion failure -- a mass-degrade would
        # quietly turn a graph arm into plain-arm numbers. Prefer the
        # retriever's per-call flag if it exposes one; otherwise delta its
        # counter across this question's attempt(s).
        degraded_before = getattr(retriever, "_degraded_count", 0)
        t_total = time.time()
        for attempt in range(1, MAX_TRIES + 1):
            done_reason["reason"] = None
            t0 = time.time()
            try:
                raw, retrieved_papers = retriever.answer_benchmark(s.question)
                latency = time.time() - t0  # the successful attempt only
                break
            except Exception as exc:
                print(
                    f"      [warn] q{i + 1} attempt {attempt}/{MAX_TRIES} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt < MAX_TRIES and not args.no_ollama_start:
                    ensure_ollama(OLLAMA_API, LLM_MODEL, restart=True)
        total = time.time() - t_total
        if hasattr(retriever, "_last_degraded"):
            degraded = bool(retriever._last_degraded)
        else:
            degraded = getattr(retriever, "_degraded_count", 0) > degraded_before
        truncated = is_truncated(raw, done_reason["reason"])

        if raw is None:
            pred = "maybe"  # last resort so one bad call doesn't abort the arm
            latency = total
            print(
                f"[{i + 1:3d}]  GT={s.final_decision:<5}  Pred={pred:<5}  !  "
                f"(skipped after {MAX_TRIES} tries)"
            )
        else:
            pred = fuzzy.extract_answer(raw)
            icon = "v" if pred == s.final_decision.lower().strip() else "x"
            retry_note = f", {total:.1f}s incl. retries" if total - latency > 0.5 else ""
            flags = (" [degraded]" if degraded else "") + (" [truncated]" if truncated else "")
            print(
                f"[{i + 1:3d}]  GT={s.final_decision:<5}  Pred={pred:<5}  {icon}  "
                f"({latency:.1f}s{retry_note}){flags}"
            )

        evaluator.record(
            s.final_decision,
            pred,
            latency,
            sample_id=s.pubid,
            failed=(raw is None),
            retrieved_papers=retrieved_papers,
            degraded=degraded,
            truncated=truncated,
        )
        if (i + 1) % CHECKPOINT_EVERY == 0:
            meta["n_done"] = len(evaluator.ids)
            evaluator.save(partial_path, quiet=True)

    n_degraded, n_truncated = sum(evaluator.degraded), sum(evaluator.truncated)
    if n_degraded:
        print(
            f"\n[GraphRAG] {n_degraded}/{len(samples)} question(s) had graph expansion "
            "fail and fell back to raw chunks -- see the [GraphRAG] ... expansion failed "
            "warnings above for why. A large count here means this arm's numbers may be "
            "closer to a plain arm's than they appear."
        )
    if n_truncated:
        print(f"[LLM] {n_truncated}/{len(samples)} answer(s) hit the num_predict cap.")

    meta["n_done"] = len(evaluator.ids)
    meta["complete"] = meta["n_done"] == len(samples)
    evaluator.report()
    evaluator.save(partial_path, quiet=True)
    if meta["complete"]:
        os.replace(partial_path, out_path)
        print(f"Results saved to {out_path}")
    else:
        print(f"Incomplete run left at {partial_path} (re-run with --resume)")


if __name__ == "__main__":
    main()
