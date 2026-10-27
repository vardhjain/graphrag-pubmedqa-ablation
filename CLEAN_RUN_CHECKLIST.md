# Clean benchmark rerun — reproducibility checklist

**Status: prepared, not executed.** This is the operator handoff for a future
PubMedQA GraphRAG ablation. It is not evidence of a new accuracy number, a
cleaned database, or a successful LLM run. Until the gates below pass and a
genuine rerun is published, the historical values in
[`RESULTS.md`](RESULTS.md) and `results/summary.json` remain **historical,
not independently replicated**.

**Compute budget:** everything through Gate 3 runs without an LLM or GPU.
The measured benchmark in Gate 4 requires a real Ollama model (local or an
already-available free runtime); do not use a paid service by default.

## Gate 0 — Freeze the protocol and protect historical artifacts

- [ ] Work from a clean, reviewed commit of `main`. Record `git rev-parse HEAD`
  and `git status --short` in the run notes. Do **not** change code, prompts,
  model identity, retrieval settings, or corpus during the six arm runs.
- [ ] Confirm the known leaked Neo4j Aura credential has been **rotated** and
  the hosted demo has its new credential. Git history can't be made safe just
  by redacting a file. Do not paste passwords into logs or reports.
- [ ] Set the comparison family **before** seeing outcomes: the cumulative
  ladder is `plain → plain_rr → graph → graph_concepts`; `graph_norr`
  compares with `plain`. `graph ↔ flat_parent` is a separate,
  **exploratory** control, not a fifth test silently added to the existing
  Holm-adjusted family. Disclose any later post-hoc analysis as exploratory.
- [ ] Use an **isolated** directory for the new raw results and summary.
  Never run `make compare` or the default `python scripts/compare.py`
  against the historic `results/` directory when preparing new evidence.
- [ ] Record dataset identity: `qiaojin/PubMedQA`, `pqa_labeled` evaluation
  split, labeled + unlabeled retrieval corpus, dataset revision/commit and
  locally observed dataset fingerprint. **The benchmark currently records
  a sample/corpus content fingerprint, not the upstream Hugging Face
  dataset revision.** Record that separately before publication.

Suggested output layout (new files only; do not replace the existing summary):

```bash
export RUN_DIR="results/clean-strict-v2"
mkdir -p "$RUN_DIR"
git rev-parse HEAD
git status --short
```

## Gate 1 — Audit the graph (no inference, no mutation)

- [ ] Select the intended **local or owned** ArangoDB database with
  `ARANGO_HOST`, `ARANGO_USER`, `ARANGO_PASS`, and `ARANGO_DB`.
  Use environment variables or an ignored local `.env`, never tracked files.
- [ ] If the historical graph has duplicate edges, **back up or use a fresh
  database** and reingest intentionally with the fixed deterministic edge keys.
  The audit itself never deletes or repairs data. Do not run a destructive
  deduplication against an unbacked-up graph.
- [ ] Run the **full structural audit**, not just the old edge-count shortcut.
  Its exit status must be zero and `ready_for_benchmark` must be true.

```bash
python scripts/audit_graph.py --report "$RUN_DIR/graph_audit.json"
```

Audit includes graph endpoint correctness, duplicate relationships, orphan
chunks, invalid parent mappings, missing embeddings and leakage-bearing Paper
fields. Equal counts of HAS_CONTEXT edges and Chunks **alone are insufficient**
because duplicate and missing links can cancel. The audit also does **not**
prove retrieval-corpus completeness, dataset identity or answer correctness.

- [ ] If any audit check fails, **stop** and investigate. Keep the audit report
  as evidence of why the run was blocked; do not use `--allow-issues` to bypass
  the publication gate.

## Gate 2 — Check equal parent context and frozen corpus (no inference)

- [ ] Ensure the vector cache represents the audited corpus. The benchmark
  runner uses `pubmed_vectors_cache.pkl`; **don't reuse a cache from before
  a graph rebuild**. Back it up/remove it after re-ingestion and allow the
  next corpus load to rebuild it, then leave both cache and database unchanged
  for every arm.
- [ ] Compare the graph and flat-paper contexts on 1,000 deterministic chunks
  using the corrected **numeric section ordering**:

```bash
python scripts/check_parent_parity.py --n 1000 --seed 42
```

This fetches stored embeddings and performs graph lookups only; no encoder,
Ollama or Gemini request is made. It is a **sample**, not a complete proof of
equal context on every possible question.

- [ ] Run the repository's tests, including the real-database contract suite
  (GitHub Actions already runs these against disposable Docker instances).
  Real database tests must not be pointed at a production endpoint:

```bash
pytest -q
ruff check src scripts tests app backend
ruff format --check src scripts tests app backend
```

- [ ] Record the model versions and content fingerprints of the encoder,
  reranker, Ollama model, prompt, selected samples and chunk corpus. Re-check
  after every arm; a change means **restart the entire experiment**.

## Gate 3 — Optional zero-cost pilot before the full run

- [ ] Confirm Ollama is locally available with the configured model and
  sufficient RAM/VRAM. The published run used `LLM_NUM_CTX=8192`;
  `config.py` defaults to 4096, so this must be set explicitly. Record
  `LLM_MODEL`, `LLM_NUM_PREDICT`, seed and Python/package versions.
- [ ] If you have no suitable **free** local/Colab hardware, stop here.
  Do not invent predictions or report the static context-parity test as an
  accuracy benchmark.
- [ ] Optionally run a **small, separate** exploratory pilot with
  `--n 5` to identify model startup/format problems. Keep it outside the
  final `RUN_DIR`; pilot outputs are not part of the confirmatory sample.

## Gate 4 — Run real, paired inference (only when free compute is available)

Every arm must use the **same** sample size, seed, parser, dataset, chunk store,
embedding model, generation settings and **code commit**. Full inference has
not been run as part of these improvements.

```bash
export LLM_NUM_CTX=8192

# Only run after all previous gates pass; can take significant local compute.
for arm in plain plain_rr graph graph_concepts graph_norr flat_parent; do
  python scripts/run_benchmark.py \
    --arm "$arm" --n 200 --seed 42 --parser strict \
    --output "$RUN_DIR/${arm}_results.json" || exit 1
done
```

The runner checkpoints to `<output>.partial`. If interrupted, rerun **the
same command** with `--resume`; it verifies checkpoint provenance. Do not
copy historical arm outputs into this directory. `strict-v2` counts
unparseable answers as incorrect rather than silently mapping them to
`maybe`.

**Stop and investigate** if any run is incomplete or has failed, degraded or
truncated calls. Invalid final answers are retained as incorrect and must be
reported, not silently filtered. If Ollama is unstable, fix the cause and
rerun affected arms under one coherent protocol instead of dropping samples.

## Gate 5 — Analyze without overwriting the historical benchmark

```bash
python scripts/compare.py \
  --results-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/analysis"

python scripts/compare_parent_baseline.py \
  --flat "$RUN_DIR/flat_parent_results.json" \
  --graph "$RUN_DIR/graph_results.json" \
  --output "$RUN_DIR/parent_comparison.json"
```

- [ ] Verify `analysis/summary.json` has all expected arms and the paired
  Holm family. It must contain **no** provenance warnings, excluded arms or
  misleading partial comparisons.
- [ ] Check the graph–flat control has exactly matched sampled questions and
  fingerprints and reports genuine `paired_n`, gains, losses, exact p-value
  and the **possibility of a null or negative graph effect**.
- [ ] Report the 56.5% majority-yes historical baseline for context, but
  recalculate that baseline on the actual new evaluation sample. Do not reuse
  the old value without verifying the new sample distribution.

`compare.py` defaults to the *historical* `results/` directory for
backward compatibility, so always pass explicit directories for a new run.

## Gate 6 — Export independently checkable evidence (no inference)

```bash
python scripts/evidence_bundle.py export \
  --results-dir "$RUN_DIR" \
  --out "$RUN_DIR/evidence"

python scripts/evidence_bundle.py verify \
  --bundle "$RUN_DIR/evidence"
```

- [ ] Verify the report says `verified: true` and `comparable: true`,
  and that all six arms and expected contrasts have been included.
- [ ] Preserve per-question JSONL, original per-arm JSON, both analysis
  summaries, the parent-control comparison, graph audit, sample/corpus/prompt
  hashes, model revisions and run commit. Review for unintended secrets or
  sensitive fields **before** publishing.
- [ ] Open a **separate results PR** showing the actual newly measured values
  (including unfavorable results), the updated methodology, class-level
  metrics and limitations. Do not silently replace the historical
  `results/summary.json` or its figure. Historical and corrected-retrieval
  experiments need distinct labels and citations.

## Completion criteria

The project may call the **new experiment reproducible** only after a clean
graph audit, frozen corpus and dataset revision, passing graph/flat parity
checks, complete matched strict-v2 inference, independently verified
per-question evidence, and an explicit published comparison.

**Current status:** PR #10 corrected and integration-tested both graph query
backends. It did **not** audit an actual historical ArangoDB corpus or rerun
the Ollama experiment. Those external-data/inference gates remain
**unverified**, so historical `+22.5 pp` should not be described as confirmed
by the corrected implementation.

Related: [EVIDENCE.md](EVIDENCE.md) ·
[PARENT_CONTROL.md](PARENT_CONTROL.md) · [RESULTS.md](RESULTS.md) ·
[GAPS.md](GAPS.md).
