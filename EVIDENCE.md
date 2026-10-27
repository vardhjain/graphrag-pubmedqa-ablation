# Reproducible benchmark evidence and graph audit

This is the publication protocol for **new** PubMedQA benchmark runs. It does
not retroactively verify the historical 200-question table: the historical
per-question predictions were not committed, and the graph likely had duplicate
HAS_CONTEXT edges.

The goal is to let a reviewer inspect **what was evaluated**, independently
recompute the reported per-sample accuracy and McNemar counts, and understand
what evidence is missing. Every operation below is free of LLM inference
unless it explicitly says "benchmark rerun."

## Step 1 — Inspect the existing graph (read-only, no GPU)

With the `ARANGO_HOST`, `ARANGO_USER`, `ARANGO_PASS`, and `ARANGO_DB`
settings pointing to your own benchmark database:

```bash
# Install python-arango if you don't have the benchmark environment.
pip install "python-arango>=8,<9"

python scripts/audit_graph.py --report results/graph_audit.json
```

The audit reads and counts all five collections and checks:

- Duplicate HAS_CONTEXT and MENTIONS source/target pairs.
- Orphaned chunks; chunks with more than one HAS_CONTEXT edge.
- Broken edge endpoints or a HAS_CONTEXT relationship to the wrong paper.
- Missing chunk embeddings or paper keys.
- Forbidden answer/question/title fields on Paper nodes.

If any condition fails, the process exits non-zero and the report has
`ready_for_benchmark: false`. To **inspect a known-dirty graph** without
failing the shell command, add `--allow-issues`; that flag changes only the
exit status, **not** the report or the database.

**Safety:** this script issues no writes, schema creation, deletions, ingestion,
or model calls. The output includes collection counts and diagnostic totals,
never credentials, free-text medical data, or connection strings. The audit
doesn't independently prove source-dataset completeness, embedding version,
or accuracy. Read the limitations field in the emitted report.

The original `python scripts/ingest.py --check-duplicates` only compared
`HAS_CONTEXT` edge count against `Chunks` count. A duplicate plus an
orphan can cancel out, so that shortcut is not a sufficiently strong gate.

## Step 2 — Repair and check parity before rerunning (user controlled)

Repair **your own** ArangoDB ingestion outside this audit. Use a new database
or back up the old one. `scripts/ingest.py` generates deterministic edge keys
for fresh ingestion. Do not run a destructive deduplication script without
checking the backup and edge schema. Run `audit_graph.py` again after any
repair.

If PR #6 (flat-parent control) is available, run its no-LLM parity preflight
against the corrected graph to verify that parent expansion reconstructs the
same context from graph links and a flat paper-to-section index. A structural
audit passing does **not** imply prompt/context parity or corpus completeness.

## Step 3 — Obtain raw predictions from a fresh experiment (inference required)

Run the standard benchmark with an available local Ollama/Colab runtime. The
runner writes files such as `results/graph_results.json`,
`results/plain_rr_results.json`, etc., **only after the corresponding arm
completes**. Keep every raw result file and use the same dataset, prompt,
seed, model, and graph/corpus snapshot across arms.

The default historical checkout contains `results/summary.json` but not
these per-question source files. **Nothing in this change recovers the missing
historical predictions or simulates rerunning the model.** Do not generate
"example results" that could be mistaken for measured model outputs.

Newer work in PR #3 versions strict answer parsing and records corpus/sample
fingerprints. Merge that work **before** generating a new confirmatory result.
Keep parsing versions separate.

## Step 4 — Export a reviewer-verifiable package (stdlib only)

```bash
python scripts/evidence_bundle.py export \
  --results-dir results \
  --out results/evidence

python scripts/evidence_bundle.py verify --bundle results/evidence
```

For PR #6's optional experiment, point `--results-dir` at
`results/parent_control` and `--out` at a distinct folder.

The resulting bundle has a `manifest.json` and one `<arm>.jsonl` per arm.
Each JSONL line includes the sampled PubMed ID, gold label, predicted label,
retrieved paper IDs, per-sample latency, and flags (failed/degraded/truncated/
invalid) when available. Missing optional fields are explicitly represented by
`null` and listed separately in the manifest. **No model chain-of-thought,
private credentials or API keys are exported.**

The manifest records SHA-256 hashes of original completed result files and
each exported JSONL, aggregate metrics independently recalculated from rows,
an allowlisted model/seed/parser/configuration metadata snapshot, and
exploratory (not multiplicity-adjusted) paired McNemar outcomes **only when
all available cross-arm provenance matches**. Missing or mismatched dataset
fingerprints and revision IDs prevent a comparable claim.

`verify` recalculates file checksums, metrics and paired counts from the
JSONL evidence alone, making it possible to independently detect edits and
check numerical consistency. It is deterministic and does not need ArangoDB,
Ollama, GPU, Internet, datasets, scikit-learn, or SciPy.

**Important limitations:** SHA-256 verifies consistency of the saved bundle,
not that a given LLM actually produced those predictions. To support
independent replication, also publish the dataset revision/sampling seed,
code revision, model identifier, input-corpus and prompt fingerprints, plus
the *clean* graph audit report. Model nondeterminism may still affect replay.

## Step 5 — Publish only real evidence

After a successful clean rerun, explicitly add the generated graph audit,
evidence folder, and updated analysis to a *separate* PR. Review provenance
and sensitivity of all new files first. Do not change the historical
`results/summary.json` in this audit tooling PR and do not treat an
exploratory paired p-value as a pre-registered, Holm-adjusted finding.

Until that happens, public statements should call the current GraphRAG lift
a **historical, not yet independently replicated** result.
