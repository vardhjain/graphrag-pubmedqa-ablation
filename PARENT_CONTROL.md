# Flat parent-document negative control

## Research question

Does a **knowledge-graph traversal** improve biomedical RAG accuracy, or
does a **full parent abstract** improve it regardless of the underlying data
structure?

The published four-arm ablation reported a +22.5 percentage-point increase
when adding parent-abstract expansion to reranked chunk RAG. That contrast
changes both **context length** and **retrieval mechanism**. It does **not**
isolate a uniquely graph-specific benefit.

This optional sixth arm, `flat_parent`, is designed to close that gap. It is
currently **implemented and unit tested, not evaluated on a real LLM**.

## Experimental design

| Setting | `graph` | `flat_parent` |
| --- | --- | --- |
| Input dataset / sampled question order | Same | Same |
| Chunk embeddings and top-k selection | Same | Same |
| Cross-encoder reranker | Enabled | Enabled |
| Parent paper source | `HAS_CONTEXT` AQL graph edges | Existing chunk-store `paper_key` and index |
| Abstract section sorting | Numeric section suffix, then chunk key | Same numeric order for parity |
| Context labels and prompt | `=== STUDY n ===` | Identical |
| LLM, temperature, token configuration | Same | Same |
| Extra concept hops | Off | Off |
| Source corpus | Same chunk rows | Same chunk rows |

The *flat* arm reads the identical already-loaded ArangoDB **Chunks**
collection as the graph arm; it does not query graph edges or use the
`Papers`/`Concepts` collections. Hence this is a flat lookup based on
stored chunk metadata, not a second database service or a graph traversal.

The expected **equal-context null** is important: if the graph is clean and
the flat index faithfully reconstructs the same abstracts, the generated
prompts are byte-identical. Any difference in LLM answers then reflects
inference nondeterminism, runtime instability, or a failed parity control,
rather than extra graph-provided evidence. A graph might still be valuable
for richer multi-hop questions or indexing, but this task does not claim to
measure those use cases.

## Zero-inference tests

```bash
pip install -r requirements-dev.txt
pytest -v tests/test_flat_parent.py tests/test_compare_parent_baseline.py
```

CPU-only deterministic fakes verify identical contexts and prompts, paper
deduplication, candidate order, deterministic indices, refusal of malformed
inputs, reproducible corpus/sample fingerprints, and a paired comparator
that accepts negative and non-significant results.

## Database preflight (no LLM or GPU)

```bash
# Point ARANGO_HOST, ARANGO_USER, ARANGO_PASS, ARANGO_DB at the repaired graph.
python scripts/ingest.py --check-duplicates
make check-parent-parity
# Or sample a larger set:
python scripts/check_parent_parity.py --n 1000 --seed 42
```

The parity preflight loads the existing embeddings (it does not run an
encoder) and checks that the exact contexts returned by live AQL traversal
match the flat index for a seeded set of real chunk IDs. This is a
**sampled** check, not a proof of database-wide integrity: run a full
edge/multiplicity audit and ensure one context edge per chunk before any LLM
evaluation. `check_parent_parity.py` also refuses degraded results.

**Integrity caveat:** historical `HAS_CONTEXT` edges may be duplicated.
The corrected graph queries now deduplicate repeated references to the same
chunk while ordering `_2` before `_10`, and the flat control uses the
same numeric section index ordering. This improves structural parity but does
**not** make the old database trustworthy or retrospectively validate its
benchmark results. Audit and repair the graph, check context parity, and
rerun both arms under the new code revision before reporting any outcome.

## Matched inference experiment (optional)

Only after the graph is clean and the parity preflight passes, run the
following on local ArangoDB and a local Ollama-compatible model. The
`benchmark-parent-control` target is NOT called by CI or `make benchmark`,
so it incurs no automatic inference cost.

```bash
make benchmark-parent-control
```

This runs `graph` and `flat_parent` afresh with the same seed, context
window, prompts and model, and writes distinct raw outputs to
`results/parent_control/`:

```text
results/parent_control/graph_results.json
results/parent_control/flat_parent_results.json
results/parent_control/comparison.json
```

To compare saved runs without inference:

```bash
make compare-parent-control
```

The comparator validates the raw, per-question ID order and labels; rejects
failed, truncated, or degraded runs; and requires matching sample/corpus
fingerprints, model/generation settings, seed and source commit. It **rejects
the historical `graph` output** because it lacks the new
`parent_control_protocol=flat-parent-v1` metadata stamp.

It reports exact paired McNemar gains/losses and p-value, along with the
graph-minus-flat accuracy difference. This is an **exploratory standalone
contrast**: do not silently add it to the original predeclared Holm family or
claim confirmatory significance. If including it in a new confirmatory
multiple-comparisons family, preregister that family's members and rerun the
complete comparison accordingly.

## Interpretation and responsible reporting

- **Near-equal accuracy with matching contexts:** supports the conclusion
  that retrieving **full abstracts** mattered, but graph edges themselves
  were not necessary for this particular single-paper QA setting.
- **Material accuracy discrepancy with equal contexts:** investigate
  nondeterministic decoding, prompts, errors, truncation, and provenance
  before claiming the graph caused it.
- **Context discrepancy:** the comparison is invalid; correct graph edges,
  chunk coverage or section sorting and rerun both arms.
- **Faster/slower retrieval:** profile graph-edge AQL expansion versus
  in-memory indexed lookup separately from LLM generation and full-pipeline
  latency. Equal-context accuracy does not imply equal operational cost.

**Published numbers remain historical and unchanged.** This PR contains no
new benchmark outcomes, makes no new performance claim, and uses only
no-GPU/no-LLM tests in CI.
