# GAPS.md — honest audit of weaknesses

Ordered most-important first. Each item: **what**, **where**, **why it matters**,
**fix** (scoped to be executable as a single task). Severity is the auditor's
call, not a formal CVSS.

The codebase is genuinely well-built — small, cohesive, tested where it counts,
and honest in its claims. Most items below are polish, not fires. But they're
real.

---

## 1. Live Neo4j credential sits in plaintext `.env` — SECURITY (HIGH)

**What:** `.env` (repo root) contains a **real, active** Neo4j AuraDB URI, user,
and password:
```
NEO4J_URI=neo4j+s://fc8ef905.databases.neo4j.io
NEO4J_USER=fc8ef905
NEO4J_PASSWORD=I4dst4SKl2qbEYpv5JVCfO4_lao_ZMXIcRZmjbyvFT8
```
**Where:** `.env` (untracked — confirmed **not** in git history, and `.gitignore`
line 18 excludes it — good). But it is a live secret readable by anything with
disk/session access, and it has been surfaced in tooling output.

**Why it matters:** These credentials grant write access to the demo graph
database. Anyone who obtains them can read or wipe the hosted demo's data. Because
the secret has now been exposed outside its intended store, it should be treated
as compromised regardless of the (correct) gitignore.

**Fix:**
1. Rotate the AuraDB password in the Neo4j console; update the Render env var
   `NEO4J_PASSWORD` and your local `.env`.
2. Consider using `.env.local` or a secrets manager for real credentials and
   keeping `.env` empty/placeholder like `.env.example`.
3. Add a `detect-private-key`-style guard is already in `.pre-commit-config.yaml`;
   confirm pre-commit is installed (`pre-commit install`) so a future secret can't
   be committed by accident.

---

## 2. ~~API error responses leak internal exception text~~ — FIXED

**What it was:** `/query` returned `detail=f"Answering failed: {exc}"` to the
client. **Fixed:** `backend/main.py`'s `query()` now logs the full exception
server-side via `logger.exception(...)` and returns a generic
`"Answering failed. Please try again."` detail; `backend/test_main.py` asserts
the raw exception text is no longer in the response body. `/ingest`'s
`dataset_id` echo was already just reflecting client-supplied input against a
known-safe allowlist, not leaking server internals, so left as-is.

---

## 3. ~~Zero test coverage for `compare.py`'s pairing logic~~ — PARTIALLY FIXED

**What it was:** No test caught a regression in `compare.py:aligned()`, the
index-alignment logic that pairs two arms' predictions by id before every
McNemar test — i.e. every published significance number. **Fixed:** added
`tests/test_compare.py` (4 cases: matched ids, shuffled ids, partially
overlapping ids, missing-`ids`-key positional fallback), run via the existing
`pythonpath = ["src", "."]` pytest config so `scripts.compare` imports
directly without touching `testpaths`.

**Still open:** `scripts/run_benchmark.py` (retry/health-check/checkpoint
logic) and `scripts/ingest.py` / `scripts/ingest_neo4j.py` remain untested —
they need a live Ollama/DB to exercise meaningfully, so covering them is a
separate, larger task (mocking the health-check/restart branches) rather than
a one-file addition.

---

## 4. ~~`decompose` / `extract` provider tasks defined but never called~~ — FIXED

**What it was:** `providers._CHAINS` defined `decompose`/`extract`/`synthesize`
tasks with Groq→Ollama chains, but only `synthesize` was ever invoked —
aspirational scaffolding that read as implemented. **Fixed (option a, delete):**
removed `call_groq`, `GROQ_API_KEY`/`GROQ_MODEL`/`GROQ_API_URL`, and the
`decompose`/`extract` chain entries from `src/kgqa/providers.py`; trimmed the
module docstring to describe only the `synthesize` task. Removed the now-dead
`GROQ_API_KEY` env var from `render.yaml` and `backend/README.md`, and the
corresponding Groq-specific tests from `tests/test_providers.py` (the
provider-chain-fallback tests were repointed at `gemini`/`ollama`, the tasks
that actually exist).

---

## 5. ~~`.env.example` pins a deprecated Gemini model~~ — FIXED

**What it was:** `.env.example` set `GEMINI_MODEL=gemini-1.5-flash` while
`providers.py` defaulted to `gemini-2.5-flash`, so following the README's
`cp .env.example .env` would silently override the good default. **Fixed:**
`.env.example` now matches (`gemini-2.5-flash`).

---

## 6. ~~Repository name inconsistent across the codebase~~ — FIXED

**What it was:** `pyproject.toml`'s Homepage/Repository/Issues URLs and two
frontend RESULTS.md links (`frontend/app/page.tsx`,
`frontend/app/benchmark/page.tsx`) still pointed at the old
`vardhjain/Knowledge_Graph_Question_Answering` slug while everything else
(git remote, README, dashboard) had moved to
`vardhjain/graphrag-pubmedqa-ablation`. **Fixed:** all three files repointed to
the canonical slug. Since the frontend is now deployed, these were live links
on the production `/benchmark` and home pages, not just repo metadata.

**Round 2 (2026-07-16) — this entry's first "FIXED" was over-scoped.** A
repo-wide sweep found four more instances the original pass never touched,
including the worst one:

- **`notebooks/01_ingest.ipynb` + `notebooks/02_benchmark.ipynb`** — both
  cloned the old slug. This was the serious one: the old repo *still exists
  and still clones*, 25 commits behind, so the documented reproduction path
  silently benchmarked stale code instead of failing loudly. Note the three
  lines (`rm -rf`, `git clone`, `%cd`) are coupled — the `%cd` was *correct*
  for the old clone, so fixing either line alone would have broken a working
  cell.
- **`CITATION.cff:9`** — `repository-code`, the field GitHub's "Cite this
  repository" widget and Zenodo read, so the dead slug propagated into every
  exported citation.
- **`CONTRIBUTING.md:11`** — cloned the *canonical* repo then `cd`-ed to the
  *old* directory name, so the contributor setup block failed on its first
  step.

All fixed and verified: no non-binary file outside historical records
(`KGQA_session_export.md`, `docs/Project_Report.pdf`) still references the old
slug. The lesson worth keeping: "I fixed the slug" is not verifiable by
spot-check — `git remote -v` and `pyproject.toml` being clean says nothing
about notebooks, citation metadata, or shell snippets in docs.

---

## 7. ~~`frontend/.env.local` points at the wrong backend port~~ — RESOLVED (stale entry)

**Re-verified 2026-07-08:** `frontend/.env.local` now points at the deployed
Render backend (`NEXT_PUBLIC_API_URL=https://graphrag-agent-api.onrender.com`),
not the stale `:8123`. This was superseded by the "Point frontend at the
deployed Render backend" commit — the audit entry was simply out of date.
Leaving this item as a record that the local-env-drift class of bug is worth
re-checking after any port/deploy-target change, since `.env.local` is
gitignored and can't be caught by CI.

---

## 8. ~~CORS default blocks the real frontend~~ — RESOLVED (stale entry)

**Re-verified 2026-07-08:** `render.yaml`'s `CORS_ORIGINS` now includes the
deployed Vercel origin (`https://graphrag-pubmedqa-ablation.vercel.app`)
alongside the local dev ports, per the "Add deployed Vercel frontend URL to
backend CORS allowlist" commit. Note the file's own comment: Render doesn't
auto-sync this field for an *existing* service from a `render.yaml` diff, so
confirm the dashboard's Environment tab matches after any future change here.

---

## 9. ~~Duplicated graph-retrieval logic across two backends~~ — PARTIALLY FIXED

**What it was:** `GraphRetriever` (ArangoDB, `graph.py`) and `Neo4jGraphRetriever`
(`neo4j_graph.py`) had near-identical `gather_studies` bodies (parent expansion +
optional concept hop + same degrade-to-raw-chunks fallback). The two also compute
"shared concept count" with subtly different semantics — AQL `COLLECT … WITH COUNT`
vs Cypher `count(DISTINCT concept)`.

**Fixed (2026-07-17):** the shared `gather_studies` orchestration is lifted into
`GraphExpansionMixin` (`src/kgqa/retrieval/base.py`); both retrievers now inherit
the identical control flow instead of carrying their own copy —
`tests/test_retrieval.py`'s `test_both_graph_backends_share_the_expansion_mixin`
pins this directly (`GraphRetriever.gather_studies is
GraphExpansionMixin.gather_studies`, same for Neo4j).

**Still open — the concept-count semantic divergence itself:** deliberately
*not* fixed here. `_CONCEPT_AQL` (`graph.py`) counts every `(seed, concept) ->
neighbour` edge; `_CONCEPT_CYPHER` (`neo4j_graph.py`)'s `count(DISTINCT concept)`
is the correct, intended semantics. Fixing the AQL side means rewriting a live
query with no way to execute or verify it in this dev environment (the test
suite fakes `db.aql.execute` entirely — see `tests/conftest.py`'s `FakeAQL`,
which pattern-matches query text rather than running it), and CLAUDE.md's own
rule is that these queries stay behavior-compatible without a fresh benchmark
run to re-validate against. Both files now carry an explicit comment
cross-referencing the other and stating which fix is needed
(`graph.py`'s `_CONCEPT_AQL`, `neo4j_graph.py`'s `_CONCEPT_CYPHER`) — fix the
AQL to count unique concept keys per neighbour the next time this file is
touched with a live ArangoDB instance available.

---

## 10. Broad `except Exception` swallows failures silently — PARTIALLY FIXED (benchmark path)

**What it was:** Several `except Exception` blocks degrade silently via `print()`
only — graph expansion failure, provider failures, DB connection failure. Real
bugs (e.g. a malformed AQL, a typo in a bind var) look identical to an expected
"DB unreachable" degrade and are hidden behind raw-chunk fallback.

**Fixed on the benchmark path (2026-07-17):** `GraphExpansionMixin.gather_studies`
(`src/kgqa/retrieval/base.py`) now increments a `self._degraded_count` on every
fallback, and `scripts/run_benchmark.py` deltas that counter per question and
prints a summary at the end of the run if any question degraded — a mass-degrade
is no longer just a `print()` line to scroll past. This is the entry's own
stated "at minimum" fix; the except itself stays broad on purpose (narrowing it
risks missing a legitimate transient error the fallback exists to protect
against).

**Still open — `service.py:84,106` and `providers.py:105`:** these are
hosted-agent-only instances of the same pattern (DB-connection degrade,
provider-chain degrade), not reachable from the benchmark. Deliberately *not*
touched: CLAUDE.md's own rule is "Error handling in the service degrades, never
crashes ... Preserve this for anything on the hosted path," so narrowing or
counting these would need a different mechanism (e.g. a metrics counter in a
long-running FastAPI process, not a linear script's end-of-run summary) and
risks that contract if done carelessly. `src/kgqa/retrieval/graph.py:121` /
`neo4j_graph.py:96` no longer apply as locations — that logic moved to the
shared mixin (GAPS #9).

**Fix:** Narrow the catches where the failure mode is known (e.g. arango/neo4j
connection exceptions) so unexpected errors propagate; or at minimum, in
`run_benchmark.py`, count and report how many questions fell back to raw chunks at
the end of the run so a mass-degrade is visible.

---

## 11. ~~AQL built with f-string interpolation~~ — FIXED (collection name);~~ingest.py~~ claim was stale

**What it was:** `ChunkStore.from_arango` built AQL by interpolating
`{collection}`, `{offset}`, `{batch}` into the query string. None of these were
user-controlled (all internal constants/ints), so this was never an injection
vulnerability -- the risk was purely that the pattern reads as "we interpolate
into AQL here," which invites copying into a future call site that isn't so
lucky.

**Fixed as a byproduct of GAPS-adjacent work (2026-07-17):** the pagination
rewrite below replaced the whole `LIMIT {offset}, {batch}` loop with a single
query, which eliminates `{offset}`/`{batch}` from the string entirely; the
remaining `{collection}` interpolation now uses ArangoDB's `@@collection` bind
syntax (`bind_vars={"@collection": collection}`) instead of an f-string. See
`src/kgqa/retrieval/base.py:113-149` and `tests/test_retrieval.py`'s
`test_from_arango_issues_a_single_query_not_offset_pagination`, which asserts
the bind var is actually used.

**Correction:** this entry's original "`ingest.py` similar" claim was
inaccurate -- `scripts/ingest.py` writes via `db.collection(name).import_bulk(...)`
(the python-arango collection API), not raw AQL, so there was never a second
instance of this pattern there. Worth remembering: a GAPS entry describing
"similar" code elsewhere is a claim, not a fact, until re-checked -- this one
apparently never was.

---

## 12. ~~`service.answer` reaches into retriever private methods~~ — FIXED

**What it was:** `service.answer` called `retriever._select(question)` directly
(the leading underscore marks `_select` as private) because it needs the
intermediate `candidates` to build `reasoning_path`, and the public surface
(`retrieve()` / `answer_benchmark()` / `chat()`) didn't expose them.

**Fixed (2026-07-17):** added a public `BaseRetriever.select(query) ->
list[Candidate]` (`src/kgqa/retrieval/base.py`) that's a thin wrapper over
`_select` — kept as the internal implementation, per this entry's own
suggested fix. `service.answer` now calls `retriever.select(question)`.
`tests/test_retrieval.py`'s `test_public_select_matches_private_select`
pins that the two return identical results.

---

## 13. Minor inconsistencies and polish — LOW / trivial

- **`frontend/README.md`** is untouched `create-next-app` boilerplate — says
  nothing about this project, the API contract, or `NEXT_PUBLIC_API_URL`. Replace
  with a few lines describing the chat + benchmark pages and the env var.
- **`FakeEncoder` uses Python `hash()`** (`tests/conftest.py:22`), which is
  salted per-process. Tests pass because hashing is stable *within* a run, but
  the vectors are meaningless across runs — fine for ranking tests, but a reader
  may assume determinism that isn't there. A comment would help.
- **`FuzzyEvaluator` defaults unparseable output to `"maybe"`**
  (`evaluation.py:27,51`). This is a documented, deliberate choice, but it biases
  errors toward the rare `maybe` class and can flatter/penalize an arm depending
  on failure rate. Worth a one-line note in RESULTS.md's methodology.
- **No `.dockerignore`** — `docker compose` only runs ArangoDB (no app image), so
  low impact, but if an app Dockerfile is ever added, `.next/`, `node_modules/`,
  and `__pycache__/` will bloat the build context.
- **`pkill`-based Ollama restart** (`scripts/run_benchmark.py:55`) is a no-op on
  Windows. Documented in a comment, but the benchmark is effectively
  Linux/Colab-only. Fine given the design, worth stating in CLAUDE.md (it is).
