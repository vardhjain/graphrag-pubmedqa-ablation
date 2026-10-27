# GAPS.md — honest audit of weaknesses

Ordered most-important first. Each item: **what**, **where**, **why it matters**,
**fix** (scoped to be executable as a single task). Severity is the auditor's
call, not a formal CVSS.

The codebase is genuinely well-built — small, cohesive, tested where it counts,
and honest in its claims. Most items below are polish, not fires. But they're
real.

---

## 1. Live Neo4j credential was published in this file — SECURITY (CRITICAL)

**What:** `.env` (repo root, untracked and gitignored) holds the real Neo4j
AuraDB URI, user, and password for the demo graph. That part is fine. The problem:
an earlier revision of **this file** quoted all three values verbatim, and GAPS.md
*is* tracked — so from commit `299a164` onward the live password sat in plaintext
in the public GitHub repo's history. The values are now redacted here
(2026-09-27), but redaction doesn't un-publish them: every clone, fork, and
GitHub's own cache of those commits still has them.

**Why it matters:** These credentials grant write access to the demo graph
database. Anyone who obtains them can read or wipe the hosted demo's data. The
secret must be treated as compromised.

**Fix (user-only — needs the Aura console):**
1. **Rotate the AuraDB password** in the Neo4j console; update the Render env var
   `NEO4J_PASSWORD` and your local `.env`. This is the only step that actually
   closes the hole — rewriting history without rotating doesn't.
2. Optionally purge the old value from history (`git filter-repo --replace-text`)
   and force-push; only worth it *after* rotation, and it breaks existing clones.
3. `detect-private-key` in `.pre-commit-config.yaml` would **not** have caught
   this — it matches PEM key headers, not passwords. The `gitleaks` hook added
   alongside this entry does; run `pre-commit install` once so it runs on commit.

The lesson worth keeping: an audit document that quotes the secret it's warning
about becomes the leak. Describe secrets by location, never by value.

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

**Since covered:** `scripts/run_benchmark.py`'s arm wiring and its new
resume/meta/atomic-checkpoint logic are exercised by `tests/test_run_benchmark.py`.

**Still open:** `scripts/ingest.py` / `scripts/ingest_neo4j.py` remain untested —
they need a live DB to exercise meaningfully, so covering them is a separate,
larger task rather than a one-file addition.

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
provider-chain-fallback tests were repointed at `gemini`/`ollama`, the providers
that actually exist). **Follow-up (2026-09-27):** `.env.example` still listed
`GROQ_API_KEY`/`GROQ_MODEL` and `LLM_CHAIN_DECOMPOSE`/`LLM_CHAIN_EXTRACT`; those
are now removed too.

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
(`docs/Project_Report.pdf`) still references the old
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

## 9. ~~Duplicated traversal logic and divergent MeSH-neighbour semantics~~ — FIXED IN SOURCE

The original ArangoDB `_CONCEPT_AQL` counted each seed-to-neighbour path,
whereas Neo4j `_CONCEPT_CYPHER` correctly counted distinct shared concepts.
Three seed papers mentioning the same concept therefore boosted one neighbour
three times, distorting the rank of related papers.

**Fixed in the query-correctness PR:** the two-stage AQL `COLLECT` first
deduplicates `(neighbour, concept)` pairs, then counts their distinct shared
concepts; ties break on the paper key. The Neo4j query retains its
`count(DISTINCT concept)` behavior. Real ArangoDB and Neo4j integration
tests exercise this semantic contrast, so it is no longer only a mocked claim.
Both retrieval implementations already share `GraphExpansionMixin`'s
fallback/deduplication orchestration.

**Benchmark status remains open:** this changes the graph-concept arm's
retrieval, so the historical accuracy and latency are **not revalidated**.
A fresh, paired strict-v2 run with saved per-question evidence is required.

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

**Still open — the broad excepts in `service.py`'s graph-connection
helpers (`_shared_db`, `_shared_neo4j_driver`) and
`providers.py`'s `call_llm` chain loop:** these are
hosted-agent-only instances of the same pattern (DB-connection degrade,
provider-chain degrade), not reachable from the benchmark. Deliberately *not*
touched: this project's rule is "Error handling in the service degrades, never
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

- ~~**`frontend/README.md`** is untouched `create-next-app` boilerplate~~ —
  **FIXED (2026-07-17):** replaced with a description of the three pages
  (`/`, `/benchmark`, `/case-study`), the `NEXT_PUBLIC_API_URL` env var, and
  the Vercel "include source files outside Root Directory" gotcha (needed
  since `/benchmark` reads `../results/summary.json` at build time).
- ~~**`FakeEncoder` uses Python `hash()`**~~ — **already stale when written**:
  `tests/conftest.py` was fixed to use an MD5-based `_stable_hash` (with a
  comment explaining exactly why) by commit `968ea5f`, "Fix flaky test
  fixture: FakeEncoder used process-randomized hash()" — before this GAPS
  entry was last touched. Worth remembering alongside GAPS #11's own
  correction: an entry can go stale by the code catching up to it, not just
  by the code drifting away from it — re-check before assuming a listed item
  is still live.
- ~~**`FuzzyEvaluator` defaults unparseable output to `"maybe"`**~~ — **FIXED
  (documented):** RESULTS.md's methodology section now states the bias. The
  behavior itself is deliberately unchanged (see #17 for the related
  mis-scoring modes).
- **No `.dockerignore`** — `docker compose` only runs ArangoDB (no app image), so
  low impact, but if an app Dockerfile is ever added, `.next/`, `node_modules/`,
  and `__pycache__/` will bloat the build context.
- **`pkill`-based Ollama restart** (`scripts/run_benchmark.py:55`) is a no-op on
  Windows. Documented in a comment, but the benchmark is effectively
  Linux/Colab-only. Fine given the design.

---

## 14. Published ArangoDB graph likely had duplicated `HAS_CONTEXT` edges — HIGH (open)

**What:** The recorded ingest counts were 209,971 `HAS_CONTEXT` edges vs
206,613 `Chunks` — a surplus of exactly 3,358, the labeled split's chunk count.
`scripts/ingest.py` created edges without deterministic `_key`s, and notebook
01 ran a labeled-only smoke ingest before the full ingest, so the labeled
papers' edges were inserted twice.

**Why it matters:** For labeled papers — the benchmark's own gold papers —
`_PARENT_AQL` rebuilt each abstract with every section twice (S0 S0 S1 S1 …),
and `graph_concepts`' shared-concept counts were inflated. Same information,
no label leakage, but the graph-arm contexts are not the method as described,
so the headline +22.5 pp is unconfirmed until re-run. Disclosed prominently in
RESULTS.md.

**Fixed so far:** `scripts/ingest.py` now writes edges with deterministic keys,
so re-ingesting is idempotent.

**Still open (needs live DB + GPU):** run the audit query (`LENGTH(HAS_CONTEXT)`
vs `LENGTH(Chunks)`) against the published database; dedupe or re-ingest; re-run
all arms and regenerate `results/`.

---

## 15. Committed `summary.json` has a stale schema; raw per-arm results never committed — MEDIUM (open)

**What:** `results/summary.json` predates the `n_failed`, Wilson CI, recall@k,
and Holm-adjusted-p fields `compare.py` now writes, and the per-arm
`results/{arm}_results.json` files it is computed from were never committed, so
nobody can regenerate or re-analyse the published run. (Holm on the committed
contrasts: 1.6e-5 / 0.163 / 0.689 — conclusions unchanged.)

**Mitigation in place:** `run_benchmark.py` now checkpoints atomically, supports
`--resume`, and records run metadata; notebook 02 can save results to Google
Drive so a Colab disconnect no longer loses them.

**Fix (needs GPU run):** on the next benchmark run, commit the per-arm JSONs
alongside the regenerated summary.

---

## 16. `graph_norr` arm has never been run — MEDIUM (open)

**What:** `--arm graph_norr` (parent expansion without the reranker — the exact
config the hosted demo serves) is wired into `run_benchmark.py`, `make
benchmark`, and notebook 02, but there are no results. The hosted demo's
accuracy is therefore unmeasured. **Fix:** include it in the next GPU run.

---

## 17. `FuzzyEvaluator` known mis-scoring modes — FIXED FOR NEW RUNS (historical behavior retained)

**What:** `extract_answer` takes the *first* yes/no/maybe match (a hedge like
"no evidence… so yes" scores `no`), matches without `` word boundaries (e.g.
"yes" inside a longer token), and relies on seeing an opening `<think>` tag to
strip reasoning. **Why unchanged:** any fix changes the published numbers
without a re-run to replace them. **Fix:** correct all three together on the
next benchmark re-run and note it in RESULTS.md.

---

## 18. ~~Lexical abstract section ordering in AQL and Cypher~~ — FIXED IN SOURCE

Originally both backends ordered chunk keys as strings, so `_10` came
before `_2`. The ArangoDB parent and concept-hop AQL, Neo4j parent and
concept-hop Cypher, and `FlatParentRetriever` now parse the **numeric section
suffix** and use a deterministic key tie-breaker. Graph-query integration
tests exercise 12-section parent and concept-hop abstracts on both engines.
The graph traversals also deduplicate repeated edges referring to the same
chunk before joining sections; the original database still needs cleanup.

**Historical benchmark:** `results/summary.json` is frozen. Correcting
section ordering is a new retrieval version, not retroactive proof of the
reported results. Rerun the full benchmark after a graph integrity audit.

---

## 19. Keep-warm workflow is disabled and never achieved its schedule — LOW (open, user-only)

**What:** GitHub disabled `.github/workflows/keep-warm.yml` for repository
inactivity, and even while enabled its cron never ran at the intended cadence
(GitHub schedules are best-effort). So the hosted backend cold-starts
(60-120s) far more often than the docs implied. **Fix:** re-enable the
workflow in the Actions tab, or use an external uptime pinger.

---

## 20. ~~No per-IP rate limiting on `/query` and `/mcp`~~ — FIXED

**What:** Both endpoints are unauthenticated. A request-body size cap and a
global concurrency limiter were added (protecting the 512MB instance from
overload), but one client can still consume the whole Gemini free-tier quota.
**Fixed:** `backend/main.py` now applies an in-memory per-client limit to POSTs
on both paths (default 10/min on `/query`, 40/min on `/mcp`; tune or disable
with `KGQA_RATE_LIMIT_PER_MIN`). The client is taken from `X-Forwarded-For`,
which a caller can spoof, so this stops casual abuse and runaway clients rather
than a determined attacker; the concurrency limiter still bounds total load.
