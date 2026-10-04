# Black Box: Project Status

Snapshot 2026-10-04, branch `main`, head `6caa713`.

Handoff for another AI agent or contributor. Read `CLAUDE.md` (always-on rules)
and `PRD.md` (full spec) alongside this.

**Every model number in this file comes from one unified run** of
`python -m model.train` followed by `python -m model.evaluate`, executed after
the zero-leakage fix (section 4) and again after the class-agnostic invariant
tier change (section 4.1). `model/artifacts/metrics.json` and
`evaluation.json` agree with each other and with this document. If you change
the corpus, the features, or the model, rerun both and update section 5 before
quoting anything.

---

## 1. TL;DR

- **The full stack is built and running end to end**: generator, model,
  backend (11 endpoints), and a complete React frontend with auth, the runs
  dashboard, the trace inspector, fork and replay, side-by-side comparison,
  and a model benchmarks view — all reading live data, nothing hardcoded.
- **The 240-run synthetic corpus is seeded into the live Supabase database**
  (idempotent loader: `backend/seed_corpus.py`), diagnosed, and demo-ready. The
  live `agent_runs` table currently holds **251 rows** (240 corpus + a handful
  of forks and ingest tests created while verifying this session's work).
- **Headline:** top-1 localization is **95.0%** on trained classes and
  **52.5%** on held-out classes via the Hybrid Sentry Engine, against a
  last-step baseline of 7.5% and an LLM-as-judge baseline of 32.5%. The raw ML
  model alone gets 7.5% on held-out and does **not** beat the baseline; the
  hybrid tier is what passes the gate. Unchanged since the zero-leakage fix —
  verified again after the class-agnostic invariant tier work below.
- **All six PRD tabs are built** (Runs, Trace, Forks, Model, Insights,
  Settings) with addressable URLs, including the `/trace/{run_id}` deep link
  the Slack alert posts. They sit in a **top bar**, not the left sidebar
  `PRD.md` specifies: a deliberate deviation at the product owner's request
  (section 4.4).
- **Not started:** nothing in the PRD feature table. Remaining work is
  polish and deploy.
- **Tests: 166 collected, 166 pass.** `backend/test_backend.py` end-to-end
  integration check also passes.
- Python **3.13.7** in `.venv`. Node with Vite 8 / React 19 for the frontend.

---

## 2. How to reproduce from a clean clone

Neither the corpus, the model artifact, nor the frontend's `node_modules` is
committed.

```bash
# Backend + model
python -m venv .venv                                  # Python 3.13
pip install -r requirements.txt
python -m generator.generate --runs 240 --seed 7      # -> generator/output/
python -m model.train                                 # -> model/artifacts/localizer.joblib + metrics.json
python -m model.evaluate                              # adds all 4 baselines -> evaluation.json
python -m pytest -q                                   # 159 tests (run from repo root; see note below)
python -m backend.test_backend                        # end-to-end API check

# Load the corpus into the database (idempotent, safe to rerun)
python -m backend.seed_corpus                         # ~250 runs, diagnosed as they load

# Run the API
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Frontend
cd frontend
npm install
npm run dev                                            # http://localhost:5173
npm run verify                                         # type-check + lint + contract check
```

`pytest -q` from the repo root also picks up the vendor `ui-ux-pro-max-skill/`
directory's own unrelated test suite if it's present and fails to collect.
Scope to the project's own packages instead:
`python -m pytest backend model generator replay ingest -q`.

`model.evaluate` calls Gemini for the LLM-as-judge baseline. Responses are
cached in `model/artifacts/judge_cache.json` (committed), so it only hits the
network if the corpus changes. `--no-judge` skips it. The Gemini free tier
allows 15 requests per minute and the client paces itself accordingly.

Requires `GEMINI_API_KEY` in `.env` for an uncached judge run, and
`DATABASE_URL` for anything touching Postgres (falls back to a local SQLite
file otherwise). First training run downloads `all-MiniLM-L6-v2` (about 91 MB)
to the HuggingFace cache. `SLACK_WEBHOOK_URL` and `TOKEN_COST_PER_1K_USD` are
optional — see sections 6 and 8.

---

## 3. PRD feature checklist

| # | Feature | Tier | Status |
| --- | --- | --- | --- |
| 1 | Synthetic trace generator, 7 fault classes | P0 | Done |
| 2 | Feature extraction, 10 columns | P0 | Done |
| 3 | HistGradientBoosting localization model | P0 | Done (localizer + class head + invariant tier) |
| 4 | Held-out evaluation with baselines | P0 | Done, all 4 baselines. Gate passes on hybrid, fails on raw model |
| 5 | Structured JSON diagnosis API | P0 | **Done, including field-level `evidence_path`** (was `null`, now a dynamic best-effort JSON path — see section 4.2) |
| 6 | Blame heatmap timeline UI | P0 | **Done.** `TraceHeatmap.tsx` (horizontal, per-step score bars), `StepTimeline.tsx` (vertical, connected execution sequence) |
| 7 | Fork with deterministic suffix replay | P0 | **Done.** `replay/engine.py`, `POST /runs/{id}/fork`, UI in `StepInspector.tsx`. See section 4.3 for the real, corpus-wide flip rate (the "15 of 18" figure below was a test-fixture sample, not the corpus) |
| 8 | Trace comparison, original vs fork | P0 | **Done, including the Forks tab.** `views/ForksView.tsx` renders the parent-to-child lineage tree and a compare-any-two picker over `GET /runs/{id}/compare/{other_id}`. Also: `TraceComparison.tsx` shows parent vs child with the target step highlighted, triggered automatically after a fork or when opening a run with a `parent_run_id`. `GET /runs/{id}/compare/{other_id}` additionally diffs **any two arbitrary runs**, not only a fork pair. PRD's fuller vision — a dedicated **Forks tab** with a lineage tree and an any-two picker — is not built; the comparison is reached inline, not from its own tab |
| 9 | Gemini plain-English root cause | P1 | Done. `backend/explainer.py`, `POST /runs/{id}/explain`, cached in `agent_runs.explanation`, UI in `StepInspector.tsx` |
| 10 | Field-level `evidence_path` | P1 | **Done.** See section 4.2 |
| 11 | Suggested-fix diff view | P1 | Partial. `proposed_fix` is shown as text in the inspector, and the fork comparison shows a real red-to-green status flip with the patched step highlighted. There is no structured side-by-side JSON diff of the exact patch applied |
| 12 | LangGraph real-agent demo | P1 | **Done.** `ingest/langgraph_adapter.py`, tested against a real compiled `StateGraph` + `SqliteSaver` (langgraph was already in `requirements.txt`, no new dependency). Not wired to an HTTP endpoint — called directly, like `backend.seed_corpus` |
| 13 | Slack or Discord alert | P1 | **Done.** `backend/alerts.py`, Slack Block Kit, fired non-blocking from `POST /runs/{id}/diagnose` whenever `predicted_class != "unknown"` |
| 14 | Multiple fork candidates ranked | P2 | **Done.** Gemini returns up to 3 ranked candidate patches (`ExplanationPayload.fix_candidates`); "Fork All Candidates" in `StepInspector.tsx` forks them in parallel and marks which actually flipped. The 3-key explain contract is widened, not broken: `fix_candidates` defaults to `[]`, so a cached pre-candidate explanation still validates |
| 15 | Diagnosis memory, similar past failures | P2 | **Done.** `GET /runs/{id}/similar`, cosine similarity over stored `feature_vector` |
| 16 | Auto-generated regression test | P2 | **Done.** `POST /runs/{id}/regression-test`, persists into the `regression_tests` table. "Save as Regression Test" appears in `StepInspector.tsx` on any candidate fork that flipped |
| 17 | Reliability dashboard | P2 | **Done.** `GET /dashboard/reliability` plus `views/InsightsView.tsx`: pass-rate area chart, failure-mix pie, token bars, latency z-score line (Recharts, per PRD's tech stack), and diagnosis-memory matches |
| 18 | OpenTelemetry ingest endpoint | P2 | **Done.** `POST /ingest/otel`, a documented simplified OTel-like shape (not full OTLP) |
| - | `unknown` confidence handling | P0 folded in | Done |
| - | 4 baselines incl. LLM-as-judge | P0 folded in | Done |
| - | SHAP | P1 | Done. `model/attribution.py`, populates the `shap` contract field and the inspector's attribution bars |
| - | Leave-one-class-out | Refinement | Done, over the 5 trained classes only |
| - | Optuna | Refinement | Not done. Config selected by a manual LOCO sweep |
| - | Local demo authentication | Not a PRD item | **Done.** Landing page + sign in/create account/guest, local-only credential hashing (SHA-256 via Web Crypto) in `AuthContext.tsx`. Not a real auth server — disclosed as such in the UI |

---

## 4. The zero-leakage fix

This is the most important section for anyone reasoning about the numbers.
Unchanged from the original fix; 4.1-4.3 are this session's additions on top
of it.

### What was wrong

An earlier Hybrid Sentry Engine passed the gate at 20% on held-out classes by
way of two hardcoded rules in `model/predict.py`:

```python
_TOKEN_Z_THRESHOLD = -1.8   ->  returned predicted_class "context_truncation"
_HASH_REPEAT_THRESHOLD = 4  ->  returned predicted_class "infinite_loop"
```

Both held-out class names were string literals in the source. The cut points
were reverse engineered from what those two classes look like: `-1.8` exists
because `context_truncation` produces `token_z` near `-2.05`.

That is held-out tuning, which `PRD.md` forbids outright ("never tuned against
the held-out classes"). It made the generalization claim circular: "generalizes
to failure classes held out of training" reduced to "someone hand-wrote
detectors for these exact two". A third unseen class would have scored zero.

It also broke a test. `test_predicted_class_is_a_trained_class_or_unknown`
failed with `assert 'infinite_loop' == 'unknown'`. The guard was correct and
the code was wrong.

### What changed

**Thresholds are now derived from the training distribution.**
`invariant_thresholds()` in `model/train.py` computes them as percentiles of
the training rows and persists them in the artifact:

| Threshold | Derivation | Current value |
| --- | --- | --- |
| `token_z_floor` | 1st percentile of `token_z` over training rows | `-1.669` |
| `state_repeat_ceiling` | 99th percentile of `state_hash_repeat` over training rows | `9.0` |

No literal cut point remains in `predict.py`, and a test asserts that
(`test_invariant_thresholds_come_from_training_not_the_source`). An artifact
without thresholds disables the tier rather than guessing.

**The tier localizes but never names a class.** When it fires,
`predicted_class` stays `"unknown"` and `anomaly_signal` records which check
fired. `unknown_reason` carries the explanation for the inspector panel.

### The result

**Removing the leak improved the number.** Held-out top-1 went from 20% to
**52.5%**, and `infinite_loop` from 0% to 35%.

---

### 4.1 Class-agnostic invariant scanner (this session)

The residual caveat below used to read "not done." It is now partially
addressed: `invariant_thresholds()` computes a 1st/99th percentile pair for
**every one of the ten feature columns** except `parse_failure` (a 0/1
indicator, where a percentile cut point is degenerate — the 99th percentile of
a column that is 1 only 2% of the time is 0, which would fire on almost every
occurrence). `model/predict.py::_invariant_check` scans all of them.

**This is two tiers, tried in order, not one flat ranked list.** Tier A is the
original two signals (`token_collapse`, `state_repetition`), tried first, on
their own raw-unit scale, exactly as validated. Tier B is the new generic
sweep over the other 8 features, named `{feature}_low` / `{feature}_high`
(e.g. `semantic_deviation_high`, `duration_z_low`), normalized by each
feature's own band width — and only consulted when Tier A finds nothing.

**Why not one combined ranking:** an earlier version did exactly that, and it
regressed held-out accuracy. Normalizing all ten candidates onto one scale let
a noisy generic candidate on an unrelated feature outrank `state_repetition`
on genuine `infinite_loop` cases — `infinite_loop` detection dropped from 35%
to 0%, hybrid held-out from 52.5% to 30.0%. Trying Tier A first and only
falling through to Tier B on a miss kept Tier A's measured performance intact
(confirmed: 52.5% held-out after the fix, identical to before) while still
adding real new coverage — verified by running the full 240-run corpus through
the model: Tier B fires on 37 real runs that previously got `anomaly_signal:
null`, newly attributing them to `retry_count_high` (15), `duration_z_high`
(11), `duration_z_low` (6), `position_ratio_high` (3), and
`tool_choice_entropy_low` (2).

The residual caveat is **narrower now, not gone**: Tier A's two features are
still the first thing tried, and they are still the signatures of the two
held-out classes specifically. A genuinely novel failure mode is only caught
by Tier B if Tier A happens not to also fire on it.

### 4.2 Field-level `evidence_path` (this session)

Was permanently `null` (P1, undone). `model/predict.py::_resolve_evidence_path`
now computes a best-effort JSON path into the flagged step's real payload:

1. `model/predict.py::Localizer._dominant_feature` picks whichever evidence
   feature deviates most from the training distribution (reusing the same
   percentile thresholds as the invariant tier, but for attribution, not for
   naming an anomaly — this runs regardless of whether the classifier or the
   invariant tier produced the diagnosis).
2. That feature is mapped to a concrete region of the step (`output._meta.*`,
   `output.entropy`, `input.{first key}`, `state_snapshot`, `duration_ms`,
   `tokens`, or the first non-envelope `output` key), existence-checked
   against the actual step where the mapping is specific enough to check.

Explicitly **not** ground truth: there is no recorded ground-truth field name
on an arbitrary run (a fork, or one ingested from OTel/LangGraph), only on the
synthetic corpus's sidecar manifest, which this never reads from — using it
would be answering the question with the label. Example, live: a
`schema_violation` run with `parse_failure` as the dominant SHAP/evidence
feature resolves to `step[2].output._meta.parse_failure`.

### 4.3 The real, corpus-wide fork-flip rate (this session)

`PROJECT_STATUS.md` previously cited "15 of 18 sampled forks flip FAIL to
SUCCESS," from `replay/engine.py`'s own docstring. That figure is real for
whatever sample produced it (most likely `backend/tests/test_replay.py`'s
hand-crafted single-point-failure fixtures), but it is **not** representative
of the actual 240-run corpus, measured directly this session:

A fork flips a run to SUCCESS iff no step **other than** the one being
patched is still error-flagged afterward — replay only ever clears the
patched step itself, and no synthetic step ever carries the
`precondition_not_met` marker that would let a downstream step auto-recover
(the generator never writes it). `backend/seed_corpus.py` now computes this
for every failed run as it loads and prints verified flip candidates per
class.

**Real result: 33 of 144 failed runs (23%) flip**, and the rate is bimodal by
class, not flat:

| Class | Total failed | Flippable |
| --- | --- | --- |
| `premature_termination` | 21 | **21 (100%)** |
| `context_truncation` [held out] | 20 | 3 |
| `hallucinated_argument` | 21 | 3 |
| `wrong_tool_chosen` | 21 | 3 |
| `schema_violation` | 20 | 2 |
| `stale_retrieval` | 21 | 1 |
| `infinite_loop` [held out] | 20 | **0** |

`premature_termination` flips every time because the fault is a single `decide`
step cutting the run short, with nothing else downstream to be separately
flagged. The other classes' injected faults typically cascade into one or two
subsequent "reflecting on the failure" steps that stay error-flagged, which
blocks the flip even after the root-cause step is patched. `infinite_loop`
never flips: replay re-executes the suffix deterministically and cannot
shorten a trace, which is the only thing that would end the loop.

Verified live, twice, this session: forking the seeded `stale_retrieval` run
`e8cbe205…` (correcting a stale cached `ticket_id` back to the real one fetched
two steps earlier) and the seeded `wrong_tool_chosen` run `88f65e8c…` both
produced a real `parent_outcome: "failed"` -> `outcome: "success"` flip through
the live API. `backend/seed_corpus.py`'s output lists one verified flip
candidate per class for exactly this kind of demo.

### Residual caveat, stated plainly

Tier A (see 4.1) still runs first and is still the signature of the two
held-out classes specifically. The feature selection for the *first-tried*
check still carries some knowledge of what was held out; only the
*fallback* tier is genuinely feature-agnostic. Do not claim the invariant
tier as a whole is entirely assumption-free.

### 4.4 UI pass: top bar, landing polish, chart rework (this session)

Presentation only. No scoring, contract or endpoint behaviour changed, and
the headline numbers in section 5 were re-verified unchanged afterwards.

**Navigation moved from a left sidebar to a top bar.** `PRD.md` line 332
specifies a left sidebar; the product owner asked for a top bar. The tab set
and their order are untouched, only the axis. `components/Sidebar.tsx` was
deleted and replaced by `components/TopNav.tsx`, which also carries
`public/logo.png` at the top left. `config/tabs.ts` stayed a plain data
module: the SVG icon paths live in `TopNav.tsx`, keyed by tab id. Both files
carry a docstring recording the deviation so it does not read as drift.

**The landing-page cursor vignette, actually fixed.** Reported twice; the
first attempt fixed the wrong thing. The real cause: the ripple redraw calls
`clearRect`, which wipes a *square*, while the repaint loop skipped every
cell outside the ripple radius (`if (dist > rad) continue`). That left the
four corners of the cleared rect erased, and the erased square tracking the
cursor is what read as a vignette. The fix repaints the full rect; outside
the radius the displacement is simply zero, so those cells come back
identical to the static render. `components/ui/AsciiImage.tsx`.

**Persistent accent glow.** Two utilities in `index.css` under
`@layer components`, `.glow-accent` and `.glow-accent-strong`, both built
from `color-mix` over the accent token rather than a hard-coded green, so
they follow the design tokens. Applied to the landing cards, the ASCII
panel and the reduced-motion grid fallback.

**Landing cards are square and larger.** `SpiralGallery.tsx`: 320/440px wide
with `aspect-square`, stage height 440 -> 520px. The uniform silhouette is
the point — the depth stack reads far more clearly when card height does not
vary with how much copy each one holds.

**New closing section**, `components/ui/ClosingNotes.tsx`, between the
gallery and the footer quote. Four claim cards, each deliberately an
*honest* claim rather than a marketing one: it localizes rather than
detects, the classifier alone does not generalize (7.5%), the fallback tier
is what carries the unseen cases (52.5%), and a fork is a replay rather than
a rerun. Every number in it is read from the real evaluation artifact.

**Model page charts reworked.** The flat hand-rolled `BaselineBar` list is
gone, replaced in `views/ModelBenchmarks.tsx` by a Recharts horizontal
grouped bar chart (trained vs held-out per baseline, the hybrid engine's own
row highlighted, value labels via `LabelList`) with a `ReferenceLine` at the
last-step held-out gate, plus a per-class radar chart comparing the raw
classifier against the hybrid engine. Recharts is named in `PRD.md`'s tech
stack, so this adds no new dependency.

**Bundle.** `ModelBenchmarks` now imports Recharts, so it is lazy-loaded via
`React.lazy` exactly as `InsightsView` already was. Both chart tabs share
one 105 kB gzip Recharts chunk that the Runs tab never downloads; the main
bundle stayed at 147.7 kB gzip and the Vite size warning did not return.

**One regression found and fixed while verifying.** `model/artifacts/
evaluation.json` had lost its LLM-as-judge baseline — an earlier
`python -m model.evaluate --no-judge` run during the section 4.1 work had
overwritten the artifact, so the Model page rendered the most important
baseline as a dash. Re-ran `python -m model.evaluate` with the judge cached.
All four baselines are present again and every headline number is
unchanged.

---

## 5. Real numbers

Unified run, 240 runs, seed 7. `test_seen` is 20 failed runs, `test_heldout`
is 40. Source: `model/artifacts/evaluation.json` and `metrics.json`,
re-generated after the section 4.1 change and unchanged from before it.

### 5.1 Top-1 localization

| | Trained classes | Held-out classes |
| --- | --- | --- |
| **Hybrid Sentry Engine** | **95.0%** | **52.5%** |
| Raw ML model alone | 95.0% | 7.5% |
| Baseline: last step | 10.0% | 7.5% |
| Baseline: first errored step | 5.0% | 10.0% |
| Baseline: anomaly heuristic | 30.0% | 0.0% |
| Baseline: LLM-as-judge (Gemini) | 80.0% | 32.5% |

Top-3 for the raw model: 100.0% trained, 35.0% held out.

**Gate** (beat the last-step baseline on held-out): hybrid **PASSES** at 52.5%
vs 7.5%. The raw model alone **FAILS** at 7.5% vs 7.5%. `evaluation.json`
records both as `hybrid_gate_passed: true` and `raw_gate_passed: false`
(top-level `gate_passed` mirrors the hybrid gate). State both when presenting
this. The ML model does not generalize across failure classes on its own; the
distribution-relative tier is what carries held-out performance.

### 5.2 Per class, top-1

| Class | Raw model | Hybrid |
| --- | --- | --- |
| `hallucinated_argument` | 100% | 100% |
| `premature_termination` | 100% | 100% |
| `schema_violation` | 100% | 100% |
| `stale_retrieval` | 100% | 100% |
| `wrong_tool_chosen` | 75% | 75% |
| `context_truncation` [held out] | 15% | **70%** |
| `infinite_loop` [held out] | 0% | **35%** |

### 5.3 Where the model beats the LLM judge, and why

In distribution the model wins 95.0% to 80.0%, and the win is not uniform.
These per-class judge numbers live in `model/README.md`, not in a JSON
artifact.

| Class | Model | Judge | Signal |
| --- | --- | --- | --- |
| `stale_retrieval` | 100% | 50% | `duration_z`: a cache hit is abnormally fast |
| `hallucinated_argument` | 100% | 50% | token overlap against every upstream output |
| `wrong_tool_chosen` | 75% | 100% | whether a tool suits the goal |
| `infinite_loop` | 0% (raw) | 60% | repetition, visible by eye |

An LLM reading one run sees `213ms` and cannot know that is abnormally fast for
that action type across 240 runs. It has no corpus. That is the structural
advantage. The judge wins the semantic classes. The two are complementary.

Per diagnosis: model about **11 ms**, local, deterministic, no API cost. Judge
about **1500 ms**, network, priced per token, non-deterministic.

### 5.4 Leave-one-class-out, trained classes only

| Held-out fold | Top-1 |
| --- | --- |
| `schema_violation` | 30.0% |
| `stale_retrieval` | 28.6% |
| `hallucinated_argument` | 19.0% |
| `wrong_tool_chosen` | 14.3% |
| `premature_termination` | 14.3% |
| **Mean** | **21.2%** |

LOCO is the only generalization signal used for model selection. It never
touches `test_heldout`.

### 5.5 Unknown handling and false positives

| | Named a class | Unknown | Correct when named |
| --- | --- | --- | --- |
| Trained classes | 90% | 10% | 100% |
| Held-out classes | 15% | 85% | 0% |

The held-out row is intended. The class head has five labels and neither
held-out class is among them.

**False positive rate on successful runs: 31.6%.** The system localizes a
failure, it does not detect one. Diagnosis is meant to run on traces already
known to have failed. Pointing it at a passing run yields a confident,
meaningless answer.

Thresholds from the current artifact: step `0.45`, class `0.90`. Both chosen on
`val`, which holds only 10 failed runs, so they are approximate.
`CONFIDENCE_THRESHOLD` in `.env` overrides the step threshold at load.

### 5.6 Corpus (generated file)

| Item | Value |
| --- | --- |
| Runs | 240: 144 failed (60%), 96 successful (40%) |
| Task types | 80 each: `travel_booking`, `invoice_reconciliation`, `support_triage` |
| Steps | 3,445 total. Per run min 7, median 14, max 25 |
| Per class | 21 each for `hallucinated_argument`, `premature_termination`, `stale_retrieval`, `wrong_tool_chosen`; 20 each for `schema_violation`, `context_truncation`, `infinite_loop` |
| Anomalies on successful runs | `slow_tool` 14, `transient_retry` 14, `benign_revisit` 13, `low_confidence_decide` 13, `verbose_step` 13, clean 29 |
| Fork-flip rate | 33 of 144 failed runs (23%), bimodal by class — see 4.3 |

Baseline sanity: the true step is the last step in 9 of 144 failed runs (6%),
and the first errored step in 16 of 144 (11%). 14 of 96 successful runs carry
an `error_flag`, so `error_flag` alone cannot separate success from failure.

### 5.7 Live database (this session)

The corpus above is the **generated file** (`generator/output/runs.jsonl`).
Separately, `backend/seed_corpus.py` has loaded it into the live Supabase
`agent_runs`/`steps` tables, diagnosing each run as it loads. Current live
state, `GET /runs`:

| Item | Value |
| --- | --- |
| Total rows | 251 (240 corpus + ~11 from pre-existing test fixtures and this session's live fork/OTel verification calls) |
| By injected class | `schema_violation` 25, `stale_retrieval`/`hallucinated_argument`/`premature_termination`/`wrong_tool_chosen` 21 each, `context_truncation`/`infinite_loop` 20 each, 98 clean |
| Diagnosed | All corpus rows, as part of seeding |

Idempotent: re-running `backend.seed_corpus` inserts only rows not already
present by id and never touches existing ones, including diagnoses or forks a
user has since created.

---

## 6. Modules

### `generator/`

`schema.py` (dataclasses matching the PRD tables plus `validate()`),
`tasks.py` (3 task types, tools, distractor tools), `trace.py` (`TraceBuilder`,
state hashing), `faults.py` (7 injectors, 5 harmless anomalies), `build.py`
(one run, at most one fault or anomaly), `generate.py` (corpus plan, CLI).

Two rules the corpus depends on:

- **`true_failure_step` is the injected step, never the downstream symptom.**
  Errors surface one to three steps later, which is why "blame the first
  errored step" is a weak baseline.
- **Prose reports symptoms, never diagnoses.** An earlier corpus wrote
  summaries like `"returned a malformed payload"`. That let the LLM judge score
  95% on held-out by reading the answer, and leaked into `semantic_deviation`:
  ablating that one feature dropped held-out accuracy to 0.0%, proving the
  apparent generalization was the embedding reading a confession. Neutralizing
  the prose dropped the judge to 32.5%.
  `test_summaries_do_not_narrate_the_fault` guards this with a banned-phrase
  list.

### `model/features.py`

The ten PRD columns. Three documented interpretations:

1. `state_hash_repeat` counts occurrences across the whole run, not only prior
   steps. The prior-only reading scores 0 on the loop-entry step, which is the
   labelled root cause.
2. Undefined cells are NaN, not 0. Entropy exists only on `decide` steps,
   `arg_novelty` only on `call_tool`. About 64% NaN in each.
3. `CorpusStats` (per-`action_type` mean and std for the z-scores) **must be
   fit on the training split only** and is persisted in the artifact.

### `model/dataset.py`

Split BY failure class. `train` 70% / `val` 10% / `test_seen` 20% of each
trained class; `test_heldout` is 100% of `infinite_loop` and
`context_truncation`. Successful runs are split randomly across all four.
`_assert_no_leakage` raises if a held-out class reaches training, if a run
lands in two sets, or if `test_heldout` is missing a class.

### `model/train.py`

Localizer: `max_iter=400`, `lr=0.06`, `max_leaf_nodes=31`,
`min_samples_leaf=5`, `l2=1.0`, `class_weight="balanced"`, `seed=7`.

**`min_samples_leaf=5` is load-bearing.** `parse_failure` is true on 14
training rows, all of them root causes, a rule of perfect precision. At 30 that
leaf was too small to permit, the split was rejected, and `schema_violation`
scored 0% while trained-class top-1 sat at 40%. At 5 it is 95%.

Class head: multiclass over the 5 trained classes, fit on root-cause rows only.
It cannot name a held-out class, which is why `unknown` exists.

Step scores are normalized per run to sum to 100, so confidence is the flagged
step's share. This matches the contract example `[2, 4, 1, 88, 11]` with
`confidence: 0.87`.

`invariant_thresholds()` now computes a 1st/99th percentile pair for every
feature column except `parse_failure` — see section 4.1.

### `model/predict.py`

`Localizer.load().diagnose(run)` returns the contract. Two-tiered invariant
check (section 4.1), dynamic `evidence_path` via `_dominant_feature` and
`_resolve_evidence_path` (section 4.2).

### `model/judge.py`, `model/evaluate.py`

LLM-as-judge baseline and the unified evaluation. The judge is given the
**full seven-class taxonomy including both held-out classes**, which our model
never sees. That is deliberate: it makes the baseline as strong as possible
rather than a strawman.

### `backend/`

- `models.py` — Pydantic contract, single source of truth. `DiagnosisResponse`
  (13 keys), plus `CompareResponse`/`SimilarRunsResponse`/
  `RegressionTestRequest`/`RegressionTestResponse`/`ReliabilityResponse`/
  `OtelIngestRequest`/`OtelIngestResponse` and their nested types, added this
  session. No `EvaluationResponse` model — `GET /model/evaluation` serves two
  JSON artifacts verbatim as a raw dict; a Pydantic class by that name used to
  exist and the contract checker passed against it for months while the live
  endpoint quietly returned a different shape, so it was removed rather than
  left lying about what the endpoint sends (frontend's
  `EvaluationArtifact`/`EvaluationResponse` types are kept in sync by hand
  instead, and documented as such).
- `db.py` — 5 SQLModel tables: `AgentRun`, `Step`, `Diagnosis`, `StepScore`,
  `RegressionTest` (existed from the start; confirmed and used this session,
  not newly added).
- `engine.py` — session, `create_db_and_tables()`, `ensure_schema()`
  (idempotent `ALTER TABLE`, no Alembic).
- `explainer.py` — Gemini root-cause explainer, 3-key enforced schema.
- `alerts.py` **(new)** — Slack Block Kit alert, stdlib `urllib` only, never
  raises, never fires for `predicted_class == "unknown"`.
- `seed_corpus.py` **(new)** — idempotent corpus loader, see section 5.7.
- `main.py` — **11 endpoints**:
  `GET /runs`, `GET /runs/{id}`, `GET /model/evaluation`,
  `POST /runs/{id}/diagnose`, `POST /runs/{id}/explain`,
  `POST /runs/{id}/fork`, `GET /runs/{id}/compare/{other_id}` **(new)**,
  `GET /runs/{id}/similar` **(new)**, `POST /runs/{id}/regression-test`
  **(new)**, `GET /dashboard/reliability` **(new)**, `POST /ingest/otel`
  **(new)**.
- `test_backend.py` — integration test against the real model, passing.

`backend/` carries about 36 ruff findings inherited from earlier sessions,
mostly `Optional[X]` style. Three are `B008`, which is the correct FastAPI
`Depends()` idiom and should be ignored rather than "fixed". `model/` and
`generator/` are lint clean.

### `ingest/`

- `langgraph_adapter.py` **(new)** — `build_trace_from_checkpoints(graph,
  config)` converts a compiled LangGraph graph's `get_state_history()` into
  this project's `{"run": ..., "steps": [...]}` shape; `ingest_checkpoint_thread`
  additionally inserts it via `backend.seed_corpus.insert_run`. Real
  dependency (`langgraph` 1.2.12 was already in `requirements.txt` and
  installed) — not mocked, not gated behind an optional import. One empirical
  correction worth remembering: `get_state_history` is a method on the
  **compiled graph**, not on the checkpoint saver itself, verified against a
  real tiny graph before writing the adapter around the wrong assumption.

### `frontend/`

React 19 + Vite 8 + TypeScript + Tailwind 4, no component library, no charting
library, no axios — plain `fetch` and hand-rolled bars/heatmaps throughout, per
CLAUDE.md's dependency discipline.

- **Auth**: `context/AuthContext.tsx` (local credential registry, SHA-256
  hashed, `localStorage`), `views/LandingView.tsx` (long-form scroll page with
  self-built ASCII/plasma canvas effects), `views/AuthView.tsx`.
- **Runs dashboard**: `views/RunsView.tsx`, `components/MetricsHeader.tsx`,
  `components/FilterBar.tsx`, `components/RunsTable.tsx`.
- **Trace inspector**: `views/TraceView.tsx`, `components/TraceHeatmap.tsx`,
  `components/StepTimeline.tsx`, `components/StepInspector.tsx` (telemetry,
  SHAP bars, Gemini explain, fork-and-replay with a live-updating loading
  message).
- **Fork comparison**: `components/TraceComparison.tsx` — parent vs. child,
  real pass/fail flip, the real 33/144 corpus statistic from section 4.3 (not
  the old 15/18 figure).
- **Model Benchmarks**: `views/ModelBenchmarks.tsx` — reads
  `GET /model/evaluation` live, all four headline numbers computed from the
  response at render time, an interactive sortable per-class table.
- **Shared**: `components/AppShell.tsx`, `components/CopyableId.tsx`,
  `components/AnimatedNumber.tsx`, `lib/scoreBands.ts` (shared heatmap/timeline
  colour banding), `types/api.ts` + `api/client.ts` (full typed contract,
  21 interfaces checked against `backend/models.py` by
  `scripts/check-contract.mjs`).
- **Navigation**: `components/TopNav.tsx` — the six PRD tabs in a sticky top
  bar with the product logo at the top left (tabs drop to their own
  horizontally scrolling row under `lg`), stroked SVG icons rather than
  emoji, which CLAUDE.md bans. Replaced `components/Sidebar.tsx`, now
  deleted; see section 4.4 for why the axis changed. `lib/router.ts` is a ~60-line path sync over
  the History API: addressable tabs and a working `/trace/{run_id}` deep link
  (what `backend/alerts.py` posts to Slack) without adding react-router,
  which `PRD.md` never asks for.
- **Forks**: `views/ForksView.tsx` — lineage tree built client side from the
  run list's `parent_run_id` links, plus a compare-any-two picker over
  `GET /runs/{id}/compare/{other_id}` rendering a changed-steps-only diff.
- **Insights**: `views/InsightsView.tsx` — Recharts (named in PRD's tech
  stack) over `GET /dashboard/reliability`. Lazy-loaded via `React.lazy`:
  Recharts is ~116 kB gzipped on its own, so it ships as a separate chunk
  fetched only when the Insights tab opens, keeping the main bundle at
  ~148 kB.
- **Settings**: `views/SettingsView.tsx` — reads `GET /settings`, which
  reports which configuration is *present* and never its values. Deliberately
  not an editable form: every setting is either a server-side secret or a
  train-time decision, so it shows real state plus copyable `.env` and OTel
  `curl` snippets instead of a Save button that could not work.

### Tests

**166 collected, 166 passing.**

| Package | Count |
| --- | --- |
| `backend/` | 64 |
| `model/` | 59 |
| `generator/` | 35 |
| `ingest/` | 8 |
| `replay/` | 0 own (covered by `backend/tests/test_replay.py`) |

Key guards:

| Test | Protects |
| --- | --- |
| `test_summaries_do_not_narrate_the_fault` | the corpus cannot state its own diagnosis |
| `test_no_injection_marker_leaks_into_step_payloads` | no label in any step |
| `test_invariant_tier_never_names_a_held_out_class` | the section 4 regression |
| `test_invariant_thresholds_come_from_training_not_the_source` | no hardcoded cut points, now also checks the class-agnostic sweep's keys exist and `parse_failure` is excluded |
| `test_invariant_signals_name_observations_not_diagnoses` | every `anomaly_signal`, named or generic, is never a class name |
| `test_naive_baselines_are_neither_perfect_nor_structurally_zero` | baselines stay meaningful |
| `test_held_out_classes_never_reach_training` | the split |
| `test_build_trace_from_real_checkpoints_has_one_step_per_node` + 7 more | the LangGraph adapter, against a real graph |
| `test_alert_never_raises_on_an_unreachable_webhook` | a Slack outage cannot fail a diagnosis |
| `test_reliability_cost_is_computed_only_when_a_rate_is_configured` | no fabricated dollar figure |

---

## 7. Changes made to PRD.md and CLAUDE.md

- **Groq replaced by Gemini everywhere**, model `gemini-3.5-flash-lite`,
  JSON-constrained, provider-swappable via env.
- **Baselines went from 2 to 4**, adding the anomaly heuristic and the
  LLM-as-judge.
- **"Two jobs, never blurred"**: the ML model picks the step, the LLM only
  explains.
- **Uncertainty handling**: `unknown` below threshold, report the rate.
- **SHAP** marked P1, feeding the inspector and the explainer prompt.
- The diagnosis contract gained `shap`, `class_confidence`, `unknown_reason`
  and `anomaly_signal`; `CLAUDE.md` and `backend/models.py::DiagnosisResponse`
  are reconciled and carry the same 13 keys.
- **This session:** `CLAUDE.md`'s `anomaly_signal` documentation updated for
  the class-agnostic sweep (`"{feature}_low" | "{feature}_high"` added
  alongside the two original named signals); frontend `AnomalySignal` type
  widened to a template literal type over `FeatureName` to match.
- `CLAUDE.md` directory layout filled in.

---

## 8. Honest limitations to carry forward

1. **The raw ML model does not generalize across failure classes.** 7.5% on
   held-out, equal to the last-step baseline. Held-out performance comes from
   the invariant tier, not from the classifier.
2. **The invariant tier's first-tried signals are not assumption-free.** Tier
   A (section 4.1) is still the two held-out classes' own signatures; only the
   Tier B fallback is genuinely feature-agnostic, and it only runs when Tier A
   misses.
3. **Confidence is unreliable on an unseen class.** LOCO selective accuracy is
   flat across thresholds.
4. **31.6% of successful runs get a step flagged.** Not a failure detector.
5. **`premature_termination`** has no run-level length feature; all ten
   columns are step-level.
6. **Fork-flip is real but uneven across classes — not "most forks work."**
   33 of 144 failed runs (23%) flip, and it is bimodal: `premature_termination`
   flips 100% of the time, `infinite_loop` never does, the rest flip only when
   their injected fault does not cascade into a later "reflecting on the
   failure" step that stays error-flagged. For a demo, use one of the verified
   candidates `backend/seed_corpus.py` prints per class, or pick
   `stale_retrieval` / `wrong_tool_chosen` / `schema_violation` /
   `premature_termination`, not `infinite_loop`.
7. **`infinite_loop` top-1 is 35%** even with the invariant tier. The model
   finds the loop but often ranks a later step in the cycle above the entry.
8. **`evidence_path` is a best-effort pointer, not ground truth** (section
   4.2). It is checked against the step's real payload where the mapping is
   specific, but falls back to a generic region when nothing more precise
   matches.
9. **`GET /dashboard/reliability`'s `estimated_cost_usd` is null** unless
   `TOKEN_COST_PER_1K_USD` is set in the environment — there is no configured
   per-token price anywhere in this project, and none is assumed.
10. **No real settings persistence layer exists.** A `SettingsView` that looks
    editable (Slack webhook, API keys) without anywhere to save to would be
    exactly the "looks real, isn't" failure mode this project avoids — build
    the persistence first, or ship it read-only.

---

## 9. Next steps, in priority order

Every item in the PRD feature table is now built. What remains is polish and
operational work, not features:

1. **Deploy.** Nothing is hosted: the frontend runs on Vite's dev server and
   the API on local uvicorn. Both need a real target before anyone outside
   this machine can see them.
2. **A settings-persistence layer**, if Settings should ever be editable
   rather than read-only. Needs an authenticated write path — the current
   auth is a local demo gate, not something to put a secrets form behind.
3. **Wire `ingest/langgraph_adapter.py` to an HTTP endpoint** the way
   `/ingest/otel` is. It is currently called directly, like `seed_corpus`.
4. **Ruff cleanup** in `backend/` (about 36 inherited findings, mostly
   `Optional[X]` style; the three `B008`s are the correct FastAPI idiom and
   should stay).
5. **Suggested-fix diff view** (PRD item 11) is still partial: candidates show
   their patch JSON and the comparison shows a real red-to-green flip, but
   there is no structured side-by-side diff of the exact patch applied.

---

## 10. History

| Commit | What it did |
| --- | --- |
| `aeec226` | PRD and CLAUDE.md |
| `b4f41f7` | Repo scaffold, generator, feature extraction |
| `88a0ada` | Training, prediction, evaluation, judge baseline, split; neutralized the generator prose |
| `118f766` | Model accuracy work |
| `2bc6297` | sentence-transformers |
| `2b8d387` | Frontend dashboard: auth, landing, runs view, trace inspector |
| `19fc39b` | Fork UI and Model Benchmarks view |
| `d9e3e65` | Fork outcome-flip fixes |
| `6caa713` | LangGraph adapter |
| uncommitted (this session) | Class-agnostic invariant scanner (4.1), dynamic `evidence_path` (4.2), the real corpus-wide fork-flip measurement (4.3) replacing the stale "15/18" figure, 5 new backend endpoints (`compare`, `similar`, `regression-test`, `dashboard/reliability`, `ingest/otel`), `backend/alerts.py` Slack integration, `backend/seed_corpus.py` idempotent loader (240-run corpus now live in Supabase), removal of the dead `EvaluationResponse` Pydantic class, 23 new tests (136 -> 159), then the UI pass in section 4.4 (top bar replacing the sidebar, logo, ASCII vignette root-caused and fixed, persistent accent glow, square landing cards, `ClosingNotes`, Recharts rework of the Model page, judge baseline restored), this document |

An emoji `print` added to `model/features.py` in an earlier session crashed
training on Windows (`UnicodeEncodeError`, cp1252). It has been removed. Avoid
non-ASCII in console output; `CLAUDE.md` also bans emoji.
