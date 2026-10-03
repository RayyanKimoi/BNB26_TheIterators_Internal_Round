# Synthetic trace generator

Produces the labelled corpus the localization model trains on. Spec: the
Failure Taxonomy and Data Model sections of `PRD.md`.

```bash
python -m generator.generate --runs 240 --seed 7
python -m pytest generator/tests -q
```

Writes to `generator/output/` (gitignored):

| File | Contents |
| --- | --- |
| `runs.jsonl` | One JSON object per line: `{"run": {...}, "steps": [...]}`, schema-exact against `agent_runs` and `steps` |
| `manifest.json` | Generation provenance: seed, counts, and per-run `anomaly` and `evidence_path` labels |

`manifest.json` is generator-side only. Feature extraction reads `runs.jsonl`
and nothing else, so nothing in the manifest can leak into training.

## Ground-truth policy

Read this before trusting an evaluation number.

**`true_failure_step` is the injected step, never a downstream symptom.** Which
step kind holds the fault depends on the class:

| Class | Anchored on | Evidence path example |
| --- | --- | --- |
| `wrong_tool_chosen` | the `decide` step | `step[7].output.chosen_tool` |
| `premature_termination` | the `decide` step | `step[9].output.chosen_tool` |
| `infinite_loop` | the first repeated `decide` | `step[6].state_snapshot` |
| `context_truncation` | the `call_llm` step | `step[8].output._meta.dropped_messages` |
| `hallucinated_argument` | the `call_tool` step | `step[11].input.flight_no` |
| `stale_retrieval` | the `call_tool` step | `step[11].output.quoted_at` |
| `schema_violation` | the `call_tool` step | `step[5].output.price` |

**The injected step is usually clean on its own.** Errors surface one to three
steps later. That is the point: it is why reading a log top-down finds the
symptom rather than the cause, and why "blame the first errored step" is a weak
baseline. The error surfaces on the injected step only about 20 percent of the
time, and only for the three classes whose root cause is the tool call itself.

**Neither naive baseline is rigged.** On seed 7, 240 runs, the true step is the
last step in 6 percent of failed runs and the first errored step in 11 percent.
Both numbers are deliberately non-zero. A fault on the terminal tool call
aborts the run outright 60 percent of the time, with no closing answer step,
which is what keeps the last-step baseline above zero. If either baseline ever
reads 0 percent, the corpus has drifted and the comparison is worthless;
`test_naive_baselines_are_neither_perfect_nor_structurally_zero` guards this.

**No step payload carries the label.** No marker field is written anywhere in
`input`, `output`, or `state_snapshot`. The class is recoverable only from the
signature. `test_no_injection_marker_leaks_into_step_payloads` guards this.

## Corpus shape

240 runs, seed 7, deterministic:

- 144 failed (60 percent), spread evenly across the seven classes, 20 or 21 each
- 96 successful (40 percent), 80 across all three task types
- 67 of the successful runs carry one harmless anomaly, 29 run clean
- 7 to 25 steps per run, median 14, 3,445 steps total

The three task types are `travel_booking`, `invoice_reconciliation`, and
`support_triage`, 80 runs each. Every failure class appears in all three, so no
class correlates with one task.

## Why successful runs carry anomalies

Without them the model learns "any anomaly equals failure" instead of what
separates a survivable anomaly from a fatal one. Five anomalies, each the
benign twin of a real signature:

| Anomaly | Mimics | Why the run still succeeds |
| --- | --- | --- |
| `slow_tool` | a `duration_z` outlier | slow, but it returned the right answer |
| `transient_retry` | `retry_count` plus `error_flag` | first attempt timed out, the retry worked |
| `verbose_step` | a `token_z` spike | a long reflection, not a truncation |
| `benign_revisit` | `state_hash_repeat` | one confirming re-check, then progress |
| `low_confidence_decide` | a `tool_choice_entropy` spike | uncertain, but it still picked correctly |

`transient_retry` matters most: 14 of the 96 successful runs carry an
`error_flag`, so `error_flag` alone cannot separate success from failure.

## Features this corpus is built to support

Every column in the PRD's feature table has a source here. The generator emits
raw observables, not features; `model/features.py` derives the ten columns.

| Feature | Where it comes from |
| --- | --- |
| `duration_z`, `token_z` | `steps.duration_ms`, `steps.tokens`, normalized per `action_type` across the corpus |
| `retry_count`, `parse_failure` | `output._meta` |
| `tool_choice_entropy` | `output.candidates` on `decide` steps |
| `semantic_deviation` | embedding of `output.summary` against the run's goal text |
| `arg_novelty` | `input` values against all upstream `output` values |
| `state_hash_repeat` | `steps.state_hash` |
| `downstream_error_count` | `steps.error_flag` |
| `position_ratio` | `step_index` over step count |

Tool arguments are sourced from upstream state wherever the task allows, which
is what makes `arg_novelty` meaningful: on a healthy run nearly every argument
has already appeared in an earlier output, and on a `hallucinated_argument` run
it has not.

## Layout

| File | Role |
| --- | --- |
| `schema.py` | `Run`, `Step`, `Trace`, the class lists, and `validate()` |
| `tasks.py` | the three task types, their tools and plausible distractor tools |
| `trace.py` | `TraceBuilder`: appends steps, carries state, hashes it |
| `faults.py` | the seven injectors and the five harmless anomalies |
| `build.py` | runs one plan, injects at most one fault or one anomaly |
| `generate.py` | corpus planning, JSONL and manifest output, CLI |

Every trace passes `validate()` at build time, so a malformed trace fails at
generation rather than at hour 20.
