# Feature extraction

Implements the ten per-step columns from the Model Specification section of
`PRD.md`. Nothing is trained here.

```bash
python -m model.features                      # separation report on the corpus
python -m model.features --out features.csv   # dump the matrix
python -m pytest model/tests -q
```

```python
from model.features import CorpusStats, FeatureExtractor, load_runs

runs = load_runs("generator/output/runs.jsonl")
stats = CorpusStats.fit(train_runs)           # TRAINING SPLIT ONLY
X = FeatureExtractor(stats).transform(one_run)        # exactly 10 columns
frame = FeatureExtractor(stats).transform_many(runs)  # + ids and labels
```

`transform()` returns the ten feature columns and nothing else, in PRD order.
`transform_many()` appends `run_id`, `step_index`, `task_type`,
`injected_class` and `is_root_cause` as separate columns, so the feature matrix
can never accidentally carry the answer.

## Three decisions a reviewer will ask about

### 1. `state_hash_repeat` counts the whole run, not just prior steps

The PRD says "times this state hash was already seen in the run". Read as
*prior* occurrences, the feature scores **0 on the step that opens a loop** and
peaks on the last step of it:

```
idx  action      prior-seen   whole-run   hash
  9  decide               0           8   7f208e4e   <<< true_failure_step
 11  decide               2           8   7f208e4e
 17  call_llm             8           8   7f208e4e
```

`true_failure_step` for `infinite_loop` is the loop entry, because that is the
step a developer must fork from. The prior-occurrence reading therefore points
the feature at the symptom and leaves the labelled cause invisible, which would
cripple one of the two held-out classes and the headline generalization number
with it.

Counting occurrences elsewhere in the run fixes this, and is consistent with
`downstream_error_count`, which the PRD already defines as forward-looking.
Diagnosis always runs on a finished trace, so the information is available.
Measured effect on the root-cause step of an `infinite_loop` run: **0.35 under
the prior reading, 10.55 under this one**, against 1.72 on steps from healthy
runs.

### 2. Undefined features are NaN, not 0.0

`tool_choice_entropy` exists only on `decide` steps and `arg_novelty` only on
`call_tool` steps. Those cells are NaN on other step kinds. Zero entropy means
"perfectly certain", which is a different claim from "no decision was made
here", and filling with zero would teach the model that every tool call was a
confident decision. HistGradientBoosting handles NaN natively and branches on
missingness. About 64 percent of rows are NaN in each of those two columns,
which is simply the share of steps of the other kinds.

### 3. `CorpusStats` must be fit on the training split alone

`duration_z` and `token_z` are measured against the same `action_type` across
the corpus, so they need corpus statistics. Fitting those over the held-out
classes would leak distributional information about classes the model is
supposed to have never seen. `CorpusStats.fit()` therefore takes an explicit
set of runs, and `to_dict()`/`from_dict()` persist it beside the model so a
single run can be scored at inference without a corpus present.

**The `python -m model.features` CLI fits on every run.** That is correct for a
descriptive report and wrong for training. The training script must split
first, then fit.

## Separation on the current corpus

240 runs, seed 7, 3,445 steps, 144 root-cause rows. Mean value on the
root-cause step of each class, against the healthy-run baseline:

| Class | Fires on | Root cause | Steps in healthy runs |
| --- | --- | --- | --- |
| `schema_violation` | `parse_failure` | 1.00 | 0.00 |
| `infinite_loop` | `state_hash_repeat` | 10.55 | 1.72 |
| `context_truncation` | `token_z` | -2.05 | 0.05 |
| `wrong_tool_chosen` | `tool_choice_entropy` | 1.00 | 0.41 |
| `hallucinated_argument` | `arg_novelty` | 0.65 | 0.16 |
| `stale_retrieval` | `duration_z` | -0.94 | 0.05 |
| `premature_termination` | `position_ratio` + `semantic_deviation` | 0.78 / 0.87 | 0.47 / 0.59 |

Every class has at least one column that moves. These are descriptive
separations, not accuracy: no model has been trained.

## Known gap

`premature_termination`'s taxonomy signature is "step count below class
median", a run-level property. All ten columns are step-level, and
`position_ratio` normalizes run length away, so the run-length signal itself
is not available to the model.

It is the weakest-separated class. What remains is a late position (0.78
against 0.47) and high semantic deviation (0.87 against 0.59) versus healthy
steps. `downstream_error_count` of 0.00 separates it cleanly from the other
six failure classes, which run 1.50 to 3.00, but not from healthy steps, which
average 0.07. Worth watching if per-class accuracy comes back weak here.

## Feature sources

| Column | Derived from |
| --- | --- |
| `duration_z` | `steps.duration_ms` against `CorpusStats.duration[action_type]` |
| `token_z` | `steps.tokens` against `CorpusStats.tokens[action_type]` |
| `retry_count` | `output._meta.retry_count` |
| `parse_failure` | `output._meta.parse_failure` |
| `tool_choice_entropy` | Shannon entropy of `output.candidates`, normalized by `log(k)` |
| `semantic_deviation` | `1 - cos(embed(output.summary), embed(goal))` |
| `arg_novelty` | share of `input` tokens absent from the goal and all upstream `output` |
| `state_hash_repeat` | occurrences of `steps.state_hash` elsewhere in the run |
| `downstream_error_count` | `steps.error_flag` on later steps |
| `position_ratio` | `step_index / len(steps)` |

Embeddings are local `all-MiniLM-L6-v2`, 384 dimensions, L2 normalized so the
dot product is the cosine similarity. The encoder loads lazily on first use
and deduplicates texts before encoding, which matters because the corpus is
heavily templated. The goal text comes from `state_snapshot.goal`, falling back
to `input.goal` and then to the first step's text, so real LangGraph and OTel
runs that store the task elsewhere still work.
