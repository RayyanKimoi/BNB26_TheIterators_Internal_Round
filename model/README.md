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

---

# Model

```bash
python -m model.train                       # train, evaluate, persist
python -m model.predict --limit 3           # diagnosis contract for sample runs
python -m pytest model/tests -q
```

`model/artifacts/localizer.joblib` holds both heads, the `CorpusStats`, the two
thresholds and the metrics from the run that produced it.

## Two heads

**Localizer.** Per-step binary classification, "is this step the root cause",
as the Model Specification frames it. Depth-1 stumps, `class_weight="balanced"`
because root-cause steps are 3.7 percent of rows. This is the head the
generalization claim is about.

**Class head.** Multiclass over the five trained classes, fit on root-cause
rows only. It *cannot* name a held-out class, because neither is among its
labels. That is not a defect, it is why `predicted_class` has an `unknown`
value: the localizer finds a step whose failure mode it has never seen, and
the class head declines to name it.

The config was chosen by leave-one-class-out over the trained classes: 21.2
percent mean for 31 leaves at `min_samples_leaf=5`, against 18.4 at depth 3,
17.4 at depth 2 and 13.4 for depth-1 stumps.

`min_samples_leaf=5` is the load-bearing setting. `parse_failure` is true on
only 14 training rows and every one is a root cause, a rule of perfect
precision. At `min_samples_leaf=30` that leaf was too small to be permitted,
the split was rejected outright, and the model scored **0 percent** on
`schema_violation` while ignoring the cleanest signal in the feature set.
Fixing it took trained-class top-1 from 40 to 95 percent. Rare, highly precise
indicators are what this problem is made of, and the leaf size has to admit
them. **`test_heldout` was not used for any tuning decision and must never be.**

## Scores are a distribution over the run's steps

The contract shows `step_scores: [2, 4, 1, 88, 11]` next to `confidence: 0.87`.
Those sum to about 100 and the confidence is the flagged step's share, so the
contract already treats scores as a distribution over one run's steps.

Reading it that way fixes a real calibration problem. The raw probability of
the top step is high in nearly every run, so thresholding it cannot express
uncertainty. A share can: when four steps tie, each takes about 25 and the run
correctly reads as `unknown`. Ranking is unaffected, since this is a positive
per-run rescaling.

## Results

240 runs, seed 7, on the neutralized corpus. `test_seen` is 20 failed runs,
`test_heldout` is 40. All four baselines.

| Top-1 | Trained | Held out |
| --- | --- | --- |
| **Black Box** | **95.0%** | 7.5% |
| baseline: last step | 10.0% | 7.5% |
| baseline: first errored step | 5.0% | 10.0% |
| baseline: anomaly heuristic | 30.0% | 0.0% |
| baseline: LLM-as-judge (Gemini) | 80.0% | 32.5% |

Top-3: 100.0% trained, 35.0% held out. Leave-one-class-out mean over the
trained classes: 21.2%.

### The gate fails on held-out classes

The PRD gate is to beat the last-step baseline on held-out classes. The model
scores 7.5% against the baseline's 7.5%. It does not beat it. **Cross-class
generalization is not a claim this project can make**, and the Model tab
should not imply otherwise.

An earlier corpus produced 25% here. That number was an artifact: the
generator wrote summaries naming each fault in plain English, and
`semantic_deviation` was embedding the confession. Ablating that single
feature dropped held-out accuracy to 0.0%, which is what exposed it. The same
narration let the LLM-as-judge score 95% on held-out by reading rather than
reasoning; once the prose was neutralized it fell to 32.5%. See
`generator/README.md`.

### Where the model wins, and why

In distribution it beats the LLM-as-judge 95.0% to 80.0%, and the win is not
uniform. It concentrates on classes whose signal is a number only meaningful
against the corpus:

| Class | Model | Judge | Signal |
| --- | --- | --- | --- |
| `stale_retrieval` | **100%** | 50% | `duration_z`: a cache hit is abnormally fast |
| `hallucinated_argument` | **100%** | 50% | token overlap against every upstream output |
| `schema_violation` | 100% | 100% | `parse_failure` flag |
| `premature_termination` | 100% | 100% | run length, no terminal tool call |
| `wrong_tool_chosen` | 75% | **100%** | whether a tool suits the goal |
| `infinite_loop` | 0% | **60%** | repetition, visible by eye [held out] |
| `context_truncation` | 15% | 5% | `token_z`: context drop [held out] |

An LLM reading one run sees `213ms` and cannot know that is 0.9 sigma fast for
that action type across 240 runs. It has no corpus. That is the structural
advantage, and `stale_retrieval` and `hallucinated_argument` are where it
shows. The classes the judge wins are the semantic ones, where reading
comprehension is the right tool. The two approaches are complementary rather
than redundant.

| Per diagnosis | Black Box | LLM-as-judge |
| --- | --- | --- |
| Latency | 11.2 ms | ~1500 ms |
| Cost | none, runs locally | priced per token, per run |
| Determinism | same input, same output | not guaranteed |

## Honest limitations

**Cross-class generalization does not work.** 7.5% on held-out, equal to the
last-step baseline. `infinite_loop` is 0%. The features are class-specific by
construction: `parse_failure` fires only for `schema_violation`,
`state_hash_repeat` only for `infinite_loop`. A model trained on five classes
has no route to the sixth. Top-3 reaches 35%, so the heatmap still puts the
right region in front of a developer, but top-1 is not there.

**Confidence is not reliable on an unseen class.** On LOCO out-of-fold
predictions selective accuracy stays flat across every threshold. The
threshold is therefore chosen in distribution, on `val`, which holds only 10
failed runs, so it is approximate. `CONFIDENCE_THRESHOLD` in `.env` overrides
it at load time.

**This localizes a failure, it does not detect one.** At the chosen threshold
every successful run in the test sets gets a step flagged. Diagnosis is meant
to run on a trace already known to have failed.

## Unknown rate

| | answered | unknown | class right when named |
| --- | --- | --- | --- |
| Trained classes | 90% | 10% | 100% |
| Held-out classes | 15% | 85% | 0% |

The held-out row is intended behaviour. The class head has five labels and
neither held-out class is among them, so declining to name a failure mode it
has never been shown is correct.

## Split

`model/dataset.py`. Four sets, split BY FAILURE CLASS:

| Set | Failed runs | Successful | Used for |
| --- | --- | --- | --- |
| `train` | 70% of each trained class | 70% | fitting both heads and `CorpusStats` |
| `val` | 10% of each trained class | 10% | choosing both thresholds |
| `test_seen` | 20% of each trained class | 10% | in-distribution accuracy |
| `test_heldout` | ALL `infinite_loop` + `context_truncation` | 10% | the generalization number |

`_assert_no_leakage` runs on every split and raises if a held-out class
reaches any training set, if a run lands in two sets, or if `test_heldout` is
missing either class. A silent leak here would invalidate every number above
while leaving the training run looking perfectly normal.
