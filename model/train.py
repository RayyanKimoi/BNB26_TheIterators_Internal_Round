"""Train the localizer and persist it.

    python -m model.train

Two heads, deliberately separate:

* **Localizer.** Per-step binary classification, "is this step the root
  cause", exactly as the Model Specification frames it. Its probability times
  100 is the suspicion score that drives the heatmap, and its argmax is the
  flagged step. This is the head the generalization claim is about.
* **Class head.** Multiclass over the five TRAINED classes, fit on root-cause
  rows only, answering "what kind of failure is this" once the step is known.

The class head cannot name a held-out class: `infinite_loop` and
`context_truncation` are not among its labels, by construction. That is not a
defect to hide, it is the reason `predicted_class` has an `unknown` value. The
honest claim is that the localizer finds a step whose failure mode it has
never seen, and the class head declines to name it. Both numbers are reported.

Thresholds are chosen on `val`, which contains only trained classes. Nothing
is ever tuned against `test_heldout`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from model.dataset import TRAIN_CLASSES, load_runs, split_by_failure_class
from model.features import FEATURE_COLUMNS, CorpusStats, Embedder, FeatureExtractor

DEFAULT_ARTIFACT = Path("model/artifacts/localizer.joblib")
DEFAULT_CORPUS = Path("generator/output/runs.jsonl")
ARTIFACT_VERSION = 1


def _localizer(seed: int) -> HistGradientBoostingClassifier:
    """Selected by leave-one-class-out over the five TRAINED classes.

    Root-cause steps are 3.7 percent of rows, so the positive class needs
    reweighting or the model answers "no" everywhere.

    `min_samples_leaf` is 5 for a specific reason. `parse_failure` is true on
    only 14 training rows and every one of them is a root cause, a rule of
    perfect precision. At the previous setting of 30 the leaf was too small to
    be allowed and the split was rejected outright, so the model never used
    the single cleanest signal in the feature set and scored 0 percent on
    `schema_violation`. Rare, highly precise indicators are exactly what this
    problem is made of, and the leaf size has to admit them.

    LOCO mean on the current corpus: 21.2% for this config, against 18.4% at
    depth 3, 17.4% at depth 2 and 13.4% for the earlier depth-1 stumps. That
    earlier config was selected on the narrated corpus, where a text feature
    dominated and the leaf-size problem was invisible. Reselected after the
    corpus was fixed. `test_heldout` was not consulted and must never be.
    """
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=5,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=False,
        random_state=seed,
    )


def _class_head(seed: int) -> HistGradientBoostingClassifier:
    # One row per failed training run, so this head sees far less data than
    # the localizer and is kept correspondingly small.
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------


def step_scores(model: Any, frame: pd.DataFrame) -> np.ndarray:
    """Raw root-cause probability per row, 0 to 100. Used for ranking only."""
    probs = model.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1]
    return probs * 100.0


def normalize_run_scores(raw: np.ndarray) -> np.ndarray:
    """Turn one run's raw scores into a distribution over its steps, summing to 100.

    The diagnosis contract shows `step_scores: [2, 4, 1, 88, 11]` alongside
    `confidence: 0.87`. Those scores sum to about 100 and the confidence is
    the flagged step's share of them, so the contract already treats the
    scores as a distribution over the steps of one run rather than as
    independent probabilities.

    Reading it that way fixes a real calibration problem. The raw probability
    of the top step is high in almost every run, including runs where the
    model is torn between four steps and runs that actually succeeded, so
    thresholding it cannot express uncertainty. A share can: when four steps
    tie, each takes about 25 and the run correctly reads as `unknown`.

    Ranking is unaffected, since this is a positive per-run rescaling.
    """
    total = float(raw.sum())
    if total <= 0:
        return np.full_like(raw, 100.0 / len(raw)) if len(raw) else raw
    return raw / total * 100.0


def _per_run(frame: pd.DataFrame, scores: np.ndarray) -> list[dict[str, Any]]:
    """Group scored rows back into runs, preserving step order."""
    out = []
    work = frame.copy()
    work["_score"] = scores
    for run_id, group in work.groupby("run_id", sort=False):
        group = group.sort_values("step_index")
        truth_rows = group.index[group["is_root_cause"]].tolist()
        truth = int(group.loc[truth_rows[0], "step_index"]) if truth_rows else None
        raw = group["_score"].to_numpy()
        out.append(
            {
                "run_id": run_id,
                "injected_class": group["injected_class"].iloc[0],
                "scores": raw,
                "normalized": normalize_run_scores(raw),
                "truth": truth,
                "frame": group,
            }
        )
    return out


def localization_accuracy(runs: list[dict[str, Any]], k: int = 1) -> tuple[float, int]:
    """Top-k accuracy over runs that have a root cause to find."""
    scored = [r for r in runs if r["truth"] is not None]
    if not scored:
        return float("nan"), 0
    hits = 0
    for run in scored:
        ranked = np.argsort(-run["scores"])[:k]
        if run["truth"] in ranked:
            hits += 1
    return hits / len(scored), len(scored)


# ---------------------------------------------------------------------------
# baselines (the three that need no LLM; the judge baseline lives in eval)
# ---------------------------------------------------------------------------


def baseline_last_step(run: dict[str, Any]) -> int:
    return len(run["scores"]) - 1


def baseline_first_error(run: dict[str, Any]) -> int:
    errors = run["frame"]["downstream_error_count"].to_numpy()
    # downstream_error_count drops to 0 at and after the final error, so the
    # first errored step is the first index where the count stops decreasing.
    flags = -np.diff(np.append(errors, 0))
    hits = np.flatnonzero(flags > 0)
    return int(hits[0]) if hits.size else len(run["scores"]) - 1


def baseline_anomaly_heuristic(run: dict[str, Any]) -> int:
    """Untrained anomaly score: the largest combined deviation on any step."""
    f = run["frame"]
    z = np.abs(f["duration_z"].to_numpy()) + np.abs(f["token_z"].to_numpy())
    z = z + f["retry_count"].to_numpy() + 3.0 * f["parse_failure"].to_numpy().astype(float)
    z = z + f["state_hash_repeat"].to_numpy() + 2.0 * np.nan_to_num(f["arg_novelty"].to_numpy())
    return int(np.argmax(z))


BASELINES = {
    "last step": baseline_last_step,
    "first errored step": baseline_first_error,
    "anomaly heuristic": baseline_anomaly_heuristic,
}


def baseline_accuracy(runs: list[dict[str, Any]], fn: Any) -> float:
    scored = [r for r in runs if r["truth"] is not None]
    if not scored:
        return float("nan")
    return sum(1 for r in scored if fn(r) == r["truth"]) / len(scored)


# ---------------------------------------------------------------------------
# threshold selection, on val only
# ---------------------------------------------------------------------------


def choose_step_threshold(runs: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    """Pick the confidence below which we answer `unknown`.

    Selective prediction: answering more often lowers accuracy among the
    answers. The threshold maximizes the harmonic mean of coverage and
    selective accuracy, but only among thresholds that keep the false
    positive rate on SUCCESSFUL runs at or below MAX_FALSE_POSITIVE_RATE.

    Diagnosis is invoked on a run already known to have failed, so a step is
    always expected to be at fault and coverage is worth paying for. The flag
    rate on successful runs is reported alongside rather than constrained,
    because this model localizes a failure, it does not detect one.
    """
    scored = [r for r in runs if r["truth"] is not None]
    healthy = [r for r in runs if r["truth"] is None]
    sweep: list[dict[str, Any]] = []
    best = (None, -1.0)
    for threshold in np.arange(0.05, 0.81, 0.05):
        answered = [r for r in scored if r["normalized"].max() / 100.0 >= threshold]
        coverage = len(answered) / len(scored) if scored else 0.0
        correct = sum(1 for r in answered if int(np.argmax(r["scores"])) == r["truth"])
        selective = correct / len(answered) if answered else 0.0
        harmonic = (
            2 * coverage * selective / (coverage + selective) if coverage + selective else 0.0
        )
        false_positive = (
            sum(1 for r in healthy if r["normalized"].max() / 100.0 >= threshold) / len(healthy)
            if healthy
            else 0.0
        )
        sweep.append(
            {
                "threshold": round(float(threshold), 2),
                "coverage": coverage,
                "selective_accuracy": selective,
                "false_positive": false_positive,
                "harmonic": harmonic,
            }
        )
        if harmonic > best[1]:
            best = (float(threshold), harmonic)
    return round(best[0] if best[0] is not None else 0.3, 2), sweep


def choose_class_threshold(probs: np.ndarray, correct: np.ndarray) -> float:
    """Lowest class confidence we are willing to name a class at.

    Chosen on val so that naming is right at least 80 percent of the time,
    falling back to the median confidence when that target is unreachable.
    """
    if probs.size == 0:
        return 0.5
    for threshold in np.arange(0.9, 0.19, -0.05):
        mask = probs >= threshold
        if mask.sum() >= max(3, 0.3 * probs.size) and correct[mask].mean() >= 0.80:
            return round(float(threshold), 2)
    return round(float(np.median(probs)), 2)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def leave_one_class_out(
    pool: list[dict[str, Any]], embedder: Embedder, seed: int
) -> tuple[dict[str, float], float, list[dict[str, Any]]]:
    """Rotate which TRAINED class is withheld, and average top-1 over the rotation.

    This is the only legitimate generalization signal available during model
    selection, because it never touches the two real held-out classes. A
    single lucky split can flatter a model; a number that survives rotation
    cannot. PRD, Model Specification: "Leave-one-class-out, if time allows".

    `pool` must contain trained-class and successful runs only.
    """
    leaked = {r["run"]["injected_class"] for r in pool} - set(TRAIN_CLASSES) - {None}
    if leaked:
        raise AssertionError(f"LOCO pool contains non-trained classes: {sorted(leaked)}")

    per_class: dict[str, float] = {}
    out_of_fold: list[dict[str, Any]] = []
    for held in TRAIN_CLASSES:
        fold_train = [r for r in pool if r["run"]["injected_class"] != held]
        fold_test = [r for r in pool if r["run"]["injected_class"] == held]
        extractor = FeatureExtractor(CorpusStats.fit(fold_train), embedder)
        ftrain = extractor.transform_many(fold_train)
        ftest = extractor.transform_many(fold_test)
        model = _localizer(seed)
        model.fit(ftrain[list(FEATURE_COLUMNS)], ftrain["is_root_cause"])
        fold_scored = _per_run(ftest, step_scores(model, ftest))
        acc, _ = localization_accuracy(fold_scored, 1)
        per_class[held] = acc
        out_of_fold.extend(fold_scored)
    return per_class, float(np.mean(list(per_class.values()))), out_of_fold


def invariant_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    """Derive the invariant tier's cut points from the TRAINING distribution.

    The invariant tier needs to know what counts as an extreme token collapse
    or an extreme state repetition. Those numbers must come from the training
    rows, never from inspecting the held-out classes: a threshold reverse
    engineered from `context_truncation` traces is held-out tuning wearing a
    disguise, and it makes any generalization claim circular.

    Percentiles of the training rows are distribution facts the model is
    already allowed to see, so they carry no information about an unseen
    failure mode beyond "this is unusual here".
    """
    token_z = frame["token_z"].astype(float).to_numpy()
    repeats = frame["state_hash_repeat"].astype(float).to_numpy()
    return {
        # 1st percentile: a token count in the bottom 1% of everything seen.
        "token_z_floor": float(np.percentile(token_z, 1)),
        # 99th percentile: a state recurring more than 99% of training steps do.
        "state_repeat_ceiling": float(np.percentile(repeats, 99)),
    }


@dataclass
class TrainingReport:
    lines: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def __str__(self) -> str:
        return "\n".join(self.lines)


def train(
    corpus_path: Path = DEFAULT_CORPUS,
    artifact_path: Path = DEFAULT_ARTIFACT,
    seed: int = 7,
    embedder: Embedder | None = None,
) -> TrainingReport:
    runs = load_runs(corpus_path)
    split = split_by_failure_class(runs, seed=seed)

    report = TrainingReport()
    report.add(f"corpus {len(runs)} runs from {corpus_path}")
    report.add(split.summary())
    report.add()

    # Corpus statistics come from the training split alone. Fitting them over
    # the held-out classes would leak their distribution into every z-score.
    stats = CorpusStats.fit(split.train)
    embedder = embedder or Embedder()
    extractor = FeatureExtractor(stats, embedder)

    frames = {
        name: extractor.transform_many(bucket)
        for name, bucket in (
            ("train", split.train),
            ("val", split.val),
            ("test_seen", split.test_seen),
            ("test_heldout", split.test_heldout),
        )
    }

    X = frames["train"][list(FEATURE_COLUMNS)]
    y = frames["train"]["is_root_cause"].to_numpy()
    localizer = _localizer(seed)
    localizer.fit(X, y)
    report.add(f"localizer      {len(X)} rows, {int(y.sum())} positive ({y.mean():.1%})")

    root = frames["train"][frames["train"]["is_root_cause"]]
    class_head = _class_head(seed)
    class_head.fit(root[list(FEATURE_COLUMNS)], root["injected_class"].to_numpy())
    report.add(
        f"class head     {len(root)} rows over {len(TRAIN_CLASSES)} trained classes"
    )
    report.add()

    thresholds = invariant_thresholds(frames["train"])
    report.add(
        f"invariant cuts  token_z_floor {thresholds['token_z_floor']:.2f}   "
        f"state_repeat_ceiling {thresholds['state_repeat_ceiling']:.1f}   "
        f"(1st/99th percentile of TRAIN rows, never of held-out)"
    )
    report.add()

    scored = {
        name: _per_run(frame, step_scores(localizer, frame)) for name, frame in frames.items()
    }

    # LOCO runs first: its out-of-fold predictions cover every trained-class
    # run from a model that never saw that class, which is both a far larger
    # sample than `val` (10 failed runs) and the right regime for a threshold
    # that has to hold up on an unseen class.
    loco_pool = split.train + split.val + split.test_seen
    loco_per_class, loco_mean, loco_oof = leave_one_class_out(loco_pool, embedder, seed)
    # The threshold is chosen on `val`, in distribution. The LOCO sweep below
    # shows why it cannot be chosen on unseen classes: there, selective
    # accuracy is flat across every threshold, so confidence carries no
    # information about correctness and no cut point beats another.
    step_threshold, sweep = choose_step_threshold(scored["val"])
    _, loco_sweep = choose_step_threshold(loco_oof)
    val_root = frames["val"][frames["val"]["is_root_cause"]]
    if len(val_root):
        val_probs = class_head.predict_proba(val_root[list(FEATURE_COLUMNS)])
        val_pred = class_head.classes_[val_probs.argmax(axis=1)]
        class_threshold = choose_class_threshold(
            val_probs.max(axis=1), val_pred == val_root["injected_class"].to_numpy()
        )
    else:
        class_threshold = 0.5

    report.add(f"thresholds chosen on val   step {step_threshold:.2f}   class {class_threshold:.2f}")
    report.add()
    n_val_failed = len([r for r in scored["val"] if r["truth"] is not None])
    report.add(f"  chosen on val, in distribution. val holds {n_val_failed} failed runs, so this")
    report.add("  is approximate; override at runtime with CONFIDENCE_THRESHOLD in .env")
    report.add()
    report.add("  threshold  coverage  selective acc  flagged (healthy runs)")
    for row in sweep:
        mark = "  <-- chosen" if row["threshold"] == step_threshold else ""
        report.add(
            f"  {row['threshold']:>9.2f}  {row['coverage']:>8.0%}  "
            f"{row['selective_accuracy']:>13.0%}  {row['false_positive']:>23.0%}{mark}"
        )
    report.add()
    report.add("  confidence reliability on an UNSEEN class (LOCO out-of-fold):")
    report.add(f"    {'threshold':>9}  {'coverage':>8}  {'selective acc':>13}")
    for row in loco_sweep[::3]:
        report.add(
            f"    {row['threshold']:>9.2f}  {row['coverage']:>8.0%}  "
            f"{row['selective_accuracy']:>13.0%}"
        )
    report.add("    Accuracy barely moves as the threshold rises, so on a class the model")
    report.add("    has never seen a high confidence does not mean a right answer.")
    report.add("    Held-out confidence is not reliable and is not claimed to be.")
    report.add()

    # -- results ---------------------------------------------------------
    report.add("TOP-1 LOCALIZATION ACCURACY")
    report.add()
    report.add(f"  {'':<22}{'top-1':>8}{'top-3':>8}{'runs':>7}")
    for name in ("test_seen", "test_heldout"):
        top1, n = localization_accuracy(scored[name], 1)
        top3, _ = localization_accuracy(scored[name], 3)
        label = "trained classes" if name == "test_seen" else "HELD-OUT classes"
        report.add(f"  {label:<22}{top1:>8.1%}{top3:>8.1%}{n:>7}")
        report.metrics[f"{name}_top1"] = top1
        report.metrics[f"{name}_top3"] = top3
        report.metrics[f"{name}_runs"] = n
    report.add()

    report.add("  per class, top-1")
    for name in ("test_seen", "test_heldout"):
        for cls in sorted({r["injected_class"] for r in scored[name] if r["injected_class"]}):
            subset = [r for r in scored[name] if r["injected_class"] == cls]
            acc, n = localization_accuracy(subset, 1)
            tag = "  [held out]" if name == "test_heldout" else ""
            report.add(f"    {cls:<24}{acc:>7.1%}{n:>5} runs{tag}")
            report.metrics[f"class_{cls}_top1"] = acc
    report.add()

    report.add("LEAVE-ONE-CLASS-OUT over the five TRAINED classes")
    report.add("  (never touches the held-out pair; the generalization sanity check)")
    report.add()
    for cls, acc in sorted(loco_per_class.items(), key=lambda kv: -kv[1]):
        report.add(f"    {cls:<24}{acc:>7.1%}")
        report.metrics[f"loco_{cls}"] = acc
    report.add(f"    {'mean':<24}{loco_mean:>7.1%}")
    report.metrics["loco_mean"] = loco_mean
    report.add()

    report.add("BASELINES, top-1 on the same runs")
    report.add()
    report.add(f"  {'':<22}{'trained':>10}{'held out':>11}")
    model_seen, _ = localization_accuracy(scored["test_seen"], 1)
    model_held, _ = localization_accuracy(scored["test_heldout"], 1)
    report.add(f"  {'Black Box model':<22}{model_seen:>10.1%}{model_held:>11.1%}")
    for label, fn in BASELINES.items():
        a = baseline_accuracy(scored["test_seen"], fn)
        b = baseline_accuracy(scored["test_heldout"], fn)
        report.add(f"  {label:<22}{a:>10.1%}{b:>11.1%}")
        report.metrics[f"baseline_{label.replace(' ', '_')}_seen"] = a
        report.metrics[f"baseline_{label.replace(' ', '_')}_heldout"] = b
    report.add()
    report.add("  LLM-as-judge baseline is not implemented yet. It belongs to the")
    report.add("  evaluation step, not here, and no number for it is claimed.")
    report.add()

    # -- unknown behaviour ------------------------------------------------
    report.add("UNKNOWN RATE")
    report.add()
    report.add(f"  {'':<22}{'answered':>10}{'unknown':>9}{'named right':>13}")
    for name in ("test_seen", "test_heldout"):
        rows = [r for r in scored[name] if r["truth"] is not None]
        named = 0
        right = 0
        for run in rows:
            top = int(np.argmax(run["scores"]))
            if run["normalized"][top] / 100.0 < step_threshold:
                continue
            feat = run["frame"].iloc[[top]][list(FEATURE_COLUMNS)]
            probs = class_head.predict_proba(feat)[0]
            if probs.max() < class_threshold:
                continue
            named += 1
            if class_head.classes_[probs.argmax()] == run["injected_class"]:
                right += 1
        total = len(rows)
        label = "trained classes" if name == "test_seen" else "HELD-OUT classes"
        report.add(
            f"  {label:<22}{named / total:>10.0%}{1 - named / total:>9.0%}"
            f"{(right / named if named else 0):>13.0%}"
        )
        report.metrics[f"{name}_named_rate"] = named / total
        report.metrics[f"{name}_class_accuracy_when_named"] = right / named if named else 0.0

    report.add()
    report.add("  A held-out run SHOULD mostly come back unknown: the class head has")
    report.add("  five labels and neither held-out class is among them. Declining to")
    report.add("  name a failure mode it has never seen is the correct behaviour.")
    report.add()

    # -- false positives on healthy runs ----------------------------------
    healthy = [
        r
        for name in ("test_seen", "test_heldout")
        for r in scored[name]
        if r["truth"] is None
    ]
    if healthy:
        flagged = sum(1 for r in healthy if r["normalized"].max() / 100.0 >= step_threshold)
        report.add(
            f"  successful runs confidently flagged: {flagged}/{len(healthy)} "
            f"({flagged / len(healthy):.0%} false positive rate)"
        )
        report.metrics["false_positive_rate"] = flagged / len(healthy)
        report.add()

    # -- persist -----------------------------------------------------------
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": ARTIFACT_VERSION,
            "localizer": localizer,
            "class_head": class_head,
            "corpus_stats": stats.to_dict(),
            "feature_columns": list(FEATURE_COLUMNS),
            "step_threshold": step_threshold,
            "class_threshold": class_threshold,
            "trained_classes": list(TRAIN_CLASSES),
            "invariant_thresholds": thresholds,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "corpus": str(corpus_path),
            "metrics": report.metrics,
        },
        artifact_path,
    )
    report.add(f"persisted {artifact_path} ({artifact_path.stat().st_size / 1024:.0f} KB)")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Black Box localizer")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--metrics-json", type=Path, default=None)
    args = parser.parse_args()

    report = train(args.corpus, args.out, args.seed)
    print(report)
    if args.metrics_json:
        args.metrics_json.write_text(json.dumps(report.metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
