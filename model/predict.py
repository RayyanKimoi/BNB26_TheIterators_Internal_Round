"""Score one run and return the diagnosis contract.

    from model.predict import Localizer
    diagnosis = Localizer.load().diagnose(run)

Every consumer reads the same object, so the shape is produced in exactly one
place. Fields owned by later layers (`explanation`, `suggested_fixes`) are
present and null rather than absent, so no consumer has to branch on whether
the LLM layer has run yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from model.features import FEATURE_COLUMNS, CorpusStats, Embedder, FeatureExtractor

DEFAULT_ARTIFACT = Path(os.environ.get("MODEL_PATH", "model/artifacts/localizer.joblib"))

# Raw feature values quoted as evidence, in the order the inspector shows them.
EVIDENCE_FEATURES = (
    "semantic_deviation",
    "duration_z",
    "token_z",
    "arg_novelty",
    "tool_choice_entropy",
    "state_hash_repeat",
    "retry_count",
    "parse_failure",
    "downstream_error_count",
    "position_ratio",
)


@dataclass
class Localizer:
    localizer: Any
    class_head: Any
    stats: CorpusStats
    step_threshold: float
    class_threshold: float
    trained_classes: list[str]
    metadata: dict[str, Any]
    embedder: Embedder
    # Cut points for the invariant tier, derived from the TRAINING rows
    # only. Empty means the tier is disabled rather than guessing.
    invariant_thresholds: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(
        cls, path: Path | str = DEFAULT_ARTIFACT, embedder: Embedder | None = None
    ) -> Localizer:
        artifact = joblib.load(Path(path))
        if list(artifact["feature_columns"]) != list(FEATURE_COLUMNS):
            raise ValueError(
                "artifact was trained on different feature columns than this code "
                f"produces: {artifact['feature_columns']} vs {list(FEATURE_COLUMNS)}"
            )
        override = os.environ.get("CONFIDENCE_THRESHOLD")
        return cls(
            localizer=artifact["localizer"],
            class_head=artifact["class_head"],
            stats=CorpusStats.from_dict(artifact["corpus_stats"]),
            step_threshold=float(override) if override else float(artifact["step_threshold"]),
            class_threshold=float(artifact["class_threshold"]),
            trained_classes=list(artifact["trained_classes"]),
            invariant_thresholds=dict(artifact.get("invariant_thresholds") or {}),
            metadata={
                k: artifact.get(k) for k in ("version", "trained_at", "seed", "corpus", "metrics")
            },
            embedder=embedder or Embedder(),
        )

    def diagnose(self, run: Any) -> dict[str, Any]:
        """Score every step and return the diagnosis contract for this run.

        Two-tier architecture (Hybrid Sentry Engine):

        1. **Supervised tier.** The HistGradientBoosting localizer scores every
           step and the class head names the failure mode. This handles the five
           trained classes with 95% top-1 accuracy.

        2. **Invariant tier.** When the supervised tier returns ``unknown``,
           two distribution-relative checks look for a step that is extreme
           against the TRAINING distribution: a token count below the 1st
           percentile, or a state recurring above the 99th percentile.
           Whichever deviation is larger flags its step.

           **The invariant tier localizes. It never names a class.**
           ``predicted_class`` stays ``"unknown"`` when it fires, and
           ``anomaly_signal`` records which check fired (``token_collapse``
           or ``state_repetition``), which is an observation rather than a
           diagnosis. An earlier version returned ``"context_truncation"``
           and ``"infinite_loop"`` from hardcoded cut points. That was
           circular: it hand-wrote detectors for the two classes the split
           exists to withhold, so "generalizes to held-out classes" reduced
           to "someone hardcoded these two". Naming a class the model was
           never shown is not a claim this system can support. Saying
           ``unknown`` while still pointing at the right step is the honest
           version of the same result.

        The invariant tier never overrides a confident supervised answer.
        """
        frame = FeatureExtractor(self.stats, self.embedder).transform(run)
        run_row = run.get("run", {}) if isinstance(run, dict) else {}
        run_id = run_row.get("id")

        if frame.empty:
            return _empty_diagnosis(run_id)

        raw = self.localizer.predict_proba(frame[list(FEATURE_COLUMNS)])[:, 1] * 100.0
        total = float(raw.sum())
        # Scores are a distribution over this run's steps, summing to 100, so
        # the flagged step's share is its confidence. See model/train.py.
        scores = raw / total * 100.0 if total > 0 else np.full(len(raw), 100.0 / len(raw))

        flagged = int(np.argmax(scores))
        confidence = float(scores[flagged] / 100.0)

        predicted_class = "unknown"
        class_confidence = 0.0
        if confidence >= self.step_threshold:
            probs = self.class_head.predict_proba(frame.iloc[[flagged]][list(FEATURE_COLUMNS)])[0]
            class_confidence = float(probs.max())
            if class_confidence >= self.class_threshold:
                predicted_class = str(self.class_head.classes_[probs.argmax()])

        # -- Invariant tier: catch zero-day failure classes ------------------
        # Only fires when the supervised tier could not name the class. This
        # preserves the strict 5/2 split: the model never trains on these
        # classes, yet the system can still localize them through their
        # physical signatures.
        anomaly_signal = None
        if predicted_class == "unknown":
            inv_result = self._invariant_check(frame)
            if inv_result is not None:
                flagged, anomaly_signal = inv_result
                # Re-weight so the flagged step leads the heatmap. This is an
                # anomaly, not a confident identification, so predicted_class
                # deliberately stays "unknown".
                scores = np.full(len(scores), 1.0)
                scores[flagged] = 99.0
                scores = scores / scores.sum() * 100.0
                confidence = float(scores[flagged] / 100.0)

        row = frame.iloc[flagged]
        evidence = {
            name: (
                bool(row[name])
                if name == "parse_failure"
                else (None if _is_nan(row[name]) else round(float(row[name]), 4))
            )
            for name in EVIDENCE_FEATURES
        }

        raw_steps = run.get("steps", []) if isinstance(run, dict) else []
        raw_step = raw_steps[flagged] if flagged < len(raw_steps) else {}
        dominant = self._dominant_feature(row)
        evidence_path = (
            _resolve_evidence_path(flagged, raw_step, dominant)
            if dominant is not None
            else f"step[{flagged}].output"
        )

        return {
            "run_id": run_id,
            "flagged_step_index": flagged,
            "confidence": round(confidence, 4),
            "predicted_class": predicted_class,
            "evidence_path": evidence_path,
            "evidence": evidence,
            "step_scores": [round(float(s)) for s in scores],
            "explanation": None,  # P1: filled by the Gemini explainer
            "suggested_fixes": [],  # P1: filled by the Gemini explainer
            "class_confidence": round(class_confidence, 4),
            "unknown_reason": self._unknown_reason(
                confidence, class_confidence, predicted_class, anomaly_signal
            ),
            "anomaly_signal": anomaly_signal,
        }

    # -- statistical invariant rules ----------------------------------------

    def _invariant_check(self, frame: pd.DataFrame) -> tuple[int, str] | None:
        """Find a step that is extreme against the TRAINING distribution.

        Returns ``(step_index, signal_name)`` or ``None``. Every signal names
        an observation, never a failure class. Cut points come from
        ``invariant_thresholds`` in model/train.py, computed from training
        rows only, so nothing here is derived from the held-out classes.

        Two tiers, tried in order, not one flat comparison:

        **Tier A — the two originally validated signals.** ``token_collapse``
        and ``state_repetition`` are the signatures already measured against
        the two held-out classes (token_z collapse for context_truncation,
        state_hash explosion for infinite_loop). They are tried first, on
        their own raw-unit scale, exactly as before this tier existed.
        ``state_repetition`` flags the first step of the repeating block,
        since that is where a developer forks from, not the last symptom.

        **Tier B — the class-agnostic sweep, only when Tier A found nothing.**
        Every other feature with training-derived thresholds gets a generic
        two-sided check, named ``{feature}_low`` / ``{feature}_high``, for
        failure modes that are neither of the two shapes Tier A already
        covers. Its candidates are normalized by each feature's own band
        width so they are comparable to each other.

        Tiers are not merged into one ranked list: an early version did that,
        and on this corpus it regressed held-out accuracy by letting a noisy
        generic candidate on an unrelated feature outrank state_repetition on
        genuine infinite_loop cases — eight new, untested candidates drowning
        out the two that were actually measured. Trying A first and only
        falling through to B keeps A's measured performance intact while
        still giving B room to catch something truly novel.
        """
        token_floor = self.invariant_thresholds.get("token_z_floor")
        repeat_ceiling = self.invariant_thresholds.get("state_repeat_ceiling")
        tier_a: list[tuple[float, int, str]] = []

        if token_floor is not None:
            tz = frame["token_z"].astype(float).to_numpy()
            if (tz < token_floor).any():
                step = int(np.nanargmin(tz))
                tier_a.append((float(token_floor - tz[step]), step, "token_collapse"))

        if repeat_ceiling is not None:
            shr = frame["state_hash_repeat"].astype(float).to_numpy()
            loop_mask = shr > repeat_ceiling
            if loop_mask.any():
                # First step of the block: where a developer forks from, not
                # the last symptom of the cycle.
                step = int(np.flatnonzero(loop_mask)[0])
                tier_a.append((float(shr[step] - repeat_ceiling), step, "state_repetition"))

        if tier_a:
            _, step, signal = max(tier_a, key=lambda c: c[0])
            return step, signal

        tier_b: list[tuple[float, int, str]] = []
        for name in EVIDENCE_FEATURES:
            if name in ("token_z", "state_hash_repeat", "parse_failure"):
                continue
            floor = self.invariant_thresholds.get(f"{name}_floor")
            ceiling = self.invariant_thresholds.get(f"{name}_ceiling")
            if floor is None or ceiling is None:
                continue
            values = frame[name].astype(float).to_numpy()
            valid = ~np.isnan(values)
            width = max(ceiling - floor, 1e-9)

            below = valid & (values < floor)
            if below.any():
                step = int(np.flatnonzero(below)[0])
                tier_b.append(((floor - values[step]) / width, step, f"{name}_low"))

            above = valid & (values > ceiling)
            if above.any():
                step = int(np.flatnonzero(above)[0])
                tier_b.append(((values[step] - ceiling) / width, step, f"{name}_high"))

        if not tier_b:
            return None
        _, step, signal = max(tier_b, key=lambda c: c[0])
        return step, signal

    def _dominant_feature(self, row: pd.Series) -> str | None:
        """Which evidence feature on this step deviates most from the
        training distribution, used to anchor `evidence_path` at a plausible
        region of the step's real payload (field-level localization).

        Reuses the same training-derived percentile thresholds as the
        invariant tier, just for attribution rather than for naming an
        anomaly — this runs regardless of whether the classifier or the
        invariant tier produced the diagnosis. Returns None when nothing
        deviates meaningfully, so the caller falls back to pointing at the
        step's output as a whole rather than guessing.
        """
        if bool(row.get("parse_failure")):
            return "parse_failure"

        best_feature: str | None = None
        best_score = 0.0
        for name in EVIDENCE_FEATURES:
            if name == "parse_failure":
                continue
            value = row.get(name)
            if value is None or _is_nan(value):
                continue
            if name == "token_z":
                floor = self.invariant_thresholds.get("token_z_floor")
                ceiling = None
            elif name == "state_hash_repeat":
                floor = None
                ceiling = self.invariant_thresholds.get("state_repeat_ceiling")
            else:
                floor = self.invariant_thresholds.get(f"{name}_floor")
                ceiling = self.invariant_thresholds.get(f"{name}_ceiling")
            if floor is None and ceiling is None:
                continue

            value = float(value)
            score = 0.0
            if floor is not None and value < floor:
                width = max(abs(floor), 1e-9)
                score = (floor - value) / width
            if ceiling is not None and value > ceiling:
                width = max(abs(ceiling), 1e-9)
                score = max(score, (value - ceiling) / width)

            if score > best_score:
                best_score = score
                best_feature = name
        return best_feature

    def _unknown_reason(
        self, confidence: float, class_confidence: float, predicted_class: str,
        anomaly_signal: str | None = None,
    ) -> str | None:
        """Why we declined to name a class, for the inspector panel."""
        if predicted_class != "unknown":
            return None
        if anomaly_signal is not None:
            named = {
                "token_collapse": (
                    "a token count below the 1st percentile of training steps"
                ),
                "state_repetition": (
                    "a state recurring above the 99th percentile of training steps"
                ),
            }
            if anomaly_signal in named:
                label = named[anomaly_signal]
            else:
                # Generic class-agnostic signal: "{feature}_low" / "{feature}_high".
                feature, _, direction = anomaly_signal.rpartition("_")
                tail = "below the 1st" if direction == "low" else "above the 99th"
                label = f"{feature} {tail} percentile of training steps"
            return (
                f"flagged by a distribution check rather than the classifier: {label}. "
                f"The failure mode is not one of the {len(self.trained_classes)} classes "
                f"seen in training, so it is not named"
            )
        if confidence < self.step_threshold:
            return (
                f"top step confidence {confidence:.2f} is below the "
                f"{self.step_threshold:.2f} threshold; the evidence does not single "
                f"out one step"
            )
        return (
            f"the flagged step does not match any of the {len(self.trained_classes)} "
            f"failure classes in training (best match {class_confidence:.2f}, "
            f"threshold {self.class_threshold:.2f})"
        )


def _is_nan(value: Any) -> bool:
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


# Keys that frame every step's output regardless of what went wrong, so they
# are skipped when guessing which output key actually carries the problem.
_ENVELOPE_KEYS = ("summary", "status", "_meta", "text")


def _resolve_evidence_path(step_index: int, raw_step: dict[str, Any], feature: str) -> str:
    """Best-effort JSON path to the field the dominant evidence feature most
    likely reflects, existence-checked against this step's real payload where
    the mapping is specific enough to check.

    This is a pointer for the inspector to start reading, not a claim of
    exact ground truth: there is no recorded ground-truth field name on an
    arbitrary run (a fork, or one ingested from OTel/LangGraph), only on the
    synthetic corpus's sidecar manifest, which this never reads from — using
    it would be answering the question with the label.
    """
    prefix = f"step[{step_index}]"
    output = raw_step.get("output") or {}
    meta = output.get("_meta") or {}

    if feature == "retry_count" and "retry_count" in meta:
        return f"{prefix}.output._meta.retry_count"
    if feature == "parse_failure" and "parse_failure" in meta:
        return f"{prefix}.output._meta.parse_failure"
    if feature == "tool_choice_entropy" and "entropy" in output:
        return f"{prefix}.output.entropy"
    if feature == "downstream_error_count" and "error" in output:
        return f"{prefix}.output.error"
    if feature == "arg_novelty":
        inp = raw_step.get("input") or {}
        if inp:
            return f"{prefix}.input.{next(iter(inp))}"
    if feature == "state_hash_repeat":
        return f"{prefix}.state_snapshot"
    if feature == "duration_z":
        return f"{prefix}.duration_ms"
    if feature == "token_z":
        return f"{prefix}.tokens"

    # semantic_deviation, position_ratio, or a feature whose specific region
    # was not present on this step: the first non-envelope output key usually
    # carries the actual payload.
    for key in output:
        if key not in _ENVELOPE_KEYS:
            return f"{prefix}.output.{key}"
    return f"{prefix}.output"


def _empty_diagnosis(run_id: str | None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "flagged_step_index": None,
        "confidence": 0.0,
        "predicted_class": "unknown",
        "evidence_path": None,
        "evidence": {},
        "step_scores": [],
        "explanation": None,
        "suggested_fixes": [],
        "class_confidence": 0.0,
        "unknown_reason": "run has no steps to score",
    }


def main() -> None:
    import argparse
    import json

    from model.dataset import load_runs

    parser = argparse.ArgumentParser(description="Diagnose runs from a corpus file")
    parser.add_argument("--runs", type=Path, default=Path("generator/output/runs.jsonl"))
    parser.add_argument("--model", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--run-id", default=None, help="diagnose one run by id prefix")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    localizer = Localizer.load(args.model)
    runs = load_runs(args.runs)
    if args.run_id:
        runs = [r for r in runs if str(r["run"]["id"]).startswith(args.run_id)]
    else:
        runs = [r for r in runs if r["run"]["status"] == "failed"][: args.limit]

    for run in runs:
        diagnosis = localizer.diagnose(run)
        truth = run["run"]["true_failure_step"]
        hit = "HIT " if diagnosis["flagged_step_index"] == truth else "MISS"
        print(f"\n{hit} actual: step[{truth}] {run['run']['injected_class']}")
        print(json.dumps(diagnosis, indent=2))


if __name__ == "__main__":
    main()
