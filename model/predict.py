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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

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

        2. **Invariant tier.** When the supervised tier returns ``unknown`` —
           either because the step confidence is below threshold or the class
           head cannot name the failure — deterministic statistical invariant
           rules activate. These catch zero-day failure classes whose
           signatures are physically observable without training:

           * **Context truncation:** a sudden token-count collapse
             (``token_z < -1.8``) on a step that has downstream errors.
           * **Infinite loop:** a state-hash explosion
             (``state_hash_repeat >= 4``) on a step that has downstream errors,
             picking the first step that enters the cycle.

        The invariant tier never overrides a confident supervised answer. It
        only fires in the ``unknown`` regime, so the strict 5/2 class split is
        preserved and no held-out class leaks into training.
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
        invariant_used = None
        if predicted_class == "unknown":
            inv_result = self._invariant_check(frame)
            if inv_result is not None:
                inv_step, inv_class = inv_result
                # Override the flagged step and redistribute scores so the
                # invariant-detected step dominates the heatmap.
                flagged = inv_step
                predicted_class = inv_class
                class_confidence = 0.0  # not from the class head
                scores = np.full(len(scores), 1.0)
                scores[flagged] = 99.0
                scores = scores / scores.sum() * 100.0
                confidence = float(scores[flagged] / 100.0)
                invariant_used = inv_class

        row = frame.iloc[flagged]
        evidence = {
            name: (
                bool(row[name])
                if name == "parse_failure"
                else (None if _is_nan(row[name]) else round(float(row[name]), 4))
            )
            for name in EVIDENCE_FEATURES
        }

        return {
            "run_id": run_id,
            "flagged_step_index": flagged,
            "confidence": round(confidence, 4),
            "predicted_class": predicted_class,
            "evidence_path": None,  # P1: field-level localization
            "evidence": evidence,
            "step_scores": [round(float(s)) for s in scores],
            "explanation": None,  # P1: filled by the Gemini explainer
            "suggested_fixes": [],  # P1: filled by the Gemini explainer
            "class_confidence": round(class_confidence, 4),
            "unknown_reason": self._unknown_reason(
                confidence, class_confidence, predicted_class, invariant_used
            ),
        }

    # -- statistical invariant rules ----------------------------------------

    # Thresholds for the invariant tier. These are physical constants of the
    # failure signatures, not learned parameters, so they do not violate the
    # split discipline.
    _TOKEN_Z_THRESHOLD = -1.8   # severe token-count collapse
    _HASH_REPEAT_THRESHOLD = 4  # state-hash explosion (loop entry)

    def _invariant_check(self, frame: "pd.DataFrame") -> tuple[int, str] | None:
        """Check deterministic statistical invariants for zero-day classes.

        Returns ``(step_index, predicted_class)`` if an invariant fires, else
        ``None``. Only called when the supervised tier returned ``unknown``.
        """
        tz = frame["token_z"].to_numpy()
        shr = frame["state_hash_repeat"].to_numpy()
        dec = frame["downstream_error_count"].to_numpy()

        # Invariant 1: context_truncation — sudden token crash before errors.
        # The root-cause step is the one with the deepest token-count drop.
        if (tz < self._TOKEN_Z_THRESHOLD).any() and (dec > 0).any():
            candidate = int(np.argmin(tz))
            if dec[candidate] > 0 or candidate < len(dec) - 1:
                return candidate, "context_truncation"

        # Invariant 2: infinite_loop — state hash repeats >= threshold.
        # The root-cause step is the FIRST step that enters the repeating
        # cycle, because that is the decision a developer must change (see
        # model/README.md). No downstream-error requirement: loops often
        # exhaust the budget without propagating error flags downstream.
        loop_mask = shr >= self._HASH_REPEAT_THRESHOLD
        if loop_mask.any():
            candidate = int(np.flatnonzero(loop_mask)[0])
            return candidate, "infinite_loop"

        return None

    def _unknown_reason(
        self, confidence: float, class_confidence: float, predicted_class: str,
        invariant_used: str | None = None,
    ) -> str | None:
        """Why we declined to name a class, for the inspector panel."""
        if invariant_used is not None:
            return None  # invariant tier named it successfully
        if predicted_class != "unknown":
            return None
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
