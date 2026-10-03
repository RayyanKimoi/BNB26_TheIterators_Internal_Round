"""Feature extraction: one run in, one row per step out.

Implements exactly the ten columns in the Model Specification section of
PRD.md, in the order that section lists them. `transform()` returns those ten
and nothing else. `transform_many()` adds identifier and label columns for the
training pipeline, kept deliberately separate so the feature matrix can never
accidentally carry the answer.

Two normalization notes that matter for the generalization claim:

* `duration_z` and `token_z` are measured against the same `action_type`
  across the corpus, which means they need corpus statistics. Those statistics
  are fit once on the TRAINING runs only (`CorpusStats.fit`) and persisted
  beside the model. Fitting them over the held-out classes too would leak
  distributional information about classes the model is supposed to have never
  seen.
* `tool_choice_entropy` and `arg_novelty` are undefined on step kinds that have
  no decision and no arguments. Those cells are NaN, not 0.0. Zero entropy
  means "perfectly certain", which is a different claim from "no decision was
  made here". HistGradientBoosting handles NaN natively and branches on
  missingness, so NaN is both more honest and more useful than a filler value.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Any

import numpy as np
import pandas as pd

# The ten columns, in the order the PRD lists them. Nothing else belongs here.
FEATURE_COLUMNS: tuple[str, ...] = (
    "duration_z",
    "token_z",
    "retry_count",
    "parse_failure",
    "tool_choice_entropy",
    "semantic_deviation",
    "arg_novelty",
    "state_hash_repeat",
    "downstream_error_count",
    "position_ratio",
)

# Identifier and label columns that transform_many() appends. Never features.
ID_COLUMNS: tuple[str, ...] = ("run_id", "step_index", "task_type", "injected_class", "is_root_cause")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_MIN_STD = 1e-6


# ---------------------------------------------------------------------------
# run access helpers, tolerant of both the generator and the DB row shape
# ---------------------------------------------------------------------------


def _steps_of(run: Any) -> list[dict[str, Any]]:
    if isinstance(run, dict) and "steps" in run:
        return list(run["steps"])
    if isinstance(run, list):
        return list(run)
    raise TypeError("run must be a dict with a 'steps' key, or a list of steps")


def _run_row(run: Any) -> dict[str, Any]:
    if isinstance(run, dict) and isinstance(run.get("run"), dict):
        return run["run"]
    return {}


def _meta(step: dict[str, Any]) -> dict[str, Any]:
    out = step.get("output")
    meta = out.get("_meta") if isinstance(out, dict) else None
    return meta if isinstance(meta, dict) else {}


def step_text(step: dict[str, Any]) -> str:
    """The text the step's output embedding is taken over."""
    out = step.get("output")
    if not isinstance(out, dict):
        return str(out)
    for key in ("summary", "text", "rationale", "detail"):
        value = out.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    payload = {k: v for k, v in out.items() if k != "_meta"}
    return json.dumps(payload, default=str)[:600]


def goal_text(steps: Sequence[dict[str, Any]]) -> str:
    """The run's task text, which every step is measured against."""
    for step in steps:
        snapshot = step.get("state_snapshot")
        if isinstance(snapshot, dict):
            goal = snapshot.get("goal")
            if isinstance(goal, str) and goal.strip():
                return goal.strip()
        inp = step.get("input")
        if isinstance(inp, dict):
            goal = inp.get("goal")
            if isinstance(goal, str) and goal.strip():
                return goal.strip()
    return step_text(steps[0]) if steps else ""


def _tokens(value: Any) -> set[str]:
    """Lowercased alphanumeric tokens from any nested payload."""
    found: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif item is not None and not isinstance(item, bool):
            found.update(t.lower() for t in _TOKEN_RE.findall(str(item)))
    return found


def normalized_entropy(dist: dict[str, float]) -> float:
    """Shannon entropy over the candidate tools, scaled to 0..1.

    Normalizing by log(k) makes decisions with different candidate-set sizes
    comparable, which matters because the generator varies that count.
    """
    weights = [float(p) for p in dist.values() if isinstance(p, (int, float)) and p > 0]
    if len(weights) < 2:
        return 0.0
    total = sum(weights)
    probs = [w / total for w in weights]
    raw = -sum(p * math.log(p) for p in probs)
    return raw / math.log(len(probs))


# ---------------------------------------------------------------------------
# corpus statistics
# ---------------------------------------------------------------------------


@dataclass
class CorpusStats:
    """Per-`action_type` mean and standard deviation for duration and tokens.

    Fit on training runs only, persisted beside the model, reused at inference
    so a single run can be scored without a corpus to compare against.
    """

    duration: dict[str, tuple[float, float]] = field(default_factory=dict)
    tokens: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def fit(cls, runs: Iterable[Any]) -> CorpusStats:
        durations: dict[str, list[float]] = {}
        tokens: dict[str, list[float]] = {}
        for run in runs:
            for step in _steps_of(run):
                action = str(step.get("action_type", "unknown"))
                durations.setdefault(action, []).append(float(step.get("duration_ms", 0)))
                tokens.setdefault(action, []).append(float(step.get("tokens", 0)))
        if not durations:
            raise ValueError("cannot fit corpus statistics on an empty set of runs")
        return cls(
            duration={k: (fmean(v), max(pstdev(v), _MIN_STD)) for k, v in durations.items()},
            tokens={k: (fmean(v), max(pstdev(v), _MIN_STD)) for k, v in tokens.items()},
        )

    def _z(self, table: dict[str, tuple[float, float]], action: str, value: float) -> float:
        stat = table.get(action)
        if stat is None:
            # An action type never seen in training. Honest answer is "unknown".
            return float("nan")
        mean, std = stat
        return (value - mean) / std

    def duration_z(self, action: str, value: float) -> float:
        return self._z(self.duration, action, value)

    def token_z(self, action: str, value: float) -> float:
        return self._z(self.tokens, action, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration": {k: list(v) for k, v in self.duration.items()},
            "tokens": {k: list(v) for k, v in self.tokens.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CorpusStats:
        return cls(
            duration={k: (float(v[0]), float(v[1])) for k, v in payload["duration"].items()},
            tokens={k: (float(v[0]), float(v[1])) for k, v in payload["tokens"].items()},
        )


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------


class Embedder:
    """Lazy local sentence-transformers encoder with a unique-text cache.

    The corpus is heavily templated, so deduplicating before encoding cuts the
    work substantially. The model loads on first use, never at import, so a
    module import does not pull torch into memory.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = self.model.encode(
                missing, normalize_embeddings=True, show_progress_bar=False, batch_size=64
            )
            for text, vector in zip(missing, np.asarray(vectors, dtype=np.float32)):
                self._cache[text] = vector
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.vstack([self._cache[t] for t in texts])

    def warm(self, texts: Sequence[str]) -> None:
        """Encode a whole corpus in one batch before per-run extraction."""
        self.encode(list(texts))


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


class FeatureExtractor:
    def __init__(self, stats: CorpusStats, embedder: Embedder | None = None) -> None:
        self.stats = stats
        self.embedder = embedder or Embedder()

    def transform(self, run: Any) -> pd.DataFrame:
        """Exactly the ten feature columns, one row per step, in PRD order."""
        steps = _steps_of(run)
        if not steps:
            return pd.DataFrame(columns=list(FEATURE_COLUMNS))

        total = len(steps)
        goal = goal_text(steps)

        # One batch for the goal plus every step, so a run costs one encode.
        texts = [goal] + [step_text(s) for s in steps]
        vectors = self.embedder.encode(texts)
        goal_vec, step_vecs = vectors[0], vectors[1:]
        # Vectors are L2-normalized, so the dot product is the cosine similarity.
        deviations = 1.0 - (step_vecs @ goal_vec)

        error_flags = [bool(s.get("error_flag")) for s in steps]
        # Errors strictly after step i, computed once as a reverse running sum.
        downstream: list[int] = [0] * total
        running = 0
        for i in range(total - 1, -1, -1):
            downstream[i] = running
            running += int(error_flags[i])

        # Occurrences of each state hash ELSEWHERE in the run, not just before
        # this step. The prior-occurrence reading scores 0 on the step that
        # opens a loop and peaks on the last step of it, which points at the
        # symptom instead of the cause and would cripple `infinite_loop`, a
        # held-out class. Diagnosis always runs on a finished trace, and
        # `downstream_error_count` is already forward-looking, so counting the
        # whole run is both consistent and the only reading that fires on the
        # labelled root cause. See model/README.md.
        hash_counts = Counter(str(s.get("state_hash", "")) for s in steps)
        upstream_tokens: set[str] = _tokens(goal)
        rows: list[dict[str, Any]] = []

        for i, step in enumerate(steps):
            action = str(step.get("action_type", "unknown"))
            meta = _meta(step)
            output = step.get("output") if isinstance(step.get("output"), dict) else {}

            repeats = hash_counts[str(step.get("state_hash", ""))] - 1

            # Entropy only exists where a choice was made.
            candidates = output.get("candidates")
            entropy = (
                normalized_entropy(candidates)
                if action == "decide" and isinstance(candidates, dict) and candidates
                else float("nan")
            )

            # Novelty only exists where there were arguments. Measured against
            # every upstream output plus the goal, which is context the agent
            # legitimately holds from the start.
            if action == "call_tool":
                arg_tokens = _tokens(step.get("input"))
                arg_novelty = (
                    len(arg_tokens - upstream_tokens) / len(arg_tokens) if arg_tokens else 0.0
                )
            else:
                arg_novelty = float("nan")

            rows.append(
                {
                    "duration_z": self.stats.duration_z(action, float(step.get("duration_ms", 0))),
                    "token_z": self.stats.token_z(action, float(step.get("tokens", 0))),
                    "retry_count": int(meta.get("retry_count") or 0),
                    "parse_failure": bool(meta.get("parse_failure")),
                    "tool_choice_entropy": entropy,
                    "semantic_deviation": float(deviations[i]),
                    "arg_novelty": arg_novelty,
                    "state_hash_repeat": repeats,
                    "downstream_error_count": downstream[i],
                    "position_ratio": i / total,
                }
            )

            upstream_tokens |= _tokens(output)

        frame = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
        return frame.astype(
            {
                "duration_z": "float64",
                "token_z": "float64",
                "retry_count": "int64",
                "parse_failure": "bool",
                "tool_choice_entropy": "float64",
                "semantic_deviation": "float64",
                "arg_novelty": "float64",
                "state_hash_repeat": "int64",
                "downstream_error_count": "int64",
                "position_ratio": "float64",
            }
        )

    def transform_many(self, runs: Sequence[Any]) -> pd.DataFrame:
        """Feature rows for many runs, with identifier and label columns appended.

        `is_root_cause` is the per-step binary target: True on
        `true_failure_step` and nowhere else. Successful runs contribute only
        negatives, which is what teaches the model that an anomaly is not
        automatically a fault.
        """
        runs = list(runs)
        if not runs:
            return pd.DataFrame(columns=[*FEATURE_COLUMNS, *ID_COLUMNS])

        # Warm the cache across the whole corpus in one pass.
        corpus_texts: list[str] = []
        for run in runs:
            steps = _steps_of(run)
            corpus_texts.append(goal_text(steps))
            corpus_texts.extend(step_text(s) for s in steps)
        self.embedder.warm(corpus_texts)

        frames: list[pd.DataFrame] = []
        for run in runs:
            steps = _steps_of(run)
            row = _run_row(run)
            frame = self.transform(run)
            truth = row.get("true_failure_step")
            frame["run_id"] = row.get("id")
            frame["step_index"] = range(len(frame))
            frame["task_type"] = row.get("task_type")
            frame["injected_class"] = row.get("injected_class")
            frame["is_root_cause"] = (
                frame["step_index"] == truth if truth is not None else False
            )
            frames.append(frame)

        return pd.concat(frames, ignore_index=True)[[*FEATURE_COLUMNS, *ID_COLUMNS]]


def extract(run: Any, stats: CorpusStats, embedder: Embedder | None = None) -> pd.DataFrame:
    """Convenience wrapper: ten feature columns for one run."""
    return FeatureExtractor(stats, embedder).transform(run)


# ---------------------------------------------------------------------------
# sanity CLI: python -m model.features
# ---------------------------------------------------------------------------


def load_runs(path: Any) -> list[dict[str, Any]]:
    """Read the generator's JSONL corpus."""
    from pathlib import Path

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _sanity_report(frame: pd.DataFrame) -> str:
    root = frame[frame["is_root_cause"]]
    other = frame[~frame["is_root_cause"]]
    lines = [
        f"rows {len(frame)}   root-cause rows {len(root)}   other {len(other)}",
        f"runs {frame['run_id'].nunique()}",
        "",
        f"{'feature':<24}{'root cause':>12}{'other':>12}{'gap (sd)':>11}{'NaN %':>8}",
    ]
    for col in FEATURE_COLUMNS:
        a, b = root[col].astype(float), other[col].astype(float)
        pooled = frame[col].astype(float).std()
        gap = (a.mean() - b.mean()) / pooled if pooled and not math.isnan(pooled) else float("nan")
        nan_pct = 100.0 * frame[col].astype(float).isna().mean()
        lines.append(f"{col:<24}{a.mean():>12.3f}{b.mean():>12.3f}{gap:>11.2f}{nan_pct:>8.1f}")

    lines += ["", "mean feature value on the root-cause step, by injected class", ""]
    classes = sorted(c for c in frame["injected_class"].dropna().unique())
    cols = ["duration_z", "token_z", "retry_count", "tool_choice_entropy",
            "semantic_deviation", "arg_novelty", "state_hash_repeat", "parse_failure"]
    lines.append(f"{'class':<24}" + "".join(f"{c[:11]:>12}" for c in cols))
    for cls in classes:
        sub = frame[(frame["injected_class"] == cls) & frame["is_root_cause"]]
        lines.append(f"{cls:<24}" + "".join(f"{sub[c].astype(float).mean():>12.2f}" for c in cols))
    return "\n".join(lines)


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Extract features and report separation")
    parser.add_argument("--runs", type=Path, default=Path("generator/output/runs.jsonl"))
    parser.add_argument("--out", type=Path, default=None, help="optional parquet/csv dump")
    args = parser.parse_args()

    runs = load_runs(args.runs)
    # NOTE: fitting stats on every run is correct only for this descriptive
    # report. Training must fit CorpusStats on the training split alone.
    stats = CorpusStats.fit(runs)
    frame = FeatureExtractor(stats).transform_many(runs)
    print(_sanity_report(frame))
    if args.out:
        if args.out.suffix == ".csv":
            frame.to_csv(args.out, index=False)
        else:
            frame.to_parquet(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
