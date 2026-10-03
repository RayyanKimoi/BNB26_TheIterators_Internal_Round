"""Held-out evaluation: the gate.

    python -m model.evaluate                 # includes the LLM-as-judge baseline
    python -m model.evaluate --no-judge      # skip the network call

Trains on five classes, holds out `infinite_loop` and `context_truncation`,
splits BY failure class, and reports top-1 and top-3 for trained and held-out
separately against all four baselines.

The gate, from the 32-hour timeline in PRD.md: the model must beat the
last-step baseline on held-out classes. If it does not, the pitch has no
centre and the correct response is to fix the generator, the features or a
leak, not to move on.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib

from generator.schema import HELD_OUT_CLASSES
from model.dataset import load_runs, split_by_failure_class
from model.features import CorpusStats, Embedder, FeatureExtractor
from model.train import (
    BASELINES,
    _per_run,
    baseline_accuracy,
    localization_accuracy,
    step_scores,
)

DEFAULT_ARTIFACT = Path("model/artifacts/localizer.joblib")
DEFAULT_CORPUS = Path("generator/output/runs.jsonl")


def _load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _rule(width: int = 78) -> str:
    return "-" * width


def evaluate(
    corpus_path: Path = DEFAULT_CORPUS,
    artifact_path: Path = DEFAULT_ARTIFACT,
    use_judge: bool = True,
) -> tuple[str, dict[str, Any]]:
    runs = load_runs(corpus_path)
    split = split_by_failure_class(runs, seed=7)
    artifact = joblib.load(artifact_path)

    extractor = FeatureExtractor(CorpusStats.from_dict(artifact["corpus_stats"]), Embedder())
    sets = {"test_seen": split.test_seen, "test_heldout": split.test_heldout}
    scored = {}
    for name, bucket in sets.items():
        frame = extractor.transform_many(bucket)
        scored[name] = _per_run(frame, step_scores(artifact["localizer"], frame))

    metrics: dict[str, Any] = {}
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("HELD-OUT EVALUATION".center(78))
    add("=" * 78)
    add("")
    add(f"  corpus            {len(runs)} runs, split by failure class, seed 7")
    add(f"  trained on        {', '.join(artifact['trained_classes'])}")
    add(f"  held out          {', '.join(HELD_OUT_CLASSES)}  (never seen in any form)")
    add(
        f"  evaluated on      {len([r for r in scored['test_seen'] if r['truth'] is not None])}"
        f" failed runs in distribution, "
        f"{len([r for r in scored['test_heldout'] if r['truth'] is not None])} held out"
    )
    add("")

    # -- judge baseline ---------------------------------------------------
    judge_acc: dict[str, float] = {}
    if use_judge:
        _load_env()
        from model.judge import LLMJudge

        judge = LLMJudge()
        add("  running LLM-as-judge baseline (cached after the first pass)")
        for name, bucket in sets.items():
            failed = [r for r in bucket if r["run"]["status"] == "failed"]
            verdicts = judge.judge_many(failed)
            hits = sum(
                1
                for run, verdict in zip(failed, verdicts)
                if verdict["root_cause_step"] == run["run"]["true_failure_step"]
            )
            judge_acc[name] = hits / len(failed) if failed else float("nan")
        add("")

    # -- the table --------------------------------------------------------
    add(_rule())
    add(f"  {'TOP-1 LOCALIZATION ACCURACY':<40}{'trained':>12}{'HELD OUT':>14}")
    add(_rule())
    seen1, n_seen = localization_accuracy(scored["test_seen"], 1)
    held1, n_held = localization_accuracy(scored["test_heldout"], 1)
    add(f"  {'Black Box (trained classifier)':<40}{seen1:>11.1%}{held1:>14.1%}")
    add("")
    rows = []
    for label, fn in BASELINES.items():
        rows.append(
            (
                label,
                baseline_accuracy(scored["test_seen"], fn),
                baseline_accuracy(scored["test_heldout"], fn),
            )
        )
    if use_judge:
        rows.append(
            ("LLM-as-judge (Gemini)", judge_acc["test_seen"], judge_acc["test_heldout"])
        )
    for label, a, b in rows:
        add(f"  {'baseline: ' + label:<40}{a:>11.1%}{b:>14.1%}")
    add(_rule())
    best_seen = max(r[1] for r in rows)
    best_held = max(r[2] for r in rows)
    add(
        f"  {'lift over best baseline':<40}"
        f"{(seen1 / best_seen if best_seen else float('inf')):>10.1f}x"
        f"{(held1 / best_held if best_held else float('inf')):>13.1f}x"
    )
    add(_rule())
    add("")

    add(f"  {'TOP-3 LOCALIZATION ACCURACY':<40}{'trained':>12}{'HELD OUT':>14}")
    add(_rule())
    seen3, _ = localization_accuracy(scored["test_seen"], 3)
    held3, _ = localization_accuracy(scored["test_heldout"], 3)
    add(f"  {'Black Box (trained classifier)':<40}{seen3:>11.1%}{held3:>14.1%}")
    add("")

    add(f"  {'PER CLASS, TOP-1':<40}{'accuracy':>12}{'runs':>14}")
    add(_rule())
    for name in ("test_seen", "test_heldout"):
        for cls in sorted({r["injected_class"] for r in scored[name] if r["injected_class"]}):
            subset = [r for r in scored[name] if r["injected_class"] == cls]
            acc, n = localization_accuracy(subset, 1)
            tag = "  [HELD OUT]" if name == "test_heldout" else ""
            add(f"  {cls:<40}{acc:>11.1%}{n:>14}{tag}")
    add("")

    metrics.update(
        {
            "test_seen_top1": seen1,
            "test_seen_top3": seen3,
            "test_heldout_top1": held1,
            "test_heldout_top3": held3,
            "test_seen_runs": n_seen,
            "test_heldout_runs": n_held,
            "baselines": {label: {"seen": a, "heldout": b} for label, a, b in rows},
            "loco_mean": artifact.get("metrics", {}).get("loco_mean"),
        }
    )

    # -- the gate ----------------------------------------------------------
    last_step_held = {r[0]: r[2] for r in rows}["last step"]
    passed = held1 > last_step_held
    add("=" * 78)
    add("  GATE: beat the last-step baseline on held-out classes")
    add(
        f"        model {held1:.1%}  vs  last step {last_step_held:.1%}   "
        f"{'PASS' if passed else 'FAIL'}"
    )
    if use_judge:
        add(
            f"        vs LLM-as-judge {judge_acc['test_heldout']:.1%} on held out, "
            f"{judge_acc['test_seen']:.1%} in distribution"
        )
    add("=" * 78)
    metrics["gate_passed"] = bool(passed)
    return "\n".join(lines), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out evaluation with all four baselines")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM baseline")
    parser.add_argument("--out", type=Path, default=Path("model/artifacts/evaluation.json"))
    args = parser.parse_args()

    table, metrics = evaluate(args.corpus, args.model, use_judge=not args.no_judge)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
