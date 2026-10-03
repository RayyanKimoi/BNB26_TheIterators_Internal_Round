"""Generate the synthetic corpus.

    python -m generator.generate --runs 240 --seed 7

Writes `runs.jsonl` (one JSON object per line, `{"run": ..., "steps": [...]}`,
schema-exact against the Data Model section of PRD.md) and `manifest.json`
(generation provenance: seed, counts, per-run anomaly labels and ground-truth
evidence paths). The manifest is generator-side only and never loaded as a
feature, so nothing in it can leak into training.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .build import build_run
from .faults import ANOMALIES
from .schema import FAILURE_CLASSES, HELD_OUT_CLASSES, Trace
from .tasks import TASK_TYPES

DEFAULT_RUNS = 240
DEFAULT_SEED = 7
FAILED_FRACTION = 0.60
# Share of successful runs that carry a harmless anomaly rather than running clean.
ANOMALY_FRACTION = 0.70
DEMO_USER_ID = "00000000-0000-4000-8000-000000000001"


def plan_corpus(total: int, rng: random.Random) -> list[tuple[str, str | None, str | None]]:
    """Return a shuffled list of (task_type, fault, anomaly) specs."""
    n_failed = round(total * FAILED_FRACTION)
    n_success = total - n_failed

    specs: list[tuple[str, str | None, str | None]] = []

    # Failed runs: spread evenly across the seven classes.
    for i in range(n_failed):
        specs.append(("", FAILURE_CLASSES[i % len(FAILURE_CLASSES)], None))

    # Successful runs: most carry one harmless anomaly, the rest run clean.
    n_anomalous = round(n_success * ANOMALY_FRACTION)
    for i in range(n_success):
        anomaly = ANOMALIES[i % len(ANOMALIES)] if i < n_anomalous else None
        specs.append(("", None, anomaly))

    rng.shuffle(specs)
    # Assign task types round-robin after the shuffle so every class is spread
    # across all three tasks rather than correlating with one.
    return [
        (TASK_TYPES[i % len(TASK_TYPES)].name, fault, anomaly)
        for i, (_, fault, anomaly) in enumerate(specs)
    ]


def generate(total: int, seed: int) -> tuple[list[Trace], list[dict]]:
    rng = random.Random(seed)
    by_name = {t.name: t for t in TASK_TYPES}
    base_time = datetime(2026, 3, 10, 8, 0, tzinfo=timezone.utc)

    traces: list[Trace] = []
    metas: list[dict] = []
    for i, (task_name, fault, anomaly) in enumerate(plan_corpus(total, rng)):
        trace, meta = build_run(
            by_name[task_name],
            rng,
            user_id=DEMO_USER_ID,
            created_at=base_time + timedelta(minutes=17 * i + rng.randint(0, 11)),
            fault=fault,
            anomaly=anomaly,
        )
        traces.append(trace)
        metas.append(meta)
    return traces, metas


def write(traces: list[Trace], metas: list[dict], out_dir: Path, seed: int, total: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    with runs_path.open("w", encoding="utf-8") as fh:
        for trace in traces:
            fh.write(json.dumps(trace.to_dict(), separators=(",", ":")) + "\n")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "requested_runs": total,
        "actual_runs": len(traces),
        "failure_classes": list(FAILURE_CLASSES),
        "held_out_classes": list(HELD_OUT_CLASSES),
        "anomalies": list(ANOMALIES),
        "runs": metas,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def report(traces: list[Trace], metas: list[dict]) -> str:
    n = len(traces)
    status = Counter(t.run.status for t in traces)
    classes = Counter(t.run.injected_class for t in traces if t.run.injected_class)
    tasks = Counter(t.run.task_type for t in traces)
    anomalies = Counter(m["anomaly"] for m in metas if m["anomaly"])
    steps = [len(t.steps) for t in traces]

    # Sanity signals a reader will want before trusting the corpus.
    failed = [t for t in traces if t.run.status == "failed"]
    truth_is_last = sum(1 for t in failed if t.run.true_failure_step == len(t.steps) - 1)
    first_error_is_truth = 0
    for t in failed:
        errs = [s.step_index for s in t.steps if s.error_flag]
        if errs and errs[0] == t.run.true_failure_step:
            first_error_is_truth += 1
    success_with_error = sum(
        1 for t in traces if t.run.status == "success" and any(s.error_flag for s in t.steps)
    )

    lines = [
        f"runs                {n}",
        f"  failed            {status['failed']} ({status['failed'] / n:.0%})",
        f"  success           {status['success']} ({status['success'] / n:.0%})",
        f"steps per run       min {min(steps)}  median {sorted(steps)[len(steps) // 2]}  max {max(steps)}",
        f"total steps         {sum(steps)}",
        "",
        "task types",
        *(f"  {k:<26}{v}" for k, v in sorted(tasks.items())),
        "",
        "injected classes",
        *(f"  {k:<26}{v}{'   [held out]' if k in HELD_OUT_CLASSES else ''}" for k, v in sorted(classes.items())),
        "",
        "harmless anomalies on successful runs",
        *(f"  {k:<26}{v}" for k, v in sorted(anomalies.items())),
        f"  {'none (clean)':<26}{status['success'] - sum(anomalies.values())}",
        "",
        "baseline sanity",
        f"  true step is last step          {truth_is_last}/{len(failed)} ({truth_is_last / len(failed):.0%})",
        f"  true step is first errored step {first_error_is_truth}/{len(failed)} ({first_error_is_truth / len(failed):.0%})",
        f"  successful runs with an error   {success_with_error}/{status['success']}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Black Box synthetic traces")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="total runs to generate")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="rng seed, for reproducibility")
    parser.add_argument(
        "--out", type=Path, default=Path("generator/output"), help="output directory"
    )
    args = parser.parse_args()

    traces, metas = generate(args.runs, args.seed)
    write(traces, metas, args.out, args.seed, args.runs)
    print(report(traces, metas))
    print(f"\nwrote {args.out / 'runs.jsonl'} and {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
