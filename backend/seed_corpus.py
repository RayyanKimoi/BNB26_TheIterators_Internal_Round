"""Idempotent loader for the synthetic corpus.

Reads `generator/output/runs.jsonl` (the 240-run corpus `generate.py`
produces) and inserts every run not already present in the database, by id.
Re-running this script is always safe: a run already in the database — from
an earlier seed, a fork, or manual testing — is left completely untouched,
including any diagnosis it already has. Nothing is ever updated or deleted.

By default, every newly-inserted run is also scored with the real Hybrid
Sentry Engine right away (the same code path `POST /runs/{id}/diagnose` uses),
so the dashboard is demo-ready immediately rather than showing an empty
heatmap until someone clicks "Run Diagnosis" 240 times. Pass --no-diagnose to
skip that and just load the rows.

Usage (run from the repo root, so backend.engine finds .env):
    python -m backend.seed_corpus
    python -m backend.seed_corpus --limit 20          # smoke test a subset
    python -m backend.seed_corpus --no-diagnose        # rows only, fast
    python -m backend.seed_corpus --corpus path/to.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session

from backend.db import AgentRun, Step
from backend.engine import create_db_and_tables, engine

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO_ROOT / "generator" / "output" / "runs.jsonl"


def load_corpus(path: Path) -> list[dict[str, Any]]:
    traces = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: malformed JSON — {exc}") from exc
    return traces


def would_flip_if_forked(steps: list[dict[str, Any]], from_step: int) -> bool:
    """Whether forking this run at `from_step` with any fix would flip it to
    SUCCESS, per the real rule in replay/engine.py: the forked step's own
    error_flag always clears once a fix is supplied, no synthetic step ever
    carries the `precondition_not_met` marker that would auto-recover a
    downstream one (the generator never writes it), and outcome_of() succeeds
    iff nothing in the trace is still erroring. So a fork flips this run iff
    no step OTHER than the one being patched is error-flagged.
    """
    return not any(
        step.get("error_flag") for i, step in enumerate(steps) if i != from_step
    )


def insert_run(session: Session, trace: dict[str, Any]) -> AgentRun | None:
    """Insert one run and its steps if `id` is not already in the database.

    Returns the inserted AgentRun (flushed, not committed), or None if it was
    already present.
    """
    run_data = trace["run"]
    run_id = run_data["id"]

    if session.get(AgentRun, run_id) is not None:
        return None

    run = AgentRun(
        id=run_id,
        user_id=run_data.get("user_id"),
        source=run_data.get("source", "synthetic"),
        task_type=run_data["task_type"],
        status=run_data["status"],
        parent_run_id=run_data.get("parent_run_id"),
        forked_at_step=run_data.get("forked_at_step"),
        fix_applied=run_data.get("fix_applied"),
        injected_class=run_data.get("injected_class"),
        true_failure_step=run_data.get("true_failure_step"),
        total_tokens=run_data["total_tokens"],
        duration_ms=run_data["duration_ms"],
        created_at=datetime.fromisoformat(run_data["created_at"]),
    )
    session.add(run)

    for step_data in trace["steps"]:
        session.add(
            Step(
                id=step_data["id"],
                run_id=step_data["run_id"],
                step_index=step_data["step_index"],
                action_type=step_data["action_type"],
                tool_name=step_data.get("tool_name"),
                input=step_data.get("input") or {},
                output=step_data.get("output") or {},
                state_snapshot=step_data.get("state_snapshot") or {},
                state_hash=step_data.get("state_hash") or "",
                duration_ms=step_data["duration_ms"],
                tokens=step_data["tokens"],
                error_flag=step_data.get("error_flag", False),
            )
        )

    session.flush()
    return run


def diagnose_run(session: Session, run: AgentRun) -> None:
    """Score one run with the real Hybrid Sentry Engine and persist it.

    Imported lazily from backend.main so this script has one source of truth
    for diagnosis logic — the exact code POST /runs/{id}/diagnose runs — and
    so importing this module never pulls in FastAPI route wiring unless
    diagnosis is actually requested.
    """
    from backend.main import _get_localizer, _persist_diagnosis, _run_to_trace_dict, _shap_for

    steps = sorted(run.steps, key=lambda s: s.step_index)
    trace_dict = _run_to_trace_dict(run, steps)
    localizer = _get_localizer()
    raw = localizer.diagnose(trace_dict)
    raw["shap"] = _shap_for(trace_dict, raw["flagged_step_index"])
    _persist_diagnosis(session, run.id, raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Path to runs.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Only load the first N traces (smoke testing)")
    parser.add_argument("--no-diagnose", action="store_true", help="Load rows only, skip scoring")
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"Corpus not found: {args.corpus}", file=sys.stderr)
        print("Generate it first: python -m generator.generate", file=sys.stderr)
        return 1

    traces = load_corpus(args.corpus)
    if args.limit is not None:
        traces = traces[: args.limit]

    print(f"Loaded {len(traces)} traces from {args.corpus}")

    create_db_and_tables()

    inserted = 0
    skipped = 0
    diagnosed = 0
    by_class: Counter[str] = Counter()
    flippable_examples: dict[str, str] = {}

    with Session(engine) as session:
        for trace in traces:
            run = insert_run(session, trace)
            if run is None:
                skipped += 1
                continue
            inserted += 1

            run_data = trace["run"]
            cls = run_data.get("injected_class")
            if cls:
                by_class[cls] += 1
                tfs = run_data.get("true_failure_step")
                if (
                    run_data["status"] == "failed"
                    and tfs is not None
                    and cls not in flippable_examples
                    and would_flip_if_forked(trace["steps"], tfs)
                ):
                    flippable_examples[cls] = run_data["id"]

            if not args.no_diagnose:
                diagnose_run(session, run)
                diagnosed += 1

            # Commit per run rather than batching 240 into one transaction:
            # a bug partway through then leaves a partially-seeded corpus you
            # can inspect and safely resume (re-running skips what landed).
            session.commit()

    print(f"\ninserted: {inserted}")
    print(f"already present (skipped): {skipped}")
    if not args.no_diagnose:
        print(f"diagnosed: {diagnosed}")

    if by_class:
        print("\ninjected_class counts (this run of the loader):")
        for cls, count in sorted(by_class.items()):
            print(f"  {cls:24s} {count}")

    if flippable_examples:
        print("\nverified fork-flip candidates (no other step is error-flagged,")
        print("so a fork from true_failure_step will flip FAIL to SUCCESS):")
        for cls, run_id in sorted(flippable_examples.items()):
            print(f"  {cls:24s} {run_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
