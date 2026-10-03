"""Trace schema for generated runs.

Mirrors the `agent_runs` and `steps` tables in the Data Model section of PRD.md
exactly. The schema is frozen at this layer: feature extraction, the API and the
UI all read these field names, so changing one here changes all of them.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

ActionType = Literal["call_llm", "call_tool", "decide"]
RunStatus = Literal["success", "failed"]
RunSource = Literal["synthetic", "langgraph", "otel"]

# The seven classes from the Failure Taxonomy section of PRD.md.
FAILURE_CLASSES: tuple[str, ...] = (
    "wrong_tool_chosen",
    "hallucinated_argument",
    "stale_retrieval",
    "premature_termination",
    "infinite_loop",
    "schema_violation",
    "context_truncation",
)

# Never present in training. The generalization number is measured on these.
HELD_OUT_CLASSES: tuple[str, ...] = ("infinite_loop", "context_truncation")


def new_uuid() -> str:
    return str(uuid.uuid4())


def state_hash(state: dict[str, Any]) -> str:
    """Stable hash of an agent state, for loop detection.

    Sorted keys and a canonical separator mean two structurally identical
    states always hash the same, which is what `state_hash_repeat` counts.
    """
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class Step:
    """One row of the `steps` table."""

    run_id: str
    step_index: int
    action_type: ActionType
    input: dict[str, Any]
    output: dict[str, Any]
    state_snapshot: dict[str, Any]
    state_hash: str
    duration_ms: int
    tokens: int
    error_flag: bool = False
    tool_name: str | None = None
    id: str = field(default_factory=new_uuid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step_index": self.step_index,
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "input": self.input,
            "output": self.output,
            "state_snapshot": self.state_snapshot,
            "state_hash": self.state_hash,
            "duration_ms": self.duration_ms,
            "tokens": self.tokens,
            "error_flag": self.error_flag,
        }


@dataclass
class Run:
    """One row of the `agent_runs` table."""

    task_type: str
    status: RunStatus
    total_tokens: int
    duration_ms: int
    created_at: datetime
    user_id: str
    source: RunSource = "synthetic"
    parent_run_id: str | None = None
    forked_at_step: int | None = None
    fix_applied: dict[str, Any] | None = None
    injected_class: str | None = None
    true_failure_step: int | None = None
    id: str = field(default_factory=new_uuid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "source": self.source,
            "task_type": self.task_type,
            "status": self.status,
            "parent_run_id": self.parent_run_id,
            "forked_at_step": self.forked_at_step,
            "fix_applied": self.fix_applied,
            "injected_class": self.injected_class,
            "true_failure_step": self.true_failure_step,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Trace:
    """A run plus its ordered steps. One line of the output JSONL."""

    run: Run
    steps: list[Step]

    def to_dict(self) -> dict[str, Any]:
        return {"run": self.run.to_dict(), "steps": [s.to_dict() for s in self.steps]}


class TraceValidationError(ValueError):
    pass


def validate(trace: Trace) -> None:
    """Fail loudly on a malformed trace.

    PRD Risks: a generator producing malformed traces found at hour 20 is fatal,
    found at hour 5 it costs ten minutes. Every generated trace runs through this.
    """
    run, steps = trace.run, trace.steps

    if not steps:
        raise TraceValidationError(f"run {run.id} has no steps")

    for i, step in enumerate(steps):
        if step.step_index != i:
            raise TraceValidationError(
                f"run {run.id}: step_index {step.step_index} out of order at position {i}"
            )
        if step.run_id != run.id:
            raise TraceValidationError(f"run {run.id}: step {i} has mismatched run_id")
        if step.action_type not in ("call_llm", "call_tool", "decide"):
            raise TraceValidationError(f"run {run.id}: step {i} bad action_type")
        if step.action_type == "call_tool" and not step.tool_name:
            raise TraceValidationError(f"run {run.id}: call_tool step {i} has no tool_name")
        if step.action_type != "call_tool" and step.tool_name is not None:
            raise TraceValidationError(f"run {run.id}: non-tool step {i} has a tool_name")
        if step.duration_ms <= 0 or step.tokens < 0:
            raise TraceValidationError(f"run {run.id}: step {i} has impossible duration/tokens")
        if step.state_hash != state_hash(step.state_snapshot):
            raise TraceValidationError(f"run {run.id}: step {i} state_hash does not match snapshot")

    if run.status == "failed":
        if run.injected_class not in FAILURE_CLASSES:
            raise TraceValidationError(f"run {run.id}: failed run has no valid injected_class")
        if run.true_failure_step is None:
            raise TraceValidationError(f"run {run.id}: failed run has no true_failure_step")
        if not 0 <= run.true_failure_step < len(steps):
            raise TraceValidationError(
                f"run {run.id}: true_failure_step {run.true_failure_step} outside 0..{len(steps) - 1}"
            )
    else:
        if run.injected_class is not None or run.true_failure_step is not None:
            raise TraceValidationError(f"run {run.id}: successful run carries ground-truth labels")

    expected_tokens = sum(s.tokens for s in steps)
    if run.total_tokens != expected_tokens:
        raise TraceValidationError(
            f"run {run.id}: total_tokens {run.total_tokens} != sum of steps {expected_tokens}"
        )
    expected_duration = sum(s.duration_ms for s in steps)
    if run.duration_ms != expected_duration:
        raise TraceValidationError(
            f"run {run.id}: duration_ms {run.duration_ms} != sum of steps {expected_duration}"
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
