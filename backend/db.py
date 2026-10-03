"""Database tables — SQLModel ORM mapping to the 6 PRD tables.

Tables:
    1. AgentRun      — agent_runs
    2. Step           — steps
    3. Diagnosis      — diagnoses
    4. StepScore      — step_scores
    5. RegressionTest — regression_tests

Uses SQLModel (Pydantic + SQLAlchemy) so each table row is also a validated
Pydantic model. Connection string comes from DATABASE_URL in .env.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from sqlalchemy import JSON, Text
from sqlmodel import Column, Field, Relationship, SQLModel


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 1. agent_runs
# ---------------------------------------------------------------------------


class AgentRun(SQLModel, table=True):
    __tablename__ = "agent_runs"

    id: str = Field(default_factory=_uuid, primary_key=True)
    user_id: Optional[str] = Field(default=None)
    source: str = Field(default="synthetic")  # synthetic / langgraph / otel
    task_type: str = Field(default="")
    status: str = Field(default="failed")  # success / failed
    # Fork lineage. parent_run_id is a real self-referential FK so a fork can
    # never point at a run that does not exist.
    parent_run_id: Optional[str] = Field(default=None, foreign_key="agent_runs.id")
    # PRD Data Model calls this `forked_at_step`. Phase 4 spec called it
    # `fork_step_index`; the PRD name is kept because the frozen schema, the
    # generator `Run` dataclass and the existing API models all use it.
    forked_at_step: Optional[int] = Field(default=None)
    fix_applied: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    # Structured Gemini root-cause analysis, written by POST /runs/{id}/explain.
    # JSON (JSONB on Postgres) holding root_cause, evidence_summary, proposed_fix.
    explanation: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    injected_class: Optional[str] = Field(default=None)
    true_failure_step: Optional[int] = Field(default=None)
    total_tokens: int = Field(default=0)
    duration_ms: int = Field(default=0)
    created_at: datetime = Field(default_factory=_now)

    # Relationships
    steps: List["Step"] = Relationship(back_populates="run")
    diagnoses: List["Diagnosis"] = Relationship(back_populates="run")


# ---------------------------------------------------------------------------
# 2. steps
# ---------------------------------------------------------------------------


class Step(SQLModel, table=True):
    __tablename__ = "steps"

    id: str = Field(default_factory=_uuid, primary_key=True)
    run_id: str = Field(foreign_key="agent_runs.id")
    step_index: int = Field(default=0)
    action_type: str = Field(default="call_tool")  # call_llm / call_tool / decide
    tool_name: Optional[str] = Field(default=None)
    input: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    output: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    state_snapshot: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    state_hash: str = Field(default="")
    duration_ms: int = Field(default=0)
    tokens: int = Field(default=0)
    error_flag: bool = Field(default=False)

    # Relationship
    run: Optional["AgentRun"] = Relationship(back_populates="steps")


# ---------------------------------------------------------------------------
# 3. diagnoses
# ---------------------------------------------------------------------------


class Diagnosis(SQLModel, table=True):
    __tablename__ = "diagnoses"

    id: str = Field(default_factory=_uuid, primary_key=True)
    run_id: str = Field(foreign_key="agent_runs.id")
    flagged_step_index: int = Field(default=0)
    confidence: float = Field(default=0.0)
    predicted_class: str = Field(default="unknown")
    evidence_path: Optional[str] = Field(default=None, sa_column=Column(Text))
    evidence: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    feature_vector: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    explanation: Optional[str] = Field(default=None, sa_column=Column(Text))
    suggested_fixes: Optional[list] = Field(default=None, sa_column=Column(JSON))
    # Extra fields from the unified contract
    shap: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    class_confidence: float = Field(default=0.0)
    unknown_reason: Optional[str] = Field(default=None, sa_column=Column(Text))
    # Set when the invariant tier flagged the step instead of the classifier.
    # Was being passed to this row without a column to land in, so it was
    # silently dropped on every write.
    anomaly_signal: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=_now)

    # Relationships
    run: Optional["AgentRun"] = Relationship(back_populates="diagnoses")
    scores: List["StepScore"] = Relationship(back_populates="diagnosis")


# ---------------------------------------------------------------------------
# 4. step_scores
# ---------------------------------------------------------------------------


class StepScore(SQLModel, table=True):
    __tablename__ = "step_scores"

    id: str = Field(default_factory=_uuid, primary_key=True)
    diagnosis_id: str = Field(foreign_key="diagnoses.id")
    step_index: int = Field(default=0)
    suspicion_score: float = Field(default=0.0)

    # Relationship
    diagnosis: Optional["Diagnosis"] = Relationship(back_populates="scores")


# ---------------------------------------------------------------------------
# 5. regression_tests
# ---------------------------------------------------------------------------


class RegressionTest(SQLModel, table=True):
    __tablename__ = "regression_tests"

    id: str = Field(default_factory=_uuid, primary_key=True)
    diagnosis_id: str = Field(foreign_key="diagnoses.id")
    assertion: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_now)
