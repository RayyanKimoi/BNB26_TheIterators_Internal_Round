"""Pydantic models — the single source of truth for the diagnosis contract.

Every consumer reads this exact shape: the UI, the Slack alert, the
regression-test generator, and the backend test script. If a field changes
here, it changes everywhere.

These models are pure data contracts (Pydantic BaseModel), NOT database
tables. Database tables are defined separately using SQLModel in db.py.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Diagnosis contract — the object returned by POST /runs/{id}/diagnose
# ---------------------------------------------------------------------------


class SuggestedFix(BaseModel):
    """One candidate fix proposed by the Gemini explainer."""

    rank: int = Field(..., description="1-indexed rank among candidate fixes")
    patch: dict[str, Any] = Field(
        default_factory=dict,
        description="The proposed change, as a JSON patch or key-value override",
    )
    rationale: str = Field("", description="Why this fix addresses the root cause")


class DiagnosisResponse(BaseModel):
    """The structured JSON diagnosis returned to every consumer.

    Shape matches the contract in PRD.md, CLAUDE.md, and the output of
    ``model.predict.Localizer.diagnose()``. This Pydantic model is the
    absolute source of truth.
    """

    run_id: str | None = Field(None, description="UUID of the diagnosed run")
    flagged_step_index: int | None = Field(
        None, description="Index of the top-ranked step (the suspected root cause)"
    )
    confidence: float = Field(
        0.0,
        description=(
            "The flagged step's share of the total suspicion score, 0 to 1. "
            "Below the step threshold the system returns predicted_class='unknown'."
        ),
    )
    predicted_class: str = Field(
        "unknown",
        description=(
            "One of the 7 failure classes, or 'unknown' when confidence is below "
            "threshold or the class head cannot name the failure mode"
        ),
    )
    evidence_path: str | None = Field(
        None,
        description=(
            "Exact field that went wrong, e.g. 'step[14].output.currency'. "
            "P1: null until field-level localization is built."
        ),
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw feature values on the flagged step (all 10 PRD columns)",
    )
    shap: dict[str, float] | None = Field(
        None,
        description=(
            "SHAP feature attributions for the flagged step. P1: null until "
            "SHAP integration is built. Keys are feature names, values are "
            "SHAP values showing each feature's contribution to the score."
        ),
    )
    step_scores: list[int] = Field(
        default_factory=list,
        description=(
            "Suspicion score per step, 0 to 100, summing to ~100. "
            "Drives the heatmap colour on the Trace screen."
        ),
    )
    explanation: str | None = Field(
        None,
        description=(
            "Gemini plain-English root cause explanation. "
            "P1: null until the Gemini explainer is built."
        ),
    )
    suggested_fixes: list[SuggestedFix] = Field(
        default_factory=list,
        description="Ranked candidate fixes proposed by the Gemini explainer",
    )
    class_confidence: float = Field(
        0.0,
        description=(
            "The class head's probability for the named class. "
            "0.0 whenever predicted_class is 'unknown', including when the "
            "invariant tier flagged the step (that tier never names a class)."
        ),
    )
    unknown_reason: str | None = Field(
        None,
        description=(
            "Why the system declined to name a class. Null when a class is "
            "named successfully. Shows in the inspector panel."
        ),
    )
    anomaly_signal: str | None = Field(
        None,
        description=(
            "Set when the step was flagged by the distribution-relative "
            "invariant tier rather than the classifier: 'token_collapse' or "
            "'state_repetition'. These name an OBSERVATION, never a failure "
            "class. Whenever this is set, predicted_class is 'unknown', "
            "because the tier localizes a step without claiming to recognize "
            "a failure mode the model was never trained on."
        ),
    )


# ---------------------------------------------------------------------------
# Request / response models for API endpoints
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    """Compact run info for the GET /runs list."""

    id: str
    source: str
    task_type: str
    status: str
    injected_class: str | None = None
    total_tokens: int
    duration_ms: int
    created_at: datetime
    parent_run_id: str | None = None
    forked_at_step: int | None = None
    has_diagnosis: bool = False
    total_steps: int = 0


class StepDetail(BaseModel):
    """One step inside a run trace."""

    id: str
    run_id: str
    step_index: int
    action_type: str
    tool_name: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
    state_hash: str = ""
    duration_ms: int = 0
    tokens: int = 0
    error_flag: bool = False


class RunDetail(BaseModel):
    """Full run trace with steps, returned by GET /runs/{id}."""

    id: str
    user_id: str | None = None
    source: str
    task_type: str
    status: str
    parent_run_id: str | None = None
    forked_at_step: int | None = None
    fix_applied: dict[str, Any] | None = None
    injected_class: str | None = None
    true_failure_step: int | None = None
    total_tokens: int
    duration_ms: int
    created_at: datetime
    steps: list[StepDetail] = Field(default_factory=list)
    diagnosis: DiagnosisResponse | None = None


# Note: there is deliberately no EvaluationResponse model here. GET
# /model/evaluation serves model/artifacts/evaluation.json and metrics.json
# verbatim as `{"evaluation": {...}, "metrics": {...}}` (see main.py); a prior
# version of this file declared a flat Pydantic shape for that endpoint that
# the endpoint never actually used, which let the frontend contract drift
# silently out of sync with the real response. frontend/src/types/api.ts
# documents the real nested shape directly instead of mirroring a model that
# does not exist on the wire.


# ---------------------------------------------------------------------------
# Phase 4: explain and fork
# ---------------------------------------------------------------------------


class ExplainRequest(BaseModel):
    """Body for POST /runs/{id}/explain."""

    step_index: int = Field(
        ..., ge=0, description="Index of the step to explain, usually the flagged step"
    )


class ExplanationPayload(BaseModel):
    """The Gemini explainer's enforced output shape.

    Stored verbatim in `agent_runs.explanation` (JSONB) and returned by the
    endpoint. The LLM never chooses the step; it explains the one the model
    flagged.
    """

    root_cause: str = Field(..., description="Why this step is the origin of the failure")
    evidence_summary: list[str] = Field(
        default_factory=list,
        description="Key JSON attributes and values that triggered the failure",
    )
    proposed_fix: str = Field(
        "", description="Code diff or modified JSON payload that resolves the issue"
    )


class ExplanationResponse(ExplanationPayload):
    """Explanation plus the context needed to render it in the inspector."""

    run_id: str
    step_index: int
    predicted_class: str = Field(
        "unknown", description="Class the model assigned to the flagged step"
    )
    shap: dict[str, float] | None = Field(
        None, description="SHAP attributions that seeded the explanation"
    )
    cached: bool = Field(
        False, description="True when returned from agent_runs.explanation without a new LLM call"
    )


class ForkRequest(BaseModel):
    """Body for POST /runs/{id}/fork."""

    from_step: int = Field(..., ge=0, description="Step index to diverge at")
    fix_payload: dict[str, Any] | None = Field(
        None, description="Patch merged into the forked step's output"
    )
    override_code: str | None = Field(
        None, description="Free-form override recorded on the fork for audit"
    )


class ForkResponse(BaseModel):
    """Result of a fork: the child run, its replayed steps, and the outcome."""

    child_run_id: str
    parent_run_id: str
    status: str = Field(..., description="'completed' once the suffix has been replayed")
    outcome: str = Field(..., description="'success' or 'failed' after re-diagnosis")
    parent_outcome: str = Field(..., description="The original run's status, for comparison")
    forked_at_step: int
    steps_replayed: int = Field(..., description="How many suffix steps were re-executed")
    steps_total: int
    fix_applied: dict[str, Any] | None = None
    steps: list[StepDetail] = Field(default_factory=list)
    diagnosis: DiagnosisResponse | None = Field(
        None, description="Fresh diagnosis of the child run"
    )
