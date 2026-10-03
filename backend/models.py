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
            "0.0 when predicted_class is 'unknown' or when the invariant tier named it."
        ),
    )
    unknown_reason: str | None = Field(
        None,
        description=(
            "Why the system declined to name a class. Null when a class is "
            "named successfully. Shows in the inspector panel."
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


class EvaluationResponse(BaseModel):
    """Model evaluation metrics served by GET /model/evaluation."""

    test_seen_top1: float
    test_seen_top3: float
    test_heldout_top1: float
    test_heldout_top3: float
    test_seen_runs: int
    test_heldout_runs: int
    baselines: dict[str, dict[str, float]]
    loco_mean: float | None = None
    gate_passed: bool
    # Hybrid engine metrics
    hybrid_seen_top1: float | None = None
    hybrid_heldout_top1: float | None = None
    hybrid_per_class: dict[str, float] | None = None
    hybrid_gate_passed: bool | None = None
