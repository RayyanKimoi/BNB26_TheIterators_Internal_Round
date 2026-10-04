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
    # PRD item 14: "Gemini proposes 2 to 3 fixes, fork all, show which passes."
    # Defaults to empty so an explainer (or a cached explanation) that only
    # returns the original three keys still validates — the three-key contract
    # is widened here, never broken.
    fix_candidates: list[SuggestedFix] = Field(
        default_factory=list,
        description="Up to 3 ranked candidate patches, each forkable on its own",
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


# ---------------------------------------------------------------------------
# Completion pass: comparison, similarity search, regression tests, the
# reliability dashboard, and OTel ingestion.
# ---------------------------------------------------------------------------


class StepDiff(BaseModel):
    """One aligned position in the two runs' step sequences.

    "Aligned" means by step_index, not by meaning — a fork's suffix is the
    same length as its parent's from `forked_at_step` on, but two arbitrary
    runs being compared may simply differ in length, which `a_present` /
    `b_present` make explicit rather than silently truncating.
    """

    step_index: int
    a_present: bool
    b_present: bool
    a_tool_name: str | None = None
    b_tool_name: str | None = None
    a_error_flag: bool | None = None
    b_error_flag: bool | None = None
    tool_changed: bool = False
    error_flag_changed: bool = False
    output_changed: bool = False
    changed_output_keys: list[str] = Field(default_factory=list)


class CompareResponse(BaseModel):
    """GET /runs/{id}/compare/{other_id}: aligned step-by-step diff."""

    run_a_id: str
    run_b_id: str
    run_a_status: str
    run_b_status: str
    steps_compared: int = Field(..., description="max(len(a.steps), len(b.steps))")
    diffs: list[StepDiff]


class SimilarRun(BaseModel):
    """One historical match from GET /runs/{id}/similar."""

    run_id: str
    similarity: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity, 1.0 = identical")
    predicted_class: str
    flagged_step_index: int
    injected_class: str | None = None


class SimilarRunsResponse(BaseModel):
    run_id: str
    compared_against: int = Field(..., description="How many other diagnosed runs were searched")
    matches: list[SimilarRun]


class RegressionTestRequest(BaseModel):
    """Body for POST /runs/{id}/regression-test.

    `diagnosis_id` is optional: omit it to assert against the run's current
    (most recent) diagnosis, which is the common case right after confirming
    a fix worked.
    """

    diagnosis_id: str | None = None
    assertion: dict[str, Any] = Field(
        ..., description="What must still hold true, e.g. {'predicted_class': 'stale_retrieval'}"
    )


class RegressionTestResponse(BaseModel):
    id: str
    diagnosis_id: str
    assertion: dict[str, Any]
    created_at: datetime


class FailureClassCount(BaseModel):
    injected_class: str
    count: int


class ReliabilityTrendPoint(BaseModel):
    """One daily bucket of the reliability trend."""

    date: str = Field(..., description="ISO date (UTC), e.g. 2026-10-04")
    total_runs: int
    success_count: int
    pass_rate: float
    total_tokens: int
    avg_duration_ms: float
    duration_zscore: float = Field(
        ..., description="This day's average duration vs. the whole period's mean/std"
    )


class ReliabilityResponse(BaseModel):
    """GET /dashboard/reliability: aggregate reliability over time.

    `estimated_cost_usd` is null unless the TOKEN_COST_PER_1K_USD environment
    variable is set. There is no configured per-token price anywhere in this
    project, and inventing one would put a fabricated dollar figure on a
    dashboard whose whole premise is reporting only measured numbers.
    """

    total_runs: int
    success_count: int
    failure_count: int
    overall_pass_rate: float
    failure_class_breakdown: list[FailureClassCount]
    trend: list[ReliabilityTrendPoint]
    total_tokens: int
    estimated_cost_usd: float | None = None
    cost_rate_configured: bool = Field(
        ..., description="Whether TOKEN_COST_PER_1K_USD was set for this read"
    )


class OtelSpan(BaseModel):
    """One span in a simplified OTel-like ingestion request.

    Not full OTLP: real OTLP is resourceSpans -> scopeSpans -> spans protobuf
    JSON, considerably more than this project's ingestion needs justify. This
    is the flattened shape ingest/otel actually consumes, documented as such
    rather than claimed to be spec-complete.
    """

    span_id: str
    name: str
    start_time_unix_nano: int
    end_time_unix_nano: int
    attributes: dict[str, Any] = Field(default_factory=dict)
    status_code: str = Field("OK", description="'OK' or 'ERROR', OTel's own convention")


class OtelIngestRequest(BaseModel):
    trace_id: str
    task_type: str = "otel_ingest"
    spans: list[OtelSpan]


class OtelIngestResponse(BaseModel):
    run_id: str
    source: str = "otel"
    steps_created: int
    status: str


class SettingsResponse(BaseModel):
    """GET /settings: which configuration is present, never its values.

    Deliberately reports booleans for every secret. A dashboard that echoed
    `SLACK_WEBHOOK_URL` or `GEMINI_API_KEY` back to the browser would put
    server credentials in client memory, the browser cache and any screen
    recording of a demo, for no benefit over "configured: yes".
    """

    gemini_configured: bool
    gemini_model: str
    slack_configured: bool
    token_cost_configured: bool
    dashboard_base_url: str
    database_dialect: str = Field(
        ..., description="e.g. 'postgresql' or 'sqlite'. Never the full URL: it holds the password"
    )
    model_artifact_present: bool
    trained_classes: list[str]
    held_out_classes: list[str]
