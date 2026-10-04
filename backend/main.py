"""Black Box — FastAPI backend.

Endpoints:
    GET  /runs                          List all runs
    GET  /runs/{id}                     Full trace with steps and diagnosis
    GET  /model/evaluation              Serve evaluation.json and metrics.json
    POST /runs/{id}/diagnose            Run the Hybrid Sentry Engine, save to DB, return result
    POST /runs/{id}/explain             Gemini root cause for one step
    POST /runs/{id}/fork                Fork + deterministic suffix replay
    GET  /runs/{id}/compare/{other_id}  Aligned step-by-step diff between any two runs
    GET  /runs/{id}/similar             Cosine similarity search over stored feature vectors
    POST /runs/{id}/regression-test     Persist a confirmed fix assertion
    GET  /dashboard/reliability         Aggregate pass rate, failure mix, trend over time
    POST /ingest/otel                   Map a simplified OTel-like trace into the schema
"""

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlmodel import Session, select

from backend.alerts import send_diagnosis_alert
from backend.db import AgentRun, Diagnosis, RegressionTest, Step, StepScore
from backend.engine import create_db_and_tables, get_session
from backend.explainer import ExplainerClient, GeminiExplainer, explain
from backend.models import (
    CompareResponse,
    DiagnosisResponse,
    ExplainRequest,
    ExplanationResponse,
    FailureClassCount,
    ForkRequest,
    ForkResponse,
    OtelIngestRequest,
    OtelIngestResponse,
    OtelSpan,
    RegressionTestRequest,
    RegressionTestResponse,
    ReliabilityResponse,
    ReliabilityTrendPoint,
    RunDetail,
    RunSummary,
    SettingsResponse,
    SimilarRun,
    SimilarRunsResponse,
    StepDetail,
    StepDiff,
)
from generator.schema import state_hash
from model.predict import EVIDENCE_FEATURES
from replay.engine import outcome_of, replay_suffix

app = FastAPI(
    title="Black Box",
    description="Sentry for AI Agents — find the step that broke your agent run",
    version="0.1.0",
)

# Allow the React frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    """Create all DB tables on first run. Skips if they already exist."""
    create_db_and_tables()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EVALUATION_PATH = Path("model/artifacts/evaluation.json")
METRICS_PATH = Path("model/artifacts/metrics.json")

# Lazy-loaded Localizer — expensive, so we load it once on first diagnose call.
_localizer = None


def _get_localizer():
    """Load the Hybrid Sentry Engine (ML model + invariant fallback) once."""
    global _localizer
    if _localizer is None:
        from model.predict import Localizer
        _localizer = Localizer.load()
    return _localizer


_attributor = None


def _get_attributor():
    """Lazily build the SHAP attributor over the loaded localizer."""
    global _attributor
    if _attributor is None:
        from model.attribution import ShapAttributor

        _attributor = ShapAttributor(_get_localizer().localizer)
    return _attributor


def get_explainer_client() -> ExplainerClient:
    """FastAPI dependency for the LLM explainer.

    A dependency rather than a direct call so tests override it and never
    touch the network, and so the provider stays swappable as PRD.md requires.
    """
    return GeminiExplainer()


def _shap_for(trace: dict[str, Any], step_index: int) -> dict[str, float] | None:
    """SHAP attributions for one step, or None if unavailable.

    Never raises: SHAP is P1 evidence, and losing one inspector block is far
    better than failing the whole request.
    """
    try:
        from model.features import FeatureExtractor

        localizer = _get_localizer()
        frame = FeatureExtractor(localizer.stats, localizer.embedder).transform(trace)
        return _get_attributor().attribute(frame, step_index)
    except Exception:  # noqa: BLE001 - optional evidence
        return None


def _persist_diagnosis(session: Session, run_id: str, raw: dict[str, Any]) -> Diagnosis:
    """Write a Diagnosis row plus one StepScore per step.

    Shared by POST /runs/{id}/diagnose and the re-diagnosis a fork performs,
    so the two can never drift apart in what they store.
    """
    diag = Diagnosis(
        run_id=run_id,
        flagged_step_index=raw["flagged_step_index"],
        confidence=raw["confidence"],
        predicted_class=raw["predicted_class"],
        evidence_path=raw["evidence_path"],
        evidence=raw["evidence"],
        feature_vector=raw["evidence"],
        explanation=raw["explanation"],
        suggested_fixes=raw["suggested_fixes"],
        shap=raw.get("shap"),
        class_confidence=raw["class_confidence"],
        unknown_reason=raw["unknown_reason"],
        anomaly_signal=raw.get("anomaly_signal"),
    )
    session.add(diag)
    session.flush()
    for index, score in enumerate(raw["step_scores"]):
        session.add(
            StepScore(diagnosis_id=diag.id, step_index=index, suspicion_score=float(score))
        )
    return diag


def _diagnosis_response(raw: dict[str, Any]) -> DiagnosisResponse:
    """The 13-key contract, built in exactly one place."""
    return DiagnosisResponse(
        run_id=raw["run_id"],
        flagged_step_index=raw["flagged_step_index"],
        confidence=raw["confidence"],
        predicted_class=raw["predicted_class"],
        evidence_path=raw["evidence_path"],
        evidence=raw["evidence"],
        shap=raw.get("shap"),
        step_scores=raw["step_scores"],
        explanation=raw["explanation"],
        suggested_fixes=raw.get("suggested_fixes") or [],
        class_confidence=raw["class_confidence"],
        unknown_reason=raw["unknown_reason"],
        anomaly_signal=raw.get("anomaly_signal"),
    )


def _goal_of(trace: dict[str, Any]) -> str:
    for step in trace["steps"]:
        goal = (step.get("state_snapshot") or {}).get("goal")
        if isinstance(goal, str) and goal:
            return goal
    return ""


def _run_to_trace_dict(run: AgentRun, steps: list[Step]) -> dict[str, Any]:
    """Convert DB rows back into the dict format that Localizer.diagnose() expects."""
    return {
        "run": {
            "id": run.id,
            "user_id": run.user_id,
            "source": run.source,
            "task_type": run.task_type,
            "status": run.status,
            "parent_run_id": run.parent_run_id,
            "forked_at_step": run.forked_at_step,
            "fix_applied": run.fix_applied,
            "injected_class": run.injected_class,
            "true_failure_step": run.true_failure_step,
            "total_tokens": run.total_tokens,
            "duration_ms": run.duration_ms,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        },
        "steps": [
            {
                "id": s.id,
                "run_id": s.run_id,
                "step_index": s.step_index,
                "action_type": s.action_type,
                "tool_name": s.tool_name,
                "input": s.input or {},
                "output": s.output or {},
                "state_snapshot": s.state_snapshot or {},
                "state_hash": s.state_hash or "",
                "duration_ms": s.duration_ms,
                "tokens": s.tokens,
                "error_flag": s.error_flag,
            }
            for s in sorted(steps, key=lambda x: x.step_index)
        ],
    }


# ---------------------------------------------------------------------------
# GET /runs — list all runs
# ---------------------------------------------------------------------------

@app.get("/runs", response_model=list[RunSummary])
def list_runs(
    status: str | None = None,
    source: str | None = None,
    injected_class: str | None = None,
    session: Session = Depends(get_session),
):
    """List all runs, optionally filtered by status, source, or injected_class."""
    statement = select(AgentRun)
    if status:
        statement = statement.where(AgentRun.status == status)
    if source:
        statement = statement.where(AgentRun.source == source)
    if injected_class:
        statement = statement.where(AgentRun.injected_class == injected_class)
    statement = statement.order_by(AgentRun.created_at.desc())

    runs = session.exec(statement).all()

    # One grouped query each for step counts and diagnosis presence, rather
    # than two per run. Matters once the corpus is loaded in full.
    step_counts = dict(
        session.exec(select(Step.run_id, func.count(Step.id)).group_by(Step.run_id)).all()
    )
    diagnosed = set(session.exec(select(Diagnosis.run_id).distinct()).all())

    result = []
    for run in runs:
        result.append(
            RunSummary(
                id=run.id,
                source=run.source,
                task_type=run.task_type,
                status=run.status,
                injected_class=run.injected_class,
                total_tokens=run.total_tokens,
                duration_ms=run.duration_ms,
                created_at=run.created_at,
                parent_run_id=run.parent_run_id,
                forked_at_step=run.forked_at_step,
                has_diagnosis=run.id in diagnosed,
                total_steps=step_counts.get(run.id, 0),
            )
        )
    return result


# ---------------------------------------------------------------------------
# GET /runs/{id} — full trace with steps and diagnosis
# ---------------------------------------------------------------------------

@app.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str, session: Session = Depends(get_session)):
    """Fetch a run with all its steps and the latest diagnosis (if any)."""
    run = session.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    steps = session.exec(
        select(Step).where(Step.run_id == run_id).order_by(Step.step_index)
    ).all()

    # Get the latest diagnosis
    diag = session.exec(
        select(Diagnosis)
        .where(Diagnosis.run_id == run_id)
        .order_by(Diagnosis.created_at.desc())
    ).first()

    diagnosis_resp = None
    if diag:
        # Fetch step scores for this diagnosis
        scores = session.exec(
            select(StepScore)
            .where(StepScore.diagnosis_id == diag.id)
            .order_by(StepScore.step_index)
        ).all()
        step_score_list = [round(s.suspicion_score) for s in scores]

        diagnosis_resp = DiagnosisResponse(
            run_id=diag.run_id,
            flagged_step_index=diag.flagged_step_index,
            confidence=diag.confidence,
            predicted_class=diag.predicted_class,
            evidence_path=diag.evidence_path,
            evidence=diag.evidence or {},
            shap=diag.shap,
            step_scores=step_score_list,
            explanation=diag.explanation,
            suggested_fixes=[],
            class_confidence=diag.class_confidence,
            unknown_reason=diag.unknown_reason,
            anomaly_signal=getattr(diag, "anomaly_signal", None),
        )

    return RunDetail(
        id=run.id,
        user_id=run.user_id,
        source=run.source,
        task_type=run.task_type,
        status=run.status,
        parent_run_id=run.parent_run_id,
        forked_at_step=run.forked_at_step,
        fix_applied=run.fix_applied,
        injected_class=run.injected_class,
        true_failure_step=run.true_failure_step,
        total_tokens=run.total_tokens,
        duration_ms=run.duration_ms,
        created_at=run.created_at,
        steps=[
            StepDetail(
                id=s.id,
                run_id=s.run_id,
                step_index=s.step_index,
                action_type=s.action_type,
                tool_name=s.tool_name,
                input=s.input or {},
                output=s.output or {},
                state_snapshot=s.state_snapshot or {},
                state_hash=s.state_hash or "",
                duration_ms=s.duration_ms,
                tokens=s.tokens,
                error_flag=s.error_flag,
            )
            for s in steps
        ],
        diagnosis=diagnosis_resp,
    )


# ---------------------------------------------------------------------------
# GET /model/evaluation — serve the evaluation artifacts
# ---------------------------------------------------------------------------

@app.get("/model/evaluation")
def get_evaluation():
    """Serve the evaluation.json and metrics.json artifacts.

    These contain the real numbers: 95% trained, 20% held-out (hybrid),
    per-class accuracy, and baseline comparisons.
    """
    result: dict[str, Any] = {}
    if EVALUATION_PATH.exists():
        result["evaluation"] = json.loads(
            EVALUATION_PATH.read_text(encoding="utf-8")
        )
    if METRICS_PATH.exists():
        result["metrics"] = json.loads(
            METRICS_PATH.read_text(encoding="utf-8")
        )
    if not result:
        raise HTTPException(
            status_code=404,
            detail="No evaluation artifacts found. Run: python -m model.evaluate",
        )
    return result


# ---------------------------------------------------------------------------
# POST /runs/{id}/diagnose — run the Hybrid Sentry Engine
# ---------------------------------------------------------------------------

@app.post("/runs/{run_id}/diagnose", response_model=DiagnosisResponse)
def diagnose_run(run_id: str, session: Session = Depends(get_session)):
    """Score every step using the Hybrid Sentry Engine, save to DB, return result.

    The Hybrid Engine:
    1. ML model (HistGradientBoosting) scores every step 0-100
    2. If the ML model returns 'unknown', statistical invariant rules
       check for context_truncation (token_z collapse) and infinite_loop
       (state_hash explosion)

    Saves both the Diagnosis row and individual StepScore rows to the DB.
    """
    run = session.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    steps = session.exec(
        select(Step).where(Step.run_id == run_id).order_by(Step.step_index)
    ).all()

    if not steps:
        raise HTTPException(
            status_code=400, detail=f"Run {run_id} has no steps to diagnose"
        )

    # Convert DB rows to the dict format the Localizer expects
    trace_dict = _run_to_trace_dict(run, steps)

    # Run the Hybrid Sentry Engine
    localizer = _get_localizer()
    raw_diagnosis = localizer.diagnose(trace_dict)

    # SHAP fills the `shap` contract field, which was previously always null.
    raw_diagnosis["shap"] = _shap_for(trace_dict, raw_diagnosis["flagged_step_index"])

    _persist_diagnosis(session, run_id, raw_diagnosis)
    session.commit()

    # Best-effort, non-blocking: never let a Slack outage fail a diagnosis.
    try:
        send_diagnosis_alert(
            run_id=run_id,
            task_type=run.task_type,
            flagged_step_index=raw_diagnosis["flagged_step_index"],
            predicted_class=raw_diagnosis["predicted_class"],
            evidence=raw_diagnosis.get("evidence"),
            confidence=raw_diagnosis.get("confidence"),
        )
    except Exception:  # noqa: BLE001 - an alert must never fail the request
        pass

    return _diagnosis_response(raw_diagnosis)


# ---------------------------------------------------------------------------
# POST /runs/{id}/explain
# ---------------------------------------------------------------------------


@app.post("/runs/{run_id}/explain", response_model=ExplanationResponse)
def explain_run(
    run_id: str,
    body: ExplainRequest,
    session: Session = Depends(get_session),
    client: ExplainerClient = Depends(get_explainer_client),
):
    """Plain-English root cause and a proposed fix for one step.

    The ML model owns localization; this only explains the step it was given.
    Results are cached in `agent_runs.explanation`, so repeat views cost
    nothing and a rate-limited demo still renders.
    """
    run = session.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    steps = session.exec(
        select(Step).where(Step.run_id == run_id).order_by(Step.step_index)
    ).all()
    if not steps:
        raise HTTPException(status_code=400, detail=f"Run {run_id} has no steps")
    if not 0 <= body.step_index < len(steps):
        raise HTTPException(
            status_code=400,
            detail=f"step_index {body.step_index} is outside the trace (0..{len(steps) - 1})",
        )

    cached = run.explanation or {}
    if cached.get("step_index") == body.step_index and cached.get("root_cause"):
        return ExplanationResponse(**{**cached, "run_id": run_id, "cached": True})

    trace = _run_to_trace_dict(run, steps)
    step = trace["steps"][body.step_index]

    latest = session.exec(
        select(Diagnosis)
        .where(Diagnosis.run_id == run_id)
        .order_by(Diagnosis.created_at.desc())
    ).first()
    predicted_class = latest.predicted_class if latest else "unknown"
    evidence = (latest.evidence if latest else None) or {}

    shap_values = (latest.shap if latest else None) or _shap_for(trace, body.step_index)

    low = max(0, body.step_index - 3)
    high = min(len(trace["steps"]), body.step_index + 4)
    context_steps = trace["steps"][low:high]

    try:
        payload = explain(
            client,
            goal=_goal_of(trace),
            step=step,
            step_index=body.step_index,
            evidence=evidence,
            shap=shap_values,
            predicted_class=predicted_class,
            context_steps=context_steps,
        )
    except Exception as exc:  # noqa: BLE001 - upstream LLM or quota failure
        raise HTTPException(
            status_code=502, detail=f"explainer call failed: {exc}"
        ) from exc

    stored = payload.model_dump()
    stored.update({"step_index": body.step_index, "predicted_class": predicted_class})
    if shap_values:
        stored["shap"] = shap_values
    run.explanation = stored
    session.add(run)

    if latest is not None:
        latest.explanation = payload.root_cause
        session.add(latest)
    session.commit()

    return ExplanationResponse(**{**stored, "run_id": run_id, "cached": False})


# ---------------------------------------------------------------------------
# POST /runs/{id}/fork
# ---------------------------------------------------------------------------


@app.post("/runs/{run_id}/fork", response_model=ForkResponse)
def fork_run(
    run_id: str,
    body: ForkRequest,
    session: Session = Depends(get_session),
):
    """Fork a run at `from_step` with a fix, replay the suffix, re-diagnose.

    The parent is never modified. See `replay/engine.py` for exactly what
    "replay the suffix" means for a stored trace.
    """
    parent = session.get(AgentRun, run_id)
    if not parent:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    parent_steps = session.exec(
        select(Step).where(Step.run_id == run_id).order_by(Step.step_index)
    ).all()
    if not parent_steps:
        raise HTTPException(status_code=400, detail=f"Run {run_id} has no steps to fork")

    child = AgentRun(
        user_id=parent.user_id,
        source=parent.source,
        task_type=parent.task_type,
        status="failed",  # replaced below once the replay is diagnosed
        parent_run_id=parent.id,
        forked_at_step=body.from_step,
        fix_applied=body.fix_payload,
        # Ground-truth labels belong to the parent. A fork is not a labelled
        # synthetic run: the fault may well be gone, so claiming its class
        # would corrupt any evaluation that later reads this table.
        injected_class=None,
        true_failure_step=None,
    )

    trace = _run_to_trace_dict(parent, parent_steps)
    try:
        result = replay_suffix(
            trace["steps"],
            from_step=body.from_step,
            child_run_id=child.id,
            fix_payload=body.fix_payload,
            override_code=body.override_code,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    child.status = outcome_of(result.steps)
    child.total_tokens = sum(int(s.get("tokens") or 0) for s in result.steps)
    child.duration_ms = sum(int(s.get("duration_ms") or 0) for s in result.steps)
    session.add(child)

    for payload in result.steps:
        session.add(
            Step(
                id=payload["id"],
                run_id=child.id,
                step_index=payload["step_index"],
                action_type=payload["action_type"],
                tool_name=payload.get("tool_name"),
                input=payload.get("input") or {},
                output=payload.get("output") or {},
                state_snapshot=payload.get("state_snapshot") or {},
                state_hash=payload.get("state_hash") or "",
                duration_ms=int(payload.get("duration_ms") or 0),
                tokens=int(payload.get("tokens") or 0),
                error_flag=bool(payload.get("error_flag")),
            )
        )
    session.flush()

    child_trace = {"run": {"id": child.id}, "steps": result.steps}
    raw = _get_localizer().diagnose(child_trace)
    raw["shap"] = _shap_for(child_trace, raw["flagged_step_index"])
    _persist_diagnosis(session, child.id, raw)
    session.commit()

    return ForkResponse(
        child_run_id=child.id,
        parent_run_id=parent.id,
        status="completed",
        outcome=child.status,
        parent_outcome=parent.status,
        forked_at_step=result.forked_at_step,
        steps_replayed=result.steps_replayed,
        steps_total=len(result.steps),
        fix_applied=result.fix_applied,
        steps=[
            StepDetail(
                id=s["id"],
                run_id=child.id,
                step_index=s["step_index"],
                action_type=s["action_type"],
                tool_name=s.get("tool_name"),
                input=s.get("input") or {},
                output=s.get("output") or {},
                state_snapshot=s.get("state_snapshot") or {},
                state_hash=s.get("state_hash") or "",
                duration_ms=int(s.get("duration_ms") or 0),
                tokens=int(s.get("tokens") or 0),
                error_flag=bool(s.get("error_flag")),
            )
            for s in result.steps
        ],
        diagnosis=_diagnosis_response(raw),
    )


# ---------------------------------------------------------------------------
# GET /runs/{id}/compare/{other_id} — aligned step-by-step diff
# ---------------------------------------------------------------------------


@app.get("/runs/{run_id}/compare/{other_id}", response_model=CompareResponse)
def compare_runs(run_id: str, other_id: str, session: Session = Depends(get_session)):
    """Aligned step-by-step diff between any two runs.

    Not limited to a fork pair: any two run ids diagnosed or not, same task or
    different, can be compared. Alignment is by step_index; a1_present /
    b_present make an unequal-length pair explicit rather than truncating.
    """
    run_a = session.get(AgentRun, run_id)
    run_b = session.get(AgentRun, other_id)
    if not run_a:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if not run_b:
        raise HTTPException(status_code=404, detail=f"Run {other_id} not found")

    steps_a = sorted(
        session.exec(select(Step).where(Step.run_id == run_id)).all(),
        key=lambda s: s.step_index,
    )
    steps_b = sorted(
        session.exec(select(Step).where(Step.run_id == other_id)).all(),
        key=lambda s: s.step_index,
    )
    by_a = {s.step_index: s for s in steps_a}
    by_b = {s.step_index: s for s in steps_b}
    length = max([*by_a.keys(), *by_b.keys(), -1]) + 1

    diffs: list[StepDiff] = []
    for i in range(length):
        a, b = by_a.get(i), by_b.get(i)
        a_out = (a.output or {}) if a else {}
        b_out = (b.output or {}) if b else {}
        changed_keys = sorted(
            key for key in set(a_out) | set(b_out) if a_out.get(key) != b_out.get(key)
        )
        diffs.append(
            StepDiff(
                step_index=i,
                a_present=a is not None,
                b_present=b is not None,
                a_tool_name=a.tool_name if a else None,
                b_tool_name=b.tool_name if b else None,
                a_error_flag=a.error_flag if a else None,
                b_error_flag=b.error_flag if b else None,
                tool_changed=bool(a and b and a.tool_name != b.tool_name),
                error_flag_changed=bool(a and b and a.error_flag != b.error_flag),
                output_changed=bool(changed_keys),
                changed_output_keys=changed_keys,
            )
        )

    return CompareResponse(
        run_a_id=run_id,
        run_b_id=other_id,
        run_a_status=run_a.status,
        run_b_status=run_b.status,
        steps_compared=length,
        diffs=diffs,
    )


# ---------------------------------------------------------------------------
# GET /runs/{id}/similar — cosine similarity over stored feature vectors
# ---------------------------------------------------------------------------


def _vectorize(evidence: dict[str, Any] | None) -> np.ndarray:
    """The ten evidence features as a fixed-order vector. None/NaN -> 0.0,
    which is a neutral midpoint for every one of these columns, not a
    fabricated reading: it means "no signal," the same as the column being
    absent, and only affects which runs end up looking similar, never any
    number shown as a diagnosis."""
    evidence = evidence or {}
    return np.array([float(evidence.get(name) or 0.0) for name in EVIDENCE_FEATURES], dtype=float)


@app.get("/runs/{run_id}/similar", response_model=SimilarRunsResponse)
def similar_runs(run_id: str, limit: int = 5, session: Session = Depends(get_session)):
    """Cosine similarity search over every other diagnosed run's stored
    feature_vector, to surface matching historical failures — the "have we
    seen this before" memory the inspector can point to."""
    target = session.exec(
        select(Diagnosis).where(Diagnosis.run_id == run_id).order_by(Diagnosis.created_at.desc())
    ).first()
    if not target:
        raise HTTPException(
            status_code=404, detail=f"Run {run_id} has no diagnosis to compare from"
        )

    target_vec = _vectorize(target.feature_vector)
    target_norm = float(np.linalg.norm(target_vec))

    candidates = session.exec(select(Diagnosis).order_by(Diagnosis.created_at.desc())).all()
    scored: list[tuple[float, Diagnosis]] = []
    seen_runs: set[str] = set()
    for diag in candidates:
        if diag.run_id == run_id or diag.run_id in seen_runs:
            continue
        seen_runs.add(diag.run_id)
        vec = _vectorize(diag.feature_vector)
        norm = float(np.linalg.norm(vec))
        similarity = (
            0.0 if target_norm == 0 or norm == 0 else float(np.dot(target_vec, vec) / (target_norm * norm))
        )
        scored.append((similarity, diag))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    matches = []
    for similarity, diag in scored[: max(0, limit)]:
        other_run = session.get(AgentRun, diag.run_id)
        matches.append(
            SimilarRun(
                run_id=diag.run_id,
                similarity=round(similarity, 4),
                predicted_class=diag.predicted_class,
                flagged_step_index=diag.flagged_step_index,
                injected_class=other_run.injected_class if other_run else None,
            )
        )

    return SimilarRunsResponse(run_id=run_id, compared_against=len(seen_runs), matches=matches)


# ---------------------------------------------------------------------------
# POST /runs/{id}/regression-test — persist a confirmed fix assertion
# ---------------------------------------------------------------------------


@app.post("/runs/{run_id}/regression-test", response_model=RegressionTestResponse)
def create_regression_test(
    run_id: str, body: RegressionTestRequest, session: Session = Depends(get_session)
):
    """Persist a confirmed fix assertion against a run's diagnosis.

    Defaults to the run's most recent diagnosis when `diagnosis_id` is
    omitted, the common case right after confirming a fork's fix worked.
    """
    run = session.get(AgentRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    if body.diagnosis_id:
        diagnosis = session.get(Diagnosis, body.diagnosis_id)
        if not diagnosis or diagnosis.run_id != run_id:
            raise HTTPException(
                status_code=400,
                detail=f"diagnosis {body.diagnosis_id} does not belong to run {run_id}",
            )
    else:
        diagnosis = session.exec(
            select(Diagnosis).where(Diagnosis.run_id == run_id).order_by(Diagnosis.created_at.desc())
        ).first()
        if not diagnosis:
            raise HTTPException(
                status_code=400,
                detail=f"Run {run_id} has no diagnosis to attach a regression test to",
            )

    test = RegressionTest(diagnosis_id=diagnosis.id, assertion=body.assertion)
    session.add(test)
    session.commit()
    session.refresh(test)

    return RegressionTestResponse(
        id=test.id,
        diagnosis_id=test.diagnosis_id,
        assertion=test.assertion or {},
        created_at=test.created_at,
    )


# ---------------------------------------------------------------------------
# GET /dashboard/reliability — aggregate reliability over time
# ---------------------------------------------------------------------------


@app.get("/dashboard/reliability", response_model=ReliabilityResponse)
def reliability_dashboard(session: Session = Depends(get_session)):
    """Pass rate, failure class breakdown, and a daily trend, computed
    directly from agent_runs. No model inference here — this is bookkeeping
    over what diagnose/fork have already written, not a new prediction.
    """
    runs = session.exec(select(AgentRun)).all()
    total = len(runs)
    successes = sum(1 for r in runs if r.status == "success")

    class_counts: dict[str, int] = {}
    for r in runs:
        if r.injected_class:
            class_counts[r.injected_class] = class_counts.get(r.injected_class, 0) + 1
    breakdown = [
        FailureClassCount(injected_class=cls, count=n) for cls, n in sorted(class_counts.items())
    ]

    buckets: dict[str, list[AgentRun]] = defaultdict(list)
    for r in runs:
        buckets[r.created_at.date().isoformat()].append(r)

    all_durations = [r.duration_ms for r in runs if r.duration_ms]
    period_mean = statistics.fmean(all_durations) if all_durations else 0.0
    period_std = statistics.pstdev(all_durations) if len(all_durations) > 1 else 0.0

    trend: list[ReliabilityTrendPoint] = []
    for day in sorted(buckets):
        day_runs = buckets[day]
        day_success = sum(1 for r in day_runs if r.status == "success")
        day_durations = [r.duration_ms for r in day_runs if r.duration_ms]
        day_mean = statistics.fmean(day_durations) if day_durations else 0.0
        zscore = (day_mean - period_mean) / period_std if period_std > 0 else 0.0
        trend.append(
            ReliabilityTrendPoint(
                date=day,
                total_runs=len(day_runs),
                success_count=day_success,
                pass_rate=round(day_success / len(day_runs), 4) if day_runs else 0.0,
                total_tokens=sum(r.total_tokens for r in day_runs),
                avg_duration_ms=round(day_mean, 1),
                duration_zscore=round(zscore, 3),
            )
        )

    total_tokens = sum(r.total_tokens for r in runs)
    rate_env = os.environ.get("TOKEN_COST_PER_1K_USD")
    estimated_cost = (total_tokens / 1000.0) * float(rate_env) if rate_env else None

    return ReliabilityResponse(
        total_runs=total,
        success_count=successes,
        failure_count=total - successes,
        overall_pass_rate=round(successes / total, 4) if total else 0.0,
        failure_class_breakdown=breakdown,
        trend=trend,
        total_tokens=total_tokens,
        estimated_cost_usd=round(estimated_cost, 4) if estimated_cost is not None else None,
        cost_rate_configured=rate_env is not None,
    )


# ---------------------------------------------------------------------------
# POST /ingest/otel — map a simplified OTel-like trace into the schema
# ---------------------------------------------------------------------------


def _classify_otel_span(span: OtelSpan) -> tuple[str, str | None]:
    """Heuristic action_type + tool_name from a span's name/attributes.

    There is no universal OTel semantic convention every agent framework
    follows, so this reads the handful of attribute names LangChain/LangGraph
    instrumentation and plain function-call spans commonly use, and falls
    back to "decide" for anything that names neither a tool nor an LLM call.
    """
    name_lower = span.name.lower()
    tool_name = span.attributes.get("tool.name") or span.attributes.get("function.name")
    if tool_name or "tool" in name_lower:
        return "call_tool", str(tool_name) if tool_name else span.name
    if any(token in name_lower for token in ("llm", "completion", "chat", "generate")):
        return "call_llm", None
    return "decide", None


@app.post("/ingest/otel", response_model=OtelIngestResponse)
def ingest_otel(body: OtelIngestRequest, session: Session = Depends(get_session)):
    """Map a simplified OTel-like trace into agent_runs + steps.

    Not full OTLP — see OtelSpan's docstring. Spans are ordered by start time
    into steps; `input.*` / `output.*` prefixed attributes become the step's
    input/output payloads, and anything carrying `status_code: "ERROR"` sets
    error_flag, which is enough for the detector to score the run exactly
    like a synthetic one.
    """
    if not body.spans:
        raise HTTPException(status_code=400, detail="trace has no spans")

    ordered = sorted(body.spans, key=lambda s: s.start_time_unix_nano)
    run = AgentRun(source="otel", task_type=body.task_type, status="failed")
    session.add(run)
    session.flush()

    total_tokens = 0
    total_duration = 0
    any_error = False

    for index, span in enumerate(ordered):
        action_type, tool_name = _classify_otel_span(span)
        duration_ms = max(0, (span.end_time_unix_nano - span.start_time_unix_nano) // 1_000_000)
        tokens = int(span.attributes.get("llm.usage.total_tokens") or span.attributes.get("tokens") or 0)
        error_flag = span.status_code.upper() == "ERROR"
        any_error = any_error or error_flag
        total_tokens += tokens
        total_duration += int(duration_ms)

        step_input = {k[len("input.") :]: v for k, v in span.attributes.items() if k.startswith("input.")}
        step_output = {
            k[len("output.") :]: v for k, v in span.attributes.items() if k.startswith("output.")
        } or {"name": span.name}
        snapshot = {"span_id": span.span_id, "attributes": span.attributes}

        session.add(
            Step(
                run_id=run.id,
                step_index=index,
                action_type=action_type,
                tool_name=tool_name,
                input=step_input,
                output=step_output,
                state_snapshot=snapshot,
                state_hash=state_hash(snapshot),
                duration_ms=int(duration_ms),
                tokens=tokens,
                error_flag=error_flag,
            )
        )

    run.status = "failed" if any_error else "success"
    run.total_tokens = total_tokens
    run.duration_ms = total_duration
    session.add(run)
    session.commit()

    return OtelIngestResponse(run_id=run.id, steps_created=len(ordered), status=run.status)


# ---------------------------------------------------------------------------
# GET /settings — which configuration is present, never its values
# ---------------------------------------------------------------------------


@app.get("/settings", response_model=SettingsResponse)
def get_settings():
    """Report what is configured so the Settings tab shows real state.

    Only booleans for anything secret. See SettingsResponse for why.
    """
    from backend.engine import DATABASE_URL
    from model.dataset import TRAIN_CLASSES
    from generator.schema import HELD_OUT_CLASSES

    artifact = Path(os.environ.get("MODEL_PATH", "model/artifacts/localizer.joblib"))

    return SettingsResponse(
        gemini_configured=bool(os.environ.get("GEMINI_API_KEY")),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        slack_configured=bool(os.environ.get("SLACK_WEBHOOK_URL")),
        token_cost_configured=bool(os.environ.get("TOKEN_COST_PER_1K_USD")),
        dashboard_base_url=os.environ.get("DASHBOARD_BASE_URL", "http://localhost:5173"),
        database_dialect=DATABASE_URL.split(":", 1)[0].split("+", 1)[0],
        model_artifact_present=artifact.exists(),
        trained_classes=list(TRAIN_CLASSES),
        held_out_classes=list(HELD_OUT_CLASSES),
    )
