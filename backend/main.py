"""Black Box — FastAPI backend.

Endpoints:
    GET  /runs                  List all runs
    GET  /runs/{id}             Full trace with steps and diagnosis
    GET  /model/evaluation      Serve evaluation.json and metrics.json
    POST /runs/{id}/diagnose    Run the Hybrid Sentry Engine, save to DB, return result
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, select

from backend.db import AgentRun, Diagnosis, Step, StepScore
from backend.engine import create_db_and_tables, get_session
from backend.models import (
    DiagnosisResponse,
    EvaluationResponse,
    RunDetail,
    RunSummary,
    StepDetail,
)

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
    result = []
    for run in runs:
        # Check if a diagnosis exists for this run
        diag = session.exec(
            select(Diagnosis).where(Diagnosis.run_id == run.id)
        ).first()
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
                has_diagnosis=diag is not None,
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

    # Save Diagnosis to DB
    diag = Diagnosis(
        run_id=run_id,
        flagged_step_index=raw_diagnosis["flagged_step_index"],
        confidence=raw_diagnosis["confidence"],
        predicted_class=raw_diagnosis["predicted_class"],
        evidence_path=raw_diagnosis["evidence_path"],
        evidence=raw_diagnosis["evidence"],
        feature_vector=raw_diagnosis["evidence"],  # same as evidence for now
        explanation=raw_diagnosis["explanation"],
        suggested_fixes=raw_diagnosis["suggested_fixes"],
        shap=raw_diagnosis.get("shap"),
        class_confidence=raw_diagnosis["class_confidence"],
        unknown_reason=raw_diagnosis["unknown_reason"],
    )
    session.add(diag)
    session.flush()  # get the diag.id

    # Save individual StepScore rows
    for i, score in enumerate(raw_diagnosis["step_scores"]):
        step_score = StepScore(
            diagnosis_id=diag.id,
            step_index=i,
            suspicion_score=float(score),
        )
        session.add(step_score)

    session.commit()
    session.refresh(diag)

    # Return the Pydantic response
    return DiagnosisResponse(
        run_id=raw_diagnosis["run_id"],
        flagged_step_index=raw_diagnosis["flagged_step_index"],
        confidence=raw_diagnosis["confidence"],
        predicted_class=raw_diagnosis["predicted_class"],
        evidence_path=raw_diagnosis["evidence_path"],
        evidence=raw_diagnosis["evidence"],
        shap=raw_diagnosis.get("shap"),
        step_scores=raw_diagnosis["step_scores"],
        explanation=raw_diagnosis["explanation"],
        suggested_fixes=[],
        class_confidence=raw_diagnosis["class_confidence"],
        unknown_reason=raw_diagnosis["unknown_reason"],
    )
