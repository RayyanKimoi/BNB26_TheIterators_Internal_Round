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
from sqlalchemy import func
from sqlmodel import Session, select

from backend.db import AgentRun, Diagnosis, Step, StepScore
from backend.engine import create_db_and_tables, get_session
from backend.explainer import ExplainerClient, GeminiExplainer, explain
from backend.models import (
    DiagnosisResponse,
    EvaluationResponse,
    ExplainRequest,
    ExplanationResponse,
    ForkRequest,
    ForkResponse,
    RunDetail,
    RunSummary,
    StepDetail,
)
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
