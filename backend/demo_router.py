"""Live demo endpoint: synthesize a fresh run, persist it, diagnose it.

`POST /demo/run-live` exists so a demo does not have to depend on a run that
was seeded hours earlier. It builds a brand new trace with the requested fault
injected, writes it to the database, and scores it with the same Hybrid Sentry
Engine that `POST /runs/{id}/diagnose` uses. The new run appears on `/runs`
immediately and its blame heatmap is ready before the response returns.

Additive by construction. No DB model, no existing endpoint and no diagnosis
code is modified here: every step reuses a helper that already exists.

* `generator.build.build_run` builds the trace, so a demo run is drawn from
  the same distribution as the training corpus rather than from a second,
  subtly different code path that would quietly invalidate the model's scores.
* `backend.seed_corpus.insert_run` persists it, so the row shape matches every
  seeded run exactly.
* `backend.seed_corpus.diagnose_run` scores it, which itself calls into
  `backend.main`, so there is exactly one implementation of diagnosis.

The flip side of reusing the real generator is that a demo run is synthetic
and its fault is injected, which is the only reason ground truth is known.
This endpoint does not claim to observe a real agent.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlmodel import Session

from backend.engine import get_session

logger = logging.getLogger("backend.demo")

router = APIRouter(prefix="/demo", tags=["demo"])

#: Matches the corpus generator's demo user so demo runs are attributable and
#: can be cleaned up as a group.
DEMO_USER_ID = "00000000-0000-4000-8000-000000000001"


class DemoRunRequest(BaseModel):
    """Body for POST /demo/run-live."""

    agent_type: str = Field(
        default="travel_booking",
        description="Which task to simulate. One of the generator's task types.",
    )
    glitch: str | None = Field(
        default="stale_retrieval",
        description=(
            "Fault class to inject, or null for a clean run that should pass. "
            "One of the seven classes in the taxonomy."
        ),
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Optional RNG seed. Omit for a genuinely fresh run each call; set "
            "it to reproduce one exactly, which is useful when rehearsing."
        ),
    )


class DemoRunResponse(BaseModel):
    """Result of POST /demo/run-live."""

    status: str
    run_id: str
    task_type: str
    injected_class: str | None
    true_failure_step: int | None
    flagged_step_index: int | None
    predicted_class: str | None
    step_count: int
    #: Whether the engine landed on the step the generator actually broke.
    #: Null for a clean run, where there is no true failure step to match.
    localized_correctly: bool | None


def _catalog() -> tuple[dict, tuple[str, ...]]:
    """Valid task types and fault classes, read from the generator itself."""
    from generator.schema import FAILURE_CLASSES
    from generator.tasks import TASKS_BY_NAME

    return TASKS_BY_NAME, FAILURE_CLASSES


@router.get("/options")
def demo_options() -> dict[str, list[str]]:
    """What `run-live` will accept. Saves callers guessing at valid values."""
    tasks, classes = _catalog()
    return {"agent_types": sorted(tasks), "glitches": sorted(classes)}


@router.post("/run-live", response_model=DemoRunResponse)
def run_live(
    body: DemoRunRequest,
    session: Session = Depends(get_session),
) -> DemoRunResponse | JSONResponse:
    """Build, persist and diagnose a fresh run in one call."""
    try:
        tasks, classes = _catalog()

        task = tasks.get(body.agent_type)
        if task is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown agent_type {body.agent_type!r}. "
                    f"Valid values: {sorted(tasks)}"
                ),
            )

        glitch = body.glitch
        if glitch is not None and glitch not in classes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown glitch {glitch!r}. "
                    f"Valid values: {sorted(classes)}, or null for a clean run."
                ),
            )

        from backend.seed_corpus import diagnose_run, insert_run
        from generator.build import build_run

        # Unseeded by default: a demo should produce a genuinely new run each
        # time, not replay one the audience has already seen.
        rng = random.Random(body.seed) if body.seed is not None else random.Random()

        trace, _meta = build_run(
            task,
            rng,
            user_id=DEMO_USER_ID,
            created_at=datetime.now(timezone.utc),
            fault=glitch,
        )

        run = insert_run(session, trace.to_dict())
        if run is None:
            # new_uuid() collided with an existing row, which should not happen.
            raise HTTPException(
                status_code=500, detail="Generated run id already exists; retry."
            )

        diagnose_run(session, run)
        session.commit()
        session.refresh(run)

        diagnosis = run.diagnoses[0] if run.diagnoses else None
        flagged = diagnosis.flagged_step_index if diagnosis else None
        truth = run.true_failure_step

        return DemoRunResponse(
            status="completed",
            run_id=run.id,
            task_type=run.task_type,
            injected_class=run.injected_class,
            true_failure_step=truth,
            flagged_step_index=flagged,
            predicted_class=diagnosis.predicted_class if diagnosis else None,
            step_count=len(run.steps),
            localized_correctly=(None if truth is None else flagged == truth),
        )

    except HTTPException:
        # 400s are the caller's answer, not a server fault; let them through
        # rather than flattening every failure into a 500.
        raise
    except Exception as exc:  # noqa: BLE001 - a demo must never take the API down
        session.rollback()
        logger.exception("demo run-live failed")
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "detail": "Could not generate a demo run.",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
