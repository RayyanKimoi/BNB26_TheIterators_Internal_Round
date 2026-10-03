"""Integration tests for POST /runs/{id}/explain and POST /runs/{id}/fork.

Both the localizer and the LLM are stubbed, so these run in milliseconds,
need no trained artifact and never touch the network. `backend/test_backend.py`
remains the end-to-end check against the real model.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from backend import main as main_module
from backend.db import AgentRun, Diagnosis, Step
from backend.engine import get_session
from backend.main import app, get_explainer_client
from backend.models import DiagnosisResponse
from replay.engine import PRECONDITION_ERROR

GOAL = "book the cheapest refundable flight"


# -- stubs ----------------------------------------------------------------


class StubLocalizer:
    """Flags the first errored step, or step 0. Shape-faithful, cheap."""

    def diagnose(self, trace):
        steps = trace["steps"]
        flagged = next((s["step_index"] for s in steps if s.get("error_flag")), 0)
        scores = [1] * len(steps)
        if steps:
            scores[flagged] = 100 - (len(steps) - 1)
        return {
            "run_id": trace.get("run", {}).get("id"),
            "flagged_step_index": flagged,
            "confidence": 0.9,
            "predicted_class": "schema_violation",
            "evidence_path": None,
            "evidence": {"parse_failure": True, "duration_z": 1.2},
            "step_scores": scores,
            "explanation": None,
            "suggested_fixes": [],
            "class_confidence": 0.95,
            "unknown_reason": None,
            "anomaly_signal": None,
        }


class StubExplainer:
    """Records the prompt it was given so tests can assert on it."""

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt: str):
        self.calls.append(prompt)
        return {
            "root_cause": "convert_currency returned a stale cached rate.",
            "evidence_summary": ["step[2].output._meta.cache_age_s = 90000"],
            "proposed_fix": '- "amount": 164.2\n+ "amount": 170.0',
        }


class BrokenExplainer:
    def generate(self, prompt: str):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


# -- fixtures -------------------------------------------------------------


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def explainer():
    return StubExplainer()


@pytest.fixture
def client(session, explainer, monkeypatch):
    # The startup hook creates tables on the real engine, which is a remote
    # Postgres. These tests use in-memory SQLite, so that round trip is pure
    # latency: leaving it in cost about 6 seconds per test.
    monkeypatch.setattr(main_module, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(main_module, "_get_localizer", lambda: StubLocalizer())
    monkeypatch.setattr(main_module, "_shap_for", lambda trace, idx: {"duration_z": 0.31})
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_explainer_client] = lambda: explainer
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def seed_run(session: Session, *, status="failed") -> str:
    """A 5-step failed run: step 2 faulty, step 4 starved by it."""
    run = AgentRun(
        id=str(uuid.uuid4()), user_id="u", source="synthetic", task_type="travel_booking",
        status=status, total_tokens=50, duration_ms=500,
        created_at=datetime.now(timezone.utc), injected_class="stale_retrieval",
        true_failure_step=2,
    )
    session.add(run)
    snap = {"goal": GOAL, "facts": {"cheapest_price": 180.0}}
    rows = [
        (0, "call_llm", None, {}, {"summary": "planning"}, False),
        (1, "decide", None, {}, {"chosen_tool": "convert_currency", "summary": "decide"}, False),
        (2, "call_tool", "convert_currency", {"amount": 180.0},
         {"amount": 164.2, "summary": "converted",
          "_meta": {"cache_hit": True, "cache_age_s": 90000, "parse_failure": False,
                    "retry_count": 0}}, False),
        (3, "call_llm", None, {}, {"summary": "reflect"}, False),
        (4, "call_tool", "book_flight", {"total_eur": 164.2},
         {"error": PRECONDITION_ERROR, "summary": "book_flight failed: required input missing"},
         True),
    ]
    for i, action, tool, inp, out, err in rows:
        session.add(Step(
            run_id=run.id, step_index=i, action_type=action, tool_name=tool,
            input=inp, output=out, state_snapshot=snap, state_hash=f"h{i}",
            duration_ms=100, tokens=10, error_flag=err,
        ))
    session.commit()
    return run.id


# -- explain --------------------------------------------------------------


def test_explain_returns_the_enforced_three_key_schema(client, session, explainer):
    run_id = seed_run(session)
    r = client.post(f"/runs/{run_id}/explain", json={"step_index": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["root_cause"]
    assert isinstance(body["evidence_summary"], list) and body["evidence_summary"]
    assert body["proposed_fix"]
    assert body["run_id"] == run_id
    assert body["step_index"] == 2
    assert body["cached"] is False
    assert len(explainer.calls) == 1


def test_explain_persists_to_agent_runs_explanation(client, session):
    run_id = seed_run(session)
    client.post(f"/runs/{run_id}/explain", json={"step_index": 2})
    session.expire_all()
    stored = session.get(AgentRun, run_id).explanation
    assert stored["root_cause"]
    assert stored["step_index"] == 2
    assert stored["proposed_fix"]


def test_explain_is_cached_on_the_second_call(client, session, explainer):
    run_id = seed_run(session)
    client.post(f"/runs/{run_id}/explain", json={"step_index": 2})
    second = client.post(f"/runs/{run_id}/explain", json={"step_index": 2})
    assert second.json()["cached"] is True
    assert len(explainer.calls) == 1, "a cached hit must not call the LLM again"


def test_explain_a_different_step_bypasses_the_cache(client, session, explainer):
    run_id = seed_run(session)
    client.post(f"/runs/{run_id}/explain", json={"step_index": 2})
    r = client.post(f"/runs/{run_id}/explain", json={"step_index": 4})
    assert r.json()["cached"] is False
    assert len(explainer.calls) == 2


def test_explain_prompt_carries_evidence_and_forbids_rechoosing_the_step(
    client, session, explainer
):
    """The LLM explains the model's choice. It must not be invited to revise it."""
    run_id = seed_run(session)
    client.post(f"/runs/{run_id}/explain", json={"step_index": 2})
    prompt = explainer.calls[0]
    assert "Do not second-guess" in prompt
    assert GOAL in prompt
    assert "shap" in prompt.lower()


def test_explain_uses_the_latest_diagnosis_class(client, session, explainer):
    run_id = seed_run(session)
    session.add(Diagnosis(
        run_id=run_id, flagged_step_index=2, confidence=0.8,
        predicted_class="stale_retrieval", evidence={"duration_z": -0.9},
    ))
    session.commit()
    client.post(f"/runs/{run_id}/explain", json={"step_index": 2})
    assert "stale_retrieval" in explainer.calls[0]


def test_explain_tells_the_model_to_not_name_an_unknown_class(client, session, explainer):
    run_id = seed_run(session)
    client.post(f"/runs/{run_id}/explain", json={"step_index": 2})
    # No diagnosis row exists, so predicted_class defaults to unknown.
    assert "could not name the failure mode" in explainer.calls[0]


def test_explain_404_on_missing_run(client):
    assert client.post("/runs/nope/explain", json={"step_index": 0}).status_code == 404


@pytest.mark.parametrize("bad", [99, 5])
def test_explain_400_on_step_index_outside_the_trace(client, session, bad):
    run_id = seed_run(session)
    r = client.post(f"/runs/{run_id}/explain", json={"step_index": bad})
    assert r.status_code == 400
    assert "outside the trace" in r.json()["detail"]


def test_explain_rejects_a_negative_step_index(client, session):
    run_id = seed_run(session)
    assert client.post(f"/runs/{run_id}/explain", json={"step_index": -1}).status_code == 422


def test_explain_502_when_the_llm_fails(client, session):
    run_id = seed_run(session)
    app.dependency_overrides[get_explainer_client] = lambda: BrokenExplainer()
    r = client.post(f"/runs/{run_id}/explain", json={"step_index": 2})
    assert r.status_code == 502
    assert "explainer call failed" in r.json()["detail"]


# -- fork -----------------------------------------------------------------


def test_fork_creates_a_child_linked_to_its_parent(client, session):
    run_id = seed_run(session)
    r = client.post(f"/runs/{run_id}/fork", json={"from_step": 2, "fix_payload": {"amount": 170.0}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parent_run_id"] == run_id
    assert body["child_run_id"] != run_id
    assert body["status"] == "completed"
    assert body["forked_at_step"] == 2
    assert body["steps_replayed"] == 3
    assert body["steps_total"] == 5

    child = session.get(AgentRun, body["child_run_id"])
    assert child.parent_run_id == run_id
    assert child.forked_at_step == 2
    assert child.fix_applied == {"amount": 170.0}


def test_fork_flips_the_outcome_when_the_fix_resolves_the_fault(client, session):
    run_id = seed_run(session)
    body = client.post(
        f"/runs/{run_id}/fork", json={"from_step": 2, "fix_payload": {"amount": 170.0}}
    ).json()
    assert body["parent_outcome"] == "failed"
    assert body["outcome"] == "success"
    assert all(not s["error_flag"] for s in body["steps"])


def test_fork_without_a_fix_does_not_flip_the_outcome(client, session):
    run_id = seed_run(session)
    body = client.post(f"/runs/{run_id}/fork", json={"from_step": 2}).json()
    assert body["outcome"] == "failed"


def test_fork_leaves_the_parent_untouched(client, session):
    run_id = seed_run(session)
    before = [(s.step_index, s.error_flag, dict(s.output)) for s in
              session.exec(__import__("sqlmodel").select(Step).where(Step.run_id == run_id)).all()]
    client.post(f"/runs/{run_id}/fork", json={"from_step": 2, "fix_payload": {"amount": 170.0}})
    session.expire_all()
    after = [(s.step_index, s.error_flag, dict(s.output)) for s in
             session.exec(__import__("sqlmodel").select(Step).where(Step.run_id == run_id)).all()]
    assert sorted(before) == sorted(after)
    assert session.get(AgentRun, run_id).status == "failed"


def test_fork_persists_child_steps_and_a_fresh_diagnosis(client, session):
    run_id = seed_run(session)
    import sqlmodel

    body = client.post(
        f"/runs/{run_id}/fork", json={"from_step": 2, "fix_payload": {"amount": 170.0}}
    ).json()
    child_id = body["child_run_id"]
    steps = session.exec(sqlmodel.select(Step).where(Step.run_id == child_id)).all()
    assert len(steps) == 5
    diags = session.exec(sqlmodel.select(Diagnosis).where(Diagnosis.run_id == child_id)).all()
    assert len(diags) == 1


def test_fork_child_carries_no_ground_truth_labels(client, session):
    """A fork is not a labelled synthetic run; claiming a class would corrupt eval."""
    run_id = seed_run(session)
    body = client.post(
        f"/runs/{run_id}/fork", json={"from_step": 2, "fix_payload": {"amount": 170.0}}
    ).json()
    child = session.get(AgentRun, body["child_run_id"])
    assert child.injected_class is None
    assert child.true_failure_step is None


def test_fork_response_diagnosis_matches_the_13_key_contract(client, session):
    run_id = seed_run(session)
    body = client.post(
        f"/runs/{run_id}/fork", json={"from_step": 2, "fix_payload": {"amount": 170.0}}
    ).json()
    assert set(body["diagnosis"]) == set(DiagnosisResponse.model_fields)


def test_fork_records_override_code_without_executing_it(client, session):
    run_id = seed_run(session)
    body = client.post(
        f"/runs/{run_id}/fork",
        json={"from_step": 2, "override_code": "raise SystemExit(1)"},
    ).json()
    assert body["steps"][2]["output"]["_override_code"] == "raise SystemExit(1)"


def test_fork_404_on_missing_run(client):
    assert client.post("/runs/nope/fork", json={"from_step": 0}).status_code == 404


def test_fork_400_on_out_of_range_step(client, session):
    run_id = seed_run(session)
    r = client.post(f"/runs/{run_id}/fork", json={"from_step": 99})
    assert r.status_code == 400
    assert "outside the parent trace" in r.json()["detail"]


def test_fork_of_a_fork_chains_lineage(client, session):
    run_id = seed_run(session)
    first = client.post(f"/runs/{run_id}/fork", json={"from_step": 2}).json()
    second = client.post(
        f"/runs/{first['child_run_id']}/fork",
        json={"from_step": 2, "fix_payload": {"amount": 170.0}},
    ).json()
    assert second["parent_run_id"] == first["child_run_id"]
    assert second["outcome"] == "success"


def test_existing_diagnose_endpoint_still_returns_the_contract(client, session):
    """Backward compatibility: Phase 4 must not change the 13-key response."""
    run_id = seed_run(session)
    r = client.post(f"/runs/{run_id}/diagnose")
    assert r.status_code == 200
    assert set(r.json()) == set(DiagnosisResponse.model_fields)
