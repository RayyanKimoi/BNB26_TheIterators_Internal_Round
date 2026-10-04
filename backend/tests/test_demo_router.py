"""Tests for the live demo endpoint.

The generator and the real localizer are both slow and the localizer needs a
model artifact, so the expensive pieces are monkeypatched for most of these:
what matters here is the router's own contract, its validation and, above all,
that an unexpected failure returns a 500 JSON body instead of a traceback.

Same fixture pattern as the other backend tests: in-memory SQLite,
create_db_and_tables patched out, no network.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from backend import main as main_module
from backend.db import AgentRun, Diagnosis, Step
from backend.engine import get_session
from backend.main import app


@pytest.fixture
def session():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def client(session, monkeypatch):
    monkeypatch.setattr(main_module, "create_db_and_tables", lambda: None)
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def fake_pipeline(monkeypatch, session):
    """Stub out generation and diagnosis, keeping the router's own logic real."""
    from backend import demo_router as dr

    class _FakeTrace:
        def __init__(self, run_id="demo-run-1", n=4):
            self.run_id = run_id
            self.n = n

        def to_dict(self):
            return {
                "run": {
                    "id": self.run_id,
                    "user_id": dr.DEMO_USER_ID,
                    "source": "synthetic",
                    "task_type": "travel_booking",
                    "status": "failed",
                    "injected_class": "stale_retrieval",
                    "true_failure_step": 2,
                    "total_tokens": 400,
                    "duration_ms": 900,
                    "created_at": "2026-10-04T00:00:00+00:00",
                },
                "steps": [
                    {
                        "id": f"{self.run_id}-s{i}",
                        "run_id": self.run_id,
                        "step_index": i,
                        "action_type": "tool_call",
                        "tool_name": "search_flights",
                        "input": {},
                        "output": {},
                        "state_snapshot": {},
                        "state_hash": f"h{i}",
                        "duration_ms": 100,
                        "tokens": 50,
                        "error_flag": False,
                    }
                    for i in range(self.n)
                ],
            }

    def fake_build_run(task, rng, **kwargs):
        return _FakeTrace(), {}

    def fake_diagnose(sess, run):
        sess.add(
            Diagnosis(
                run_id=run.id,
                flagged_step_index=2,
                confidence=0.9,
                predicted_class="stale_retrieval",
                evidence_path="step[2].output",
                evidence={},
                feature_vector={},
                explanation=None,
                suggested_fixes=[],
                class_confidence=0.9,
                unknown_reason=None,
            )
        )
        sess.flush()

    import generator.build as gb
    import backend.seed_corpus as sc

    monkeypatch.setattr(gb, "build_run", fake_build_run)
    monkeypatch.setattr(sc, "diagnose_run", fake_diagnose)
    return fake_build_run


# -- options ----------------------------------------------------------------


def test_options_lists_real_generator_values(client):
    body = client.get("/demo/options").json()
    assert "travel_booking" in body["agent_types"]
    assert "stale_retrieval" in body["glitches"]
    # The seven-class taxonomy, not a hand-maintained copy of it.
    assert len(body["glitches"]) == 7


# -- happy path -------------------------------------------------------------


def test_run_live_creates_persists_and_diagnoses(client, session, fake_pipeline):
    response = client.post(
        "/demo/run-live",
        json={"agent_type": "travel_booking", "glitch": "stale_retrieval"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "completed"
    assert body["run_id"]
    assert body["predicted_class"] == "stale_retrieval"
    assert body["localized_correctly"] is True

    # Actually in the database, not just in the response.
    run = session.get(AgentRun, body["run_id"])
    assert run is not None
    assert run.injected_class == "stale_retrieval"
    assert len(run.steps) == body["step_count"]
    assert len(run.diagnoses) == 1


def test_defaults_match_the_documented_payload(client, fake_pipeline):
    """An empty body is the spec's example payload."""
    body = client.post("/demo/run-live", json={}).json()
    assert body["status"] == "completed"
    assert body["task_type"] == "travel_booking"


def test_localized_correctly_is_false_on_a_miss(client, session, monkeypatch, fake_pipeline):
    import backend.seed_corpus as sc

    def wrong(sess, run):
        sess.add(
            Diagnosis(
                run_id=run.id,
                flagged_step_index=0,  # truth is 2
                confidence=0.5,
                predicted_class="wrong_tool_chosen",
                evidence_path="step[0].output",
                evidence={},
                feature_vector={},
                explanation=None,
                suggested_fixes=[],
                class_confidence=0.5,
                unknown_reason=None,
            )
        )
        sess.flush()

    monkeypatch.setattr(sc, "diagnose_run", wrong)
    body = client.post("/demo/run-live", json={}).json()
    assert body["localized_correctly"] is False


# -- validation -------------------------------------------------------------


def test_unknown_agent_type_is_400_not_500(client):
    response = client.post(
        "/demo/run-live", json={"agent_type": "nope", "glitch": "stale_retrieval"}
    )
    assert response.status_code == 400
    assert "Unknown agent_type" in response.json()["detail"]


def test_unknown_glitch_is_400_not_500(client):
    response = client.post(
        "/demo/run-live", json={"agent_type": "travel_booking", "glitch": "bogus"}
    )
    assert response.status_code == 400
    assert "Unknown glitch" in response.json()["detail"]


def test_null_glitch_is_allowed(client, fake_pipeline):
    """A clean run is a legitimate demo: it should pass, not fail validation."""
    response = client.post(
        "/demo/run-live", json={"agent_type": "travel_booking", "glitch": None}
    )
    assert response.status_code == 200


# -- failure containment ----------------------------------------------------


def test_generator_explosion_returns_500_json_not_a_crash(client, monkeypatch):
    import generator.build as gb

    def boom(*args, **kwargs):
        raise RuntimeError("generator exploded")

    monkeypatch.setattr(gb, "build_run", boom)
    response = client.post("/demo/run-live", json={})

    assert response.status_code == 500
    body = response.json()
    assert body["status"] == "failed"
    assert "RuntimeError" in body["error"]


def test_diagnosis_explosion_returns_500_json(client, monkeypatch, fake_pipeline):
    import backend.seed_corpus as sc

    def boom(sess, run):
        raise ValueError("model artifact missing")

    monkeypatch.setattr(sc, "diagnose_run", boom)
    response = client.post("/demo/run-live", json={})

    assert response.status_code == 500
    assert response.json()["status"] == "failed"


def test_failed_run_is_rolled_back(client, session, monkeypatch, fake_pipeline):
    """A half-written run must not survive a failure partway through."""
    import backend.seed_corpus as sc

    monkeypatch.setattr(
        sc, "diagnose_run", lambda sess, run: (_ for _ in ()).throw(ValueError("boom"))
    )
    client.post("/demo/run-live", json={})

    from sqlmodel import select

    assert session.exec(select(AgentRun)).all() == []
    assert session.exec(select(Step)).all() == []


# -- additivity -------------------------------------------------------------


def test_existing_endpoints_are_untouched(client):
    """Mounting the router must not disturb the routes that were already there."""
    assert client.get("/runs").status_code == 200
    assert client.get("/settings").status_code == 200


def test_demo_routes_are_registered_once(client):
    """Assert against the OpenAPI schema, not app.routes.

    This FastAPI version wraps an included router in an _IncludedRouter that
    carries no `.path`, so walking app.routes finds nothing and a naive
    assertion there would pass or fail for reasons unrelated to mounting.
    The schema is the actual public surface and is stable across versions.
    """
    paths = app.openapi()["paths"]
    assert "/demo/run-live" in paths
    assert "/demo/options" in paths
    assert "post" in paths["/demo/run-live"]
    # Mounting the router must not have shadowed the existing surface.
    assert "/runs/{run_id}/diagnose" in paths
