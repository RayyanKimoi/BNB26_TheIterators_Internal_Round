"""Integration tests for the Phase 1 completion endpoints and Slack alerts.

compare, similar, regression-test and the reliability dashboard touch only
stored rows — no model or LLM involved — so most of these seed Diagnosis rows
directly rather than running a real diagnosis. otel ingestion is tested
end-to-end against a small synthetic span list. Mirrors the fixture pattern
in test_phase4_api.py: in-memory SQLite, create_db_and_tables monkeypatched
out, no network.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool

from backend import alerts
from backend import main as main_module
from backend.db import AgentRun, Diagnosis, RegressionTest, Step
from backend.engine import get_session
from backend.main import app


# -- fixtures ---------------------------------------------------------------


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


def seed_run(
    session: Session, *, status="failed", injected_class="stale_retrieval", total_tokens=50,
    duration_ms=500, task_type="travel_booking",
) -> AgentRun:
    run = AgentRun(
        id=str(uuid.uuid4()), source="synthetic", task_type=task_type, status=status,
        total_tokens=total_tokens, duration_ms=duration_ms,
        created_at=datetime.now(timezone.utc), injected_class=injected_class,
        true_failure_step=1 if status == "failed" else None,
    )
    session.add(run)
    rows = [
        (0, "call_llm", None, {}, {"summary": "planning"}, False),
        (1, "call_tool", "convert_currency", {"amount": 180.0}, {"amount": 164.2, "summary": "converted"}, status == "failed"),
        (2, "call_llm", None, {}, {"summary": "reflect"}, False),
    ]
    for i, action, tool, inp, out, err in rows:
        session.add(
            Step(
                run_id=run.id, step_index=i, action_type=action, tool_name=tool,
                input=inp, output=out, state_snapshot={}, state_hash=f"h{i}",
                duration_ms=100, tokens=10, error_flag=err,
            )
        )
    session.commit()
    session.refresh(run)
    return run


def seed_diagnosis(session: Session, run_id: str, *, feature_vector: dict, predicted_class="stale_retrieval", flagged=1) -> Diagnosis:
    diag = Diagnosis(
        run_id=run_id, flagged_step_index=flagged, confidence=0.9,
        predicted_class=predicted_class, evidence=feature_vector, feature_vector=feature_vector,
        class_confidence=0.9,
    )
    session.add(diag)
    session.commit()
    session.refresh(diag)
    return diag


# -- GET /runs/{id}/compare/{other_id} --------------------------------------


def test_compare_flags_tool_and_error_changes(client, session):
    a = seed_run(session, status="failed")
    b = seed_run(session, status="success")
    r = client.get(f"/runs/{a.id}/compare/{b.id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_a_id"] == a.id
    assert body["run_b_id"] == b.id
    assert body["steps_compared"] == 3
    step1_diff = next(d for d in body["diffs"] if d["step_index"] == 1)
    assert step1_diff["a_error_flag"] is True
    assert step1_diff["b_error_flag"] is False
    assert step1_diff["error_flag_changed"] is True


def test_compare_handles_unequal_length_runs(client, session):
    a = seed_run(session)
    b = seed_run(session)
    session.add(
        Step(
            run_id=b.id, step_index=3, action_type="call_tool", tool_name="extra",
            input={}, output={}, state_snapshot={}, state_hash="h3",
            duration_ms=10, tokens=1, error_flag=False,
        )
    )
    session.commit()
    r = client.get(f"/runs/{a.id}/compare/{b.id}")
    body = r.json()
    assert body["steps_compared"] == 4
    extra = next(d for d in body["diffs"] if d["step_index"] == 3)
    assert extra["a_present"] is False
    assert extra["b_present"] is True


def test_compare_404_on_missing_run(client, session):
    a = seed_run(session)
    r = client.get(f"/runs/{a.id}/compare/does-not-exist")
    assert r.status_code == 404


# -- GET /runs/{id}/similar --------------------------------------------------


def test_similar_ranks_closer_vector_higher(client, session):
    target = seed_run(session)
    close = seed_run(session)
    far = seed_run(session)

    seed_diagnosis(session, target.id, feature_vector={"duration_z": 2.0, "token_z": 0.0, "semantic_deviation": 0.5})
    seed_diagnosis(session, close.id, feature_vector={"duration_z": 2.1, "token_z": 0.1, "semantic_deviation": 0.5})
    seed_diagnosis(session, far.id, feature_vector={"duration_z": -3.0, "token_z": 5.0, "semantic_deviation": 0.0})

    r = client.get(f"/runs/{target.id}/similar")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["compared_against"] == 2
    ids_in_order = [m["run_id"] for m in body["matches"]]
    assert ids_in_order[0] == close.id, "the near-identical vector must rank first"


def test_similar_404_without_a_diagnosis(client, session):
    run = seed_run(session)
    r = client.get(f"/runs/{run.id}/similar")
    assert r.status_code == 404


# -- POST /runs/{id}/regression-test -----------------------------------------


def test_regression_test_persists_against_latest_diagnosis(client, session):
    run = seed_run(session)
    diag = seed_diagnosis(session, run.id, feature_vector={"duration_z": 1.0})
    r = client.post(
        f"/runs/{run.id}/regression-test",
        json={"assertion": {"predicted_class": "stale_retrieval"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["diagnosis_id"] == diag.id
    assert body["assertion"] == {"predicted_class": "stale_retrieval"}

    stored = session.exec(
        select(RegressionTest).where(RegressionTest.diagnosis_id == diag.id)
    ).first()
    assert stored is not None


def test_regression_test_400_without_any_diagnosis(client, session):
    run = seed_run(session)
    r = client.post(f"/runs/{run.id}/regression-test", json={"assertion": {}})
    assert r.status_code == 400


# -- GET /dashboard/reliability -----------------------------------------------


def test_reliability_dashboard_aggregates_real_rows(client, session):
    seed_run(session, status="success", total_tokens=100)
    seed_run(session, status="failed", injected_class="wrong_tool_chosen", total_tokens=200)
    seed_run(session, status="failed", injected_class="wrong_tool_chosen", total_tokens=300)

    r = client.get("/dashboard/reliability")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_runs"] == 3
    assert body["success_count"] == 1
    assert body["failure_count"] == 2
    assert body["overall_pass_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert body["total_tokens"] == 600
    wtc = next(c for c in body["failure_class_breakdown"] if c["injected_class"] == "wrong_tool_chosen")
    assert wtc["count"] == 2
    assert len(body["trend"]) >= 1
    # No TOKEN_COST_PER_1K_USD is configured in the test environment: cost
    # must be null, never a guessed number.
    assert body["cost_rate_configured"] is False
    assert body["estimated_cost_usd"] is None


def test_reliability_cost_is_computed_only_when_a_rate_is_configured(client, session, monkeypatch):
    seed_run(session, status="success", total_tokens=1000)
    monkeypatch.setenv("TOKEN_COST_PER_1K_USD", "0.002")
    r = client.get("/dashboard/reliability")
    body = r.json()
    assert body["cost_rate_configured"] is True
    assert body["estimated_cost_usd"] == pytest.approx(0.002, abs=1e-6)


# -- POST /ingest/otel --------------------------------------------------------


def _span(span_id, name, start_ns, end_ns, attributes=None, status_code="OK"):
    return {
        "span_id": span_id,
        "name": name,
        "start_time_unix_nano": start_ns,
        "end_time_unix_nano": end_ns,
        "attributes": attributes or {},
        "status_code": status_code,
    }


def test_otel_ingest_maps_spans_into_steps(client, session):
    payload = {
        "trace_id": "trace-1",
        "task_type": "customer_support",
        "spans": [
            _span("s1", "llm.chat.completion", 0, 500_000_000, {"llm.usage.total_tokens": 42}),
            _span("s2", "tool.fetch_ticket", 500_000_000, 700_000_000, {"tool.name": "fetch_ticket", "output.ticket_id": 4821}),
            _span("s3", "tool.send_reply", 700_000_000, 900_000_000, {"tool.name": "send_reply"}, status_code="ERROR"),
        ],
    }
    r = client.post("/ingest/otel", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["steps_created"] == 3
    assert body["status"] == "failed"  # one ERROR span -> the run failed

    run = session.get(AgentRun, body["run_id"])
    assert run.source == "otel"
    assert run.total_tokens == 42
    assert run.duration_ms == 500 + 200 + 200

    steps = sorted(
        session.exec(select(Step).where(Step.run_id == run.id)).all(),
        key=lambda s: s.step_index,
    )
    assert steps[0].action_type == "call_llm"
    assert steps[1].action_type == "call_tool"
    assert steps[1].tool_name == "fetch_ticket"
    assert steps[1].output.get("ticket_id") == 4821
    assert steps[2].error_flag is True


def test_otel_ingest_rejects_an_empty_trace(client):
    r = client.post("/ingest/otel", json={"trace_id": "empty", "spans": []})
    assert r.status_code in (400, 422)


# -- Slack alerts -------------------------------------------------------------


def test_slack_payload_includes_run_and_evidence(monkeypatch):
    monkeypatch.setenv("DASHBOARD_BASE_URL", "http://localhost:5173")
    payload = alerts.build_slack_payload(
        run_id="abc-123",
        task_type="travel_booking",
        flagged_step_index=2,
        predicted_class="schema_violation",
        evidence_summary="parse_failure=True",
    )
    text_blob = str(payload)
    assert "abc-123" in text_blob
    assert "schema_violation" in text_blob
    assert "http://localhost:5173/trace/abc-123" in text_blob


def test_alert_never_fires_for_unknown(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/anything")
    sent = alerts.send_diagnosis_alert(
        run_id="r1", task_type="t", flagged_step_index=0,
        predicted_class="unknown", evidence={},
    )
    assert sent is False


def test_alert_skips_silently_without_a_webhook_url(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    sent = alerts.send_diagnosis_alert(
        run_id="r1", task_type="t", flagged_step_index=0,
        predicted_class="stale_retrieval", evidence={},
    )
    assert sent is False


def test_alert_never_raises_on_an_unreachable_webhook(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://this-host-does-not-exist.invalid/webhook")
    monkeypatch.setattr(alerts, "REQUEST_TIMEOUT_S", 1.0)
    sent = alerts.send_diagnosis_alert(
        run_id="r1", task_type="t", flagged_step_index=0,
        predicted_class="stale_retrieval", evidence={"duration_z": 1.2},
    )
    assert sent is False  # failed, but did not raise


def test_slack_payload_carries_confidence_and_no_emoji():
    payload = alerts.build_slack_payload(
        run_id="abc", task_type="travel_booking", flagged_step_index=2,
        predicted_class="schema_violation", evidence_summary="parse_failure=True",
        confidence=0.9168,
    )
    blob = str(payload)
    assert "92%" in blob, "confidence is rendered as a percentage"
    # CLAUDE.md bans emoji icons project-wide, alerts included.
    assert not any(ord(ch) > 0x2100 for ch in blob), "no emoji in the alert payload"


# -- multi-candidate fixes (PRD item 14) --------------------------------------


def test_candidate_patches_are_parsed_from_json_strings():
    from backend.explainer import _parse_candidates

    parsed = _parse_candidates(
        [
            {"rank": 2, "patch_json": '{"currency": "EUR"}', "rationale": "second"},
            {"rank": 1, "patch_json": '{"ticket_id": 4821}', "rationale": "first"},
        ]
    )
    assert [c["rank"] for c in parsed] == [1, 2], "candidates come back rank-sorted"
    assert parsed[0]["patch"] == {"ticket_id": 4821}


def test_malformed_candidate_is_dropped_not_fatal():
    from backend.explainer import _parse_candidates

    parsed = _parse_candidates(
        [
            {"rank": 1, "patch_json": "not json at all", "rationale": "bad"},
            {"rank": 2, "patch_json": "[1, 2, 3]", "rationale": "not an object"},
            {"rank": 3, "patch_json": '{"ok": true}', "rationale": "good"},
        ]
    )
    assert len(parsed) == 1, "only the well-formed object candidate survives"
    assert parsed[0]["patch"] == {"ok": True}


def test_candidates_are_capped_at_three():
    from backend.explainer import _parse_candidates

    parsed = _parse_candidates(
        [{"rank": i, "patch_json": f'{{"k": {i}}}', "rationale": ""} for i in range(1, 7)]
    )
    assert len(parsed) == 3


# -- GET /settings -------------------------------------------------------------


def test_settings_reports_presence_never_secret_values(client, monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/super-secret-path")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-do-not-leak-me")
    r = client.get("/settings")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["slack_configured"] is True
    assert body["gemini_configured"] is True
    blob = r.text
    assert "super-secret-path" not in blob, "the webhook URL must never reach the browser"
    assert "sk-do-not-leak-me" not in blob, "the API key must never reach the browser"
    # The dialect is safe to show; the full URL holds the password.
    assert "://" not in body["database_dialect"]


def test_settings_reports_the_class_split(client):
    body = client.get("/settings").json()
    assert set(body["held_out_classes"]) == {"infinite_loop", "context_truncation"}
    assert "stale_retrieval" in body["trained_classes"]
    assert not set(body["trained_classes"]) & set(body["held_out_classes"])


def test_explanation_payload_still_validates_without_candidates():
    """The original three-key response must keep working: the contract is
    widened by fix_candidates, never broken by it."""
    from backend.models import ExplanationPayload

    payload = ExplanationPayload.model_validate(
        {"root_cause": "x", "evidence_summary": ["y"], "proposed_fix": "z"}
    )
    assert payload.fix_candidates == []
