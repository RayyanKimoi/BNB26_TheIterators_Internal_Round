"""Backend integration test — proves the full pipeline works.

Inserts a dummy trace into the database, calls POST /runs/{id}/diagnose
via the FastAPI test client, and validates the response against the
Pydantic DiagnosisResponse schema.

Run:
    python -m backend.test_backend
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

# -- Bootstrap: create the app and DB tables before anything else -----------
from backend.main import app
from backend.engine import create_db_and_tables, engine
from backend.db import AgentRun, Step
from backend.models import DiagnosisResponse

from sqlmodel import Session


def _make_dummy_trace() -> tuple[AgentRun, list[Step]]:
    """Create a minimal but realistic failed run with a schema_violation.

    This is a 5-step trace where step 2 is a call_tool with parse_failure=True,
    which is the signature the ML model uses to detect schema_violation.
    """
    run_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Build 5 steps: decide -> call_tool -> call_tool(FAULT) -> call_llm -> call_tool
    steps_data = [
        {
            "step_index": 0,
            "action_type": "decide",
            "tool_name": None,
            "input": {"goal": "Book a flight from NYC to London for next Tuesday"},
            "output": {
                "summary": "Evaluating available tools for flight search",
                "candidates": {"search_flights": 0.8, "check_weather": 0.15, "book_hotel": 0.05},
            },
            "state_snapshot": {"goal": "Book a flight from NYC to London for next Tuesday", "step": 0},
            "duration_ms": 120,
            "tokens": 450,
            "error_flag": False,
        },
        {
            "step_index": 1,
            "action_type": "call_tool",
            "tool_name": "search_flights",
            "input": {"from": "NYC", "to": "London", "date": "2026-10-10"},
            "output": {
                "summary": "Found 3 available flights",
                "flights": [
                    {"airline": "BA", "price": 850, "departure": "08:00"},
                    {"airline": "AA", "price": 920, "departure": "14:30"},
                ],
            },
            "state_snapshot": {"goal": "Book a flight from NYC to London for next Tuesday", "step": 1, "flights_found": True},
            "duration_ms": 340,
            "tokens": 620,
            "error_flag": False,
        },
        {
            # THIS IS THE FAULTY STEP — parse_failure triggers schema_violation
            "step_index": 2,
            "action_type": "call_tool",
            "tool_name": "book_flight",
            "input": {"flight_id": "BA-123", "passenger": "John Doe"},
            "output": {
                "summary": "Attempted to book flight but response was malformed",
                "_meta": {"parse_failure": True},
            },
            "state_snapshot": {"goal": "Book a flight from NYC to London for next Tuesday", "step": 2, "booking_attempted": True},
            "duration_ms": 890,
            "tokens": 380,
            "error_flag": True,
        },
        {
            "step_index": 3,
            "action_type": "call_llm",
            "tool_name": None,
            "input": {"context": "Booking failed, need to retry or try alternative"},
            "output": {
                "summary": "Analyzing the failed booking attempt to determine next steps",
            },
            "state_snapshot": {"goal": "Book a flight from NYC to London for next Tuesday", "step": 3, "error_occurred": True},
            "duration_ms": 200,
            "tokens": 550,
            "error_flag": True,
        },
        {
            "step_index": 4,
            "action_type": "call_tool",
            "tool_name": "search_flights",
            "input": {"from": "NYC", "to": "London", "date": "2026-10-10", "retry": True},
            "output": {
                "summary": "Retried flight search but task ultimately failed",
            },
            "state_snapshot": {"goal": "Book a flight from NYC to London for next Tuesday", "step": 4, "failed": True},
            "duration_ms": 300,
            "tokens": 400,
            "error_flag": True,
        },
    ]

    # Compute state hashes
    import hashlib

    def _hash(state: dict) -> str:
        canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    run = AgentRun(
        id=run_id,
        user_id=user_id,
        source="synthetic",
        task_type="travel_booking",
        status="failed",
        injected_class="schema_violation",
        true_failure_step=2,
        total_tokens=sum(s["tokens"] for s in steps_data),
        duration_ms=sum(s["duration_ms"] for s in steps_data),
        created_at=now,
    )

    steps = []
    for s in steps_data:
        steps.append(
            Step(
                id=str(uuid.uuid4()),
                run_id=run_id,
                step_index=s["step_index"],
                action_type=s["action_type"],
                tool_name=s["tool_name"],
                input=s["input"],
                output=s["output"],
                state_snapshot=s["state_snapshot"],
                state_hash=_hash(s["state_snapshot"]),
                duration_ms=s["duration_ms"],
                tokens=s["tokens"],
                error_flag=s["error_flag"],
            )
        )

    return run, steps


def main() -> None:
    print("=" * 70)
    print("BLACK BOX — Backend Integration Test")
    print("=" * 70)
    print()

    # 1. Create tables
    print("[1/5] Creating database tables...")
    create_db_and_tables()
    print("      Done. Tables created in the database.")
    print()

    # 2. Insert dummy trace
    print("[2/5] Inserting a dummy 5-step failed trace (schema_violation)...")
    run, steps = _make_dummy_trace()
    run_id = run.id  # capture before session closes
    num_steps = len(steps)
    with Session(engine) as session:
        session.add(run)
        for step in steps:
            session.add(step)
        session.commit()
    print(f"      Run ID: {run_id}")
    print(f"      Steps:  {num_steps}")
    print(f"      Fault:  schema_violation at step 2")
    print()

    # 3. Test GET /runs
    print("[3/5] Testing GET /runs ...")
    client = TestClient(app)
    resp = client.get("/runs")
    assert resp.status_code == 200, f"GET /runs failed: {resp.status_code}"
    runs_list = resp.json()
    assert len(runs_list) >= 1, "No runs returned"
    found = any(r["id"] == run_id for r in runs_list)
    assert found, f"Inserted run {run_id} not found in GET /runs"
    print(f"      OK — {len(runs_list)} run(s) returned, our run is in the list")
    print()

    # 4. Test GET /runs/{id}
    print(f"[4/5] Testing GET /runs/{run_id[:8]}... ...")
    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200, f"GET /runs/{{id}} failed: {resp.status_code}"
    run_detail = resp.json()
    assert run_detail["id"] == run_id
    assert len(run_detail["steps"]) == 5
    assert run_detail["diagnosis"] is None  # not diagnosed yet
    print(f"      OK — run returned with {len(run_detail['steps'])} steps, no diagnosis yet")
    print()

    # 5. Test POST /runs/{id}/diagnose (THE BIG ONE)
    print(f"[5/5] Testing POST /runs/{run_id[:8]}... /diagnose ...")
    print("      Loading Hybrid Sentry Engine (ML model + invariant fallback)...")
    resp = client.post(f"/runs/{run_id}/diagnose")
    assert resp.status_code == 200, f"POST /diagnose failed: {resp.status_code} — {resp.text}"
    diagnosis_json = resp.json()

    # Validate against Pydantic schema
    diagnosis = DiagnosisResponse(**diagnosis_json)
    print("      Pydantic validation: PASSED")
    print()

    # Print the diagnosis
    print("-" * 70)
    print("DIAGNOSIS RESULT (Pydantic-validated JSON)")
    print("-" * 70)
    print(json.dumps(diagnosis_json, indent=2))
    print("-" * 70)
    print()

    # Assertions
    errors = []
    if diagnosis.run_id != run_id:
        errors.append(f"run_id mismatch: {diagnosis.run_id} != {run_id}")
    if diagnosis.flagged_step_index is None:
        errors.append("flagged_step_index is None")
    if not (0.0 <= diagnosis.confidence <= 1.0):
        errors.append(f"confidence out of range: {diagnosis.confidence}")
    if diagnosis.predicted_class not in (
        "wrong_tool_chosen", "hallucinated_argument", "stale_retrieval",
        "premature_termination", "infinite_loop", "schema_violation",
        "context_truncation", "unknown"
    ):
        errors.append(f"unexpected predicted_class: {diagnosis.predicted_class}")
    if len(diagnosis.step_scores) != 5:
        errors.append(f"step_scores length {len(diagnosis.step_scores)} != 5")
    if sum(diagnosis.step_scores) < 90 or sum(diagnosis.step_scores) > 110:
        errors.append(f"step_scores sum {sum(diagnosis.step_scores)} not ~100")

    # Check the diagnosis was saved to DB
    resp2 = client.get(f"/runs/{run_id}")
    run_after = resp2.json()
    if run_after["diagnosis"] is None:
        errors.append("Diagnosis not persisted to DB — GET /runs/{id} still shows null")

    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
        print()
        print("TEST: FAILED")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED:")
        print(f"  - run_id matches")
        print(f"  - flagged_step_index = {diagnosis.flagged_step_index}")
        print(f"  - confidence = {diagnosis.confidence}")
        print(f"  - predicted_class = {diagnosis.predicted_class}")
        print(f"  - step_scores length = {len(diagnosis.step_scores)}, sum = {sum(diagnosis.step_scores)}")
        print(f"  - class_confidence = {diagnosis.class_confidence}")
        print(f"  - unknown_reason = {diagnosis.unknown_reason}")
        print(f"  - Pydantic schema validation: PASSED")
        print(f"  - DB persistence: PASSED")
        print()

        if diagnosis.flagged_step_index == 2:
            print(f"  BONUS: Model correctly identified step 2 as the root cause! (ground truth: step 2)")
        else:
            print(f"  NOTE: Model flagged step {diagnosis.flagged_step_index} (ground truth was step 2)")

        print()
        print("=" * 70)
        print("TEST: PASSED — Backend is ready for frontend integration")
        print("=" * 70)


if __name__ == "__main__":
    main()
