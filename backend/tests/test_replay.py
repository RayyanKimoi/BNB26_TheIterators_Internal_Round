"""Unit tests for deterministic suffix replay.

No model, no database, no network. These pin the propagation rule that
decides whether a fork flips the outcome, which is the single most
consequential piece of logic in the fork path.
"""

from __future__ import annotations

import pytest

from generator.schema import state_hash
from replay.engine import PRECONDITION_ERROR, outcome_of, replay_suffix


def make_steps() -> list[dict]:
    """A 6-step failed trace: step 2 is faulty, steps 4 and 5 are starved."""

    def snap(facts):
        return {"goal": "book the cheapest refundable flight", "facts": dict(facts)}

    raw = [
        (0, "call_llm", None, {}, {"summary": "planning"}, snap({}), False),
        (1, "decide", None, {}, {"chosen_tool": "search_flights", "summary": "decide"},
         snap({}), False),
        (2, "call_tool", "convert_currency", {"amount": 180.0},
         {"amount": 164.2, "currency": "EUR", "summary": "converted",
          "_meta": {"parse_failure": True, "cache_hit": True, "cache_age_s": 90000,
                    "retry_count": 0}},
         snap({"cheapest_price": 180.0}), False),
        (3, "call_llm", None, {}, {"summary": "reflect"},
         snap({"cheapest_price": 180.0}), False),
        (4, "call_tool", "book_flight", {"total_eur": 164.2},
         {"error": PRECONDITION_ERROR, "detail": "missing total",
          "summary": "book_flight failed: required input missing"},
         snap({"cheapest_price": 180.0}), True),
        (5, "call_tool", "send_receipt", {},
         {"error": "upstream_timeout", "detail": "smtp down",
          "summary": "send_receipt timed out"},
         snap({"cheapest_price": 180.0}), True),
    ]
    return [
        {
            "id": f"parent-step-{i}", "run_id": "parent", "step_index": i,
            "action_type": a, "tool_name": t, "input": inp, "output": out,
            "state_snapshot": st, "state_hash": state_hash(st),
            "duration_ms": 100 + i, "tokens": 10 + i, "error_flag": err,
        }
        for i, a, t, inp, out, st, err in raw
    ]


def test_prefix_is_copied_verbatim():
    parent = make_steps()
    result = replay_suffix(parent, from_step=2, child_run_id="child", fix_payload={"amount": 170.0})
    for i in range(2):
        assert result.steps[i]["output"] == parent[i]["output"]
        assert result.steps[i]["action_type"] == parent[i]["action_type"]
        assert result.steps[i]["run_id"] == "child"
        assert result.steps[i]["id"] != parent[i]["id"]  # fresh row identity


def test_parent_steps_are_not_mutated():
    """The original run is immutable. This is a PRD requirement, not a detail."""
    parent = make_steps()
    before = [dict(s["output"]) for s in parent]
    replay_suffix(parent, from_step=2, child_run_id="child", fix_payload={"amount": 170.0})
    assert [dict(s["output"]) for s in parent] == before
    assert all(s["run_id"] == "parent" for s in parent)


def test_fix_payload_is_merged_and_fault_telemetry_cleared():
    result = replay_suffix(
        make_steps(), from_step=2, child_run_id="child", fix_payload={"amount": 170.0}
    )
    forked = result.steps[2]
    assert forked["output"]["amount"] == 170.0
    assert forked["output"]["currency"] == "EUR"  # untouched keys survive
    assert forked["output"]["_meta"]["parse_failure"] is False
    assert forked["output"]["_meta"]["cache_hit"] is False
    assert forked["output"]["_meta"]["patched"] is True
    assert forked["error_flag"] is False


def test_only_precondition_failures_recover():
    """A fix upstream cannot repair an unrelated downstream fault."""
    result = replay_suffix(
        make_steps(), from_step=2, child_run_id="child", fix_payload={"amount": 170.0}
    )
    assert result.steps[4]["error_flag"] is False     # was precondition_not_met
    assert result.steps[5]["error_flag"] is True      # was upstream_timeout
    assert result.steps_recovered == 1
    assert outcome_of(result.steps) == "failed"       # the timeout still fails it


def test_outcome_flips_when_every_error_was_caused_by_the_fault():
    steps = make_steps()[:5]  # drop the independent timeout
    result = replay_suffix(steps, from_step=2, child_run_id="child", fix_payload={"amount": 170.0})
    assert outcome_of(result.steps) == "success"
    assert result.steps_recovered == 1


def test_no_fix_means_no_recovery():
    result = replay_suffix(make_steps(), from_step=2, child_run_id="child")
    assert result.steps_recovered == 0
    assert outcome_of(result.steps) == "failed"
    assert any("no fix supplied" in n for n in result.notes)


def test_override_code_is_recorded_but_not_executed():
    result = replay_suffix(
        make_steps(), from_step=2, child_run_id="child", override_code="amount = 170.0"
    )
    assert result.steps[2]["output"]["_override_code"] == "amount = 170.0"
    assert any("not executed" in n for n in result.notes)


def test_state_hash_is_recomputed_and_consistent():
    result = replay_suffix(
        make_steps(), from_step=2, child_run_id="child", fix_payload={"amount": 170.0}
    )
    for step in result.steps:
        assert step["state_hash"] == state_hash(step["state_snapshot"])
    # The fix must be visible in the state the suffix sees.
    assert result.steps[4]["state_snapshot"]["facts"]["amount"] == 170.0


def test_step_indices_stay_dense_and_ordered():
    result = replay_suffix(make_steps(), from_step=3, child_run_id="child", fix_payload={"x": 1})
    assert [s["step_index"] for s in result.steps] == list(range(len(result.steps)))


def test_counts_are_reported_accurately():
    result = replay_suffix(make_steps(), from_step=2, child_run_id="child", fix_payload={"x": 1})
    assert result.forked_at_step == 2
    assert result.steps_replayed == 4  # steps 2,3,4,5
    assert len(result.steps) == 6


@pytest.mark.parametrize("bad", [-1, 6, 99])
def test_out_of_range_fork_is_rejected(bad):
    with pytest.raises(ValueError, match="outside the parent trace"):
        replay_suffix(make_steps(), from_step=bad, child_run_id="child")


def test_empty_trace_is_rejected():
    with pytest.raises(ValueError, match="no steps"):
        replay_suffix([], from_step=0, child_run_id="child")


def test_forking_at_the_last_step_replays_only_that_step():
    result = replay_suffix(make_steps(), from_step=5, child_run_id="child", fix_payload={"x": 1})
    assert result.steps_replayed == 1
    assert len(result.steps) == 6


# -- the closing report ---------------------------------------------------


def closing_report_steps() -> list[dict]:
    """A failed run whose last step is the narrative 'goal not met' summary."""
    steps = make_steps()[:5]  # drop the independent timeout
    snap = {"goal": "book the cheapest refundable flight", "facts": {}}
    steps.append({
        "id": "parent-step-5", "run_id": "parent", "step_index": 5,
        "action_type": "call_llm", "tool_name": None, "input": {},
        "output": {"text": "Run did not complete the goal.",
                   "summary": "Run did not complete the goal.",
                   "_meta": {"retry_count": 0, "parse_failure": False}},
        "state_snapshot": snap, "state_hash": state_hash(snap),
        "duration_ms": 120, "tokens": 40, "error_flag": True,
    })
    return steps


def test_closing_report_is_rewritten_when_the_fix_resolves_everything():
    result = replay_suffix(
        closing_report_steps(), from_step=2, child_run_id="child",
        fix_payload={"amount": 170.0},
    )
    assert outcome_of(result.steps) == "success"
    assert result.steps[-1]["error_flag"] is False
    assert "after the fix" in result.steps[-1]["output"]["summary"]
    assert any("closing report re-derived" in n for n in result.notes)


def test_closing_report_survives_when_a_real_failure_remains():
    """An unrelated fault must keep the run failed, report included."""
    steps = closing_report_steps()
    steps.insert(5, {
        "id": "parent-step-x", "run_id": "parent", "step_index": 5,
        "action_type": "call_tool", "tool_name": "send_receipt", "input": {},
        "output": {"error": "upstream_timeout", "summary": "send_receipt timed out"},
        "state_snapshot": steps[4]["state_snapshot"], "state_hash": steps[4]["state_hash"],
        "duration_ms": 90, "tokens": 12, "error_flag": True,
    })
    for i, s in enumerate(steps):
        s["step_index"] = i
    result = replay_suffix(
        steps, from_step=2, child_run_id="child", fix_payload={"amount": 170.0}
    )
    assert outcome_of(result.steps) == "failed"
    assert result.steps[-1]["error_flag"] is True
    assert any("left as a failure" in n for n in result.notes)


def test_closing_report_is_not_rewritten_without_a_fix():
    result = replay_suffix(closing_report_steps(), from_step=2, child_run_id="child")
    assert outcome_of(result.steps) == "failed"
