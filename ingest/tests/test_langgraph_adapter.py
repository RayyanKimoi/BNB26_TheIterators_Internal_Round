"""Integration test against a real LangGraph graph and a real SqliteSaver.

No mocking of LangGraph itself: a tiny two-node graph actually runs, its
checkpoints go into a real in-memory SQLite connection, and the adapter reads
them back exactly as it would for a real agent. This is deliberately not a
hand-rolled fake checkpoint shape — LangGraph's checkpoint internals have
changed across versions before, and a fake would silently drift from whatever
version is actually installed.
"""

from __future__ import annotations

import sqlite3
from typing import TypedDict

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from sqlmodel import Session, SQLModel, create_engine, select
from sqlmodel.pool import StaticPool

from backend.db import AgentRun, Step
from ingest.langgraph_adapter import build_trace_from_checkpoints, ingest_checkpoint_thread


class _State(TypedDict):
    messages: list[str]
    step_count: int


def _node_a(state: _State) -> dict:
    return {"messages": state["messages"] + ["node_a ran"], "step_count": state["step_count"] + 1}


def _node_b(state: _State) -> dict:
    return {"messages": state["messages"] + ["node_b ran"], "step_count": state["step_count"] + 1}


def _compiled_graph():
    graph = StateGraph(_State)
    graph.add_node("fetch_tool", _node_a)
    graph.add_node("llm_generate", _node_b)
    graph.set_entry_point("fetch_tool")
    graph.add_edge("fetch_tool", "llm_generate")
    graph.add_edge("llm_generate", END)
    return graph.compile(checkpointer=SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)))


def _run_graph():
    app = _compiled_graph()
    config = {"configurable": {"thread_id": "test-thread"}}
    app.invoke({"messages": [], "step_count": 0}, config=config)
    return app, config


def test_build_trace_from_real_checkpoints_has_one_step_per_node():
    app, config = _run_graph()
    trace = build_trace_from_checkpoints(app, config, task_type="demo")

    assert trace["run"]["source"] == "langgraph"
    assert trace["run"]["task_type"] == "demo"
    assert trace["run"]["status"] == "success"
    assert trace["run"]["injected_class"] is None, "a real run carries no synthetic ground truth"

    steps = trace["steps"]
    assert len(steps) == 2  # fetch_tool, then llm_generate
    assert [s["step_index"] for s in steps] == [0, 1]


def test_node_names_are_classified_by_their_hints():
    app, config = _run_graph()
    trace = build_trace_from_checkpoints(app, config)
    steps = trace["steps"]

    assert steps[0]["action_type"] == "call_tool"
    assert steps[0]["tool_name"] == "fetch_tool"
    assert steps[1]["action_type"] == "call_llm"
    assert steps[1]["tool_name"] is None


def test_state_accumulates_correctly_across_steps():
    app, config = _run_graph()
    trace = build_trace_from_checkpoints(app, config)
    steps = trace["steps"]

    assert steps[0]["output"]["messages"] == ["node_a ran"]
    assert steps[1]["output"]["messages"] == ["node_a ran", "node_b ran"]
    assert steps[1]["input"] == steps[0]["output"], "step i's input is step i-1's output"


def test_duration_is_real_elapsed_time_not_fabricated():
    app, config = _run_graph()
    trace = build_trace_from_checkpoints(app, config)
    for step in trace["steps"]:
        assert step["duration_ms"] >= 0
        assert isinstance(step["duration_ms"], int)


def test_state_hash_matches_the_shared_implementation():
    from generator.schema import state_hash

    app, config = _run_graph()
    trace = build_trace_from_checkpoints(app, config)
    for step in trace["steps"]:
        assert step["state_hash"] == state_hash(step["state_snapshot"])


def test_raises_on_a_thread_with_no_history():
    app = _compiled_graph()
    with pytest.raises(ValueError):
        build_trace_from_checkpoints(
            app, {"configurable": {"thread_id": "never-ran"}}
        )


class _ErrorState(TypedDict):
    messages: list[str]
    step_count: int
    error: str | None


def test_error_in_state_marks_the_run_failed():
    # A dedicated schema with `error` as a declared field: LangGraph's state
    # channels are built from the schema's own annotations, so a key a node
    # returns but the schema never declared is silently dropped rather than
    # merged in.
    graph = StateGraph(_ErrorState)

    def failing_node(state: _ErrorState) -> dict:
        return {"messages": state["messages"], "step_count": state["step_count"] + 1, "error": "boom"}

    graph.add_node("tool_call", failing_node)
    graph.set_entry_point("tool_call")
    graph.add_edge("tool_call", END)
    app = graph.compile(checkpointer=SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)))
    config = {"configurable": {"thread_id": "t2"}}
    app.invoke({"messages": [], "step_count": 0}, config=config)

    trace = build_trace_from_checkpoints(app, config)
    assert trace["run"]["status"] == "failed"
    assert trace["steps"][-1]["error_flag"] is True


def test_ingest_checkpoint_thread_persists_through_the_shared_inserter():
    app, config = _run_graph()
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        run = ingest_checkpoint_thread(session, app, config, task_type="support")
        session.commit()
        assert run is not None
        assert session.get(AgentRun, run.id) is not None
        steps = session.exec(select(Step).where(Step.run_id == run.id)).all()
        assert len(steps) == 2
