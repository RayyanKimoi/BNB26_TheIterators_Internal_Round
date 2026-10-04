"""LangGraph checkpointer adapter.

Converts a LangGraph thread's checkpoint history (from any
`BaseCheckpointSaver` — `SqliteSaver` for a local agent, most commonly) into
this project's Run/Step schema, so a real agent run can be ingested next to
the synthetic corpus through the same tables the rest of the system already
reads.

Usage:
    from langgraph.checkpoint.sqlite import SqliteSaver
    app = graph.compile(checkpointer=SqliteSaver(sqlite3.connect("checkpoints.db")))
    config = {"configurable": {"thread_id": "..."}}
    app.invoke(initial_state, config=config)
    trace = build_trace_from_checkpoints(app, config, task_type="support_ticket")
    # trace is {"run": {...}, "steps": [...]}, the exact shape generator/
    # produces — load it with backend.seed_corpus.insert_run, or go through
    # ingest_checkpoint_thread() below to do both in one call.

`get_state_history()` returns `StateSnapshot`s newest-first, each holding the
full accumulated state at that point and which node is `next` to run. This
walks them oldest-first and turns each consecutive pair into one Step: the
node that ran between them, the state before and after, and real elapsed time
from LangGraph's own checkpoint timestamps.

One correction worth recording: `get_state_history` is a method on the
*compiled graph* (`CompiledStateGraph`, returned by `graph.compile(...)`),
which reconstructs typed `StateSnapshot`s using the graph's own channel
schema — not on the checkpoint saver itself (`SqliteSaver.list()` returns
lower-level, untyped `CheckpointTuple`s instead). This module's functions
take the compiled graph for exactly that reason, verified empirically against
LangGraph 1.2.12 rather than assumed from the method's name.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from generator.schema import new_uuid, state_hash

# LangGraph does not itself distinguish an LLM call from a tool call from a
# routing decision — that is a convention each graph's author chooses node
# names around. This reads node names the same pragmatic way the OTel adapter
# reads span names: a short hint list, never a claim of certainty.
_LLM_HINTS = ("llm", "chat", "generate", "completion", "model")
_DECIDE_HINTS = ("route", "decide", "choice", "branch", "check", "should_")


def _classify_node(node_name: str) -> tuple[str, str | None]:
    lowered = node_name.lower()
    if any(hint in lowered for hint in _LLM_HINTS):
        return "call_llm", None
    if any(hint in lowered for hint in _DECIDE_HINTS):
        return "decide", None
    return "call_tool", node_name


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def build_trace_from_checkpoints(
    graph: Any,
    config: dict[str, Any],
    *,
    task_type: str = "langgraph_ingest",
    user_id: str | None = None,
) -> dict[str, Any]:
    """Build one Run + its Steps from a LangGraph thread's checkpoint history.

    `graph` is a compiled graph (`StateGraph(...).compile(checkpointer=...)`)
    that has already run the thread named in `config`
    (`{"configurable": {"thread_id": "..."}}`). Returns the same
    `{"run": {...}, "steps": [...]}` shape the generator produces.

    Raises `ValueError` if the thread has no checkpoints — a caller bug (wrong
    thread_id, or asking before the graph has run), not something to paper
    over with an empty trace.
    """
    history = list(graph.get_state_history(config))
    if not history:
        raise ValueError(f"no checkpoint history for thread {config!r}")

    # Newest-first -> oldest-first, so step 0 is the initial state.
    history = list(reversed(history))

    run_id = new_uuid()
    steps: list[dict[str, Any]] = []
    total_tokens = 0
    total_duration_ms = 0
    any_error = False
    real_index = 0

    for index in range(len(history) - 1):
        before, after = history[index], history[index + 1]
        node_name = (before.next or ("unknown",))[0]
        # LangGraph's own bookkeeping transition from "nothing" to the input
        # state the caller passed to invoke() — not a node the agent ran, so
        # it is not a step.
        if node_name == "__start__":
            continue
        action_type, tool_name = _classify_node(node_name)

        before_ts = _parse_timestamp(before.created_at)
        after_ts = _parse_timestamp(after.created_at)
        duration_ms = (
            max(0, int((after_ts - before_ts).total_seconds() * 1000))
            if before_ts and after_ts
            else 0
        )

        output = dict(after.values or {})
        # Best-effort: only set if the graph's own state reports it. No token
        # count is fabricated when a graph does not track one itself.
        tokens = int(output.get("tokens") or output.get("token_count") or 0)
        error_flag = bool(output.get("error"))

        total_tokens += tokens
        total_duration_ms += duration_ms
        any_error = any_error or error_flag

        steps.append(
            {
                "id": new_uuid(),
                "run_id": run_id,
                "step_index": real_index,
                "action_type": action_type,
                "tool_name": tool_name,
                "input": dict(before.values or {}),
                "output": output,
                "state_snapshot": output,
                "state_hash": state_hash(output),
                "duration_ms": duration_ms,
                "tokens": tokens,
                "error_flag": error_flag,
            }
        )
        real_index += 1

    started_at = _parse_timestamp(history[0].created_at) or datetime.now(timezone.utc)
    run = {
        "id": run_id,
        "user_id": user_id,
        "source": "langgraph",
        "task_type": task_type,
        "status": "failed" if any_error else "success",
        "parent_run_id": None,
        "forked_at_step": None,
        "fix_applied": None,
        # Ground truth belongs to the synthetic corpus, never to a real run.
        "injected_class": None,
        "true_failure_step": None,
        "total_tokens": total_tokens,
        "duration_ms": total_duration_ms,
        "created_at": started_at.isoformat(),
    }
    return {"run": run, "steps": steps}


def ingest_checkpoint_thread(
    session: Any,
    graph: Any,
    config: dict[str, Any],
    *,
    task_type: str = "langgraph_ingest",
    user_id: str | None = None,
):
    """Build a trace from a LangGraph thread and insert it into `session`.

    Delegates to `backend.seed_corpus.insert_run` so a real run lands through
    the exact same idempotent insertion path the synthetic corpus loader
    uses — one place owns "how a trace dict becomes AgentRun + Step rows."
    Imported lazily to avoid a backend <-> ingest import cycle at module load.
    """
    from backend.seed_corpus import insert_run

    trace = build_trace_from_checkpoints(graph, config, task_type=task_type, user_id=user_id)
    return insert_run(session, trace)
