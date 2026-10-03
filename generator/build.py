"""Build one trace: run the plan, inject at most one fault or one anomaly.

Ground-truth policy, which the README restates and the evaluation depends on:

* `true_failure_step` is the step the fault was injected on, never a downstream
  symptom. For `wrong_tool_chosen` and `premature_termination` that is the
  `decide` step, for `context_truncation` the `call_llm` step, otherwise the
  `call_tool` step.
* The injected step is usually clean on its own. Errors surface one to three
  steps later, which is exactly why "blame the first errored step" is a weak
  baseline. It surfaces immediately on the injected step only
  `IMMEDIATE_ERROR_PROBABILITY` of the time, so that baseline scores a real
  number rather than a rigged zero.
* No step payload carries an injection marker. The label is recoverable only
  from the signature.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from . import faults
from .schema import Run, Trace, new_uuid, validate
from .tasks import TaskType
from .trace import TraceBuilder, confident_distribution, flat_distribution

REFLECT_PROBABILITY = 0.45
IMMEDIATE_ERROR_PROBABILITY = 0.20
DOWNSTREAM_ERROR_PROBABILITY = 0.55


def _rival_tools(task: TaskType, chosen: str, rng: random.Random, k: int = 3) -> list[str]:
    pool = [n for n in (*task.tools, *task.distractors) if n != chosen]
    rng.shuffle(pool)
    return pool[:k]


def _starved_result(tool_name: str, rng: random.Random) -> dict[str, Any]:
    missing = rng.choice(["identifier", "total", "status field", "record"])
    return {
        "error": "precondition_not_met",
        "detail": f"{tool_name} could not resolve the required {missing} from prior context.",
        "summary": f"{tool_name} failed: the required {missing} was not available from earlier steps.",
    }


def build_run(
    task: TaskType,
    rng: random.Random,
    *,
    user_id: str,
    created_at: datetime,
    fault: str | None = None,
    anomaly: str | None = None,
) -> tuple[Trace, dict[str, Any]]:
    """Return a validated trace plus generation metadata for the manifest."""
    b = TraceBuilder(new_uuid(), task.goal, rng)
    plan = list(task.plan)
    truth_step: int | None = None
    evidence_path: str | None = None
    degraded = False

    b.call_llm(
        f"Goal: {task.goal} I will call {', '.join(plan)} in order.",
        prompt="Draft a plan for the goal.",
    )

    if fault == "premature_termination":
        inject_move = rng.randint(2, len(plan) - 2)
    elif fault is not None:
        inject_move = rng.randint(1, len(plan) - 1)
    else:
        inject_move = -1
    anomaly_move = rng.randrange(len(plan)) if anomaly else -1
    # A fault on the terminal call sometimes aborts the run outright, with no
    # closing answer step. Without this the faulty step could never be the last
    # step and the "always blame the last step" baseline would score a rigged
    # zero rather than a real number.
    aborted = fault is not None and inject_move == len(plan) - 1 and rng.random() < 0.6

    for move, tool_name in enumerate(plan):
        spec = task.tools[tool_name]
        is_fault = fault is not None and move == inject_move
        is_anomaly = anomaly is not None and move == anomaly_move

        # -- structural faults end the run here ---------------------------
        if is_fault and fault == "premature_termination":
            rivals = _rival_tools(task, "finish", rng)
            truth_step = b.decide(
                "finish",
                confident_distribution(rng, "finish", rivals),
                "The results so far look sufficient. Returning the answer.",
            )
            evidence_path = f"step[{truth_step}].output.chosen_tool"
            b.call_llm(
                faults.failure_summary(fault, task.goal),
                prompt="Produce the final answer.",
                token_scale=rng.uniform(0.42, 0.68),
            )
            degraded = True
            break

        if is_fault and fault == "infinite_loop":
            frozen = b.snapshot()
            iterations = rng.randint(4, 7)
            for i in range(iterations):
                rivals = _rival_tools(task, tool_name, rng)
                dist = (
                    confident_distribution(rng, tool_name, rivals)
                    if i == 0
                    else flat_distribution(rng, tool_name, rivals)
                )
                idx = b.decide(
                    tool_name,
                    dist,
                    f"Attempt {i + 1}: calling {tool_name} again, the state has not advanced.",
                    frozen_state=frozen,
                )
                if i == 0:
                    truth_step = idx
                    evidence_path = f"step[{idx}].state_snapshot"
                args = spec.args_fn(b.facts, rng)
                result = spec.result_fn(args, b.facts, rng)
                result["summary"] = (
                    f"{tool_name} returned the same payload as the previous attempt. "
                    f"Agent state is unchanged after {i + 1} attempts."
                )
                b.call_tool(
                    spec,
                    args,
                    result,
                    retry_count=i + 1,
                    commit=False,
                    error_flag=i >= iterations - 2,
                    frozen_state=frozen,
                )
            b.call_llm(
                faults.failure_summary(fault, task.goal),
                prompt="Produce the final answer.",
                error_flag=True,
            )
            degraded = True
            break

        # -- decide -------------------------------------------------------
        chosen = tool_name
        rationale = f"Calling {tool_name} to make progress toward the goal."
        uncertain = is_anomaly and anomaly == "low_confidence_decide"

        if is_fault and fault == "wrong_tool_chosen":
            chosen = rng.choice(list(task.distractors))
            rationale = f"{chosen} may provide what the goal needs here."
            uncertain = True

        rivals = _rival_tools(task, chosen, rng)
        dist = (flat_distribution if uncertain else confident_distribution)(rng, chosen, rivals)
        decide_idx = b.decide(chosen, dist, rationale)

        if is_fault and fault == "wrong_tool_chosen":
            truth_step = decide_idx
            evidence_path = f"step[{decide_idx}].output.chosen_tool"

        # -- benign revisit: one extra uncommitted pass, same state hash ---
        if is_anomaly and anomaly == "benign_revisit":
            revisit_args = spec.args_fn(b.facts, rng)
            revisit_result = spec.result_fn(revisit_args, b.facts, rng)
            revisit_result["summary"] = (
                f"Re-checked {tool_name} to confirm the earlier value before continuing."
            )
            b.call_tool(spec, revisit_args, revisit_result, commit=False)
            b.decide(
                tool_name,
                confident_distribution(rng, tool_name, _rival_tools(task, tool_name, rng)),
                f"Value confirmed. Proceeding with {tool_name}.",
            )

        # -- the tool call -------------------------------------------------
        call_spec = task.tool(chosen)
        args = call_spec.args_fn(b.facts, rng)
        result = call_spec.result_fn(args, b.facts, rng)
        kwargs: dict[str, Any] = {}
        # Only the classes whose root cause is this very tool call may surface
        # an error on it. Otherwise error_flag would contradict the payload,
        # and a step that is not the root cause would look like one.
        tool_anchored = fault in ("hallucinated_argument", "stale_retrieval", "schema_violation")
        immediate_error = is_fault and tool_anchored and rng.random() < IMMEDIATE_ERROR_PROBABILITY

        if degraded and not is_fault and rng.random() < DOWNSTREAM_ERROR_PROBABILITY:
            result = _starved_result(chosen, rng)
            kwargs["error_flag"] = True
            kwargs["commit"] = False

        if is_fault and fault == "hallucinated_argument":
            args, field, value = faults.hallucinate_arg(args, rng)
            result = call_spec.result_fn(args, b.facts, rng)
            result["summary"] = (
                f"{chosen} returned a record for {field}={value}, which does not "
                f"appear in any earlier step output."
            )
            truth_step = b.next_index
            evidence_path = f"step[{truth_step}].input.{field}"
            degraded = True

        elif is_fault and fault == "stale_retrieval":
            result, _stale_args, age_s = faults.stale_payload(call_spec, args, b.facts, rng)
            kwargs.update(cache_hit=True, cache_age_s=age_s, duration_scale=rng.uniform(0.12, 0.3))
            truth_step = b.next_index
            evidence_path = f"step[{truth_step}].output.quoted_at"
            degraded = True

        elif is_fault and fault == "schema_violation":
            result, field = faults.break_schema(result, call_spec, rng)
            kwargs.update(parse_failure=True, commit=False)
            truth_step = b.next_index
            evidence_path = f"step[{truth_step}].output.{field}"
            degraded = True

        elif is_fault and fault == "wrong_tool_chosen":
            degraded = True

        if immediate_error:
            kwargs["error_flag"] = True
            result["summary"] = f"{result.get('summary', '')} The caller rejected this result."

        if is_anomaly and anomaly == "slow_tool":
            kwargs["duration_scale"] = rng.uniform(4.0, 6.5)

        if is_anomaly and anomaly == "transient_retry":
            # First attempt errors, the retry succeeds. A successful run that
            # still carries an error_flag, so error_flag alone cannot separate.
            b.call_tool(
                call_spec,
                args,
                {
                    "error": "upstream_timeout",
                    "detail": f"{chosen} timed out, retrying.",
                    "summary": f"{chosen} timed out on the first attempt and will be retried.",
                },
                retry_count=1,
                error_flag=True,
                commit=False,
                duration_scale=rng.uniform(1.8, 3.0),
            )
            kwargs["retry_count"] = 2

        tool_idx = b.call_tool(call_spec, args, result, **kwargs)

        if is_fault and fault in ("hallucinated_argument", "stale_retrieval", "schema_violation"):
            truth_step = tool_idx

        # -- context truncation lands on the reflection step ---------------
        if is_fault and fault == "context_truncation":
            dropped = rng.randint(2, max(2, len(b.steps) // 2))
            truth_step = b.call_llm(
                faults.truncated_reflection(rng, dropped),
                prompt="Summarize progress so far.",
                token_scale=rng.uniform(0.16, 0.32),
                dropped_messages=dropped,
            )
            evidence_path = f"step[{truth_step}].output._meta.dropped_messages"
            # The window that fell away took the established facts with it.
            for key in list(b.facts)[: max(1, len(b.facts) // 2)]:
                b.facts.pop(key)
            degraded = True

        elif aborted:
            # The run died on this step. No further steps are emitted.
            pass
        elif degraded and rng.random() < REFLECT_PROBABILITY:
            b.call_llm(
                faults.off_goal_reflection(chosen),
                prompt="Check the last result against the goal.",
                token_scale=rng.uniform(0.7, 1.1),
            )
        elif degraded:
            pass  # a degraded run never reports progress
        elif is_anomaly and anomaly == "verbose_step":
            b.call_llm(
                f"Working through the {chosen} result in detail before continuing.",
                prompt="Check the last result against the goal.",
                token_scale=rng.uniform(2.4, 3.2),
            )
        elif rng.random() < REFLECT_PROBABILITY:
            b.call_llm(
                f"{chosen} returned what the next step needs. Continuing.",
                prompt="Check the last result against the goal.",
            )

    # -- closing answer ----------------------------------------------------
    status = "failed" if fault else "success"
    if fault not in ("premature_termination", "infinite_loop") and not aborted:
        if status == "failed":
            b.call_llm(
                faults.failure_summary(fault, task.goal),  # type: ignore[arg-type]
                prompt="Produce the final answer.",
                error_flag=True,
            )
        else:
            b.call_llm(
                f"Goal complete. {task.goal} Result recorded.",
                prompt="Produce the final answer.",
            )

    steps = b.steps
    run = Run(
        id=b.run_id,
        user_id=user_id,
        source="synthetic",
        task_type=task.name,
        status=status,  # type: ignore[arg-type]
        injected_class=fault,
        true_failure_step=truth_step,
        total_tokens=sum(s.tokens for s in steps),
        duration_ms=sum(s.duration_ms for s in steps),
        created_at=created_at,
    )
    trace = Trace(run=run, steps=steps)
    validate(trace)

    meta = {
        "run_id": run.id,
        "task_type": task.name,
        "status": status,
        "injected_class": fault,
        "true_failure_step": truth_step,
        "evidence_path": evidence_path,
        "anomaly": anomaly,
        "step_count": len(steps),
    }
    return trace, meta
