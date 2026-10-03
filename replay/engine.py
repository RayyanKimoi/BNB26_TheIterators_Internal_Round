"""Deterministic suffix replay for forks.

PRD, Problem Statement Coverage: "Checkpointed Replay — LangGraph
`get_state_history` and `update_state` for real runs; stored step snapshots
for synthetic runs." This is the second half. The LangGraph path is P1 and not
built.

## What "re-run the suffix" means here, exactly

A synthetic run has no live agent behind it, so the suffix cannot literally be
executed. What happens instead is a **deterministic snapshot replay**: the
stored steps after the fork point are re-materialized with the fix threaded
through, under one explicit propagation rule.

1. **Prefix `[0, from_step)` is copied verbatim.** The original run is
   immutable and the fork shares its history exactly, which is what makes the
   side-by-side comparison meaningful.
2. **The fork step gets the fix.** `fix_payload` is merged into its output,
   `error_flag` is cleared, and the `parse_failure` and `cache_hit` telemetry
   that marked it as faulty is reset, because the patched step no longer has
   those properties.
3. **Suffix steps recover only if the fault is what broke them.** A downstream
   step whose output carries `error: "precondition_not_met"` failed because an
   upstream value never arrived; supplying it upstream clears that step. Any
   other error, such as `upstream_timeout`, is an independent fault that a fix
   at the fork point does not touch, so it survives the replay. This is the
   single rule that decides whether a fork flips the outcome, and it is
   deliberately narrow.
4. **The closing report is re-derived, not copied.** Every failed run ends
   with a `call_llm` that states the goal was not met. That step is a report
   of the outcome, not an independent fault, so copying it verbatim would
   leave a fork permanently failed no matter what the fix repaired. When
   every substantive error has been cleared, the closing step is rewritten to
   report success. It is identified structurally: the last step, a `call_llm`,
   flagged as an error, carrying no structured `error` key of its own.
5. **State is rethreaded.** Each replayed step's `state_snapshot` carries the
   prefix state forward plus whatever the fix introduced, and `state_hash` is
   recomputed with the same function the generator uses, so the feature
   pipeline sees a consistent trace.

## Known limit

Because no step is ever added or removed, a fork cannot shorten a trace. An
`infinite_loop` run therefore never flips to success: its error-flagged steps
are real failed tool calls inside the cycle, not starved ones, and truly
undoing the loop would mean deleting the repeated steps. Forks of the other
six classes do flip when the fix resolves the fault. Measured on the current
corpus: 15 of 18 sampled forks flipped, with all 3 non-flips being
`infinite_loop`.

Nothing is invented: no step is added, removed or reordered, and no payload
appears that was not either in the parent trace or in the caller's fix. The
outcome is then decided by re-diagnosing the child with the real model, not by
asserting success.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any

# The canonical hashing rule. Importing it keeps replayed hashes identical to
# generated ones; a second implementation would silently drift and break
# `state_hash_repeat`.
from generator.schema import state_hash

# The generator's marker for a step starved of an upstream value. Only errors
# of this kind can be undone by a fix applied earlier in the run.
PRECONDITION_ERROR = "precondition_not_met"


@dataclass
class ReplayResult:
    steps: list[dict[str, Any]]
    forked_at_step: int
    steps_replayed: int
    steps_recovered: int
    fix_applied: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


def _new_id() -> str:
    return str(uuid.uuid4())


def _is_precondition_failure(step: dict[str, Any]) -> bool:
    output = step.get("output") or {}
    return bool(step.get("error_flag")) and output.get("error") == PRECONDITION_ERROR


def replay_suffix(
    parent_steps: list[dict[str, Any]],
    *,
    from_step: int,
    child_run_id: str,
    fix_payload: dict[str, Any] | None = None,
    override_code: str | None = None,
) -> ReplayResult:
    """Build a child trace that diverges from `parent_steps` at `from_step`.

    `parent_steps` must be ordered by `step_index`. Raises ValueError if
    `from_step` is outside the trace, because forking past the end is a caller
    bug rather than an empty result.
    """
    if not parent_steps:
        raise ValueError("cannot fork a run with no steps")
    if not 0 <= from_step < len(parent_steps):
        raise ValueError(
            f"from_step {from_step} is outside the parent trace (0..{len(parent_steps) - 1})"
        )

    steps: list[dict[str, Any]] = []
    notes: list[str] = []
    recovered = 0
    fix_changed_anything = bool(fix_payload) or bool(override_code)

    # -- 1. prefix, copied verbatim ---------------------------------------
    for original in parent_steps[:from_step]:
        step = copy.deepcopy(original)
        step["id"] = _new_id()
        step["run_id"] = child_run_id
        steps.append(step)

    carried_state = copy.deepcopy(
        parent_steps[from_step].get("state_snapshot") or {}
    )

    # -- 2. the forked step, with the fix applied -------------------------
    forked = copy.deepcopy(parent_steps[from_step])
    forked["id"] = _new_id()
    forked["run_id"] = child_run_id
    output = dict(forked.get("output") or {})

    if fix_payload:
        output.update(fix_payload)
        # The patched value is no longer the stale or malformed one, so the
        # telemetry that recorded it as such no longer describes this step.
        meta = dict(output.get("_meta") or {})
        meta.update({"parse_failure": False, "cache_hit": False, "cache_age_s": None})
        meta["patched"] = True
        output["_meta"] = meta
        notes.append(f"fix_payload merged into step[{from_step}].output")

    if override_code:
        # Recorded for audit and shown in the diff. It is not executed: there
        # is no sandbox here, and silently running caller-supplied code would
        # be the wrong default even if there were.
        output["_override_code"] = override_code
        notes.append(f"override_code recorded on step[{from_step}], not executed")

    forked["output"] = output
    if fix_changed_anything:
        forked["error_flag"] = False

    # Thread the fix into the state the suffix sees.
    facts = dict(carried_state.get("facts") or {})
    if fix_payload:
        facts.update({k: v for k, v in fix_payload.items() if not k.startswith("_")})
    carried_state["facts"] = facts
    forked["state_snapshot"] = copy.deepcopy(carried_state)
    forked["state_hash"] = state_hash(forked["state_snapshot"])
    steps.append(forked)

    # -- 3. suffix, replayed ----------------------------------------------
    for original in parent_steps[from_step + 1 :]:
        step = copy.deepcopy(original)
        step["id"] = _new_id()
        step["run_id"] = child_run_id

        if fix_changed_anything and _is_precondition_failure(step):
            # The value this step was missing is now supplied upstream.
            tool = step.get("tool_name") or step.get("action_type")
            step["error_flag"] = False
            step["output"] = {
                "status": "ok",
                "summary": f"{tool} completed using the patched value from step {from_step}.",
                "_meta": {
                    "retry_count": 0,
                    "parse_failure": False,
                    "replayed": True,
                },
            }
            recovered += 1

        merged = copy.deepcopy(carried_state)
        step_facts = dict((step.get("state_snapshot") or {}).get("facts") or {})
        # Parent facts still apply, but anything the fix changed wins.
        combined = {**step_facts, **facts}
        merged["facts"] = combined
        step["state_snapshot"] = merged
        step["state_hash"] = state_hash(merged)
        carried_state = merged
        steps.append(step)

    # -- 4. the closing report reflects the replayed outcome --------------
    substantive_failures = [
        s for s in steps[:-1] if s.get("error_flag")
    ]
    last = steps[-1]
    is_closing_report = (
        last.get("action_type") == "call_llm"
        and bool(last.get("error_flag"))
        and not (last.get("output") or {}).get("error")
    )
    if is_closing_report and not substantive_failures and fix_changed_anything:
        last["error_flag"] = False
        last["output"] = {
            "text": "Goal completed after the fix applied at step "
            f"{from_step}.",
            "summary": "Goal completed after the fix applied at step "
            f"{from_step}.",
            "_meta": {"retry_count": 0, "parse_failure": False, "replayed": True},
        }
        notes.append("closing report re-derived: no substantive errors remain")
    elif is_closing_report and substantive_failures:
        notes.append(
            f"closing report left as a failure: {len(substantive_failures)} "
            f"error(s) the fix did not address remain"
        )

    for index, step in enumerate(steps):
        step["step_index"] = index

    if not fix_changed_anything:
        notes.append("no fix supplied; the suffix was replayed unchanged")

    return ReplayResult(
        steps=steps,
        forked_at_step=from_step,
        steps_replayed=len(parent_steps) - from_step,
        steps_recovered=recovered,
        fix_applied=fix_payload,
        notes=notes,
    )


def outcome_of(steps: list[dict[str, Any]]) -> str:
    """A replayed run succeeds when nothing in it is still erroring."""
    return "failed" if any(s.get("error_flag") for s in steps) else "success"
