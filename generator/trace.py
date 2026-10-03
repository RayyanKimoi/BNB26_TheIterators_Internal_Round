"""Incremental trace builder.

Appends schema-exact `steps` rows while carrying agent state forward, so that
`state_hash` and the arg-sourcing behaviour that `arg_novelty` depends on are
both produced naturally rather than patched on afterwards.
"""

from __future__ import annotations

import math
import random
from typing import Any

from .schema import Step, state_hash
from .tasks import ToolSpec

# Base latency and token bands per action type, before fault or anomaly scaling.
DECIDE_LATENCY = (300, 900)
DECIDE_TOKENS = (180, 420)
TOOL_TOKENS = (40, 160)
LLM_LATENCY = (600, 1800)
LLM_TOKENS = (300, 900)


def entropy(dist: dict[str, float]) -> float:
    """Shannon entropy in nats, for the record. Feature extraction recomputes it."""
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def confident_distribution(rng: random.Random, chosen: str, others: list[str]) -> dict[str, float]:
    """A decision the agent is sure about: one tool takes most of the mass."""
    top = rng.uniform(0.78, 0.94)
    rest = 1.0 - top
    dist = {chosen: round(top, 4)}
    if others:
        weights = [rng.uniform(0.2, 1.0) for _ in others]
        total = sum(weights)
        for name, w in zip(others, weights):
            dist[name] = round(rest * w / total, 4)
    return dist


def flat_distribution(rng: random.Random, chosen: str, others: list[str]) -> dict[str, float]:
    """An uncertain decision: near-uniform mass, which is the entropy spike."""
    names = [chosen, *others]
    weights = [rng.uniform(0.85, 1.15) for _ in names]
    total = sum(weights)
    return {n: round(w / total, 4) for n, w in zip(names, weights)}


class TraceBuilder:
    def __init__(self, run_id: str, goal: str, rng: random.Random) -> None:
        self.run_id = run_id
        self.goal = goal
        self.rng = rng
        self.steps: list[Step] = []
        self.state: dict[str, Any] = {"goal": goal, "facts": {}}

    # -- state ------------------------------------------------------------

    def commit(self, keys: tuple[str, ...], result: dict[str, Any]) -> None:
        for key in keys:
            if key in result:
                self.state["facts"][key] = result[key]

    def snapshot(self) -> dict[str, Any]:
        import copy

        return copy.deepcopy(self.state)

    @property
    def facts(self) -> dict[str, Any]:
        return self.state["facts"]

    @property
    def next_index(self) -> int:
        return len(self.steps)

    # -- step emitters ----------------------------------------------------

    def _append(
        self,
        action_type: str,
        inp: dict[str, Any],
        out: dict[str, Any],
        duration_ms: int,
        tokens: int,
        *,
        tool_name: str | None = None,
        error_flag: bool = False,
        frozen_state: dict[str, Any] | None = None,
    ) -> int:
        snap = frozen_state if frozen_state is not None else self.snapshot()
        step = Step(
            run_id=self.run_id,
            step_index=len(self.steps),
            action_type=action_type,  # type: ignore[arg-type]
            tool_name=tool_name,
            input=inp,
            output=out,
            state_snapshot=snap,
            state_hash=state_hash(snap),
            duration_ms=max(1, duration_ms),
            tokens=max(0, tokens),
            error_flag=error_flag,
        )
        self.steps.append(step)
        return step.step_index

    def decide(
        self,
        chosen_tool: str,
        candidates: dict[str, float],
        rationale: str,
        *,
        duration_scale: float = 1.0,
        token_scale: float = 1.0,
        frozen_state: dict[str, Any] | None = None,
    ) -> int:
        rng = self.rng
        out = {
            "chosen_tool": chosen_tool,
            "candidates": candidates,
            "entropy": round(entropy(candidates), 4),
            "rationale": rationale,
            "summary": rationale,
            "_meta": {"retry_count": 0, "parse_failure": False},
        }
        inp = {
            "goal": self.goal,
            "known_facts": sorted(self.facts.keys()),
            "candidate_tools": sorted(candidates.keys()),
        }
        return self._append(
            "decide",
            inp,
            out,
            int(rng.uniform(*DECIDE_LATENCY) * duration_scale),
            int(rng.uniform(*DECIDE_TOKENS) * token_scale),
            frozen_state=frozen_state,
        )

    def call_tool(
        self,
        spec: ToolSpec,
        args: dict[str, Any],
        result: dict[str, Any],
        *,
        duration_scale: float = 1.0,
        token_scale: float = 1.0,
        retry_count: int = 0,
        parse_failure: bool = False,
        cache_hit: bool = False,
        cache_age_s: int | None = None,
        error_flag: bool = False,
        commit: bool = True,
        frozen_state: dict[str, Any] | None = None,
    ) -> int:
        rng = self.rng
        out = dict(result)
        out["_meta"] = {
            "retry_count": retry_count,
            "parse_failure": parse_failure,
            "cache_hit": cache_hit,
            "cache_age_s": cache_age_s,
        }
        idx = self._append(
            "call_tool",
            dict(args),
            out,
            int(rng.uniform(*spec.latency) * duration_scale),
            int(rng.uniform(*TOOL_TOKENS) * token_scale),
            tool_name=spec.name,
            error_flag=error_flag,
            frozen_state=frozen_state,
        )
        # State advances only on a clean call, which is why a fault starves
        # every downstream step of the fact it needed.
        if commit and not error_flag and not parse_failure:
            self.commit(spec.state_keys, result)
        return idx

    def call_llm(
        self,
        text: str,
        *,
        prompt: str = "",
        duration_scale: float = 1.0,
        token_scale: float = 1.0,
        dropped_messages: int = 0,
        error_flag: bool = False,
        frozen_state: dict[str, Any] | None = None,
    ) -> int:
        rng = self.rng
        tokens = int(rng.uniform(*LLM_TOKENS) * token_scale)
        out = {
            "text": text,
            "summary": text,
            "_meta": {
                "retry_count": 0,
                "parse_failure": False,
                "dropped_messages": dropped_messages,
                "context_tokens": tokens,
            },
        }
        inp = {
            "prompt": prompt or "Reflect on progress toward the goal.",
            "context_step_count": max(0, len(self.steps) - dropped_messages),
        }
        return self._append(
            "call_llm",
            inp,
            out,
            int(rng.uniform(*LLM_LATENCY) * duration_scale),
            tokens,
            error_flag=error_flag,
            frozen_state=frozen_state,
        )
