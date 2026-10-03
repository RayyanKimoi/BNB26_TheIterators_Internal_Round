"""LLM-as-judge baseline.

The fourth baseline from the Model Specification, and the one the pitch exists
to beat: competing tools read a single run with an LLM and ask it which step
broke. This implements that honestly and as strongly as possible.

Two deliberate choices that favour the baseline:

* The judge is given the **full seven-class taxonomy**, including
  `infinite_loop` and `context_truncation`. Our model has never seen those two
  in any form. On the held-out comparison the judge therefore knows about
  failure modes the model does not, which is the opposite of a strawman. If
  the model still wins there, the claim is airtight.
* The judge sees every step with the same observable fields the feature
  pipeline reads: action, tool, duration, tokens, error flag, retry count and
  the output summary. It is not handicapped on information.

Responses are cached to disk by content hash, so a rerun costs nothing and the
numbers in the results table are reproducible without the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generator.schema import FAILURE_CLASSES

CACHE_PATH = Path("model/artifacts/judge_cache.json")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

TAXONOMY = """\
wrong_tool_chosen      the agent called a plausible but incorrect tool
hallucinated_argument  a tool argument held a value absent from all prior context
stale_retrieval        a cached result from an earlier, different query was served
premature_termination  the run stopped before the goal condition was met
infinite_loop          the agent repeated a cycle until the step budget ran out
schema_violation       a tool returned output breaking its declared schema
context_truncation     earlier messages fell out of the prompt window"""

PROMPT = """\
You are debugging a failed AI agent run. Exactly one step is the ROOT CAUSE of \
the failure. Later steps often show errors, but those are symptoms: name the \
step where the problem originated, because that is the step a developer would \
fork the run from to fix it.

Failure taxonomy:
{taxonomy}

Goal of the run:
{goal}

Trace ({n} steps):
{trace}

Return the index of the single root-cause step."""


def _step_line(step: dict[str, Any]) -> str:
    out = step.get("output") or {}
    meta = out.get("_meta") or {}
    summary = out.get("summary") or out.get("text") or json.dumps(
        {k: v for k, v in out.items() if k != "_meta"}, default=str
    )[:240]
    bits = [
        f"[{step['step_index']:>2}]",
        f"{step['action_type']:<9}",
        f"{(step.get('tool_name') or '-'):<26}",
        f"{step.get('duration_ms', 0):>5}ms",
        f"{step.get('tokens', 0):>4}tok",
    ]
    if step.get("error_flag"):
        bits.append("ERROR")
    if meta.get("retry_count"):
        bits.append(f"retry={meta['retry_count']}")
    if meta.get("parse_failure"):
        bits.append("parse_failure")
    if meta.get("cache_hit"):
        bits.append(f"cache_hit age={meta.get('cache_age_s')}s")
    bits.append(f"| {summary}")
    return " ".join(bits)


def render_trace(run: dict[str, Any]) -> tuple[str, str]:
    steps = run["steps"]
    goal = ""
    for step in steps:
        snapshot = step.get("state_snapshot") or {}
        if isinstance(snapshot.get("goal"), str):
            goal = snapshot["goal"]
            break
    return goal, "\n".join(_step_line(s) for s in steps)


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause_step": {"type": "integer"},
        "failure_class": {"type": "string", "enum": list(FAILURE_CLASSES)},
        "reason": {"type": "string"},
    },
    "required": ["root_cause_step", "failure_class", "reason"],
}


@dataclass
class LLMJudge:
    model: str = DEFAULT_MODEL
    cache_path: Path = CACHE_PATH
    max_retries: int = 6
    # Gemini free tier allows 15 requests per minute for flash-lite, so pace
    # at just under that. Cached runs cost nothing and skip the wait.
    min_interval_s: float = 4.3
    _cache: dict[str, Any] | None = None
    _client: Any = None
    _last_call: float = 0.0

    def __post_init__(self) -> None:
        if self._cache is None:
            self._cache = (
                json.loads(self.cache_path.read_text(encoding="utf-8"))
                if self.cache_path.exists()
                else {}
            )

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai

            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set; cannot run the judge baseline")
            self._client = genai.Client(api_key=key)
        return self._client

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=1), encoding="utf-8")

    def judge(self, run: dict[str, Any]) -> dict[str, Any]:
        """Return {'root_cause_step', 'failure_class', 'reason'} for one run."""
        goal, trace = render_trace(run)
        prompt = PROMPT.format(
            taxonomy=TAXONOMY, goal=goal, n=len(run["steps"]), trace=trace
        )
        key = hashlib.sha256((self.model + prompt).encode("utf-8")).hexdigest()[:32]
        assert self._cache is not None
        if key in self._cache:
            return self._cache[key]

        from google.genai import types

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                wait = self.min_interval_s - (time.monotonic() - self._last_call)
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.monotonic()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                        temperature=0.0,
                    ),
                )
                parsed = json.loads(response.text)
                n = len(run["steps"])
                # A judge that names a step outside the trace has failed to
                # answer the question; clamp and record it rather than crash.
                parsed["root_cause_step"] = max(0, min(n - 1, int(parsed["root_cause_step"])))
                self._cache[key] = parsed
                self._save()
                return parsed
            except Exception as exc:  # noqa: BLE001 - rate limits, 5xx, bad JSON
                last_error = exc
                # A 429 carries the server's own retryDelay. Honour it rather
                # than guessing, otherwise every retry burns more quota.
                match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)", str(exc))
                if match:
                    time.sleep(int(match.group(1)) + 2)
                elif "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    time.sleep(45)
                else:
                    time.sleep(3.0 * (attempt + 1))
        raise RuntimeError(f"judge failed after {self.max_retries} attempts: {last_error}")

    def judge_many(self, runs: list[dict[str, Any]], progress: bool = True) -> list[dict[str, Any]]:
        out = []
        for i, run in enumerate(runs, 1):
            out.append(self.judge(run))
            if progress and i % 10 == 0:
                print(f"    judged {i}/{len(runs)}", flush=True)
        return out
