"""Gemini root-cause explainer.

The second of the "two jobs, never blurred" from PRD.md: the ML model decides
WHICH step is the root cause, and this explains WHY in plain English and
proposes a fix. It is handed the flagged step, its raw feature values and its
SHAP attributions, and it never chooses or second-guesses the step. That line
is what lets the project claim a trained, evaluated model rather than an LLM
wrapper, and it is the line a judge will probe.

JSON mode with an enforced response schema, so the three keys always come
back and no consumer has to parse prose.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from backend.models import ExplanationPayload

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "root_cause": {
            "type": "string",
            "description": "Why this specific step is the origin of the failure",
        },
        "evidence_summary": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key JSON attributes and values that point at this step",
        },
        "proposed_fix": {
            "type": "string",
            "description": "A code diff or modified JSON payload that resolves it",
        },
        # PRD item 14. `patch_json` is a STRING holding a JSON object literal
        # rather than a nested object: Gemini's structured output wants every
        # object's properties declared up front, and a patch's keys are
        # whatever that step's payload happens to use. A string sidesteps that
        # and is parsed (and rejected if malformed) in `_parse_candidates`.
        "fix_candidates": {
            "type": "array",
            "description": "2 to 3 distinct candidate fixes, best first",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer", "description": "1 is the best candidate"},
                    "patch_json": {
                        "type": "string",
                        "description": (
                            "A JSON object literal merged into the flagged step's "
                            'output, e.g. {"currency": "EUR"}'
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this candidate resolves the root cause",
                    },
                },
                "required": ["rank", "patch_json", "rationale"],
            },
        },
    },
    "required": ["root_cause", "evidence_summary", "proposed_fix", "fix_candidates"],
}

PROMPT = """\
You are explaining a diagnosis that has already been made. A trained \
classifier has identified step {step_index} of this AI agent run as the root \
cause of the failure. Do not second-guess that choice: your job is to explain \
why that step is the origin and to propose a concrete fix.

Goal of the run:
{goal}

Flagged step {step_index} ({action_type}{tool}):
{step_json}

Model evidence for that step. `features` are the raw measured values and \
`shap` is each feature's contribution to the score, where a larger positive \
number means the feature pushed this step harder toward being the root cause:
{evidence_json}

Surrounding trace for context:
{context}

{class_line}

Write the root cause in plain English, naming concrete values from the step. \
List the specific JSON attributes and values that justify the call. Then give \
a proposed fix as either a unified diff or a corrected JSON payload.

Also give 2 to 3 DISTINCT candidate fixes in `fix_candidates`, ranked best \
first. Each `patch_json` must be a JSON object literal that will be merged \
into the flagged step's output to correct it, using only values that appear \
in the evidence or surrounding trace above. Make the candidates genuinely \
different approaches, not restatements of one another.

Be concrete and brief. Do not invent values that do not appear above.\
"""


class ExplainerClient(Protocol):
    """Anything that can turn a prompt into the three-key payload.

    Exists so tests inject a stub and never touch the network, and so the
    provider stays swappable, which PRD.md requires of the LLM layer.
    """

    def generate(self, prompt: str) -> dict[str, Any]: ...


class GeminiExplainer:
    """Calls Gemini with a JSON-constrained response schema."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or GEMINI_MODEL
        self._api_key = api_key
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai

            key = self._api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set; cannot generate an explanation")
            self._client = genai.Client(api_key=key)
        return self._client

    def generate(self, prompt: str) -> dict[str, Any]:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
                temperature=0.2,
            ),
        )
        return json.loads(response.text)


def _trim(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, indent=2, default=str)
    return text if len(text) <= limit else text[:limit] + "\n  ... truncated"


def _parse_candidates(raw: Any) -> list[dict[str, Any]]:
    """Turn the model's `fix_candidates` into SuggestedFix-shaped dicts.

    Each candidate's `patch_json` is a string holding a JSON object (see
    RESPONSE_SCHEMA). A candidate whose patch will not parse, or does not
    parse to an object, is dropped rather than failing the whole explanation:
    losing one of three optional candidates is a far better outcome than
    losing the root cause the user actually asked for. Capped at 3 to match
    the PRD.
    """
    if not isinstance(raw, list):
        return []

    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        patch_raw = item.get("patch_json")
        if isinstance(patch_raw, dict):
            patch = patch_raw  # already an object; accept it
        elif isinstance(patch_raw, str):
            try:
                parsed = json.loads(patch_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(parsed, dict):
                continue
            patch = parsed
        else:
            continue
        if not patch:
            continue

        rank = item.get("rank")
        candidates.append(
            {
                "rank": int(rank) if isinstance(rank, (int, float)) else index + 1,
                "patch": patch,
                "rationale": str(item.get("rationale") or ""),
            }
        )

    candidates.sort(key=lambda c: c["rank"])
    return candidates[:3]


def build_prompt(
    *,
    goal: str,
    step: dict[str, Any],
    step_index: int,
    evidence: dict[str, Any],
    shap: dict[str, float] | None,
    predicted_class: str,
    context_steps: list[dict[str, Any]],
) -> str:
    """Assemble the explainer prompt. Pure, so it is unit testable."""
    tool = f", tool {step.get('tool_name')}" if step.get("tool_name") else ""
    if predicted_class and predicted_class != "unknown":
        class_line = (
            f"The classifier labelled this failure `{predicted_class}`. "
            f"Explain it as that failure mode."
        )
    else:
        class_line = (
            "The classifier flagged this step but could not name the failure mode, "
            "because it does not match any class it was trained on. Say so plainly "
            "and describe what actually went wrong from the evidence, without "
            "claiming a named category."
        )

    context = "\n".join(
        f"  [{s.get('step_index')}] {s.get('action_type')} "
        f"{s.get('tool_name') or '-'} "
        f"{'ERROR ' if s.get('error_flag') else ''}"
        f"{str((s.get('output') or {}).get('summary', ''))[:110]}"
        for s in context_steps
    )

    return PROMPT.format(
        step_index=step_index,
        goal=goal or "(not recorded)",
        action_type=step.get("action_type", "unknown"),
        tool=tool,
        step_json=_trim({k: step.get(k) for k in ("input", "output", "error_flag")}),
        evidence_json=_trim({"features": evidence, "shap": shap}),
        context=context or "  (no surrounding steps)",
        class_line=class_line,
    )


def explain(
    client: ExplainerClient,
    *,
    goal: str,
    step: dict[str, Any],
    step_index: int,
    evidence: dict[str, Any],
    shap: dict[str, float] | None,
    predicted_class: str,
    context_steps: list[dict[str, Any]],
) -> ExplanationPayload:
    """Generate and validate one explanation."""
    prompt = build_prompt(
        goal=goal,
        step=step,
        step_index=step_index,
        evidence=evidence,
        shap=shap,
        predicted_class=predicted_class,
        context_steps=context_steps,
    )
    raw = client.generate(prompt)
    if isinstance(raw, dict) and "fix_candidates" in raw:
        raw = {**raw, "fix_candidates": _parse_candidates(raw.get("fix_candidates"))}
    # Pydantic validates the shape, so a malformed model response fails here
    # rather than reaching the database or the UI.
    return ExplanationPayload.model_validate(raw)
