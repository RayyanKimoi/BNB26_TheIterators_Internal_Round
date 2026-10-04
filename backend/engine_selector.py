"""Diagnosis engine selection, with the local Hybrid Sentry Engine as the floor.

A thin routing layer in front of diagnosis. The caller names a provider; if
that provider is unavailable, slow, misconfigured or returns something that is
not a valid diagnosis, this falls back to the local engine and the request
still succeeds. Diagnosis never fails because an optional remote is down.

Nothing in `model/predict.py` or `backend/main.py`'s existing diagnosis path is
modified. This module only wraps them.

Two design decisions worth reading before changing anything here:

**Remote URLs come from server-side environment variables, never from the
request.** The `X-Engine-Provider` header selects *which configured provider*
to use; it cannot supply a URL. Letting a client name an arbitrary endpoint
that the server then POSTs an internal trace to is server-side request forgery,
and the trace is exactly the sort of payload an attacker would want exfiltrated.
An unconfigured provider falls back to local rather than trusting the caller.

**The local engine is the only one that is actually wired today.** No
`ENGINE_*_URL` is set in a normal deployment, so every provider resolves to
local and the diagnosis is identical to what the endpoint produced before this
module existed. That is intentional: the routing is real, the remote providers
are opt-in, and the response says which engine actually ran rather than which
one was asked for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("backend.engine_selector")

#: Hard ceiling on a remote engine call. The local engine is always available,
#: so waiting longer for a remote one is strictly worse than falling back.
REMOTE_TIMEOUT_S = 2.0

#: The canonical provider ids, and the aliases that map onto them. The
#: frontend's selector (`frontend/src/lib/engineProvider.ts`) and the original
#: backend spec use different names for the same two providers, so both are
#: accepted and normalized here rather than forcing either side to change.
PROVIDER_ALIASES: dict[str, str] = {
    "default": "default",
    "local": "default",
    "hybrid": "default",
    # Hosted frontier model, bring-your-own-key.
    "custom_llm": "custom_llm",
    "byo_llm": "custom_llm",
    # Customer-operated scoring service.
    "webhook": "webhook",
    "enterprise_webhook": "webhook",
    # On-prem / air-gapped inference server. Same transport as a webhook.
    "air_gapped": "air_gapped",
    # Hosted LLM judge on Groq's OpenAI-compatible API.
    "groq": "groq",
}

#: Groq speaks the OpenAI chat-completions shape, so this needs no SDK and
#: adds no dependency: httpx is already here for the Slack webhook.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

#: Overridable because Groq retires model ids faster than this repo changes,
#: and because model availability differs per account: a 404 from Groq means
#: "your key cannot use that model", not "the endpoint is wrong". List what a
#: key can actually reach with GET https://api.groq.com/openai/v1/models.
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"

#: LLM inference needs far longer than a scoring webhook. REMOTE_TIMEOUT_S is
#: 2s because a service that exists to return a precomputed diagnosis should
#: be fast; a model generating tokens is a different kind of wait, and holding
#: it to 2s would mean every Groq call times out and falls back, which would
#: look like the provider being broken rather than being slow.
GROQ_TIMEOUT_S = float(os.environ.get("GROQ_TIMEOUT_S", "20"))

#: Environment variable holding each remote provider's endpoint. Absent means
#: the provider is not configured, which means fall back to local.
PROVIDER_ENV_VAR: dict[str, str] = {
    "custom_llm": "ENGINE_CUSTOM_LLM_URL",
    "webhook": "ENGINE_WEBHOOK_URL",
    "air_gapped": "ENGINE_AIR_GAPPED_URL",
}

#: Keys a remote response must carry to be usable in place of a local
#: diagnosis. Deliberately the load-bearing subset of the 13-key contract:
#: anything missing these would break `_persist_diagnosis` or the UI.
REQUIRED_KEYS: frozenset[str] = frozenset(
    {"flagged_step_index", "predicted_class", "step_scores"}
)


def normalize_provider(provider_type: str | None) -> str:
    """Map a request-supplied provider name onto a canonical id.

    Unknown values resolve to `"default"` rather than raising: an unrecognised
    header is a reason to use the engine that always works, not to reject an
    otherwise valid diagnosis request.
    """
    if not provider_type:
        return "default"
    return PROVIDER_ALIASES.get(provider_type.strip().lower(), "default")


def _run_local(trace_data: dict[str, Any]) -> dict[str, Any]:
    """Call the existing local Hybrid Sentry Engine, unchanged.

    Imported lazily and through `backend.main`'s accessor so the model is
    loaded exactly once per process and shared with the rest of the app,
    rather than this module building a second copy of a heavyweight artifact.
    """
    from backend.main import _get_localizer

    result = _get_localizer().diagnose(trace_data)
    result["engine_used"] = "default"
    return result


def _is_valid_remote_diagnosis(payload: Any) -> bool:
    """Whether a remote response can stand in for a local diagnosis."""
    if not isinstance(payload, dict):
        return False
    if not REQUIRED_KEYS.issubset(payload.keys()):
        return False
    scores = payload.get("step_scores")
    if not isinstance(scores, list) or not scores:
        return False
    return all(isinstance(value, (int, float)) for value in scores)


async def _run_remote(
    provider: str,
    endpoint: str,
    feature_vector: dict[str, Any] | None,
    trace_data: dict[str, Any],
) -> dict[str, Any] | None:
    """POST the trace to a configured remote engine.

    Returns the remote diagnosis, or None if it failed in any way. Never
    raises: every failure path here has a working local fallback.
    """
    body = {
        "provider": provider,
        "trace": trace_data,
        # Sent as a hint so a remote scorer does not have to recompute what we
        # already have. The local engine ignores it and extracts its own.
        "feature_vector": feature_vector or {},
    }

    try:
        async with httpx.AsyncClient(timeout=REMOTE_TIMEOUT_S) as client:
            response = await client.post(endpoint, json=body)
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException:
        logger.warning(
            "Engine %s timed out after %.1fs, falling back to local",
            provider,
            REMOTE_TIMEOUT_S,
        )
        return None
    except (httpx.HTTPError, OSError, ValueError) as exc:
        logger.warning("Engine %s failed (%s), falling back to local", provider, exc)
        return None

    if not _is_valid_remote_diagnosis(payload):
        logger.warning(
            "Engine %s returned a payload missing %s, falling back to local",
            provider,
            sorted(REQUIRED_KEYS - set(payload if isinstance(payload, dict) else {})),
        )
        return None

    payload["engine_used"] = provider
    return payload


def _judge_to_contract(
    parsed: dict[str, Any],
    trace_data: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Shape an LLM judge verdict into the 13-key diagnosis contract.

    An LLM returns a point estimate: one step index and one class name. It does
    not return a blame distribution, feature values or attributions, and this
    does not pretend otherwise:

    * `step_scores` is a spike, not a distribution. The named step takes
      everything left after giving every other step 1, so the heatmap shows a
      single spike rather than the graded falloff the local engine produces.
      That visual difference is honest: the judge genuinely has no opinion
      about the relative blame of the steps it did not pick.
    * `evidence` is empty and `shap` is None because no features were
      extracted. Inventing numbers to fill those panels would be fabricating
      evidence, which CLAUDE.md rules out.
    * `class_confidence` is 0.0 because an LLM gives no calibrated probability.
      A number read off the model's own prose would be a guess wearing a
      decimal point.
    """
    steps = trace_data.get("steps") or []
    n = max(len(steps), 1)
    flagged = max(0, min(n - 1, int(parsed.get("root_cause_step", 0))))

    scores = [1] * n
    scores[flagged] = max(1, 100 - (n - 1))

    total = sum(scores) or 1
    return {
        "run_id": trace_data.get("run_id", ""),
        "flagged_step_index": flagged,
        "confidence": round(scores[flagged] / total, 4),
        "predicted_class": parsed.get("failure_class") or "unknown",
        "evidence_path": f"step[{flagged}].output",
        "evidence": {},
        "shap": None,
        "step_scores": scores,
        "explanation": parsed.get("reason") or None,
        "suggested_fixes": [],
        "class_confidence": 0.0,
        "unknown_reason": None,
        "anomaly_signal": None,
        "engine_used": "groq",
        "engine_model": model,
    }


async def _run_groq(
    trace_data: dict[str, Any],
    api_key: str,
) -> dict[str, Any] | None:
    """Diagnose via Groq, using the existing LLM-as-judge prompt.

    Reuses `model/judge.py`'s prompt, taxonomy and trace rendering rather than
    writing a second one. That keeps this provider honest in a specific way:
    it is the same LLM-as-judge baseline the Model tab measures, just served by
    Groq instead of Gemini. Whatever that baseline scores is what this scores.

    Returns None on any failure, so the caller falls back to the local engine.
    """
    try:
        from model.judge import PROMPT, TAXONOMY, render_trace
    except Exception as exc:  # noqa: BLE001 - optional path, never fatal
        logger.warning("Groq requested but the judge prompt is unavailable: %s", exc)
        return None

    steps = trace_data.get("steps") or []
    if not steps:
        return None

    # render_trace indexes required step fields directly, so a trace from an
    # unusual source (a partial OTel ingest, say) can raise here. That must
    # fall back like any other provider failure, not 500 the endpoint.
    try:
        goal, rendered = render_trace(trace_data)
        prompt = PROMPT.format(
            taxonomy=TAXONOMY, goal=goal, n=len(steps), trace=rendered
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Groq could not render this trace (%s), using local engine", exc)
        return None
    model = os.environ.get("GROQ_MODEL", "").strip() or GROQ_DEFAULT_MODEL

    body = {
        "model": model,
        "temperature": 0.0,
        # Groq implements OpenAI's JSON mode, which requires the word "json"
        # to appear in the conversation. The schema is restated in the system
        # turn so the reply parses without a repair pass.
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You localize failures in AI agent traces. Reply with a single "
                    "json object and nothing else, with keys: root_cause_step "
                    "(integer), failure_class (string), reason (string)."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=GROQ_TIMEOUT_S) as client:
            response = await client.post(
                GROQ_URL,
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            parsed = json.loads(content)
    except httpx.TimeoutException:
        logger.warning(
            "Groq timed out after %.1fs, falling back to local", GROQ_TIMEOUT_S
        )
        return None
    except (httpx.HTTPError, OSError, ValueError, KeyError, IndexError) as exc:
        logger.warning("Groq call failed (%s), falling back to local", exc)
        return None

    if not isinstance(parsed, dict) or "root_cause_step" not in parsed:
        logger.warning("Groq returned no root_cause_step, falling back to local")
        return None

    try:
        return _judge_to_contract(parsed, trace_data, model)
    except (TypeError, ValueError) as exc:
        logger.warning("Groq verdict was unusable (%s), falling back to local", exc)
        return None


async def run_selected_diagnosis(
    provider_type: str,
    feature_vector: dict[str, Any] | None,
    trace_data: dict[str, Any],
    api_key: str | None = None,
) -> dict[str, Any]:
    """Diagnose `trace_data` using the named provider, falling back to local.

    Always returns a usable diagnosis dict. The returned `engine_used` key
    names the engine that actually produced the result, which is not always
    the one that was requested: a remote that is unconfigured, slow, broken or
    malformed resolves to `"default"`. Callers that care about the difference
    should read `engine_used` rather than assuming `provider_type` was honoured.

    `feature_vector` is forwarded to remote providers as a hint and ignored by
    the local engine, which extracts its own features from the trace.
    """
    provider = normalize_provider(provider_type)

    if provider == "default":
        return _run_local(trace_data)

    if provider == "groq":
        # Server-side config wins. A key held in the server's environment is
        # the safe arrangement; a caller-supplied one is the bring-your-own-key
        # convenience, accepted but never stored and never logged.
        key = os.environ.get("GROQ_API_KEY", "").strip() or (api_key or "").strip()
        if not key:
            logger.info("Groq requested but no API key is available, using local engine")
            return _run_local(trace_data)
        groq = await _run_groq(trace_data, key)
        return groq if groq is not None else _run_local(trace_data)

    env_var = PROVIDER_ENV_VAR.get(provider)
    endpoint = os.environ.get(env_var, "").strip() if env_var else ""
    if not endpoint:
        logger.info(
            "Engine %s requested but %s is unset, using local engine",
            provider,
            env_var,
        )
        return _run_local(trace_data)

    remote = await _run_remote(provider, endpoint, feature_vector, trace_data)
    if remote is not None:
        return remote

    return _run_local(trace_data)


def run_selected_diagnosis_sync(
    provider_type: str,
    feature_vector: dict[str, Any] | None,
    trace_data: dict[str, Any],
    api_key: str | None = None,
) -> dict[str, Any]:
    """Blocking bridge to `run_selected_diagnosis`.

    `POST /runs/{id}/diagnose` is a sync endpoint, because it does blocking
    SQLModel work that would stall the event loop if it were async. FastAPI
    therefore runs it in a worker thread, where no event loop is running and
    `asyncio.run` is safe.

    The local-only path is kept entirely out of asyncio: when no remote is
    involved there is nothing to await, and spinning up an event loop to call
    a synchronous model would be pure overhead on the hot path.
    """
    provider = normalize_provider(provider_type)
    if provider == "default":
        return _run_local(trace_data)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_selected_diagnosis(provider, feature_vector, trace_data, api_key)
        )

    # Called from inside a running loop, where asyncio.run would raise. The
    # caller should await run_selected_diagnosis directly; falling back to the
    # local engine keeps the request working instead of failing on a misuse.
    logger.warning(
        "run_selected_diagnosis_sync called from a running event loop; "
        "using the local engine. Await run_selected_diagnosis instead."
    )
    return _run_local(trace_data)
