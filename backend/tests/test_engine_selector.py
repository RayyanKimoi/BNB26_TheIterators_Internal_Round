"""Tests for the diagnosis engine selector and its fallback behaviour.

The point of this module is that diagnosis never fails because an optional
remote engine is unavailable, so most of these assert a fallback: unconfigured,
timed out, HTTP error, and malformed response all have to land on the local
engine with `engine_used == "default"`.

No network and no real model. `_run_local` is monkeypatched to a sentinel so a
test can tell local from remote by looking at the result, and httpx is stubbed
per test to simulate each failure mode.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from backend import engine_selector


LOCAL_SENTINEL = {
    "run_id": "local-run",
    "flagged_step_index": 3,
    "predicted_class": "stale_retrieval",
    "step_scores": [10, 20, 30, 40],
    "engine_used": "default",
}

REMOTE_OK = {
    "run_id": "remote-run",
    "flagged_step_index": 7,
    "predicted_class": "schema_violation",
    "step_scores": [5, 95],
}

TRACE = {"run_id": "r1", "steps": [{"step_index": 0}]}


@pytest.fixture
def local(monkeypatch):
    """Replace the real localizer call with a cheap sentinel."""
    monkeypatch.setattr(
        engine_selector, "_run_local", lambda trace_data: dict(LOCAL_SENTINEL)
    )
    return LOCAL_SENTINEL


def _run(provider, feature_vector=None, trace=None):
    return asyncio.run(
        engine_selector.run_selected_diagnosis(
            provider, feature_vector, trace if trace is not None else TRACE
        )
    )


class _StubClient:
    """Minimal async context manager standing in for httpx.AsyncClient."""

    seen_headers: dict = {}

    def __init__(self, handler):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json, headers=None):  # noqa: A002 - httpx's signature
        self.last_headers = headers or {}
        _StubClient.seen_headers = self.last_headers
        return self._handler(url, json)


def _install(monkeypatch, handler):
    monkeypatch.setattr(
        engine_selector.httpx,
        "AsyncClient",
        lambda **kwargs: _StubClient(handler),
    )


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._payload


# -- provider normalization -------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "default"),
        ("", "default"),
        ("default", "default"),
        ("DEFAULT", "default"),
        ("  local  ", "default"),
        ("nonsense", "default"),
        ("custom_llm", "custom_llm"),
        # The frontend selector's own ids must resolve, not silently degrade.
        ("byo_llm", "custom_llm"),
        ("webhook", "webhook"),
        ("enterprise_webhook", "webhook"),
        ("air_gapped", "air_gapped"),
    ],
)
def test_normalize_provider(raw, expected):
    assert engine_selector.normalize_provider(raw) == expected


# -- routing ----------------------------------------------------------------


def test_default_uses_local(local):
    result = _run("default")
    assert result["engine_used"] == "default"
    assert result["flagged_step_index"] == 3


def test_unknown_provider_uses_local(local):
    assert _run("not-a-provider")["engine_used"] == "default"


def test_unconfigured_remote_falls_back(local, monkeypatch):
    monkeypatch.delenv("ENGINE_WEBHOOK_URL", raising=False)
    result = _run("webhook")
    assert result["engine_used"] == "default"


def test_blank_env_var_counts_as_unconfigured(local, monkeypatch):
    monkeypatch.setenv("ENGINE_WEBHOOK_URL", "   ")
    assert _run("webhook")["engine_used"] == "default"


def test_configured_remote_success(local, monkeypatch):
    monkeypatch.setenv("ENGINE_WEBHOOK_URL", "https://scoring.internal/diagnose")
    seen = {}

    def handler(url, json):
        seen["url"] = url
        seen["body"] = json
        return _Response(dict(REMOTE_OK))

    _install(monkeypatch, handler)
    result = _run("webhook", feature_vector={"duration_z": 2.1})

    assert result["engine_used"] == "webhook"
    assert result["flagged_step_index"] == 7
    assert seen["url"] == "https://scoring.internal/diagnose"
    # The feature vector is forwarded as a hint so a remote need not recompute.
    assert seen["body"]["feature_vector"] == {"duration_z": 2.1}
    assert seen["body"]["trace"] == TRACE


def test_alias_reaches_the_canonical_env_var(local, monkeypatch):
    """byo_llm is the frontend's name for custom_llm, so it must read that var."""
    monkeypatch.setenv("ENGINE_CUSTOM_LLM_URL", "https://llm.internal/diagnose")
    _install(monkeypatch, lambda url, json: _Response(dict(REMOTE_OK)))
    assert _run("byo_llm")["engine_used"] == "custom_llm"


# -- fallback paths ---------------------------------------------------------


def test_timeout_falls_back(local, monkeypatch):
    monkeypatch.setenv("ENGINE_WEBHOOK_URL", "https://slow.internal/diagnose")

    def handler(url, json):
        raise httpx.TimeoutException("too slow")

    _install(monkeypatch, handler)
    result = _run("webhook")
    assert result["engine_used"] == "default"
    assert result["flagged_step_index"] == 3


def test_http_error_falls_back(local, monkeypatch):
    monkeypatch.setenv("ENGINE_WEBHOOK_URL", "https://bad.internal/diagnose")
    _install(monkeypatch, lambda url, json: _Response({}, status=500))
    assert _run("webhook")["engine_used"] == "default"


def test_connection_error_falls_back(local, monkeypatch):
    monkeypatch.setenv("ENGINE_WEBHOOK_URL", "https://down.internal/diagnose")

    def handler(url, json):
        raise httpx.ConnectError("refused")

    _install(monkeypatch, handler)
    assert _run("webhook")["engine_used"] == "default"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"flagged_step_index": 1},
        {"flagged_step_index": 1, "predicted_class": "x"},
        # step_scores present but unusable
        {"flagged_step_index": 1, "predicted_class": "x", "step_scores": []},
        {"flagged_step_index": 1, "predicted_class": "x", "step_scores": "nope"},
        {"flagged_step_index": 1, "predicted_class": "x", "step_scores": [1, "two"]},
        "not a dict",
        None,
    ],
)
def test_malformed_remote_response_falls_back(local, monkeypatch, payload):
    """A remote that answers with garbage must not poison the database."""
    monkeypatch.setenv("ENGINE_WEBHOOK_URL", "https://weird.internal/diagnose")
    _install(monkeypatch, lambda url, json: _Response(payload))
    assert _run("webhook")["engine_used"] == "default"


def test_invalid_json_falls_back(local, monkeypatch):
    monkeypatch.setenv("ENGINE_WEBHOOK_URL", "https://weird.internal/diagnose")

    class _BadJson:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError("not json")

    _install(monkeypatch, lambda url, json: _BadJson())
    assert _run("webhook")["engine_used"] == "default"


# -- sync bridge ------------------------------------------------------------


def test_sync_bridge_default_skips_asyncio(local):
    result = engine_selector.run_selected_diagnosis_sync("default", None, TRACE)
    assert result["engine_used"] == "default"


def test_sync_bridge_runs_remote(local, monkeypatch):
    monkeypatch.setenv("ENGINE_WEBHOOK_URL", "https://scoring.internal/diagnose")
    _install(monkeypatch, lambda url, json: _Response(dict(REMOTE_OK)))
    result = engine_selector.run_selected_diagnosis_sync("webhook", None, TRACE)
    assert result["engine_used"] == "webhook"


def test_sync_bridge_inside_running_loop_falls_back(local, monkeypatch):
    """Misuse from an async context degrades to local instead of raising."""
    monkeypatch.setenv("ENGINE_WEBHOOK_URL", "https://scoring.internal/diagnose")
    _install(monkeypatch, lambda url, json: _Response(dict(REMOTE_OK)))

    async def inner():
        return engine_selector.run_selected_diagnosis_sync("webhook", None, TRACE)

    assert asyncio.run(inner())["engine_used"] == "default"


# -- the SSRF guard ---------------------------------------------------------


def test_request_cannot_supply_an_endpoint(local, monkeypatch):
    """A provider name is not a URL.

    The whole point of resolving endpoints from the environment is that a
    caller cannot make the server POST an internal trace somewhere of their
    choosing. A URL passed as the provider must resolve to local, not be
    dialled.
    """
    monkeypatch.delenv("ENGINE_WEBHOOK_URL", raising=False)

    def explode(url, json):
        raise AssertionError(f"server dialled a caller-supplied URL: {url}")

    _install(monkeypatch, explode)
    result = _run("https://attacker.example/collect")
    assert result["engine_used"] == "default"


# -- groq -------------------------------------------------------------------

GROQ_OK = {
    "choices": [
        {
            "message": {
                "content": '{"root_cause_step": 2, "failure_class": "stale_retrieval", '
                '"reason": "step 2 served a cached quote"}'
            }
        }
    ]
}

GROQ_TRACE = {
    "run_id": "g1",
    "steps": [
        {
            "step_index": i,
            "action_type": "tool_call",
            "tool_name": "search_flights",
            "duration_ms": 120,
            "tokens": 40,
            "error_flag": False,
            "state_snapshot": {"goal": "book a trip"},
            "output": {"summary": f"result {i}"},
        }
        for i in range(5)
    ],
}


def test_groq_without_a_key_falls_back(local, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    result = _run("groq", trace=GROQ_TRACE)
    assert result["engine_used"] == "default"


def test_groq_uses_server_key(local, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "server-side-key")
    seen = {}

    def handler(url, json):
        seen["url"] = url
        return _Response(GROQ_OK)

    _install(monkeypatch, handler)
    result = _run("groq", trace=GROQ_TRACE)

    assert result["engine_used"] == "groq"
    assert seen["url"] == engine_selector.GROQ_URL
    assert result["flagged_step_index"] == 2
    assert result["predicted_class"] == "stale_retrieval"


def test_groq_accepts_a_caller_supplied_key(local, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    _install(monkeypatch, lambda url, json: _Response(GROQ_OK))
    result = asyncio.run(
        engine_selector.run_selected_diagnosis("groq", None, GROQ_TRACE, "byo-key")
    )
    assert result["engine_used"] == "groq"


def test_server_key_wins_over_caller_key(local, monkeypatch):
    """A server-configured key must not be overridable by a request header."""
    monkeypatch.setenv("GROQ_API_KEY", "server-side-key")
    _install(monkeypatch, lambda url, json: _Response(GROQ_OK))
    asyncio.run(
        engine_selector.run_selected_diagnosis("groq", None, GROQ_TRACE, "caller-key")
    )
    assert _StubClient.seen_headers["Authorization"] == "Bearer server-side-key"


def test_groq_contract_shape_is_honest(local, monkeypatch):
    """No invented evidence, and step_scores is a spike that still sums to 100."""
    monkeypatch.setenv("GROQ_API_KEY", "k")
    _install(monkeypatch, lambda url, json: _Response(GROQ_OK))
    result = _run("groq", trace=GROQ_TRACE)

    assert result["evidence"] == {}
    assert result["shap"] is None
    assert result["class_confidence"] == 0.0
    assert result["unknown_reason"] is None
    assert sum(result["step_scores"]) == 100
    assert result["step_scores"][2] == max(result["step_scores"])
    assert result["explanation"] == "step 2 served a cached quote"


def test_groq_out_of_range_step_is_clamped(local, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    payload = {
        "choices": [
            {"message": {"content": '{"root_cause_step": 99, "failure_class": "x", "reason": "r"}'}}
        ]
    }
    _install(monkeypatch, lambda url, json: _Response(payload))
    result = _run("groq", trace=GROQ_TRACE)
    assert result["flagged_step_index"] == 4  # last valid index of 5 steps


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"content": "not json"}}]},
        {"choices": [{"message": {"content": '{"no_step": 1}'}}]},
        {},
    ],
)
def test_groq_malformed_response_falls_back(local, monkeypatch, payload):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    _install(monkeypatch, lambda url, json: _Response(payload))
    assert _run("groq", trace=GROQ_TRACE)["engine_used"] == "default"


def test_groq_timeout_falls_back(local, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")

    def handler(url, json):
        raise httpx.TimeoutException("slow")

    _install(monkeypatch, handler)
    assert _run("groq", trace=GROQ_TRACE)["engine_used"] == "default"


def test_groq_empty_trace_falls_back(local, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    _install(monkeypatch, lambda url, json: _Response(GROQ_OK))
    assert _run("groq", trace={"run_id": "x", "steps": []})["engine_used"] == "default"


def test_groq_model_is_overridable(local, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    seen = {}

    def handler(url, json):
        seen["model"] = json["model"]
        return _Response(GROQ_OK)

    _install(monkeypatch, handler)
    result = _run("groq", trace=GROQ_TRACE)
    assert seen["model"] == "llama-3.1-8b-instant"
    assert result["engine_model"] == "llama-3.1-8b-instant"
