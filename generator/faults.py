"""The seven fault classes and the harmless anomalies.

Taxonomy and signatures come from the Failure Taxonomy section of PRD.md. Each
injector produces the observable signature for its class and nothing else: no
marker field is written into any step, so the only way to recover the label
from a trace is to learn the signature. Ground truth lives in `agent_runs`
(`injected_class`, `true_failure_step`) and in the sidecar manifest, never in
the step payloads.
"""

from __future__ import annotations

import random
from typing import Any

from .tasks import ToolSpec

# Harmless anomalies for successful runs. Without these the model learns
# "any anomaly equals failure" instead of what separates a survivable anomaly
# from a fatal one (PRD, Failure Taxonomy).
ANOMALIES: tuple[str, ...] = (
    "slow_tool",
    "transient_retry",
    "verbose_step",
    "benign_revisit",
    "low_confidence_decide",
)

# Which step index each class blames, expressed as the step kind the fault
# lives on. Used by build.py and documented in the README.
TRUTH_ANCHOR: dict[str, str] = {
    "wrong_tool_chosen": "decide",
    "hallucinated_argument": "call_tool",
    "stale_retrieval": "call_tool",
    "premature_termination": "decide",
    "infinite_loop": "decide",
    "schema_violation": "call_tool",
    "context_truncation": "call_llm",
}


def break_schema(
    result: dict[str, Any], spec: ToolSpec, rng: random.Random
) -> tuple[dict[str, Any], str]:
    """Return output that violates the tool's declared schema.

    Signature: parse-failure flag, type mismatch.
    """
    out = dict(result)
    candidates = [f for f in spec.output_schema if f in out]
    if not candidates:
        candidates = list(spec.output_schema)
    field = candidates[rng.randrange(len(candidates))]
    declared = spec.output_schema[field]

    if declared == "float":
        out[field] = f"{rng.randint(1, 4)} {rng.randint(100, 999)},{rng.randint(10, 99)} EUR"
    elif declared == "int":
        out[field] = None
    elif declared == "bool":
        out[field] = "yes"
    elif declared == "list":
        out[field] = {"error": "unexpected object where array was declared"}
    else:
        out[field] = ["unexpected", "array"]

    out["summary"] = (
        f"Tool {spec.name} returned a malformed payload: field {field} was not "
        f"the declared {declared}."
    )
    return out, field


def stale_payload(
    spec: ToolSpec,
    args: dict[str, Any],
    state: dict[str, Any],
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Serve a cached result belonging to an earlier, different query.

    Signature: timestamp gap, low similarity to the current query. Returns the
    stale result, the arguments it was actually computed for, and its age.
    """
    stale_args = dict(args)
    for key, replacement in (
        ("destination", "MAD"),
        ("origin", "BCN"),
        ("from_currency", "GBP"),
        ("to_currency", "USD"),
        ("period", "2025-11"),
        ("vendor", "Contoso Metals"),
        ("query", "invoice export formatting"),
        ("customer_id", f"CUST-{rng.randint(5000, 5999)}"),
        ("vendor_id", f"VEND-{rng.randint(700, 799)}"),
        ("flight_no", f"IB{rng.randint(100, 999)}"),
        ("ticket_id", 3190),
        ("feature", "api_webhooks"),
    ):
        if key in stale_args:
            stale_args[key] = replacement

    result = spec.result_fn(stale_args, state, rng)
    age_s = rng.randint(38_000, 410_000)
    result["quoted_at"] = f"2025-1{rng.randint(0, 2)}-{rng.randint(10, 28):02d}T0{rng.randint(1, 9)}:12:00Z"
    changed = {k: v for k, v in stale_args.items() if args.get(k) != v}
    result["summary"] = (
        f"Cache hit. {result.get('summary', '')} This result was computed for "
        f"{changed} roughly {age_s // 3600} hours ago, not for the current query."
    )
    return result, stale_args, age_s


# Argument fields worth fabricating, per the identifier shape the task uses.
_FABRICATIONS: dict[str, Any] = {
    "flight_no": "QX7741",
    "invoice_id": "INV-99812",
    "vendor_id": "VEND-8842",
    "customer_id": "CUST-9940",
    "ticket_id": 99317,
    "amount": 8431.77,
    "total_eur": 8431.77,
    "invoice_total": 8431.77,
    "payment_total": 1204.03,
    "delta": 7227.74,
    "feature": "bulk_sso_provisioning",
    "query": "quota exceeded on nightly rollup",
    "passenger": "R. Delacroix",
}


def hallucinate_arg(
    args: dict[str, Any], rng: random.Random
) -> tuple[dict[str, Any], str, Any]:
    """Insert a field value absent from all prior context.

    Signature: argument unseen in any upstream step output.
    """
    out = dict(args)
    options = [k for k in out if k in _FABRICATIONS]
    if not options:
        field = next(iter(out)) if out else "reference"
        value = f"REF-{rng.randint(90000, 99999)}"
    else:
        field = options[rng.randrange(len(options))]
        value = _FABRICATIONS[field]
    out[field] = value
    return out, field, value


def truncated_reflection(rng: random.Random, dropped: int) -> str:
    """Text produced after earlier messages fell out of the prompt window.

    Signature: token count drop, reference to absent content.
    """
    return (
        "Continuing from the earlier result. I no longer have the specific "
        "values from the previous steps in context, so I will proceed with the "
        "figure referenced above."
    )


def off_goal_reflection(tool_name: str) -> str:
    return (
        f"The {tool_name} result does not contain the field the next step needs. "
        f"Proceeding with an assumed value."
    )


def failure_summary(class_name: str, task_goal: str) -> str:
    reasons = {
        "wrong_tool_chosen": "the wrong tool was queried, so the required fact was never retrieved",
        "hallucinated_argument": "an identifier that never appeared upstream was passed to a tool",
        "stale_retrieval": "a cached result from a different query was used as if it were current",
        "premature_termination": "the run stopped before the goal condition was met",
        "infinite_loop": "the agent re-entered the same state until the step budget was exhausted",
        "schema_violation": "a tool returned a payload that did not match its declared schema",
        "context_truncation": "earlier messages left the prompt window and the needed values were lost",
    }
    return f"Run did not complete the goal. Cause: {reasons[class_name]}."
