"""Guards on the generated corpus.

The expensive failure mode for this project is a malformed or leaky corpus
discovered after the model is trained. These tests assert three things: traces
match the frozen schema, each class actually exhibits the signature the
Failure Taxonomy claims for it, and no step payload carries the label.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone

import pytest

from generator.build import build_run
from generator.faults import ANOMALIES, TRUTH_ANCHOR
from generator.generate import generate
from generator.schema import FAILURE_CLASSES, validate
from generator.tasks import TASK_TYPES, TASKS_BY_NAME

USER = "00000000-0000-4000-8000-000000000001"
WHEN = datetime(2026, 3, 10, 8, 0, tzinfo=timezone.utc)


def make(fault=None, anomaly=None, task=None, seed=0):
    rng = random.Random(seed)
    return build_run(
        task or TASK_TYPES[seed % len(TASK_TYPES)],
        rng,
        user_id=USER,
        created_at=WHEN,
        fault=fault,
        anomaly=anomaly,
    )


def runs_of(fault: str, n: int = 24):
    """Several runs of one class, spread across all three task types."""
    return [make(fault=fault, task=TASK_TYPES[i % 3], seed=i) for i in range(n)]


# -- schema ---------------------------------------------------------------


def test_corpus_validates_and_is_json_serializable():
    traces, metas = generate(60, seed=11)
    assert len(traces) == 60
    for trace in traces:
        validate(trace)
        json.dumps(trace.to_dict())  # must survive the JSONL round trip
    assert len(metas) == 60


def test_step_indices_are_dense_and_ordered():
    for trace, _ in [make(fault=c, seed=i) for i, c in enumerate(FAILURE_CLASSES)]:
        assert [s.step_index for s in trace.steps] == list(range(len(trace.steps)))


def test_successful_runs_carry_no_ground_truth():
    for anomaly in (*ANOMALIES, None):
        trace, meta = make(anomaly=anomaly, seed=3)
        assert trace.run.status == "success"
        assert trace.run.injected_class is None
        assert trace.run.true_failure_step is None
        assert meta["anomaly"] == anomaly


def test_generation_is_deterministic():
    a, _ = generate(30, seed=5)
    b, _ = generate(30, seed=5)
    strip = lambda ts: [
        {**t.to_dict(), "run": {**t.run.to_dict(), "id": None},
         "steps": [{**s, "id": None, "run_id": None} for s in (x.to_dict() for x in t.steps)]}
        for t in ts
    ]
    assert strip(a) == strip(b)


# -- ground truth ---------------------------------------------------------


@pytest.mark.parametrize("fault", FAILURE_CLASSES)
def test_ground_truth_is_in_range_and_on_the_right_step_kind(fault):
    for trace, meta in runs_of(fault):
        run = trace.run
        assert run.status == "failed"
        assert run.injected_class == fault
        idx = run.true_failure_step
        assert idx is not None and 0 <= idx < len(trace.steps)
        assert trace.steps[idx].action_type == TRUTH_ANCHOR[fault]
        assert meta["evidence_path"], "every fault must name the field it broke"


@pytest.mark.parametrize("fault", FAILURE_CLASSES)
def test_no_injection_marker_leaks_into_step_payloads(fault):
    """The label must be recoverable only from the signature, never from a flag."""
    forbidden = ("injected", "injected_class", "fault", "ground_truth", "is_bug", "true_failure_step")
    for trace, _ in runs_of(fault, n=9):
        blob = json.dumps([s.to_dict() for s in trace.steps]).lower()
        for marker in forbidden:
            assert marker not in blob, f"{fault}: step payloads mention {marker!r}"


# -- per-class signatures from the Failure Taxonomy -----------------------


def test_wrong_tool_chosen_picks_a_distractor_with_an_entropy_spike():
    for trace, _ in runs_of("wrong_tool_chosen"):
        task = TASKS_BY_NAME[trace.run.task_type]
        step = trace.steps[trace.run.true_failure_step]
        chosen = step.output["chosen_tool"]
        assert chosen in task.distractors, "the fault must select a plausible wrong tool"
        assert chosen not in task.tools
        # Near-uniform candidate mass is the entropy spike the taxonomy names.
        assert max(step.output["candidates"].values()) < 0.6


def test_hallucinated_argument_value_is_absent_from_all_upstream_output():
    for trace, meta in runs_of("hallucinated_argument"):
        idx = trace.run.true_failure_step
        field = meta["evidence_path"].rsplit(".", 1)[-1]
        value = trace.steps[idx].input[field]
        upstream = json.dumps([s.output for s in trace.steps[:idx]])
        assert str(value) not in upstream, f"{field}={value} was visible upstream"


def test_stale_retrieval_has_a_cache_hit_and_a_timestamp_gap():
    for trace, _ in runs_of("stale_retrieval"):
        meta = trace.steps[trace.run.true_failure_step].output["_meta"]
        assert meta["cache_hit"] is True
        assert meta["cache_age_s"] > 36_000, "a stale result must be measurably old"


def test_premature_termination_is_short_and_never_calls_the_terminal_tool():
    lengths, full_lengths = [], []
    for trace, _ in runs_of("premature_termination"):
        task = TASKS_BY_NAME[trace.run.task_type]
        called = {s.tool_name for s in trace.steps if s.tool_name}
        assert task.terminal_tool not in called
        assert trace.steps[trace.run.true_failure_step].output["chosen_tool"] == "finish"
        lengths.append(len(trace.steps))
    for trace, _ in [make(anomaly=None, task=TASK_TYPES[i % 3], seed=i) for i in range(24)]:
        full_lengths.append(len(trace.steps))
    # "Step count below class median" is the taxonomy's stated signature.
    assert sorted(lengths)[len(lengths) // 2] < sorted(full_lengths)[len(full_lengths) // 2]


def test_infinite_loop_repeats_a_state_hash_with_a_climbing_retry_count():
    for trace, _ in runs_of("infinite_loop"):
        hashes = [s.state_hash for s in trace.steps]
        top = max(hashes.count(h) for h in set(hashes))
        assert top >= 8, "the loop must revisit the same state repeatedly"
        retries = [
            s.output["_meta"]["retry_count"]
            for s in trace.steps
            if s.action_type == "call_tool" and s.output["_meta"]["retry_count"]
        ]
        assert retries == sorted(retries) and max(retries) >= 4


def test_schema_violation_sets_parse_failure_and_breaks_a_declared_type():
    for trace, meta in runs_of("schema_violation"):
        step = trace.steps[trace.run.true_failure_step]
        assert step.output["_meta"]["parse_failure"] is True
        task = TASKS_BY_NAME[trace.run.task_type]
        spec = task.tool(step.tool_name)
        field = meta["evidence_path"].rsplit(".", 1)[-1]
        declared = spec.output_schema[field]
        actual = step.output[field]
        kinds = {"float": float, "int": int, "bool": bool, "str": str, "list": list}
        assert not isinstance(actual, kinds[declared]) or actual is None


def test_context_truncation_drops_tokens_sharply_against_the_run_baseline():
    for trace, _ in runs_of("context_truncation"):
        step = trace.steps[trace.run.true_failure_step]
        assert step.action_type == "call_llm"
        assert step.output["_meta"]["dropped_messages"] >= 2
        others = [
            s.tokens
            for s in trace.steps
            if s.action_type == "call_llm" and s.step_index != step.step_index
        ]
        assert step.tokens < 0.5 * (sum(others) / len(others))


# -- corpus shape ---------------------------------------------------------


def test_corpus_proportions_and_class_coverage():
    traces, _ = generate(240, seed=7)
    failed = [t for t in traces if t.run.status == "failed"]
    assert 0.55 <= len(failed) / len(traces) <= 0.65

    per_class = {c: sum(1 for t in failed if t.run.injected_class == c) for c in FAILURE_CLASSES}
    assert all(v >= 15 for v in per_class.values()), per_class
    assert max(per_class.values()) - min(per_class.values()) <= 2

    assert len({t.run.task_type for t in traces}) == 3
    # Successful runs must include error-carrying ones, or error_flag alone
    # would separate success from failure and the model learns nothing.
    assert any(
        t.run.status == "success" and any(s.error_flag for s in t.steps) for t in traces
    )


def test_naive_baselines_are_neither_perfect_nor_structurally_zero():
    """Both PRD baselines must score a real number on this corpus."""
    traces, _ = generate(240, seed=7)
    failed = [t for t in traces if t.run.status == "failed"]

    last = sum(1 for t in failed if t.run.true_failure_step == len(t.steps) - 1) / len(failed)
    first_err = 0
    for t in failed:
        errs = [s.step_index for s in t.steps if s.error_flag]
        if errs and errs[0] == t.run.true_failure_step:
            first_err += 1
    first_err /= len(failed)

    assert 0.01 < last < 0.5, f"last-step baseline at {last:.0%} looks rigged"
    assert 0.01 < first_err < 0.5, f"first-error baseline at {first_err:.0%} looks rigged"


# -- the corpus must not narrate its own faults ---------------------------

# Phrases that state a diagnosis rather than report an observation. An earlier
# version of the generator wrote these, which let an LLM-as-judge score 95
# percent by reading the answer off the page and leaked the label into the
# semantic_deviation feature. See generator/README.md.
DIAGNOSTIC_PHRASES = (
    "does not appear in any earlier",
    "malformed",
    "not for the current query",
    "cache hit",
    "i no longer have",
    "the same payload as the previous",
    "state is unchanged",
    "state has not advanced",
    "look sufficient",
    "may provide what the goal needs",
    "does not contain the field",
    "assumed value",
    "cause:",
    "was not available from earlier steps",
    "re-checked",
    "will be retried",
    "returned what the next step needs",
)


@pytest.mark.parametrize("fault", FAILURE_CLASSES)
def test_summaries_do_not_narrate_the_fault(fault):
    """Prose reports what a step returned, never whether it was wrong."""
    for trace, _ in runs_of(fault, n=9):
        for step in trace.steps:
            text = json.dumps(step.to_dict()).lower()
            for phrase in DIAGNOSTIC_PHRASES:
                assert phrase not in text, (
                    f"{fault}: step[{step.step_index}] narrates the diagnosis "
                    f"with {phrase!r}"
                )


def test_root_cause_step_is_not_textually_distinctive():
    """The root-cause summary must not stand out from its own run by wording.

    A cheap proxy for the leak that mattered: if the root-cause step is the
    only step mentioning a diagnostic word, any reader finds it for free.
    """
    for fault in FAILURE_CLASSES:
        for trace, _ in runs_of(fault, n=6):
            root = trace.steps[trace.run.true_failure_step]
            text = str(root.output.get("summary", "")).lower()
            for phrase in DIAGNOSTIC_PHRASES:
                assert phrase not in text
