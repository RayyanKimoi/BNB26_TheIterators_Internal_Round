"""Guards on the feature contract.

The column list, its order and its dtypes are consumed by training, the
diagnosis API and the inspector panel, so they are treated as frozen. Most
tests run against a deterministic stub encoder; one exercises the real
all-MiniLM-L6-v2 model.
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from generator.build import build_run
from generator.schema import FAILURE_CLASSES
from generator.tasks import TASK_TYPES
from model.features import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    CorpusStats,
    Embedder,
    FeatureExtractor,
    normalized_entropy,
    step_text,
)

WHEN = datetime(2026, 3, 10, 8, 0, tzinfo=timezone.utc)
USER = "00000000-0000-4000-8000-000000000001"


class StubEmbedder(Embedder):
    """Deterministic hash-based unit vectors. No torch, no download."""

    def encode(self, texts):  # type: ignore[override]
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) - 127.5
            out.append(vec / np.linalg.norm(vec))
        return np.vstack(out) if out else np.zeros((0, 32), dtype=np.float32)


def make(fault=None, anomaly=None, seed=0):
    rng = random.Random(seed)
    trace, meta = build_run(
        TASK_TYPES[seed % len(TASK_TYPES)], rng,
        user_id=USER, created_at=WHEN, fault=fault, anomaly=anomaly,
    )
    return trace.to_dict(), meta


@pytest.fixture(scope="module")
def corpus():
    runs = []
    for i, cls in enumerate(FAILURE_CLASSES * 3):
        runs.append(make(fault=cls, seed=i)[0])
    for i in range(12):
        runs.append(make(seed=100 + i)[0])
    return runs


@pytest.fixture(scope="module")
def extractor(corpus):
    return FeatureExtractor(CorpusStats.fit(corpus), StubEmbedder())


# -- the column contract --------------------------------------------------


def test_returns_exactly_the_ten_prd_columns_in_order(extractor, corpus):
    frame = extractor.transform(corpus[0])
    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 10


def test_one_row_per_step(extractor, corpus):
    for run in corpus[:10]:
        assert len(extractor.transform(run)) == len(run["steps"])


def test_dtypes_match_the_prd_types(extractor, corpus):
    frame = extractor.transform(corpus[0])
    assert frame["parse_failure"].dtype == bool
    for col in ("retry_count", "state_hash_repeat", "downstream_error_count"):
        assert pd.api.types.is_integer_dtype(frame[col]), col
    for col in ("duration_z", "token_z", "tool_choice_entropy",
                "semantic_deviation", "arg_novelty", "position_ratio"):
        assert pd.api.types.is_float_dtype(frame[col]), col


def test_feature_columns_carry_no_identifier_or_label(extractor, corpus):
    frame = extractor.transform(corpus[0])
    assert not set(frame.columns) & set(ID_COLUMNS)
    for banned in ("injected", "true_failure", "label", "target", "root_cause"):
        assert not any(banned in c for c in frame.columns)


def test_extraction_is_deterministic(extractor, corpus):
    a = extractor.transform(corpus[3])
    b = extractor.transform(corpus[3])
    pd.testing.assert_frame_equal(a, b)


def test_empty_run_yields_an_empty_frame_with_the_right_columns(extractor):
    frame = extractor.transform({"run": {}, "steps": []})
    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert frame.empty


def test_accepts_a_bare_list_of_steps(extractor, corpus):
    run = corpus[0]
    assert len(extractor.transform(run["steps"])) == len(run["steps"])


# -- individual feature definitions ---------------------------------------


def test_position_ratio_is_index_over_total(extractor, corpus):
    run = corpus[0]
    frame = extractor.transform(run)
    n = len(run["steps"])
    assert frame["position_ratio"].tolist() == [i / n for i in range(n)]


def test_downstream_error_count_counts_only_steps_after(extractor, corpus):
    for run in corpus[:12]:
        frame = extractor.transform(run)
        flags = [bool(s["error_flag"]) for s in run["steps"]]
        expected = [sum(flags[i + 1:]) for i in range(len(flags))]
        assert frame["downstream_error_count"].tolist() == expected


def test_state_hash_repeat_counts_the_whole_run_not_just_prior_steps(extractor, corpus):
    """The loop-entry step must see the repeats, or infinite_loop is unlearnable."""
    for run in corpus[:12]:
        frame = extractor.transform(run)
        hashes = [s["state_hash"] for s in run["steps"]]
        assert frame["state_hash_repeat"].tolist() == [hashes.count(h) - 1 for h in hashes]


def test_undefined_features_are_nan_not_zero(extractor, corpus):
    """Zero entropy means certainty. 'No decision here' must read as missing."""
    for run in corpus[:12]:
        frame = extractor.transform(run)
        for i, step in enumerate(run["steps"]):
            if step["action_type"] != "decide":
                assert np.isnan(frame["tool_choice_entropy"].iloc[i])
            if step["action_type"] != "call_tool":
                assert np.isnan(frame["arg_novelty"].iloc[i])
            else:
                assert not np.isnan(frame["arg_novelty"].iloc[i])


def test_normalized_entropy_bounds():
    assert normalized_entropy({"a": 1.0}) == 0.0
    assert normalized_entropy({"a": 0.5, "b": 0.5}) == pytest.approx(1.0)
    assert normalized_entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(1.0)
    assert 0.0 < normalized_entropy({"a": 0.9, "b": 0.1}) < 0.6


def test_semantic_deviation_is_a_cosine_distance(extractor, corpus):
    for run in corpus[:8]:
        values = extractor.transform(run)["semantic_deviation"]
        assert values.between(0.0, 2.0).all()


def test_arg_novelty_is_a_fraction(extractor, corpus):
    for run in corpus[:8]:
        values = extractor.transform(run)["arg_novelty"].dropna()
        assert values.between(0.0, 1.0).all()


# -- corpus statistics ----------------------------------------------------


def test_corpus_stats_round_trip(corpus):
    stats = CorpusStats.fit(corpus)
    again = CorpusStats.from_dict(stats.to_dict())
    assert again.duration == stats.duration
    assert again.tokens == stats.tokens


def test_corpus_stats_are_per_action_type(corpus):
    stats = CorpusStats.fit(corpus)
    assert set(stats.duration) == {"call_llm", "call_tool", "decide"}
    # A tool call is cheaper than an LLM call, so the means must differ.
    assert stats.tokens["call_tool"][0] < stats.tokens["call_llm"][0]


def test_unknown_action_type_yields_nan_rather_than_a_guess(corpus):
    stats = CorpusStats.fit(corpus)
    assert np.isnan(stats.duration_z("never_seen", 100.0))


def test_z_scores_use_the_supplied_stats_not_the_run(corpus):
    """Inference on a single run must reuse the persisted training stats."""
    stats = CorpusStats.fit(corpus)
    frame = FeatureExtractor(stats, StubEmbedder()).transform(corpus[0])
    step = corpus[0]["steps"][0]
    mean, std = stats.duration[step["action_type"]]
    assert frame["duration_z"].iloc[0] == pytest.approx((step["duration_ms"] - mean) / std)


def test_fit_on_empty_corpus_raises():
    with pytest.raises(ValueError):
        CorpusStats.fit([])


# -- labels ---------------------------------------------------------------


def test_transform_many_labels_exactly_one_root_cause_per_failed_run(extractor, corpus):
    frame = extractor.transform_many(corpus)
    assert list(frame.columns) == [*FEATURE_COLUMNS, *ID_COLUMNS]
    for run in corpus:
        rows = frame[frame["run_id"] == run["run"]["id"]]
        truth = run["run"]["true_failure_step"]
        if run["run"]["status"] == "failed":
            assert rows["is_root_cause"].sum() == 1
            assert rows[rows["is_root_cause"]]["step_index"].iloc[0] == truth
        else:
            assert rows["is_root_cause"].sum() == 0


def test_successful_runs_contribute_only_negatives(extractor):
    runs = [make(anomaly=a, seed=i)[0] for i, a in enumerate(
        ["slow_tool", "transient_retry", "verbose_step", "benign_revisit", "low_confidence_decide"])]
    frame = FeatureExtractor(CorpusStats.fit(runs), StubEmbedder()).transform_many(runs)
    assert not frame["is_root_cause"].any()


# -- the signatures the taxonomy claims, measured through the features ----


def _root_rows(fault, n=18):
    """Root-cause rows for one class, against steps from healthy runs.

    The contrast group is deliberately the healthy runs rather than every
    non-root step, because the non-root steps of a faulty run are contaminated
    by the fault: every step inside an infinite loop shares the repeated state
    hash, so including them would hide the very signature under test.
    """
    runs = [make(fault=fault, seed=i)[0] for i in range(n)]
    base = [make(seed=500 + i)[0] for i in range(n)]
    extractor = FeatureExtractor(CorpusStats.fit(runs + base), StubEmbedder())
    frame = extractor.transform_many(runs + base)
    healthy = frame[frame["injected_class"].isna()]
    return frame[frame["is_root_cause"]], healthy


def test_schema_violation_shows_on_parse_failure():
    root, other = _root_rows("schema_violation")
    assert root["parse_failure"].all()
    assert other["parse_failure"].mean() < 0.01


def test_infinite_loop_shows_on_state_hash_repeat():
    root, other = _root_rows("infinite_loop")
    assert root["state_hash_repeat"].mean() > 3 * other["state_hash_repeat"].mean()


def test_context_truncation_shows_as_a_token_drop():
    root, other = _root_rows("context_truncation")
    assert root["token_z"].mean() < -1.0 < other["token_z"].mean() + 1.0


def test_hallucinated_argument_shows_on_arg_novelty():
    root, other = _root_rows("hallucinated_argument")
    assert root["arg_novelty"].mean() > other["arg_novelty"].dropna().mean()


def test_wrong_tool_chosen_shows_as_an_entropy_spike():
    root, other = _root_rows("wrong_tool_chosen")
    assert root["tool_choice_entropy"].mean() > other["tool_choice_entropy"].dropna().mean()


def test_premature_termination_shows_as_late_position_with_no_downstream_errors():
    root, other = _root_rows("premature_termination")
    assert root["downstream_error_count"].max() == 0
    assert root["position_ratio"].mean() > other["position_ratio"].mean()


def test_stale_retrieval_shows_as_an_unusually_fast_call():
    root, other = _root_rows("stale_retrieval")
    assert root["duration_z"].mean() < other["duration_z"].mean()


# -- the real encoder -----------------------------------------------------


@pytest.mark.slow
def test_real_minilm_encoder_separates_on_topic_from_off_topic():
    embedder = Embedder()
    run, _ = make(fault="wrong_tool_chosen", seed=1)
    stats = CorpusStats.fit([run])
    frame = FeatureExtractor(stats, embedder).transform(run)
    assert frame["semantic_deviation"].between(0.0, 2.0).all()
    # 384 dimensions, L2 normalized, and the cache must dedupe.
    vectors = embedder.encode(["one", "two", "one"])
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert np.array_equal(vectors[0], vectors[2])


@pytest.mark.slow
def test_step_text_prefers_summary():
    run, _ = make(seed=2)
    step = run["steps"][2]
    assert step_text(step) == step["output"]["summary"]
