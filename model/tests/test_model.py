"""Guards on the split, the training artifact and the diagnosis contract.

The split tests are the important ones. If a held-out class ever reaches
training, every number in the pitch becomes meaningless, and nothing about
the training run would look wrong from the outside.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

import numpy as np
import pytest

from generator.build import build_run
from generator.schema import FAILURE_CLASSES, HELD_OUT_CLASSES
from generator.tasks import TASK_TYPES
from model.dataset import TRAIN_CLASSES, split_by_failure_class
from model.features import FEATURE_COLUMNS
from model.predict import Localizer
from model.tests.test_features import StubEmbedder
from model.train import (
    BASELINES,
    choose_step_threshold,
    normalize_run_scores,
    train,
)

WHEN = datetime(2026, 3, 10, 8, 0, tzinfo=timezone.utc)
USER = "00000000-0000-4000-8000-000000000001"


@pytest.fixture(scope="module")
def corpus():
    """A small corpus with every class and a healthy share of successes."""
    runs = []
    seed = 0
    for cls in FAILURE_CLASSES:
        for _ in range(8):
            rng = random.Random(seed)
            trace, _ = build_run(TASK_TYPES[seed % 3], rng, user_id=USER,
                                 created_at=WHEN, fault=cls)
            runs.append(trace.to_dict())
            seed += 1
    for _ in range(30):
        rng = random.Random(seed)
        anomaly = ["slow_tool", "transient_retry", "verbose_step", None][seed % 4]
        trace, _ = build_run(TASK_TYPES[seed % 3], rng, user_id=USER,
                             created_at=WHEN, anomaly=anomaly)
        runs.append(trace.to_dict())
        seed += 1
    return runs


# -- the split, which the entire claim depends on -------------------------


def test_held_out_classes_never_reach_training(corpus):
    split = split_by_failure_class(corpus, seed=7)
    for bucket in (split.train, split.val, split.test_seen):
        for run in bucket:
            assert run["run"]["injected_class"] not in HELD_OUT_CLASSES


def test_test_heldout_contains_exactly_the_held_out_classes(corpus):
    split = split_by_failure_class(corpus, seed=7)
    failed = {r["run"]["injected_class"] for r in split.test_heldout
              if r["run"]["status"] == "failed"}
    assert failed == set(HELD_OUT_CLASSES)


def test_no_run_appears_in_two_splits(corpus):
    split = split_by_failure_class(corpus, seed=7)
    ids = [r["run"]["id"] for b in (split.train, split.val, split.test_seen, split.test_heldout)
           for r in b]
    assert len(ids) == len(set(ids)) == len(corpus)


def test_every_trained_class_appears_in_train(corpus):
    split = split_by_failure_class(corpus, seed=7)
    present = {r["run"]["injected_class"] for r in split.train} - {None}
    assert present == set(TRAIN_CLASSES)


def test_successful_runs_reach_every_split(corpus):
    split = split_by_failure_class(corpus, seed=7)
    for bucket in (split.train, split.val, split.test_seen, split.test_heldout):
        assert any(r["run"]["status"] == "success" for r in bucket)


def test_split_rejects_an_unrecognized_class(corpus):
    bad = [dict(r) for r in corpus[:4]]
    bad[0] = {**bad[0], "run": {**bad[0]["run"], "injected_class": "cosmic_ray"}}
    with pytest.raises(ValueError, match="unrecognized"):
        split_by_failure_class(bad, seed=7)


def test_split_is_deterministic_for_a_seed(corpus):
    a = split_by_failure_class(corpus, seed=7)
    b = split_by_failure_class(corpus, seed=7)
    assert [r["run"]["id"] for r in a.train] == [r["run"]["id"] for r in b.train]


# -- scoring --------------------------------------------------------------


def test_normalized_scores_sum_to_one_hundred():
    raw = np.array([3.0, 70.0, 12.0, 1.0])
    out = normalize_run_scores(raw)
    assert out.sum() == pytest.approx(100.0)
    assert int(np.argmax(out)) == int(np.argmax(raw))  # ranking preserved


def test_normalized_scores_handle_an_all_zero_run():
    out = normalize_run_scores(np.zeros(5))
    assert out.sum() == pytest.approx(100.0)


def test_threshold_selection_prefers_coverage_with_accuracy():
    """A run set where the confident answers are the right ones."""
    runs = []
    for i in range(20):
        correct = i < 14
        scores = np.full(10, 1.0)
        scores[3] = 90.0 if correct else 2.0
        runs.append({"scores": scores, "normalized": normalize_run_scores(scores),
                     "truth": 3, "frame": None})
    threshold, sweep = choose_step_threshold(runs)
    assert 0.0 < threshold < 1.0
    assert any(row["coverage"] > 0 for row in sweep)


@pytest.mark.parametrize("name", list(BASELINES))
def test_baselines_return_a_valid_step_index(corpus, name):
    from model.features import CorpusStats, FeatureExtractor
    from model.train import _per_run, step_scores

    extractor = FeatureExtractor(CorpusStats.fit(corpus), StubEmbedder())
    frame = extractor.transform_many(corpus[:10])
    import sklearn.dummy

    dummy = sklearn.dummy.DummyClassifier(strategy="stratified").fit(
        frame[list(FEATURE_COLUMNS)], frame["is_root_cause"]
    )
    for run in _per_run(frame, step_scores(dummy, frame)):
        idx = BASELINES[name](run)
        assert 0 <= idx < len(run["scores"])


# -- end to end -----------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory, corpus):
    import json

    path = tmp_path_factory.mktemp("corpus") / "runs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in corpus), encoding="utf-8")
    artifact = tmp_path_factory.mktemp("artifacts") / "localizer.joblib"
    report = train(path, artifact, seed=7, embedder=StubEmbedder())
    return artifact, report


def test_training_persists_a_loadable_artifact(trained):
    artifact, _ = trained
    assert artifact.exists()
    model = Localizer.load(artifact, embedder=StubEmbedder())
    assert model.trained_classes == list(TRAIN_CLASSES)
    assert 0.0 < model.step_threshold < 1.0


def test_training_reports_both_accuracies_and_loco(trained):
    _, report = trained
    for key in ("test_seen_top1", "test_heldout_top1", "test_heldout_top3", "loco_mean"):
        assert key in report.metrics
        assert 0.0 <= report.metrics[key] <= 1.0


def test_artifact_rejects_mismatched_feature_columns(trained, monkeypatch):
    artifact, _ = trained
    monkeypatch.setattr("model.predict.FEATURE_COLUMNS", ("a", "b"))
    with pytest.raises(ValueError, match="different feature columns"):
        Localizer.load(artifact, embedder=StubEmbedder())


# -- the diagnosis contract -----------------------------------------------


CONTRACT_KEYS = {
    "run_id", "flagged_step_index", "confidence", "predicted_class",
    "evidence_path", "evidence", "step_scores", "explanation", "suggested_fixes",
}


def test_diagnosis_matches_the_contract_shape(trained, corpus):
    artifact, _ = trained
    model = Localizer.load(artifact, embedder=StubEmbedder())
    failed = [r for r in corpus if r["run"]["status"] == "failed"][:8]
    for run in failed:
        d = model.diagnose(run)
        assert CONTRACT_KEYS <= set(d)
        assert d["run_id"] == run["run"]["id"]
        assert len(d["step_scores"]) == len(run["steps"])
        assert 0 <= d["flagged_step_index"] < len(run["steps"])
        assert all(isinstance(s, int) for s in d["step_scores"])


def test_step_scores_sum_to_about_one_hundred(trained, corpus):
    artifact, _ = trained
    model = Localizer.load(artifact, embedder=StubEmbedder())
    for run in corpus[:8]:
        d = model.diagnose(run)
        assert sum(d["step_scores"]) == pytest.approx(100, abs=len(run["steps"]))


def test_confidence_is_the_flagged_step_share(trained, corpus):
    artifact, _ = trained
    model = Localizer.load(artifact, embedder=StubEmbedder())
    for run in corpus[:8]:
        d = model.diagnose(run)
        assert d["confidence"] == pytest.approx(
            d["step_scores"][d["flagged_step_index"]] / 100.0, abs=0.01
        )


def test_predicted_class_is_a_trained_class_or_unknown(trained, corpus):
    artifact, _ = trained
    model = Localizer.load(artifact, embedder=StubEmbedder())
    for run in corpus:
        cls = model.diagnose(run)["predicted_class"]
        assert cls == "unknown" or cls in TRAIN_CLASSES
        # A held-out class can never be named: it is not a label of the head.
        assert cls not in HELD_OUT_CLASSES


def test_low_confidence_returns_unknown_with_a_reason(trained, corpus):
    artifact, _ = trained
    model = Localizer.load(artifact, embedder=StubEmbedder())
    model.step_threshold = 0.99  # nothing can clear this
    d = model.diagnose(corpus[0])
    assert d["predicted_class"] == "unknown"
    assert "below" in d["unknown_reason"]


def test_confidence_threshold_env_override(trained, monkeypatch):
    artifact, _ = trained
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.42")
    assert Localizer.load(artifact, embedder=StubEmbedder()).step_threshold == 0.42


def test_empty_run_is_handled(trained):
    artifact, _ = trained
    model = Localizer.load(artifact, embedder=StubEmbedder())
    d = model.diagnose({"run": {"id": "x"}, "steps": []})
    assert d["predicted_class"] == "unknown"
    assert d["step_scores"] == []
    assert CONTRACT_KEYS <= set(d)


def test_evidence_quotes_raw_feature_values(trained, corpus):
    artifact, _ = trained
    model = Localizer.load(artifact, embedder=StubEmbedder())
    d = model.diagnose(corpus[0])
    assert "semantic_deviation" in d["evidence"]
    assert isinstance(d["evidence"]["parse_failure"], bool)
    # NaN is reported as null, never as a filler number.
    for value in d["evidence"].values():
        assert value is None or isinstance(value, (int, float, bool))
