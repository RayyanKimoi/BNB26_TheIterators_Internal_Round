"""Corpus loading and the split.

The split is BY FAILURE CLASS, never randomly. `infinite_loop` and
`context_truncation` runs never appear in training in any form, which is what
makes the held-out localization number a generalization claim rather than a
restatement of the training set.

Four sets:

| Set | Failed runs | Successful runs | Used for |
| --- | --- | --- | --- |
| `train` | 70% of each trained class | 70% | fitting both heads and `CorpusStats` |
| `val` | 10% of each trained class | 10% | choosing the confidence threshold |
| `test_seen` | 20% of each trained class | 10% | in-distribution accuracy |
| `test_heldout` | ALL of both held-out classes | 10% | the generalization number |

Successful runs carry no class, so they are divided randomly and appear in
every set. Both test sets need them: a model that flags a step confidently on
a run that actually succeeded is producing a false positive, and that only
shows up if successful runs are scored too.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generator.schema import FAILURE_CLASSES, HELD_OUT_CLASSES

TRAIN_CLASSES: tuple[str, ...] = tuple(c for c in FAILURE_CLASSES if c not in HELD_OUT_CLASSES)

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.10


def load_runs(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@dataclass
class Split:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test_seen: list[dict[str, Any]]
    test_heldout: list[dict[str, Any]]

    def summary(self) -> str:
        def describe(name: str, runs: list[dict[str, Any]]) -> str:
            failed = [r for r in runs if r["run"]["status"] == "failed"]
            classes = sorted({r["run"]["injected_class"] for r in failed})
            return (
                f"  {name:<14}{len(runs):>4} runs  "
                f"{len(failed):>3} failed  {len(runs) - len(failed):>3} success  "
                f"{', '.join(classes) if classes else 'no failure classes'}"
            )

        return "\n".join(
            [
                describe("train", self.train),
                describe("val", self.val),
                describe("test_seen", self.test_seen),
                describe("test_heldout", self.test_heldout),
            ]
        )


def split_by_failure_class(runs: list[dict[str, Any]], seed: int = 7) -> Split:
    """Partition runs so the held-out classes never touch training."""
    rng = random.Random(seed)

    by_class: dict[str | None, list[dict[str, Any]]] = {}
    for run in runs:
        by_class.setdefault(run["run"]["injected_class"], []).append(run)

    unknown = set(by_class) - set(FAILURE_CLASSES) - {None}
    if unknown:
        raise ValueError(f"corpus contains unrecognized failure classes: {sorted(unknown)}")

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test_seen: list[dict[str, Any]] = []
    test_heldout: list[dict[str, Any]] = []

    for cls in HELD_OUT_CLASSES:
        # Whole class, untouched by training. This is the point of the split.
        test_heldout.extend(by_class.get(cls, []))

    for cls in TRAIN_CLASSES:
        group = list(by_class.get(cls, []))
        rng.shuffle(group)
        n_train = round(len(group) * TRAIN_FRACTION)
        n_val = round(len(group) * VAL_FRACTION)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test_seen.extend(group[n_train + n_val :])

    successes = list(by_class.get(None, []))
    rng.shuffle(successes)
    n = len(successes)
    a, b, c = round(n * 0.70), round(n * 0.80), round(n * 0.90)
    train.extend(successes[:a])
    val.extend(successes[a:b])
    test_seen.extend(successes[b:c])
    test_heldout.extend(successes[c:])

    for bucket in (train, val, test_seen, test_heldout):
        rng.shuffle(bucket)

    _assert_no_leakage(train, val, test_seen, test_heldout)
    return Split(train=train, val=val, test_seen=test_seen, test_heldout=test_heldout)


def _assert_no_leakage(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test_seen: list[dict[str, Any]],
    test_heldout: list[dict[str, Any]],
) -> None:
    """The guarantees the whole pitch rests on, checked every time we split."""
    for name, bucket in (("train", train), ("val", val), ("test_seen", test_seen)):
        offenders = {
            r["run"]["injected_class"]
            for r in bucket
            if r["run"]["injected_class"] in HELD_OUT_CLASSES
        }
        if offenders:
            raise AssertionError(f"{name} contains held-out classes {sorted(offenders)}")

    ids = [r["run"]["id"] for bucket in (train, val, test_seen, test_heldout) for r in bucket]
    if len(ids) != len(set(ids)):
        raise AssertionError("a run appears in more than one split")

    heldout_failed = {
        r["run"]["injected_class"] for r in test_heldout if r["run"]["status"] == "failed"
    }
    if heldout_failed != set(HELD_OUT_CLASSES):
        raise AssertionError(
            f"test_heldout must contain exactly {sorted(HELD_OUT_CLASSES)}, got {sorted(heldout_failed)}"
        )
