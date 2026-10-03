"""SHAP attributions for a flagged step.

Answers "which features drove this step's score", at the level of the model's
own reasoning rather than the raw feature values. The inspector panel shows
these beside `evidence`, and the Gemini explainer prompt is seeded from them
(PRD, Model Specification: "SHAP for model-level evidence").

This explains the model. It does not change it, so it is safe under a model
freeze: no retraining, no threshold change, no effect on `step_scores` or on
which step is flagged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from model.features import FEATURE_COLUMNS


class ShapAttributor:
    """Lazily built TreeExplainer over the localizer.

    `shap` and its explainer are both expensive to import and construct, so
    neither happens until the first attribution is requested. A failure here
    degrades to `None` rather than taking down a diagnosis: SHAP is P1
    evidence, and an inspector panel missing one block is far better than an
    endpoint returning 500.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._explainer: Any = None
        self._unavailable = False

    @property
    def explainer(self) -> Any:
        if self._explainer is None and not self._unavailable:
            try:
                import shap

                self._explainer = shap.TreeExplainer(self._model)
            except Exception:  # noqa: BLE001 - optional evidence, never fatal
                self._unavailable = True
        return self._explainer

    def attribute(self, frame: pd.DataFrame, step_index: int) -> dict[str, float] | None:
        """SHAP value per feature for one row, or None if unavailable.

        Positive means the feature pushed this step toward "root cause".
        """
        if frame.empty or not 0 <= step_index < len(frame):
            return None
        explainer = self.explainer
        if explainer is None:
            return None
        row = frame.iloc[[step_index]][list(FEATURE_COLUMNS)]
        try:
            values = np.asarray(explainer.shap_values(row))
        except Exception:  # noqa: BLE001 - optional evidence, never fatal
            return None

        # shap returns (n_rows, n_features) for binary trees, and
        # (n_rows, n_features, n_classes) for some versions. Take the
        # positive class either way.
        values = values[0]
        if values.ndim == 2:
            values = values[:, -1]
        if values.shape[0] != len(FEATURE_COLUMNS):
            return None
        return {
            name: round(float(value), 4)
            for name, value in zip(FEATURE_COLUMNS, values)
        }


def top_drivers(shap_values: dict[str, float] | None, limit: int = 4) -> list[tuple[str, float]]:
    """The features that pushed the score up most, largest first.

    Used to seed the explainer prompt, which should talk about what drove the
    decision rather than recite all ten columns.
    """
    if not shap_values:
        return []
    positive = [(k, v) for k, v in shap_values.items() if v > 0]
    return sorted(positive, key=lambda kv: -kv[1])[:limit]
