"""Protocol and abstract definitions for Stage 5 semantic classifiers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SemanticClassifier(Protocol):
    """Protocol for turn-level semantic sentiment classifiers."""

    labels: tuple[str, ...]
    licence_code: str
    licence_weights: str
    checkpoint_revision: str
    feature_names: tuple[str, ...]

    def predict(
        self,
        windows: list[str],
        features: np.ndarray,  # (batch, n_features), already scaled
    ) -> np.ndarray:  # (batch, n_labels) calibrated probs
        """Predict calibrated label probabilities for a batch of context windows and features."""
        ...
