"""Stub implementation of SemanticClassifier for offline and testing paths."""

from __future__ import annotations

import hashlib

import numpy as np

from convaudio.licences import assert_licence_allowed
from convaudio.stage5_semantic.features import DYNAMICS_FEATURE_NAMES


class StubSemanticClassifier:
    """Deterministic hash-based classifier for tests and stub pipeline runs."""

    labels: tuple[str, ...] = ("negative", "neutral", "positive")
    licence_code: str = "N/A"
    licence_weights: str = "N/A"
    checkpoint_revision: str = "stub-v1"
    feature_names: tuple[str, ...] = DYNAMICS_FEATURE_NAMES

    def __init__(
        self,
        licence_code: str = "N/A",
        licence_weights: str = "N/A",
        checkpoint_revision: str = "stub-v1",
    ) -> None:
        self.licence_code = licence_code
        self.licence_weights = licence_weights
        self.checkpoint_revision = checkpoint_revision
        assert_licence_allowed(self.licence_code, self.licence_weights)

    def predict(
        self,
        windows: list[str],
        features: np.ndarray,
    ) -> np.ndarray:
        """Deterministically predict label probabilities derived from sha256 hash of window string."""
        n_samples = len(windows)
        if n_samples == 0:
            return np.empty((0, len(self.labels)), dtype=np.float32)

        if features.shape[0] != n_samples:
            raise ValueError(
                f"Batch size mismatch: {len(windows)} windows vs {features.shape[0]} feature rows"
            )

        probs_list: list[list[float]] = []
        for win in windows:
            digest = hashlib.sha256(win.encode("utf-8")).digest()
            # Derive deterministic positive weights for negative, neutral, positive
            w0 = float(int.from_bytes(digest[0:4], "big") % 1000 + 10)
            w1 = float(int.from_bytes(digest[4:8], "big") % 1000 + 10)
            w2 = float(int.from_bytes(digest[8:12], "big") % 1000 + 10)
            total = w0 + w1 + w2
            probs_list.append([w0 / total, w1 / total, w2 / total])

        return np.array(probs_list, dtype=np.float32)
