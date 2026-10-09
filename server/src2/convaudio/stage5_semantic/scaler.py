"""Feature scaler for Stage 5 tabular dynamics features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class DynamicsScaler:
    """Standard scaler (z-score) fitted exclusively on training dynamics features."""

    def __init__(
        self,
        feature_names: tuple[str, ...],
        mean: np.ndarray | None = None,
        scale: np.ndarray | None = None,
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.mean_ = np.asarray(mean, dtype=np.float32) if mean is not None else None
        self.scale_ = np.asarray(scale, dtype=np.float32) if scale is not None else None
        self._is_fitted = mean is not None and scale is not None

    def fit(self, X: np.ndarray) -> DynamicsScaler:
        """Fit mean and standard deviation from training data only."""
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected X with shape (n_samples, {len(self.feature_names)}), got {X.shape}"
            )
        self.mean_ = np.mean(X, axis=0).astype(np.float32)
        std = np.std(X, axis=0).astype(np.float32)
        # Avoid division by zero
        self.scale_ = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features using loaded/fitted mean and scale."""
        if not self._is_fitted or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted or loaded before transform() is called.")
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self.feature_names)}, got {X.shape[1]}"
            )
        return (np.asarray(X, dtype=np.float32) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step (training only)."""
        return self.fit(X).transform(X)

    def to_dict(self) -> dict[str, Any]:
        """Serialize scaler parameters to dictionary."""
        if not self._is_fitted or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Cannot serialize unfitted scaler.")
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DynamicsScaler:
        """Construct scaler from parameter dictionary."""
        names = tuple(data["feature_names"])
        mean = np.array(data["mean"], dtype=np.float32)
        scale = np.array(data["scale"], dtype=np.float32)
        return cls(feature_names=names, mean=mean, scale=scale)

    def save(self, path: Path | str) -> None:
        """Save scaler parameters to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> DynamicsScaler:
        """Load scaler from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
