"""Protocol interface for frame-level emotion encoders."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameEmotionEncoder(Protocol):
    """Protocol for frame-level valence and arousal feature extraction."""

    frame_rate_hz: float
    licence_code: str
    licence_weights: str
    checkpoint_revision: str

    def encode(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Return (n_frames, 2) float32: valence, arousal per frame."""
        ...
