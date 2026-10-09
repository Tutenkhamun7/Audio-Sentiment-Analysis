"""Deterministic stub implementation of FrameEmotionEncoder."""

from __future__ import annotations

import numpy as np

from convaudio.licences import assert_licence_allowed


class StubFrameEmotionEncoder:
    """Deterministic, offline stub frame emotion encoder for testing."""

    frame_rate_hz: float
    licence_code: str
    licence_weights: str
    checkpoint_revision: str

    def __init__(
        self,
        frame_rate_hz: float = 50.0,
        licence_code: str = "N/A",
        licence_weights: str = "N/A",
        checkpoint_revision: str = "stub-rev-001",
    ) -> None:
        self.frame_rate_hz = frame_rate_hz
        self.licence_code = licence_code
        self.licence_weights = licence_weights
        self.checkpoint_revision = checkpoint_revision

        # Enforce licence policy at construction
        assert_licence_allowed(self.licence_code, self.licence_weights)

    def encode(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Return deterministic (n_frames, 2) float32 array: valence, arousal."""
        n_samples = len(wav)
        duration_s = n_samples / sr if sr > 0 else 0.0
        n_frames = max(1, int(round(duration_s * self.frame_rate_hz)))

        # Baseline valence and arousal with slight smooth variation
        v_base = 0.2
        a_base = 0.4
        t = np.linspace(0, 1, n_frames, dtype=np.float32)

        valence = np.clip(v_base + 0.1 * np.sin(2 * np.pi * t), -1.0, 1.0).astype(np.float32)
        arousal = np.clip(a_base + 0.1 * np.cos(2 * np.pi * t), -1.0, 1.0).astype(np.float32)

        return np.stack([valence, arousal], axis=1)
