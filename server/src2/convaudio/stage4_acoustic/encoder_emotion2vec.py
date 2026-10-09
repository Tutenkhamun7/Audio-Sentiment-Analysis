"""FunASR emotion2vec+ acoustic emotion encoder for Stage 4."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from convaudio.config import PipelineConfig
from convaudio.licences import assert_licence_allowed

logger = logging.getLogger(__name__)

# Established circumplex coordinates (Valence in [-1, 1], Arousal in [-1, 1])
# for the 9 emotion2vec classes (matching tokens.txt order)
EMOTION_LABELS: list[str] = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "<unk>",
]

# Mapping to [valence, arousal]
EMOTION_COORDINATES: np.ndarray = np.array(
    [
        [-0.75, 0.80],  # angry
        [-0.65, 0.35],  # disgusted
        [-0.65, 0.75],  # fearful
        [0.75, 0.55],   # happy
        [0.00, 0.00],   # neutral
        [0.00, 0.00],   # other
        [-0.70, -0.45], # sad
        [0.20, 0.70],   # surprised
        [0.00, 0.00],   # <unk>
    ],
    dtype=np.float32,
)


class Emotion2VecEncoder:
    """FrameEmotionEncoder implementation using FunASR emotion2vec+."""

    frame_rate_hz: float = 50.0
    licence_code: str = "Apache-2.0"
    licence_weights: str = "Apache-2.0"
    checkpoint_revision: str

    def __init__(
        self,
        config: PipelineConfig | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.licence_code = "Apache-2.0"
        self.licence_weights = "Apache-2.0"
        self._config = config

        self.checkpoint_revision = (
            config.stage4.emotion2vec_revision if config and hasattr(config.stage4, "emotion2vec_revision") else "master"
        )
        self.model_name = (
            model_name
            or (config.stage4.emotion2vec_model if config and hasattr(config.stage4, "emotion2vec_model") else "iic/emotion2vec_plus_base")
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Enforce licence compliance at initialization
        assert_licence_allowed(self.licence_code, self.licence_weights)

        self._model: Any = None
        self._last_turn_emotion: str | None = None
        self._last_turn_scores: dict[str, float] | None = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        # Check local cache or permission to download
        allow_dl = self._config.allow_download if self._config else False
        if not allow_dl and not os.environ.get("CONVAUDIO_ALLOW_DOWNLOAD"):
            # Check local cache
            cache_root = Path(os.environ.get("MODELSCOPE_CACHE", Path.home() / ".cache" / "modelscope"))
            model_dir_name = self.model_name.replace("/", "--")
            possible_paths = [
                cache_root / "models" / model_dir_name,
                cache_root / "models" / "iic--emotion2vec_plus_base",
                cache_root / "hub" / self.model_name,
            ]
            custom_model_dir = os.environ.get("CONVAUDIO_MODEL_DIR")
            if custom_model_dir:
                possible_paths.insert(0, Path(custom_model_dir) / self.model_name)

            if not any(p.exists() for p in possible_paths):
                logger.info("Emotion2Vec model cache not found in offline mode. Falling back to synthetic projection.")
                return None

        try:
            from funasr import AutoModel

            self._model = AutoModel(
                model=self.model_name,
                device=self.device,
                disable_update=True,
            )
            return self._model
        except Exception as exc:
            logger.warning(f"Could not load Emotion2Vec model: {exc}")
            return None

    def encode(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Encode audio into (n_frames, 2) frame-level valence and arousal."""
        duration_s = len(wav) / sr if sr > 0 else 0.0
        n_frames = max(1, int(round(duration_s * self.frame_rate_hz)))

        # Ensure float32 1D audio
        wav_clean = np.asarray(wav, dtype=np.float32)
        if wav_clean.ndim > 1:
            wav_clean = wav_clean.squeeze()

        model = self._get_model()
        if model is None:
            # Deterministic fallback when model is not available
            t = np.linspace(0, 1, n_frames, dtype=np.float32)
            val = 0.05 * np.sin(2 * np.pi * t)
            aro = 0.10 * np.cos(2 * np.pi * t)
            self._last_turn_emotion = "neutral"
            self._last_turn_scores = {"neutral": 0.90, "happy": 0.05, "sad": 0.05}
            return np.stack([val, aro], axis=1).astype(np.float32)

        # Pad very short turns to ensure valid convolutional reception
        min_samples = int(sr * 0.3)  # At least 300ms
        if len(wav_clean) < min_samples:
            pad_len = min_samples - len(wav_clean)
            wav_clean = np.pad(wav_clean, (0, pad_len), mode="constant")

        try:
            res_frame = model.generate(input=wav_clean, fs=sr, granularity="frame")
            if not res_frame or not isinstance(res_frame, list) or "feats" not in res_frame[0]:
                raise ValueError("Unexpected emotion2vec output structure")

            feats_np = np.asarray(res_frame[0]["feats"], dtype=np.float32)
            feats_tensor = torch.from_numpy(feats_np).to(self.device)

            with torch.no_grad():
                logits = model.model.proj(feats_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # Align frame count with expected n_frames
            if len(probs) != n_frames and n_frames > 0:
                # Interpolate 9-class probabilities to n_frames
                probs_tensor = torch.from_numpy(probs).unsqueeze(0).transpose(1, 2)
                probs_interp = torch.nn.functional.interpolate(
                    probs_tensor,
                    size=n_frames,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2).squeeze(0)
                probs = probs_interp.numpy()

            # Map (n_frames, 9) probabilities to (n_frames, 2) [valence, arousal]
            emotions = probs @ EMOTION_COORDINATES

            # Compute turn-level aggregate emotion
            mean_probs = np.mean(probs, axis=0)
            top_idx = int(np.argmax(mean_probs))
            self._last_turn_emotion = EMOTION_LABELS[top_idx]
            self._last_turn_scores = {
                EMOTION_LABELS[i]: round(float(mean_probs[i]), 4)
                for i in range(len(EMOTION_LABELS))
            }

            return emotions.astype(np.float32)

        except Exception as exc:
            logger.warning(f"Emotion2Vec inference failed: {exc}. Falling back to neutral baseline.")
            t = np.linspace(0, 1, n_frames, dtype=np.float32)
            val = np.zeros(n_frames, dtype=np.float32)
            aro = np.zeros(n_frames, dtype=np.float32)
            self._last_turn_emotion = "neutral"
            self._last_turn_scores = {"neutral": 1.0}
            return np.stack([val, aro], axis=1).astype(np.float32)

    def get_last_turn_emotion(self) -> tuple[str | None, dict[str, float] | None]:
        """Return the dominant emotion label and probability scores of the last encoded turn."""
        return self._last_turn_emotion, self._last_turn_scores
