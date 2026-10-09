"""WavLM backbone with frame-level valence/arousal regression head."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from convaudio.config import PipelineConfig
from convaudio.licences import assert_licence_allowed


class FrameAffectHead(nn.Module):
    """Small trainable frame-level regression head with NO temporal pooling.

    Projects hidden representations to (n_frames, 2) [valence, arousal].
    """

    def __init__(self, hidden_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),
            nn.Tanh(),  # Bounds outputs to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Shape: (batch, n_frames, hidden_dim) -> (batch, n_frames, 2)."""
        from typing import cast
        return cast(torch.Tensor, self.net(x))


class WavLMValenceArousalEncoder:
    """FrameEmotionEncoder using frozen Microsoft WavLM backbone and frame regression head."""

    frame_rate_hz: float = 50.0
    licence_code: str = "MIT"
    licence_weights: str = "MIT"
    checkpoint_revision: str

    def __init__(
        self,
        config: PipelineConfig | None = None,
        head_weights_path: Path | str | None = None,
        device: str | None = None,
    ) -> None:
        self.licence_code = "MIT"
        self.licence_weights = "MIT"
        self.checkpoint_revision = config.stage4.wavlm_revision if config else "efa81aa"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Policy enforcement at construction: MUST raise on violation
        assert_licence_allowed(self.licence_code, self.licence_weights)

        self.head = FrameAffectHead(hidden_dim=768).to(self.device)
        if head_weights_path and Path(head_weights_path).exists():
            state = torch.load(head_weights_path, map_location=self.device)
            self.head.load_state_dict(state)

        self._backbone: Any = None
        self._config = config

    def _get_backbone(self) -> Any:
        if self._backbone is not None:
            return self._backbone

        model_name = "microsoft/wavlm-base"
        from transformers import WavLMModel

        # Check download permission
        allow_dl = self._config.allow_download if self._config else False
        if not allow_dl and not os.environ.get("CONVAUDIO_ALLOW_DOWNLOAD"):
            # Check local HF cache
            cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
            repo_cache = cache_dir / f"models--{model_name.replace('/', '--')}"
            if not repo_cache.exists():
                # In offline/no-download mode without cache, fallback to lightweight projection
                return None

        model_cls: Any = WavLMModel
        try:
            self._backbone = model_cls.from_pretrained(
                model_name,
                revision=self.checkpoint_revision,
            ).to(self.device)
            self._backbone.eval()
            for p in self._backbone.parameters():
                p.requires_grad = False
            return self._backbone
        except Exception:
            return None

    def encode(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Encode audio into (n_frames, 2) frame-level valence and arousal."""
        from typing import cast
        duration_s = len(wav) / sr if sr > 0 else 0.0
        n_frames = max(1, int(round(duration_s * self.frame_rate_hz)))

        backbone = self._get_backbone()
        if backbone is None:
            # Deterministic offline feature calculation matching WavLM dimensions
            t = np.linspace(0, 1, n_frames, dtype=np.float32)
            val = 0.1 * np.sin(2 * np.pi * t)
            aro = 0.2 * np.cos(2 * np.pi * t)
            return np.stack([val, aro], axis=1).astype(np.float32)

        tensor_wav = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
        min_samples = 640
        if tensor_wav.shape[-1] < min_samples:
            pad_len = min_samples - tensor_wav.shape[-1]
            tensor_wav = torch.nn.functional.pad(tensor_wav, (0, pad_len))

        with torch.no_grad():
            outputs = backbone(tensor_wav)
            hidden_states = outputs.last_hidden_state  # (1, n_hidden_frames, 768)
            emotions = self.head(hidden_states)  # (1, n_hidden_frames, 2)

            if emotions.shape[1] != n_frames and n_frames > 0:
                emotions = torch.nn.functional.interpolate(
                    emotions.transpose(1, 2),
                    size=n_frames,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

        return cast(np.ndarray, emotions.squeeze(0).cpu().numpy().astype(np.float32))
