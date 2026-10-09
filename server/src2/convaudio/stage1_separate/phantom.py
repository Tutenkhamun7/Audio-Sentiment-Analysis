"""Phantom source detection and rejection for separator outputs."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np
import torch


def filter_phantom_sources(
    sources: Sequence[torch.Tensor | np.ndarray],
    active_durations_s: Sequence[float],
    min_speaker_speech_s: float = 1.0,
    silent_rms_threshold: float = 1e-4,
) -> Tuple[list[int], list[dict[str, Any]]]:
    """Filter out phantom / dummy speaker sources below minimum speech or near-silent energy.

    Returns:
        (valid_source_indices, rejected_sources_diagnostics)
    """
    valid_indices: list[int] = []
    rejected: list[dict[str, Any]] = []

    for idx, (src, dur_s) in enumerate(zip(sources, active_durations_s)):
        # Calculate RMS energy
        if isinstance(src, torch.Tensor):
            arr = src.detach().cpu().numpy()
        else:
            arr = src
        rms = float(np.sqrt(np.mean(arr**2) + 1e-12))

        if dur_s < min_speaker_speech_s:
            rejected.append(
                {
                    "source_idx": idx,
                    "reason": "below_min_speech",
                    "speech_s": round(dur_s, 3),
                    "measured_rms": round(rms, 6),
                }
            )
        elif rms < silent_rms_threshold:
            rejected.append(
                {
                    "source_idx": idx,
                    "reason": "near_silent_energy",
                    "speech_s": round(dur_s, 3),
                    "measured_rms": round(rms, 6),
                }
            )
        else:
            valid_indices.append(idx)

    return valid_indices, rejected
