"""Channel analysis for stereo-vs-mono speaker separation decision."""

from __future__ import annotations

from typing import Tuple

import torch


def analyze_channels(
    waveform: torch.Tensor,
    corr_threshold: float = 0.6,
    min_energy_ratio: float = 0.05,
) -> Tuple[bool, float, float]:
    """Evaluate whether a multi-channel waveform represents already-separated speaker channels.

    Returns:
        (is_speaker_separated, inter_channel_correlation, energy_ratio)
    """
    if waveform.shape[0] < 2:
        return False, 1.0, 0.0

    ch0 = waveform[0].float()
    ch1 = waveform[1].float()

    # Per-channel RMS energy
    rms0 = float(torch.sqrt(torch.mean(ch0**2)).item())
    rms1 = float(torch.sqrt(torch.mean(ch1**2)).item())

    max_rms = max(rms0, rms1)
    min_rms = min(rms0, rms1)
    energy_ratio = min_rms / (max_rms + 1e-8)

    # Inter-channel Pearson correlation
    c0 = ch0 - ch0.mean()
    c1 = ch1 - ch1.mean()
    norm0 = float(torch.norm(c0).item())
    norm1 = float(torch.norm(c1).item())

    if norm0 * norm1 > 1e-8:
        corr = float((torch.dot(c0, c1) / (norm0 * norm1)).item())
    else:
        corr = 0.0

    is_separated = (corr < corr_threshold) and (energy_ratio >= min_energy_ratio)
    return is_separated, corr, energy_ratio
