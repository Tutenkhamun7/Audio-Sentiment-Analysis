"""Fast stereo-vs-mono branching logic (instant bypass for clean stereo calls)."""

from __future__ import annotations

from typing import Tuple
import torch


def analyze_channels(
    waveform: torch.Tensor,
    corr_threshold: float = 0.6,
    min_energy_ratio: float = 0.05,
) -> Tuple[bool, float, float]:
    """Examine 2-channel audio correlation to decide if channels represent isolated speakers."""
    if waveform.shape[0] < 2:
        return False, 1.0, 0.0

    ch0 = waveform[0].float()
    ch1 = waveform[1].float()

    e0 = float(torch.sum(ch0**2).item())
    e1 = float(torch.sum(ch1**2).item())

    if e0 <= 1e-9 or e1 <= 1e-9:
        return False, 0.0, 0.0

    ratio = min(e0, e1) / max(e0, e1)
    if ratio < min_energy_ratio:
        return False, 1.0, ratio

    # Pearson correlation
    ch0_c = ch0 - ch0.mean()
    ch1_c = ch1 - ch1.mean()
    denom = torch.sqrt(torch.sum(ch0_c**2) * torch.sum(ch1_c**2)) + 1e-9
    corr = float(torch.abs(torch.sum(ch0_c * ch1_c) / denom).item())

    is_clean_stereo = (corr < corr_threshold) and (ratio >= min_energy_ratio)
    return is_clean_stereo, corr, ratio


def decide_branch(
    waveform: torch.Tensor,
    corr_threshold: float = 0.6,
    min_energy_ratio: float = 0.05,
    force_branch: str | None = None,
) -> Tuple[str, dict[str, float]]:
    """Determine whether to process as stereo_split, mono_separated, or no_split."""
    if force_branch:
        fb = force_branch.strip().lower()
        if fb in ("stereo", "stereo_split"):
            return "stereo_split", {"forced": 1.0}
        elif fb in ("mono", "mono_separated"):
            return "mono_separated", {"forced": 1.0}
        elif fb in ("no_split", "none", "single"):
            return "no_split", {"forced": 1.0}

    if waveform.shape[0] < 2:
        return "mono_separated", {"inter_channel_correlation": 1.0, "channel_energy_ratio": 0.0}

    is_stereo, corr, ratio = analyze_channels(
        waveform,
        corr_threshold=corr_threshold,
        min_energy_ratio=min_energy_ratio,
    )
    metrics = {"inter_channel_correlation": corr, "channel_energy_ratio": ratio}
    return ("stereo_split" if is_stereo else "mono_separated"), metrics
