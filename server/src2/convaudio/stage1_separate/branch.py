"""Branching logic for stereo-vs-mono separation paths."""

from __future__ import annotations

from typing import Tuple

import torch

from convaudio.io.channels import analyze_channels


def decide_branch(
    waveform: torch.Tensor,
    corr_threshold: float = 0.6,
    min_energy_ratio: float = 0.05,
    force_branch: str | None = None,
) -> Tuple[str, dict[str, float]]:
    """Determine whether to route to clean stereo branch or mono mixture separator.

    Returns:
        (branch_name, metrics_dict)
        branch_name is 'stereo_split' or 'mono_separated'
    """
    if force_branch:
        fb = force_branch.strip().lower()
        if fb == "stereo":
            return "stereo_split", {"forced": 1.0}
        elif fb == "mono":
            return "mono_separated", {"forced": 1.0}
        elif fb in ("no_split", "none", "single", "nosplit"):
            return "no_split", {"forced": 1.0}
        else:
            raise ValueError(
                f"Invalid force_branch option '{force_branch}'. Must be 'stereo', 'mono', or 'no_split'."
            )

    is_sep, corr, ratio = analyze_channels(
        waveform,
        corr_threshold=corr_threshold,
        min_energy_ratio=min_energy_ratio,
    )

    metrics = {
        "inter_channel_correlation": corr,
        "channel_energy_ratio": ratio,
    }

    if is_sep:
        return "stereo_split", metrics
    return "mono_separated", metrics
