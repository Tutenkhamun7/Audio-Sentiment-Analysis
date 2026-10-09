"""Frame-level overlap mask operations, resampling, and interruption analysis."""

from __future__ import annotations

from typing import Tuple, cast

import numpy as np

from convaudio.stage2_timeline.schema import Turn


def overlap_ratio_for(
    mask: np.ndarray,
    start: float,
    end: float,
    frame_rate_hz: float,
) -> Tuple[float, float, int, int]:
    """Calculate overlap ratio, clean speech duration (s), clean frames, and overlap frames in window.

    Returns:
        (overlap_ratio, clean_speech_s, clean_frames_count, overlap_frames_count)
    """
    dur = end - start
    if dur <= 0:
        return 0.0, 0.0, 0, 0

    s_idx = max(0, int(round(start * frame_rate_hz)))
    e_idx = min(len(mask), int(round(end * frame_rate_hz)))

    if e_idx <= s_idx:
        return 0.0, float(dur), 0, 0

    sub_mask = mask[s_idx:e_idx]
    overlap_frames = int(np.sum(sub_mask))
    total_frames = len(sub_mask)
    clean_frames = total_frames - overlap_frames

    ratio = overlap_frames / total_frames if total_frames > 0 else 0.0
    clean_speech_s = clean_frames / frame_rate_hz

    return float(ratio), float(clean_speech_s), clean_frames, overlap_frames


def clean_frames_for(
    mask: np.ndarray,
    start: float,
    end: float,
    frame_rate_hz: float,
) -> np.ndarray:
    """Return slice of mask inverted (~mask) for the given time window (clean frames = True)."""
    s_idx = max(0, int(round(start * frame_rate_hz)))
    e_idx = min(len(mask), int(round(end * frame_rate_hz)))
    if e_idx <= s_idx:
        return np.zeros(0, dtype=bool)
    return ~mask[s_idx:e_idx]


def resample_mask(mask: np.ndarray, from_hz: float, to_hz: float) -> np.ndarray:
    """Resample a frame-level boolean mask from from_hz to to_hz.

    Preserves timing pattern accurately using time-index mapping.
    """
    if len(mask) == 0:
        return np.zeros(0, dtype=bool)

    duration_s = len(mask) / from_hz
    n_out_frames = int(round(duration_s * to_hz))
    if n_out_frames == 0:
        return np.zeros(0, dtype=bool)

    # Output frame centers in seconds
    t_out = (np.arange(n_out_frames, dtype=np.float64) + 0.5) / to_hz
    # Map to input frame indices
    src_indices = np.clip(np.floor(t_out * from_hz).astype(int), 0, len(mask) - 1)

    return cast(np.ndarray, mask[src_indices])


def detect_interruptions(
    turns: list[Turn],
    interrupt_window_s: float = 1.0,
) -> int:
    """Detect speaker interruptions and calculate preceding gap across turns in-place.

    Interruption rule:
      Speaker B starts while A is still active (B.start < A.end and B.start >= A.start and B.speaker != A.speaker)
      AND A stops within interrupt_window_s of B starting (A.end - B.start <= interrupt_window_s).
      -> B interrupts A (B.overlap.interrupts = A.speaker)
      -> A interrupted by B (A.overlap.interrupted_by = B.speaker)

    preceding_gap_s:
      previous turn's end subtracted from this turn's start (this.start - prev.end).
      Negative means this turn began before the previous one ended.
      None for the first turn.

    Returns:
      Total count of interruptions detected.
    """
    # Sort turns by start time
    turns.sort(key=lambda t: (t.start, t.end))

    # 1. Compute preceding_gap_s sequentially
    for i in range(len(turns)):
        if i == 0:
            turns[i].overlap.preceding_gap_s = None
        else:
            prev_turn = turns[i - 1]
            gap = turns[i].start - prev_turn.end
            turns[i].overlap.preceding_gap_s = round(float(gap), 4)

    # 2. Interruption detection
    n_interruptions = 0
    for i in range(len(turns)):
        curr = turns[i]
        # Check all subsequent turns that start while curr is still active
        for j in range(i + 1, len(turns)):
            cand = turns[j]
            if cand.start >= curr.end:
                # Started after curr ended, can't interrupt curr
                break
            if cand.speaker != curr.speaker and cand.start >= curr.start:
                # cand started while curr is active
                active_overlap = curr.end - cand.start
                if 0 < active_overlap <= interrupt_window_s:
                    cand.overlap.interrupts = curr.speaker
                    curr.overlap.interrupted_by = cand.speaker
                    n_interruptions += 1

    return n_interruptions
