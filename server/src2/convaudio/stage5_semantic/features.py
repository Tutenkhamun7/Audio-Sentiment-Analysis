"""Dynamics feature vector extraction for Stage 5 semantic classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from convaudio.stage2_timeline.schema import Turn

DYNAMICS_FEATURE_NAMES: tuple[str, ...] = (
    "overlap_ratio",
    "interrupts_flag",
    "interrupted_by_flag",
    "preceding_gap_s",
    "gap_is_null",
    "turn_duration_s",
    "words_per_sec",
    "n_words",
    "asr_avg_logprob",
    "sep_cosine",
    "is_separated",
    "turn_index_norm",
)


def extract_turn_dynamics(turn: Turn, turn_idx: int, n_turns: int) -> dict[str, float]:
    """Extract tabular dynamics feature dictionary for a single turn."""
    # Overlap features
    overlap_ratio = float(turn.overlap.ratio)
    interrupts_flag = 1.0 if turn.overlap.interrupts is not None else 0.0
    interrupted_by_flag = 1.0 if turn.overlap.interrupted_by is not None else 0.0

    # Gap feature with sign preservation and explicit indicator
    if turn.overlap.preceding_gap_s is None:
        preceding_gap_s = 0.0
        gap_is_null = 1.0
    else:
        preceding_gap_s = float(turn.overlap.preceding_gap_s)
        gap_is_null = 0.0

    # Duration and rate features
    turn_duration_s = max(float(turn.end - turn.start), 1e-4)
    if turn.word_timings is not None:
        n_words = float(len(turn.word_timings))
    elif turn.text is not None and turn.text.strip():
        n_words = float(len(turn.text.strip().split()))
    else:
        n_words = 0.0

    words_per_sec = n_words / turn_duration_s

    # Quality & trust features
    asr_avg_logprob = (
        float(turn.quality.asr_avg_logprob)
        if turn.quality.asr_avg_logprob is not None
        else 0.0
    )
    sep_cosine = (
        float(turn.quality.sep_cosine)
        if turn.quality.sep_cosine is not None
        else 1.0
    )
    is_separated = 1.0 if turn.provenance == "separated" else 0.0

    # Normalised position in call
    turn_index_norm = float(turn_idx) / float(max(n_turns, 1))

    return {
        "overlap_ratio": overlap_ratio,
        "interrupts_flag": interrupts_flag,
        "interrupted_by_flag": interrupted_by_flag,
        "preceding_gap_s": preceding_gap_s,
        "gap_is_null": gap_is_null,
        "turn_duration_s": turn_duration_s,
        "words_per_sec": words_per_sec,
        "n_words": n_words,
        "asr_avg_logprob": asr_avg_logprob,
        "sep_cosine": sep_cosine,
        "is_separated": is_separated,
        "turn_index_norm": turn_index_norm,
    }


def extract_features_array(
    turns: list[Turn],
    feature_names: tuple[str, ...] = DYNAMICS_FEATURE_NAMES,
) -> np.ndarray:
    """Extract a 2D numpy array of dynamics features for all turns.

    Raises ValueError if expected feature_names do not match DYNAMICS_FEATURE_NAMES.
    """
    if tuple(feature_names) != DYNAMICS_FEATURE_NAMES:
        raise ValueError(
            f"Feature order mismatch: expected {DYNAMICS_FEATURE_NAMES}, got {tuple(feature_names)}"
        )

    n_turns = len(turns)
    if n_turns == 0:
        return np.empty((0, len(DYNAMICS_FEATURE_NAMES)), dtype=np.float32)

    rows: list[list[float]] = []
    for idx, turn in enumerate(turns):
        feat_dict = extract_turn_dynamics(turn, idx, n_turns)
        rows.append([feat_dict[name] for name in feature_names])

    return np.array(rows, dtype=np.float32)
