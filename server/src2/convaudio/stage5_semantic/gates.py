"""Abstention gates and temperature calibration for Stage 5 semantic analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from convaudio.stage2_timeline.schema import Turn


def check_abstention_pre_model(
    turn: Turn,
    min_words: int = 1,
    min_asr_logprob: float = -1.0,
) -> str | None:
    """Evaluate pre-model abstention gates in strict precedence order.

    Order:
    1. text is null or empty -> NO_TEXT
    2. n_words < min_words -> NO_TEXT
    3. asr_avg_logprob < min_asr_logprob -> LOW_ASR_CONFIDENCE
    """
    # Gate 1: text null or empty
    if turn.text is None or not turn.text.strip():
        return "NO_TEXT"

    # Gate 2: word count
    if turn.word_timings is not None:
        n_words = len(turn.word_timings)
    else:
        n_words = len(turn.text.strip().split())

    if n_words < min_words:
        return "NO_TEXT"

    # Gate 3: asr_avg_logprob
    if (
        turn.quality.asr_avg_logprob is not None
        and turn.quality.asr_avg_logprob < min_asr_logprob
    ):
        return "LOW_ASR_CONFIDENCE"

    return None


def check_abstention_post_model(
    probs: np.ndarray,
    min_conf: float = 0.45,
) -> str | None:
    """Evaluate post-model confidence gate.

    Order:
    4. max(probs) < min_conf -> LOW_SEM_CONF
    """
    if probs.size == 0:
        return "LOW_SEM_CONF"
    if float(np.max(probs)) < min_conf:
        return "LOW_SEM_CONF"
    return None


def check_turn_abstention(
    turn: Turn,
    probs: np.ndarray | None = None,
    min_words: int = 1,
    min_asr_logprob: float = -1.0,
    min_conf: float = 0.45,
) -> str | None:
    """Check all gates in strict documented precedence order:

    1. text null or empty -> NO_TEXT
    2. n_words < min_words -> NO_TEXT
    3. asr_avg_logprob < min_asr_logprob -> LOW_ASR_CONFIDENCE
    4. max(probs) < min_conf -> LOW_SEM_CONF
    """
    pre_reason = check_abstention_pre_model(
        turn, min_words=min_words, min_asr_logprob=min_asr_logprob
    )
    if pre_reason is not None:
        return pre_reason

    if probs is not None:
        post_reason = check_abstention_post_model(probs, min_conf=min_conf)
        if post_reason is not None:
            return post_reason

    return None


def scale_temperature(probs_or_logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to probabilities or logits."""
    if temperature <= 0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature}")

    # If inputs look like probabilities (sum to ~1.0 per row), convert to logits first
    arr = np.asarray(probs_or_logits, dtype=np.float64)
    if np.all(arr >= 0.0) and np.allclose(np.sum(arr, axis=-1, keepdims=True), 1.0, atol=1e-2):
        # Treat as probs: invert to log-probabilities
        eps = 1e-12
        logits = np.log(np.clip(arr, eps, 1.0))
    else:
        logits = arr

    scaled_logits = logits / temperature
    # Stable softmax
    shifted = scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probs: np.ndarray = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
    return probs.astype(np.float32)
