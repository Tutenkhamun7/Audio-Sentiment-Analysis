"""Tests for Step 9: Acoustic pooling, abstention gates, and precedence (Tests 9, 10, 11)."""

import numpy as np

from convaudio.stage4_acoustic.pooling import check_abstention, pool_acoustic_features


def test_9_abstention_clean_speech_below_threshold() -> None:
    """Test 9: abstention: clean speech below 1.0 s -> acoustic is None and reason == 'SHORT_TURN'."""
    frame_emotions = np.ones((50, 2), dtype=np.float32) * 0.5
    mask = np.zeros(50, dtype=bool)

    features, reason = pool_acoustic_features(
        frame_emotions=frame_emotions,
        mask=mask,
        clean_speech_s=0.75,  # below 1.0s
        overlap_ratio=0.1,
        sep_cosine=0.9,
        min_clean_speech_s=1.0,
    )

    assert features is None
    assert reason == "SHORT_TURN"


def test_10_abstention_precedence_short_turn_first() -> None:
    """Test 10: abstention precedence: a turn failing two gates reports SHORT_TURN first."""
    # Fails both SHORT_TURN (clean_speech_s < 1.0) and HIGH_OVERLAP (overlap_ratio > 0.8)
    reason = check_abstention(
        clean_speech_s=0.4,
        overlap_ratio=0.95,
        sep_cosine=0.2,  # also fails LOW_SEP_QUALITY (< 0.5)
        min_clean_speech_s=1.0,
        max_overlap_ratio=0.8,
        min_sep_cosine=0.5,
    )
    assert reason == "SHORT_TURN"

    # Fails both HIGH_OVERLAP and LOW_SEP_QUALITY, but clean_speech_s >= 1.0
    reason_high_overlap = check_abstention(
        clean_speech_s=1.5,
        overlap_ratio=0.85,
        sep_cosine=0.3,
        min_clean_speech_s=1.0,
        max_overlap_ratio=0.8,
        min_sep_cosine=0.5,
    )
    assert reason_high_overlap == "HIGH_OVERLAP"


def test_11_masked_pooling_ignores_masked_frames() -> None:
    """Test 11: masked pooling ignores masked frames — inject a huge sentinel value into masked positions and assert the pooled output is unchanged."""
    n_frames = 100
    # Clean frames: indices 0..69 (70 frames)
    # Masked frames: indices 70..99 (30 frames)
    mask = np.zeros(n_frames, dtype=bool)
    mask[70:] = True

    rng = np.random.default_rng(123)
    frame_emotions = rng.uniform(-0.5, 0.5, size=(n_frames, 2)).astype(np.float32)

    feat_baseline, reason_baseline = pool_acoustic_features(
        frame_emotions=frame_emotions.copy(),
        mask=mask,
        clean_speech_s=1.4,
        overlap_ratio=0.3,
        sep_cosine=0.9,
    )

    assert feat_baseline is not None
    assert reason_baseline is None

    # Inject huge sentinel values into masked positions
    corrupted_emotions = frame_emotions.copy()
    corrupted_emotions[70:, :] = 999999.0

    feat_corrupted, reason_corrupted = pool_acoustic_features(
        frame_emotions=corrupted_emotions,
        mask=mask,
        clean_speech_s=1.4,
        overlap_ratio=0.3,
        sep_cosine=0.9,
    )

    assert feat_corrupted is not None
    assert feat_corrupted.valence == feat_baseline.valence
    assert feat_corrupted.arousal == feat_baseline.arousal
    assert feat_corrupted.conf == feat_baseline.conf
    assert feat_corrupted.pooled_frames == feat_baseline.pooled_frames
    assert feat_corrupted.masked_frames == feat_baseline.masked_frames
