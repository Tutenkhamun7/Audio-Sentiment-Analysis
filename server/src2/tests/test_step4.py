"""Tests for Step 4: Overlap mask resampling and interruption detection (Tests 5, 6)."""

import numpy as np

from convaudio.stage2_timeline.overlap import detect_interruptions, resample_mask
from convaudio.stage2_timeline.schema import Turn, TurnOverlap, TurnQuality


def test_5_resample_mask_correct_between_frame_rates() -> None:
    """Test 5: resample_mask is correct between two different frame rates — assert on a known pattern, not just on output length."""
    # Pattern at 50 Hz: 1 second False, 1 second True, 1 second False -> total 3 seconds (150 frames)
    # Frames 0..49: False, 50..99: True, 100..149: False
    mask_50hz = np.zeros(150, dtype=bool)
    mask_50hz[50:100] = True

    # Resample to 100 Hz: duration 3s -> 300 frames
    # Frames 0..99: False, 100..199: True, 200..299: False
    mask_100hz = resample_mask(mask_50hz, from_hz=50.0, to_hz=100.0)
    assert len(mask_100hz) == 300
    assert not np.any(mask_100hz[:99])
    assert np.all(mask_100hz[101:198])
    assert not np.any(mask_100hz[201:])

    # Resample back from 100 Hz to 25 Hz: duration 3s -> 75 frames
    # Frames 0..24: False, 25..49: True, 50..74: False
    mask_25hz = resample_mask(mask_100hz, from_hz=100.0, to_hz=25.0)
    assert len(mask_25hz) == 75
    assert not np.any(mask_25hz[:24])
    assert np.all(mask_25hz[26:48])
    assert not np.any(mask_25hz[51:])


def test_6_interruption_detection_programmed_overlap() -> None:
    """Test 6: interruption detection: a programmed overlap yields the correct interrupts / interrupted_by pair AND a negative preceding_gap_s."""
    # Turn 0: SPEAKER_00 from 1.0 to 3.0
    # Turn 1: SPEAKER_01 from 2.2 to 4.0
    # -> Turn 1 starts at 2.2, while Turn 0 ends at 3.0.
    #    Active overlap: 3.0 - 2.2 = 0.8s <= 1.0s (interrupt_window_s).
    #    So SPEAKER_01 interrupts SPEAKER_00, and SPEAKER_00 is interrupted_by SPEAKER_01!
    #    preceding_gap_s of Turn 1 is 2.2 - 3.0 = -0.8s!
    turn0 = Turn(
        turn_id="turn_000",
        speaker="SPEAKER_00",
        start=1.0,
        end=3.0,
        provenance="separated",
        overlap=TurnOverlap(ratio=0.4, clean_speech_s=1.2),
        quality=TurnQuality(sep_cosine=0.9),
    )
    turn1 = Turn(
        turn_id="turn_001",
        speaker="SPEAKER_01",
        start=2.2,
        end=4.0,
        provenance="separated",
        overlap=TurnOverlap(ratio=0.44, clean_speech_s=1.0),
        quality=TurnQuality(sep_cosine=0.85),
    )
    turn2 = Turn(
        turn_id="turn_002",
        speaker="SPEAKER_00",
        start=4.5,
        end=5.5,
        provenance="separated",
        overlap=TurnOverlap(ratio=0.0, clean_speech_s=1.0),
        quality=TurnQuality(sep_cosine=0.95),
    )

    turns = [turn0, turn1, turn2]
    n_interrupts = detect_interruptions(turns, interrupt_window_s=1.0)

    assert n_interrupts == 1
    # Check preceding gaps
    assert turn0.overlap.preceding_gap_s is None
    assert turn1.overlap.preceding_gap_s == -0.8
    assert turn2.overlap.preceding_gap_s == 0.5

    # Check interruption attribution
    assert turn1.overlap.interrupts == "SPEAKER_00"
    assert turn0.overlap.interrupted_by == "SPEAKER_01"
    assert turn2.overlap.interrupts is None
    assert turn2.overlap.interrupted_by is None
