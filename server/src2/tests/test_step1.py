"""Tests for Step 1: Schema, Validation, Licences (Tests 4, 7, 8, 14, 15)."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from convaudio.cli import app
from convaudio.errors import LicenceViolation, TimelineValidationError
from convaudio.licences import assert_licence_allowed
from convaudio.stage2_timeline.schema import (
    AcousticFeatures,
    AudioMeta,
    ConversationTimeline,
    CorpusStats,
    Diagnostics,
    SpeakerMeta,
    Turn,
    TurnOverlap,
    TurnQuality,
)
from convaudio.stage2_timeline.validate import validate_timeline


def make_valid_timeline() -> ConversationTimeline:
    """Helper to create a fully valid ConversationTimeline."""
    return ConversationTimeline(
        schema_version="1.0.0",
        call_id="call_test_001",
        created_utc="2026-10-05T12:00:00Z",
        audio_meta=AudioMeta(
            path="audio.wav",
            sha256="abc123def456",
            original_sr=16000,
            channels=2,
            duration_s=10.0,
            branch="stereo_split",
        ),
        frame_rate_hz=50.0,
        n_frames=500,
        overlap_mask="overlap_mask.npy",
        speakers={
            "SPEAKER_00": SpeakerMeta(
                stream_path="streams/SPEAKER_00.wav",
                total_speech_s=5.0,
                embedding_key="SPEAKER_00",
            ),
            "SPEAKER_01": SpeakerMeta(
                stream_path="streams/SPEAKER_01.wav",
                total_speech_s=4.0,
                embedding_key="SPEAKER_01",
            ),
        },
        corpus_stats=CorpusStats(
            total_speech_s=9.0,
            overlap_speech_s=0.5,
            overlap_ratio=0.055,
            n_turns=2,
            n_interruptions=0,
        ),
        turns=[
            Turn(
                turn_id="turn_000",
                speaker="SPEAKER_00",
                start=0.5,
                end=3.0,
                provenance="clean",
                overlap=TurnOverlap(
                    ratio=0.0,
                    clean_speech_s=2.5,
                    interrupts=None,
                    interrupted_by=None,
                    preceding_gap_s=None,
                ),
                quality=TurnQuality(
                    sep_cosine=0.95,
                    asr_avg_logprob=-0.2,
                    no_speech_prob=0.01,
                    compression_ratio=1.2,
                ),
                text="Hello world",
                word_timings=None,
                word_timings_source="asr_fallback",
                acoustic=AcousticFeatures(
                    valence=0.5,
                    arousal=-0.1,
                    conf=0.9,
                    pooled_frames=125,
                    masked_frames=0,
                ),
                acoustic_abstain_reason=None,
            ),
            Turn(
                turn_id="turn_001",
                speaker="SPEAKER_01",
                start=3.2,
                end=4.0,
                provenance="clean",
                overlap=TurnOverlap(
                    ratio=0.0,
                    clean_speech_s=0.8,
                    interrupts=None,
                    interrupted_by=None,
                    preceding_gap_s=0.2,
                ),
                quality=TurnQuality(
                    sep_cosine=0.92,
                    asr_avg_logprob=-0.3,
                    no_speech_prob=0.05,
                    compression_ratio=1.1,
                ),
                text="Hi there",
                word_timings=None,
                word_timings_source="asr_fallback",
                acoustic=None,
                acoustic_abstain_reason="SHORT_TURN",
            ),
        ],
        diagnostics=Diagnostics(
            rejected_sources=[],
            alignment_failures=0,
            acoustic_abstentions=1,
            abstain_reasons={"SHORT_TURN": 1, "HIGH_OVERLAP": 0, "LOW_SEP_QUALITY": 0},
            model_revisions={},
            stage_durations_s={"s1": 0.1, "s2": 0.05, "s3": 0.2, "s4": 0.1},
        ),
    )


def test_4_frame_level_mask_survives_json_roundtrip(tmp_path: Path) -> None:
    """Test 4: frame-level mask survives a JSON round-trip with frame_rate_hz intact."""
    tl = make_valid_timeline()
    out_file = tmp_path / "timeline.json"
    tl.to_json(out_file)

    loaded = ConversationTimeline.from_json(out_file)
    assert loaded.frame_rate_hz == 50.0
    assert loaded.n_frames == 500
    assert loaded.overlap_mask == "overlap_mask.npy"
    assert loaded.call_id == tl.call_id
    assert len(loaded.turns) == 2

    # Also validate the saved json
    validate_timeline(out_file)


def test_7_validate_timeline_rejects_missing_frame_rate_hz() -> None:
    """Test 7: validate_timeline() REJECTS a timeline missing frame_rate_hz."""
    tl = make_valid_timeline()
    data = tl.to_dict()
    del data["frame_rate_hz"]

    with pytest.raises(TimelineValidationError):
        validate_timeline(data)


def test_8_validate_timeline_rejects_turn_with_both_or_neither_acoustic_and_reason() -> None:
    """Test 8: validate_timeline() REJECTS a turn with both acoustic and acoustic_abstain_reason non-null, and one with both null."""
    # Case 1: both non-null
    tl_both = make_valid_timeline()
    data_both = tl_both.to_dict()
    data_both["turns"][0]["acoustic_abstain_reason"] = "SHORT_TURN"  # turn 0 already has acoustic
    with pytest.raises(TimelineValidationError):
        validate_timeline(data_both)

    # Case 2: both null
    tl_neither = make_valid_timeline()
    data_neither = tl_neither.to_dict()
    data_neither["turns"][0]["acoustic"] = None
    data_neither["turns"][0]["acoustic_abstain_reason"] = None
    with pytest.raises(TimelineValidationError):
        validate_timeline(data_neither)


def test_14_assert_licence_allowed_raises_on_cc_by_nc() -> None:
    """Test 14: assert_licence_allowed raises LicenceViolation on 'CC-BY-NC-4.0'."""
    with pytest.raises(LicenceViolation):
        assert_licence_allowed(code_licence="MIT", weights_licence="CC-BY-NC-4.0")

    with pytest.raises(LicenceViolation):
        assert_licence_allowed(code_licence="CC-BY-NC-4.0", weights_licence="MIT")


def test_15_convaudio_licences_check_exits_nonzero_on_injected_denylist(tmp_path: Path) -> None:
    """Test 15: convaudio licences check exits non-zero on an injected deny-listed entry."""
    runner = CliRunner()
    bad_manifest = tmp_path / "bad_manifest.yaml"
    bad_manifest.write_text(
        """
- name: "injected_bad_model"
  code_licence: "MIT"
  weights_licence: "CC-BY-NC-4.0"
        """.strip(),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["licences", "check", "--manifest", str(bad_manifest)])
    assert result.exit_code != 0
    assert "Licence policy violations detected" in result.output
