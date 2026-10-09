"""Unit tests for dialogue extraction and formatting utility."""

from pathlib import Path

from convaudio.report.dialogue import format_dialogue, get_dialogue_turns, print_dialogue
from convaudio.stage2_timeline.schema import (
    AudioMeta,
    ConversationTimeline,
    CorpusStats,
    Diagnostics,
    SpeakerMeta,
    Turn,
    TurnOverlap,
    TurnQuality,
)


def _make_dummy_timeline() -> ConversationTimeline:
    return ConversationTimeline(
        schema_version="1.0.0",
        call_id="call_test",
        created_utc="2026-10-05T00:00:00Z",
        audio_meta=AudioMeta(
            path="test.wav",
            sha256="abc",
            original_sr=16000,
            channels=1,
            duration_s=10.0,
            branch="mono_separated",
        ),
        frame_rate_hz=50.0,
        n_frames=500,
        overlap_mask="overlap.npy",
        speakers={
            "SPEAKER_00": SpeakerMeta(stream_path="s0.wav", total_speech_s=5.0, embedding_key="SPEAKER_00"),
            "SPEAKER_01": SpeakerMeta(stream_path="s1.wav", total_speech_s=5.0, embedding_key="SPEAKER_01"),
        },
        corpus_stats=CorpusStats(
            total_speech_s=10.0,
            overlap_speech_s=0.0,
            overlap_ratio=0.0,
            n_turns=3,
            n_interruptions=0,
        ),
        turns=[
            Turn(
                turn_id="turn_000",
                speaker="SPEAKER_00",
                start=0.0,
                end=2.0,
                provenance="separated",
                overlap=TurnOverlap(ratio=0.0, clean_speech_s=2.0),
                quality=TurnQuality(),
                text="Hello, how can I help you?",
            ),
            Turn(
                turn_id="turn_001",
                speaker="SPEAKER_01",
                start=2.5,
                end=5.0,
                provenance="separated",
                overlap=TurnOverlap(ratio=0.0, clean_speech_s=2.5),
                quality=TurnQuality(),
                text="I would like to book a cab.",
            ),
            Turn(
                turn_id="turn_002",
                speaker="SPEAKER_00",
                start=5.2,
                end=6.0,
                provenance="separated",
                overlap=TurnOverlap(ratio=0.0, clean_speech_s=0.8),
                quality=TurnQuality(),
                text="Sure thing.",
            ),
        ],
        outcome=None,
        diagnostics=Diagnostics(),
    )


def test_format_dialogue_humanized() -> None:
    tl = _make_dummy_timeline()
    text = format_dialogue(tl, humanize_speakers=True)
    expected = (
        "Speaker 1 - Hello, how can I help you?\n"
        "Speaker 2 - I would like to book a cab.\n"
        "Speaker 1 - Sure thing."
    )
    assert text == expected


def test_format_dialogue_raw_speakers() -> None:
    tl = _make_dummy_timeline()
    text = format_dialogue(tl, humanize_speakers=False)
    expected = (
        "SPEAKER_00 - Hello, how can I help you?\n"
        "SPEAKER_01 - I would like to book a cab.\n"
        "SPEAKER_00 - Sure thing."
    )
    assert text == expected


def test_get_dialogue_turns() -> None:
    tl = _make_dummy_timeline()
    turns = get_dialogue_turns(tl)
    assert turns == [
        ("Speaker 1", "Hello, how can I help you?"),
        ("Speaker 2", "I would like to book a cab."),
        ("Speaker 1", "Sure thing."),
    ]


def test_print_dialogue_file(tmp_path: Path, capsys: None) -> None:
    tl = _make_dummy_timeline()
    file_path = tmp_path / "timeline.json"
    tl.to_json(file_path)

    # Should run from file path or directory without error
    print_dialogue(file_path)
    print_dialogue(tmp_path)
