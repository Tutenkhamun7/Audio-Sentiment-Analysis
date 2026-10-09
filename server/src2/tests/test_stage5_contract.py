"""Tests 1-6, 10, 11, 12 for Stage 5 semantic contracts, window builder, features, and schema."""

import numpy as np
import pytest

from convaudio.errors import TimelineValidationError
from convaudio.stage2_timeline.schema import (
    AcousticFeatures,
    AudioMeta,
    ConversationTimeline,
    CorpusStats,
    SemanticFeatures,
    SpeakerMeta,
    Turn,
    TurnOverlap,
    TurnQuality,
)
from convaudio.stage2_timeline.validate import validate_timeline
from convaudio.stage5_semantic.features import (
    DYNAMICS_FEATURE_NAMES,
    extract_features_array,
    extract_turn_dynamics,
)
from convaudio.stage5_semantic.window import build_context_window


def _make_dummy_turn(
    turn_id: str = "t_0",
    speaker: str = "SPEAKER_00",
    text: str | None = "hello there",
    preceding_gap_s: float | None = 0.5,
    overlap_ratio: float = 0.0,
    interrupts: str | None = None,
    interrupted_by: str | None = None,
    provenance: str = "clean",
    start: float = 0.0,
    end: float = 2.0,
    asr_avg_logprob: float | None = -0.2,
    sep_cosine: float | None = 0.95,
) -> Turn:
    return Turn(
        turn_id=turn_id,
        speaker=speaker,
        start=start,
        end=end,
        provenance=provenance,
        overlap=TurnOverlap(
            ratio=overlap_ratio,
            clean_speech_s=end - start,
            interrupts=interrupts,
            interrupted_by=interrupted_by,
            preceding_gap_s=preceding_gap_s,
        ),
        quality=TurnQuality(
            sep_cosine=sep_cosine,
            asr_avg_logprob=asr_avg_logprob,
            no_speech_prob=0.01,
            compression_ratio=1.1,
        ),
        text=text,
        word_timings=None,
        word_timings_source="asr_fallback" if text else None,
        acoustic=AcousticFeatures(
            valence=0.0,
            arousal=0.0,
            conf=0.9,
            pooled_frames=10,
            masked_frames=0,
        ),
    )


def _make_dummy_timeline(schema_version: str = "1.1.0", turns: list[Turn] | None = None) -> ConversationTimeline:
    if turns is None:
        turns = [_make_dummy_turn()]
    return ConversationTimeline(
        schema_version=schema_version,
        call_id="call_test_001",
        created_utc="2026-10-06T00:00:00Z",
        audio_meta=AudioMeta(
            path="test.wav",
            sha256="123456",
            original_sr=16000,
            channels=2,
            duration_s=10.0,
            branch="stereo_split",
        ),
        frame_rate_hz=50.0,
        n_frames=500,
        overlap_mask="overlap.npy",
        speakers={
            "SPEAKER_00": SpeakerMeta(stream_path="s0.wav", total_speech_s=5.0, embedding_key="s0")
        },
        corpus_stats=CorpusStats(
            total_speech_s=5.0,
            overlap_speech_s=0.0,
            overlap_ratio=0.0,
            n_turns=len(turns),
            n_interruptions=0,
        ),
        turns=turns,
    )


# 1. Window builder: exactly one >>> marker, correct speaker tags, target turn last
def test_1_window_builder_format() -> None:
    turns = [
        _make_dummy_turn("t0", "SPEAKER_00", "the refund window closed in March"),
        _make_dummy_turn("t1", "SPEAKER_01", "no that is not what I was told"),
        _make_dummy_turn("t2", "SPEAKER_00", "I understand let me check"),
        _make_dummy_turn("t3", "SPEAKER_01", "fine"),
    ]
    roles = {"SPEAKER_00": "AGENT", "SPEAKER_01": "CUSTOMER"}
    win = build_context_window(turns, target_idx=3, n_context=3, speaker_roles=roles)

    assert win.count(">>> ") == 1
    lines = win.split("\n")
    assert len(lines) == 4
    assert lines[0] == "AGENT: the refund window closed in March"
    assert lines[1] == "CUSTOMER: no that is not what I was told"
    assert lines[2] == "AGENT: I understand let me check"
    assert lines[3] == ">>> CUSTOMER: fine"


# 2. Window builder skips empty-text turns and walks further back to fill n_context
def test_2_window_builder_skips_empty() -> None:
    turns = [
        _make_dummy_turn("t0", "SPEAKER_00", "valid context turn 0"),
        _make_dummy_turn("t1", "SPEAKER_01", ""),  # empty
        _make_dummy_turn("t2", "SPEAKER_00", None),  # None
        _make_dummy_turn("t3", "SPEAKER_01", "   "),  # whitespace
        _make_dummy_turn("t4", "SPEAKER_00", "valid context turn 4"),
        _make_dummy_turn("t5", "SPEAKER_01", "target turn"),
    ]
    win = build_context_window(turns, target_idx=5, n_context=2)
    lines = win.split("\n")
    # Should skip t1, t2, t3 and take t4 and t0
    assert len(lines) == 3
    assert "valid context turn 0" in lines[0]
    assert "valid context turn 4" in lines[1]
    assert lines[2] == ">>> SPEAKER_01: target turn"


# 3. Window builder truncates the oldest turn, never the target, when over token limit
def test_3_window_builder_truncates_oldest_never_target() -> None:
    turns = [
        _make_dummy_turn("t0", "SPEAKER_00", "first oldest turn that should be dropped"),
        _make_dummy_turn("t1", "SPEAKER_01", "second turn that might stay"),
        _make_dummy_turn("t2", "SPEAKER_00", "target turn keep always"),
    ]

    # Mock tokenizer returning 1 token per word
    def mock_tok(s: str) -> list[str]:
        return s.split()

    # Line 0 is 8 tokens, Line 1 is 6 tokens, Line 2 is 6 tokens.
    # Total is 20 tokens. With max_tokens=13, line 0 is dropped and lines 1 & 2 remain.
    win = build_context_window(turns, target_idx=2, n_context=2, tokenizer=mock_tok, max_tokens=13)
    lines = win.split("\n")
    assert len(lines) == 2
    assert "first oldest turn" not in win
    assert "second turn that might stay" in win
    assert lines[-1] == ">>> SPEAKER_00: target turn keep always"

    # Even if max_tokens is tiny (e.g. 1), target turn is NEVER dropped
    win_tiny = build_context_window(turns, target_idx=2, n_context=2, tokenizer=mock_tok, max_tokens=1)
    tiny_lines = win_tiny.split("\n")
    assert len(tiny_lines) == 1
    assert tiny_lines[0] == ">>> SPEAKER_00: target turn keep always"


# 4. First turn: preceding_gap_s null -> imputed 0.0 AND gap_is_null == 1
def test_4_first_turn_gap_imputation() -> None:
    turn = _make_dummy_turn("t0", preceding_gap_s=None)
    feats = extract_turn_dynamics(turn, turn_idx=0, n_turns=5)
    assert feats["preceding_gap_s"] == 0.0
    assert feats["gap_is_null"] == 1.0


# 5. Negative preceding_gap_s survives into the feature vector with its sign intact
def test_5_negative_gap_preserves_sign() -> None:
    turn = _make_dummy_turn("t1", preceding_gap_s=-1.25)
    feats = extract_turn_dynamics(turn, turn_idx=1, n_turns=5)
    assert feats["preceding_gap_s"] == -1.25
    assert feats["gap_is_null"] == 0.0

    arr = extract_features_array([turn])
    gap_idx = DYNAMICS_FEATURE_NAMES.index("preceding_gap_s")
    assert arr[0, gap_idx] == -1.25


# 6. Feature order mismatch between persisted model and runtime raises
def test_6_feature_order_mismatch_raises() -> None:
    turn = _make_dummy_turn()
    mismatched_order = tuple(reversed(DYNAMICS_FEATURE_NAMES))
    with pytest.raises(ValueError, match="Feature order mismatch"):
        extract_features_array([turn], feature_names=mismatched_order)


# 10. validate_timeline() rejects a turn with both semantic and semantic_abstain_reason non-null, and one with both null
def test_10_validate_timeline_semantic_mutual_exclusion() -> None:
    t = _make_dummy_turn()

    # Case A: Both non-null in 1.1.0 -> raises TimelineValidationError
    t.semantic = SemanticFeatures(
        probs={"negative": 0.1, "neutral": 0.2, "positive": 0.7},
        label="positive",
        conf=0.7,
        calibrated=True,
        model_revision="stub_v1",
    )
    t.semantic_abstain_reason = "NO_TEXT"
    tl_both = _make_dummy_timeline(schema_version="1.1.0", turns=[t])
    with pytest.raises(TimelineValidationError, match="both 'semantic' and 'semantic_abstain_reason' non-null"):
        validate_timeline(tl_both)

    # Case B: Both null in 1.1.0 -> raises TimelineValidationError
    t.semantic = None
    t.semantic_abstain_reason = None
    tl_neither = _make_dummy_timeline(schema_version="1.1.0", turns=[t])
    with pytest.raises(TimelineValidationError, match="both 'semantic' and 'semantic_abstain_reason' null"):
        validate_timeline(tl_neither)

    # Case C: Exactly one non-null -> validates cleanly
    t.semantic = SemanticFeatures(
        probs={"negative": 0.1, "neutral": 0.2, "positive": 0.7},
        label="positive",
        conf=0.7,
        calibrated=True,
        model_revision="stub_v1",
    )
    t.semantic_abstain_reason = None
    tl_valid = _make_dummy_timeline(schema_version="1.1.0", turns=[t])
    validate_timeline(tl_valid)

    # Case D: Abstained turn -> validates cleanly
    t.semantic = None
    t.semantic_abstain_reason = "LOW_SEM_CONF"
    tl_abstained = _make_dummy_timeline(schema_version="1.1.0", turns=[t])
    validate_timeline(tl_abstained)


# 11. A 1.0.0 timeline loads without error; Stage 5 writes 1.1.0
def test_11_backward_compatibility_1_0_0() -> None:
    # 1.0.0 timeline with semantic null loads cleanly and validates
    t = _make_dummy_turn()
    t.semantic = None
    t.semantic_abstain_reason = None
    tl_100 = _make_dummy_timeline(schema_version="1.0.0", turns=[t])

    # Must validate cleanly without error
    validate_timeline(tl_100)

    # Converting to dict and back
    data = tl_100.to_dict()
    loaded = ConversationTimeline.from_dict(data)
    assert loaded.schema_version == "1.0.0"
    assert loaded.turns[0].semantic is None
    assert loaded.turns[0].semantic_abstain_reason is None


# 12. probs keys exactly equal classifier.labels
def test_12_probs_keys_match_classifier_labels() -> None:
    labels = ("negative", "neutral", "positive")
    t = _make_dummy_turn()
    t.semantic = SemanticFeatures(
        probs={"negative": 0.2, "neutral": 0.5, "positive": 0.3},
        label="neutral",
        conf=0.5,
        calibrated=True,
        model_revision="stub_v1",
    )
    t.semantic_abstain_reason = None
    tl = _make_dummy_timeline(schema_version="1.1.0", turns=[t])
    validate_timeline(tl)

    assert tuple(t.semantic.probs.keys()) == labels


# 8. Abstention: empty text -> NO_TEXT; low asr_avg_logprob -> LOW_ASR_CONFIDENCE; low max-prob -> LOW_SEM_CONF
def test_8_abstention_individual_gates() -> None:
    from convaudio.stage5_semantic.gates import check_turn_abstention

    # Case A: Empty text
    t_empty = _make_dummy_turn(text="")
    assert check_turn_abstention(t_empty, probs=np.array([0.1, 0.8, 0.1])) == "NO_TEXT"

    # Case B: Low asr_avg_logprob (< -1.0)
    t_low_asr = _make_dummy_turn(text="clear text here", asr_avg_logprob=-1.5)
    assert check_turn_abstention(t_low_asr, probs=np.array([0.1, 0.8, 0.1])) == "LOW_ASR_CONFIDENCE"

    # Case C: Low max prob (< 0.45)
    t_low_conf = _make_dummy_turn(text="clear text here", asr_avg_logprob=-0.2)
    low_probs = np.array([0.34, 0.33, 0.33])
    assert check_turn_abstention(t_low_conf, probs=low_probs) == "LOW_SEM_CONF"

    # Case D: Valid turn passing all gates
    good_probs = np.array([0.1, 0.8, 0.1])
    assert check_turn_abstention(t_low_conf, probs=good_probs) is None


# 9. Abstention precedence: a turn failing multiple gates reports the FIRST in documented order
def test_9_abstention_precedence_order() -> None:
    from convaudio.stage5_semantic.gates import check_turn_abstention

    # Fails both NO_TEXT and LOW_ASR_CONFIDENCE -> must return NO_TEXT (first)
    t_multi_1 = _make_dummy_turn(text="", asr_avg_logprob=-2.0)
    assert check_turn_abstention(t_multi_1, probs=np.array([0.3, 0.3, 0.4])) == "NO_TEXT"

    # Fails both LOW_ASR_CONFIDENCE and LOW_SEM_CONF -> must return LOW_ASR_CONFIDENCE (second)
    t_multi_2 = _make_dummy_turn(text="some text", asr_avg_logprob=-2.5)
    assert check_turn_abstention(t_multi_2, probs=np.array([0.3, 0.3, 0.4])) == "LOW_ASR_CONFIDENCE"


# 13. assert_licence_allowed raises on a CC-BY-NC classifier
def test_13_licence_rejection_for_nc() -> None:
    from convaudio.errors import LicenceViolation
    from convaudio.licences import assert_licence_allowed

    # Permissive code with NC weights must raise LicenceViolation
    with pytest.raises(LicenceViolation, match="violates licence policy"):
        assert_licence_allowed("MIT", "CC-BY-NC-4.0")

    # Non-commercial code must also raise LicenceViolation
    with pytest.raises(LicenceViolation, match="violates licence policy"):
        assert_licence_allowed("CC-BY-NC-4.0", "MIT")

