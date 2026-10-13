"""Comprehensive test suite for app service."""

from __future__ import annotations

import io
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf
import torch

from app.api.app import app
from app.core.audio import crossfade_splice, load_audio_from_bytes, slice_waveform
from app.core.config import Settings
from app.engine.branch import analyze_channels, decide_branch
from app.engine.overlap_separator import extract_acoustic_profile


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_granular_device_settings():
    """Verify that every component can have its device independently swapped."""
    cfg = Settings(
        default_device="cpu",
        diarization_device="cuda:0" if torch.cuda.is_available() else "cpu",
        asr_device="cpu",
        acoustic_device="cpu",
    )

    assert cfg.get_device("asr") == "cpu"
    assert cfg.get_device("acoustic") == "cpu"
    # Unset component falls back to default_device
    assert cfg.get_device("alignment") == "cpu"
    if torch.cuda.is_available():
        assert cfg.get_device("diarization") == "cuda:0"


def test_in_memory_audio_loading():
    """Verify decoding raw audio bytes directly into float32 tensors."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)

    buf = io.BytesIO()
    sf.write(buf, sine, sr, format="WAV")
    raw_bytes = buf.getvalue()

    waveform, orig_sr, dur = load_audio_from_bytes(raw_bytes, target_sr=16000)
    assert orig_sr == 16000
    assert pytest.approx(dur, 0.01) == 1.0
    assert waveform.shape == (1, 16000)
    assert waveform.dtype == torch.float32


def test_in_memory_slicing_and_crossfade():
    """Verify slicing and boundary crossfading in RAM."""
    wav = torch.ones(1000)
    sliced = slice_waveform(wav, 0.01, 0.03, sr=1000)
    assert sliced.shape[-1] == 20

    base = torch.zeros(500)
    patch = torch.ones(200)
    crossfade_splice(base, patch, 100, 300, fade_len=10)
    assert base[95] == 0.0
    assert base[150] == 1.0
    assert base[305] == 0.0


def test_stereo_branch_detection():
    """Verify clean stereo detection for call-center channels."""
    # Channel 0: Speaker A
    # Channel 1: Speaker B (independent)
    ch0 = torch.randn(16000)
    ch1 = torch.randn(16000)
    stereo_wav = torch.stack([ch0, ch1], dim=0)

    is_stereo, corr, ratio = analyze_channels(stereo_wav, corr_threshold=0.6)
    assert is_stereo is True
    assert corr < 0.2

    branch, _ = decide_branch(stereo_wav)
    assert branch == "stereo_split"

    # Mono test
    mono_wav = ch0.unsqueeze(0)
    branch_mono, _ = decide_branch(mono_wav)
    assert branch_mono == "mono_separated"


def test_extract_acoustic_profile():
    """Verify timbral extraction from audio slice."""
    sr = 16000
    t = torch.linspace(0, 0.5, sr // 2)
    tone = torch.sin(2 * np.pi * 500 * t)
    prof = extract_acoustic_profile(tone, sr)
    assert prof.shape == (13,)
    assert pytest.approx(torch.linalg.norm(prof).item(), 0.01) == 1.0


def test_health_check_endpoint(client: TestClient):
    """Verify the /api/v1/health endpoint returns status and device routing."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "device_routing" in data
    assert "diarization" in data["device_routing"]
    assert "asr" in data["device_routing"]


def test_analyze_endpoint_no_split(client: TestClient):
    """Verify the /api/v1/analyze endpoint on a synthetic WAV in no_split mode."""
    sr = 16000
    dur = 2.0
    t = np.linspace(0, dur, int(dur * sr), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 400 * t)

    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV")
    buf.seek(0)

    # Use no_split and disable neural models to test fast API roundtrip
    files = {"file": ("sample.wav", buf, "audio/wav")}
    data = {
        "force_branch": "no_split",
        "enable_overlap_separation": False,
        "align_words": False,
        "predict_emotion": False,
    }

    resp = client.post("/api/v1/analyze", files=files, data=data)
    assert resp.status_code == 200
    res = resp.json()

    assert "call_id" in res
    assert res["metrics"]["branch_used"] == "no_split"
    assert res["metrics"]["speaker_count"] == 1
    assert len(res["turns"]) >= 1


def test_semantic_sentiment_analysis():
    """Verify semantic emotion classification, context conditioning, and batching."""
    from app.engine.semantic import analyze_text_affect, analyze_text_sentiment, analyze_texts_batch

    lbl, score = analyze_text_sentiment("Thank you so much, this is great!")
    assert lbl in ("happy", "positive")
    assert score > 0.5

    lbl_neg, score_neg = analyze_text_sentiment("This is a terrible delay and problem.")
    assert lbl_neg in ("angry", "sad", "fearful", "disgusted", "negative")
    assert score_neg > 0.5

    # Full distribution scores
    dominant, conf, scores = analyze_text_affect("Thank you so much, this is great!")
    assert dominant in ("happy", "positive")
    assert "happy" in scores or "neutral" in scores

    batch = analyze_texts_batch(
        ["Thank you!", "Horrible delay.", "The cab is outside."],
        prior_texts=[None, "We missed our flight.", None],
    )
    assert len(batch) == 3
    assert batch[0][0] in ("happy", "positive")
    assert batch[1][0] in ("angry", "sad", "fearful", "disgusted", "negative")


def test_turn_sentiment_schema():
    """Verify SentimentAnalysis integrates cleanly in TurnDetail with scores and sarcasm."""
    from app.schemas.response import EmotionAnalysis, SentimentAnalysis, TurnDetail

    td = TurnDetail(
        turn_id="turn_001",
        speaker="SPEAKER_00",
        start=0.0,
        end=2.0,
        duration=2.0,
        text="Oh fantastic, my cab is 40 minutes late.",
        emotion=EmotionAnalysis(
            dominant_emotion="angry",
            is_sarcastic=True,
            incongruence_score=0.82,
        ),
        sentiment=SentimentAnalysis(label="happy", score=0.88, scores={"happy": 0.88, "neutral": 0.12}),
        is_sarcastic=True,
    )
    assert td.sentiment is not None
    assert td.sentiment.label == "happy"
    assert td.sentiment.scores["happy"] == 0.88
    assert td.is_sarcastic is True
    assert td.emotion.is_sarcastic is True
    assert td.emotion.incongruence_score == 0.82


def test_multilingual_turn_segmentation():
    """Verify pause splitting, abbreviation protection, and multilingual CJK/Indic terminal punctuation."""
    from app.engine.segmenter import group_words_into_turns
    from app.schemas.response import WordTiming

    test_words = [
        # Sentence 1 with Dr. abbreviation (should NOT split on Dr.)
        WordTiming(word="Hello", start=1.0, end=1.3),
        WordTiming(word="Dr.", start=1.4, end=1.7),
        WordTiming(word="Smith,", start=1.8, end=2.2),
        WordTiming(word="welcome.", start=2.3, end=2.8),
        # 1.5s natural pause -> should trigger turn split
        WordTiming(word="I", start=4.3, end=4.5),
        WordTiming(word="have", start=4.6, end=4.8),
        WordTiming(word="your", start=4.9, end=5.1),
        WordTiming(word="ticket.", start=5.2, end=5.5),
        # Multilingual CJK terminal punctuation (。)
        WordTiming(word="非常感谢您。", start=5.6, end=6.2),
    ]

    turns = group_words_into_turns(test_words, speaker="SPEAKER_00", max_pause_s=1.2)
    assert len(turns) == 2
    assert "Dr. Smith, welcome." in turns[0].text
    assert turns[0].start == 1.0
    assert turns[0].end == 2.8
    assert turns[1].start == 4.3
    assert "非常感谢您。" in turns[1].text


def test_backchannel_classification():
    """Verify acoustic backchannels (<0.8s during other speaker's turn) are correctly tagged."""
    from app.engine.diarizer import RawTurn
    from app.engine.segmenter import assemble_interleaved_dialogue

    t1 = RawTurn(speaker="SPEAKER_00", start=1.0, end=4.0, text="I was wondering about the ticket")
    # Quick backchannel "Yeah" while SPEAKER_00 is talking
    t_bc = RawTurn(speaker="SPEAKER_01", start=2.0, end=2.4, text="Yeah")
    t2 = RawTurn(speaker="SPEAKER_00", start=4.2, end=6.0, text="and if it is confirmed yet.")

    interleaved = assemble_interleaved_dialogue(
        {"SPEAKER_00": [t1, t2], "SPEAKER_01": [t_bc]},
        backchannel_duration_s=0.8,
    )

    assert len(interleaved) == 3
    assert interleaved[1].speaker == "SPEAKER_01"
    assert interleaved[1].turn_type == "backchannel"
    assert interleaved[0].turn_type == "speech"
    assert interleaved[2].turn_type == "speech"


def test_monologue_duration_capping():
    """Verify long run-on monologues without pauses are split at duration caps."""
    from app.engine.segmenter import group_words_into_turns
    from app.schemas.response import WordTiming

    # 12-second continuous speech with no pauses > 0.2s
    words = [
        WordTiming(word=f"word_{i}", start=round(i * 0.5, 2), end=round((i * 0.5) + 0.3, 2))
        for i in range(24)
    ]

    turns = group_words_into_turns(
        words,
        speaker="SPEAKER_00",
        max_pause_s=1.2,
        soft_turn_duration_s=4.0,
        max_turn_duration_s=6.0,
    )

    # Monologue should be capped and split into at least 2 turns
    assert len(turns) >= 2
    for t in turns:
        assert t.duration <= 7.0


def test_context_conditioned_sarcasm_and_affect():
    """Verify that prior context conditions text affect and detects modal incongruence."""
    from app.engine.semantic import format_context_prompt, analyze_texts_batch

    # 1. Verify prompt formatting within 512 token budget
    prompt = format_context_prompt("Oh, fantastic!", prior_text="We have cancelled your flight.")
    assert "Context: We have cancelled your flight." in prompt
    assert "Utterance: Oh, fantastic!" in prompt

    # 2. Compare standalone vs context-conditioned affect
    batch = analyze_texts_batch(
        ["Thank you so much, this is wonderful!", "Thank you so much, this is wonderful!"],
        prior_texts=[None, "You completely ruined my day."],
    )
    # Standalone praise yields happy / joy
    assert batch[0][0] in ("happy", "positive")
    # Contextualized with negative situation drops joy and activates negative affect
    scores_context = batch[1][2]
    assert scores_context.get("happy", 0.0) < batch[0][2].get("happy", 1.0)


def test_whisperx_word_speaker_assignment():
    """Verify that continuous ASR words are mapped to speakers via temporal intersection and floor inertia."""
    from app.engine.aligner import assign_words_to_speakers
    from app.engine.diarizer import RawTurn
    from app.schemas.response import WordTiming

    # Diarization intervals:
    # SPEAKER_00 speaks [0.0 - 2.5]
    # SPEAKER_01 speaks [2.8 - 4.5]
    diar_turns = [
        RawTurn(speaker="SPEAKER_00", start=0.0, end=2.5),
        RawTurn(speaker="SPEAKER_01", start=2.8, end=4.5),
    ]

    words = [
        WordTiming(word="Tell", start=0.1, end=0.4),
        WordTiming(word="me", start=0.45, end=0.7),
        WordTiming(word="about", start=0.75, end=1.1),
        WordTiming(word="Harrods.", start=1.15, end=1.8),
        # Response by child after 1.1s gap
        WordTiming(word="It", start=2.9, end=3.1),
        WordTiming(word="was", start=3.15, end=3.35),
        WordTiming(word="fantastic.", start=3.4, end=4.1),
    ]

    assigned = assign_words_to_speakers(words, diar_turns)

    assert len(assigned) == 7
    # First 4 words belong to SPEAKER_00
    for w in assigned[:4]:
        assert w.speaker == "SPEAKER_00"
    # Next 3 words belong to SPEAKER_01
    for w in assigned[4:]:
        assert w.speaker == "SPEAKER_01"


def test_group_aligned_words_into_turns_dialogue():
    """Verify that mapped words are segmented into discrete conversation turns across speakers."""
    from app.engine.segmenter import group_aligned_words_into_turns
    from app.schemas.response import WordTiming

    words = [
        WordTiming(word="Tell", start=0.1, end=0.4, speaker="SPEAKER_00"),
        WordTiming(word="me", start=0.45, end=0.7, speaker="SPEAKER_00"),
        WordTiming(word="about", start=0.75, end=1.1, speaker="SPEAKER_00"),
        WordTiming(word="Harrods.", start=1.15, end=1.8, speaker="SPEAKER_00"),
        WordTiming(word="It", start=2.9, end=3.1, speaker="SPEAKER_01"),
        WordTiming(word="was", start=3.15, end=3.35, speaker="SPEAKER_01"),
        WordTiming(word="fantastic.", start=3.4, end=4.1, speaker="SPEAKER_01"),
    ]

    turns = group_aligned_words_into_turns(words)
    assert len(turns) == 2
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].text == "Tell me about Harrods."
    assert turns[1].speaker == "SPEAKER_01"
    assert turns[1].text == "It was fantastic."


def test_acoustic_confidence_gating_and_abstention(monkeypatch):
    """Verify that ambiguous/low-confidence acoustic predictions default to neutral or abstain."""
    from app.engine.acoustic import extract_turn_emotions
    from app.engine.diarizer import RawTurn
    from app.core.model_registry import ModelRegistry

    # Mock pipe that returns ambiguous scores: angry is barely highest, but confidence is low
    class MockPipeAmbiguous:
        def generate(self, *args, **kwargs):
            return [{
                "labels": ["angry", "neutral", "happy"],
                "scores": [0.38, 0.35, 0.27],
            }]

    # Mock pipe that returns extremely low confidence (< 0.35)
    class MockPipeLowConf:
        def generate(self, *args, **kwargs):
            return [{
                "labels": ["angry", "neutral", "happy"],
                "scores": [0.32, 0.31, 0.30],
            }]

    # Mock pipe that returns clear confident emotion
    class MockPipeConfident:
        def generate(self, *args, **kwargs):
            return [{
                "labels": ["angry", "neutral", "happy"],
                "scores": [0.85, 0.10, 0.05],
            }]

    reg = ModelRegistry.get_instance()
    dummy_wav = torch.randn(1, 16000)

    # Case 1: Ambiguous distribution (margin = 0.03 < 0.10, top = 0.38 < 0.50) -> should default to neutral
    monkeypatch.setattr(reg, "get_acoustic_encoder", lambda: MockPipeAmbiguous())
    t1 = RawTurn(speaker="SPEAKER_00", start=0.0, end=1.0)
    extract_turn_emotions(dummy_wav, [t1], sr=16000, min_acoustic_confidence=0.50, min_top_margin=0.10)
    assert t1.emotion.dominant_emotion == "neutral"
    assert t1.emotion.abstained is False

    # Case 2: Extreme low confidence (< 0.35) -> should abstain with LOW_ACOUSTIC_CONF
    monkeypatch.setattr(reg, "get_acoustic_encoder", lambda: MockPipeLowConf())
    t2 = RawTurn(speaker="SPEAKER_00", start=0.0, end=1.0)
    extract_turn_emotions(dummy_wav, [t2], sr=16000, min_acoustic_confidence=0.50, min_top_margin=0.10)
    assert t2.emotion.abstained is True
    assert t2.emotion.abstain_reason == "LOW_ACOUSTIC_CONF"

    # Case 3: High confidence emotion (top = 0.85 >= 0.50, margin = 0.75 >= 0.10) -> preserves angry
    monkeypatch.setattr(reg, "get_acoustic_encoder", lambda: MockPipeConfident())
    t3 = RawTurn(speaker="SPEAKER_00", start=0.0, end=1.0)
    extract_turn_emotions(dummy_wav, [t3], sr=16000, min_acoustic_confidence=0.50, min_top_margin=0.10)
    assert t3.emotion.dominant_emotion == "angry"
    assert t3.emotion.abstained is False


def test_overlap_acoustic_energy_word_attribution():
    """Verify that assign_words_to_speakers disambiguates overlapping words using stream acoustic energy."""
    from app.engine.aligner import assign_words_to_speakers
    from app.engine.diarizer import RawTurn
    from app.schemas.response import WordTiming

    sr = 16000
    # Overlap interval from 2.0 to 3.0 where both speakers are active
    diar_turns = [
        RawTurn(speaker="SPEAKER_00", start=1.0, end=3.0),
        RawTurn(speaker="SPEAKER_01", start=2.0, end=4.0),
    ]

    # Create speaker streams: SPEAKER_00 has high energy at [2.1 - 2.4], SPEAKER_01 has high energy at [2.5 - 2.9]
    st0 = torch.zeros(sr * 5)
    st1 = torch.zeros(sr * 5)
    st0[int(2.1 * sr) : int(2.4 * sr)] = 0.8
    st1[int(2.5 * sr) : int(2.9 * sr)] = 0.8

    words = [
        WordTiming(word="ClearTurn0", start=1.2, end=1.8),
        WordTiming(word="OverlapWordSpk0", start=2.1, end=2.4),
        WordTiming(word="OverlapWordSpk1", start=2.5, end=2.9),
        WordTiming(word="ClearTurn1", start=3.2, end=3.8),
    ]

    assigned = assign_words_to_speakers(
        words=words,
        diarization_turns=diar_turns,
        fallback_speaker="SPEAKER_00",
        speaker_streams={"SPEAKER_00": st0, "SPEAKER_01": st1},
        sr=sr,
    )

    assert assigned[0].speaker == "SPEAKER_00"
    assert assigned[1].speaker == "SPEAKER_00"  # Higher energy in st0
    assert assigned[2].speaker == "SPEAKER_01"  # Higher energy in st1
    assert assigned[3].speaker == "SPEAKER_01"


def test_per_speaker_turn_assembly_with_overlap():
    """Verify that simultaneous overlapping speech from both speakers forms coherent concurrent turns without 1-word shredding."""
    from app.engine.segmenter import assemble_interleaved_dialogue, group_words_into_turns
    from app.schemas.response import WordTiming

    # SPEAKER_00 speaks a complete sentence from 10.0 to 13.0
    spk0_words = [
        WordTiming(word="The", start=10.0, end=10.3, speaker="SPEAKER_00"),
        WordTiming(word="driver's", start=10.4, end=10.9, speaker="SPEAKER_00"),
        WordTiming(word="name", start=11.0, end=11.3, speaker="SPEAKER_00"),
        WordTiming(word="is", start=11.4, end=11.6, speaker="SPEAKER_00"),
        WordTiming(word="Vikram.", start=11.7, end=12.5, speaker="SPEAKER_00"),
    ]

    # SPEAKER_01 interjects simultaneously from 10.8 to 11.8
    spk1_words = [
        WordTiming(word="Yeah,", start=10.8, end=11.2, speaker="SPEAKER_01"),
        WordTiming(word="got", start=11.3, end=11.5, speaker="SPEAKER_01"),
        WordTiming(word="it.", start=11.6, end=11.8, speaker="SPEAKER_01"),
    ]

    # Group words per speaker
    per_speaker = {
        "SPEAKER_00": group_words_into_turns(spk0_words, speaker="SPEAKER_00"),
        "SPEAKER_01": group_words_into_turns(spk1_words, speaker="SPEAKER_01"),
    }

    assert len(per_speaker["SPEAKER_00"]) == 1
    assert per_speaker["SPEAKER_00"][0].text == "The driver's name is Vikram."

    assert len(per_speaker["SPEAKER_01"]) == 1
    assert per_speaker["SPEAKER_01"][0].text == "Yeah, got it."

    # Assemble dialogue
    assembled = assemble_interleaved_dialogue(per_speaker)
    assert len(assembled) == 2
    # SPEAKER_00 starts first
    assert assembled[0].speaker == "SPEAKER_00"
    assert assembled[0].text == "The driver's name is Vikram."
    # SPEAKER_01 overlaps cleanly
    assert assembled[1].speaker == "SPEAKER_01"
    assert assembled[1].text == "Yeah, got it."


def test_per_speaker_consecutive_turn_merging():
    """Verify that adjacent segments of the same speaker across brief pauses merge even when an interjection intervenes."""
    from app.engine.diarizer import RawTurn, merge_consecutive_turns

    turns = [
        RawTurn(speaker="SPEAKER_00", start=1.0, end=3.0),
        RawTurn(speaker="SPEAKER_01", start=2.5, end=3.2),  # Interjection
        RawTurn(speaker="SPEAKER_00", start=3.3, end=5.0),  # SPEAKER_00 continues after 0.3s pause
    ]

    merged = merge_consecutive_turns(turns, max_gap_s=0.8, min_duration_s=0.25)
    # SPEAKER_00 should be merged across the 0.3s pause (from 1.0 to 5.0)
    spk0_turns = [t for t in merged if t.speaker == "SPEAKER_00"]
    assert len(spk0_turns) == 1
    assert spk0_turns[0].start == 1.0
    assert spk0_turns[0].end == 5.0

    # SPEAKER_01 should remain intact
    spk1_turns = [t for t in merged if t.speaker == "SPEAKER_01"]
    assert len(spk1_turns) == 1
    assert spk1_turns[0].start == 2.5
    assert spk1_turns[0].end == 3.2


def test_post_analysis_endpoint():
    """Verify that /api/v1/post-analysis accepts analyze output and returns structured LLM call analysis."""
    from fastapi.testclient import TestClient
    from main import create_app

    client = TestClient(create_app())
    payload = {
        "call_id": "test_call_001",
        "speakers": ["Agent", "Customer"],
        "metrics": {
            "duration_s": 24.5,
            "total_speech_s": 22.0,
            "overlap_s": 1.2,
            "overlap_ratio": 0.05,
            "interruption_count": 1,
            "speaker_count": 2,
            "branch_used": "mono",
        },
        "turns": [
            {
                "turn_id": "turn_001",
                "speaker": "Customer",
                "start": 0.5,
                "end": 4.2,
                "duration": 3.7,
                "text": "My cab was delayed and I am very frustrated.",
                "emotion": {"dominant_emotion": "angry"},
                "sentiment": {"label": "negative"},
            },
            {
                "turn_id": "turn_002",
                "speaker": "Agent",
                "start": 4.5,
                "end": 8.5,
                "duration": 4.0,
                "text": "I apologize for the delay. I have dispatched an express vehicle right away.",
                "emotion": {"dominant_emotion": "neutral"},
                "sentiment": {"label": "positive"},
            },
            {
                "turn_id": "turn_003",
                "speaker": "Customer",
                "start": 9.0,
                "end": 12.0,
                "duration": 3.0,
                "text": "Thank you so much, that solves my issue.",
                "emotion": {"dominant_emotion": "happy"},
                "sentiment": {"label": "positive"},
            },
        ],
    }

    resp = client.post("/api/v1/post-analysis", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["call_id"] == "test_call_001"
    assert "executive_summary" in data
    assert data["call_outcome"] in ["Resolved", "Pending Follow-up", "Escalated", "Inquiry Answered"]
    assert "customer_satisfaction_score" in data
    assert "sentiment_trajectory" in data
    assert "agent_performance" in data
    assert isinstance(data["action_items"], list)
    assert len(data["action_items"]) > 0
    assert "risk_assessment" in data
    assert isinstance(data["key_moments"], list)




