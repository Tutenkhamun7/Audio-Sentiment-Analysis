"""Comprehensive test suite for src3 service."""

from __future__ import annotations

import io
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf
import torch

from src3.api.app import app
from src3.core.audio import crossfade_splice, load_audio_from_bytes, slice_waveform
from src3.core.config import Settings
from src3.engine.branch import analyze_channels, decide_branch
from src3.engine.overlap_separator import extract_acoustic_profile


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
    """Verify semantic sentiment classification and batching."""
    from src3.engine.semantic import analyze_text_sentiment, analyze_texts_batch

    lbl, score = analyze_text_sentiment("Thank you so much, this is great!")
    assert lbl == "positive"
    assert score > 0.5

    lbl_neg, score_neg = analyze_text_sentiment("This is a terrible delay and problem.")
    assert lbl_neg == "negative"
    assert score_neg > 0.5

    batch = analyze_texts_batch(["Thank you!", "Horrible delay.", "The cab is outside."])
    assert len(batch) == 3
    assert batch[0][0] == "positive"
    assert batch[1][0] == "negative"


def test_turn_sentiment_schema():
    """Verify SentimentAnalysis integrates cleanly in TurnDetail."""
    from src3.schemas.response import SentimentAnalysis, TurnDetail

    td = TurnDetail(
        turn_id="turn_001",
        speaker="SPEAKER_00",
        start=0.0,
        end=2.0,
        duration=2.0,
        text="Hello, thank you!",
        sentiment=SentimentAnalysis(label="positive", score=0.98),
    )
    assert td.sentiment is not None
    assert td.sentiment.label == "positive"
    assert td.sentiment.score == 0.98
