"""Tests for Emotion2VecEncoder in Stage 4 acoustic affect extraction."""

from pathlib import Path

import numpy as np

from convaudio.config import PipelineConfig
from convaudio.pipeline import PipelineRunner
from convaudio.stage4_acoustic.encoder_emotion2vec import (
    EMOTION_COORDINATES,
    EMOTION_LABELS,
    Emotion2VecEncoder,
)
from convaudio.stage4_acoustic.interfaces import FrameEmotionEncoder
from tests.fixtures.synth import generate_clean_stereo_call


def test_emotion2vec_encoder_protocol_and_licence() -> None:
    """Verify Emotion2VecEncoder satisfies FrameEmotionEncoder Protocol and has approved licence."""
    encoder = Emotion2VecEncoder()
    assert isinstance(encoder, FrameEmotionEncoder)
    assert encoder.licence_code == "Apache-2.0"
    assert encoder.licence_weights == "Apache-2.0"
    assert encoder.frame_rate_hz == 50.0


def test_emotion2vec_encoder_output_shape_and_bounds() -> None:
    """Verify Emotion2VecEncoder produces bounded float32 [valence, arousal] frames."""
    encoder = Emotion2VecEncoder()
    # 2 seconds of audio at 16kHz
    wav = np.zeros(32000, dtype=np.float32)
    emotions = encoder.encode(wav, sr=16000)

    # 2.0s * 50Hz = 100 frames
    assert emotions.shape == (100, 2)
    assert emotions.dtype == np.float32
    # Valence and arousal bounded in [-1, 1]
    assert np.all(emotions >= -1.0)
    assert np.all(emotions <= 1.0)

    label, scores = encoder.get_last_turn_emotion()
    assert label in EMOTION_LABELS
    assert scores is not None
    assert isinstance(scores, dict)


def test_emotion2vec_pipeline_integration(tmp_path: Path) -> None:
    """Verify full pipeline run with stage4.encoder='emotion2vec'."""
    wav_path = tmp_path / "call.wav"
    generate_clean_stereo_call(wav_path, duration_s=4.0, sr=16000)

    cfg = PipelineConfig()
    cfg.stage4.encoder = "emotion2vec"

    # Run with stub_models=True so pipeline uses stubs for stages 1 & 3
    # but verify emotion2vec encoder selection in non-stub stage4 mode
    runner = PipelineRunner(config=cfg, stub_models=True)
    tl = runner.run(wav_path, run_dir=tmp_path / "run_e2v")

    assert tl.diagnostics.model_revisions.get("emotion_encoder") is not None
    assert len(tl.turns) > 0
    # Every turn has valid acoustic or abstain reason
    for turn in tl.turns:
        assert (turn.acoustic is not None) ^ (turn.acoustic_abstain_reason is not None)
