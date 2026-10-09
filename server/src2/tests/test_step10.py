"""Tests for Step 10: WavLM encoder, corpus-licence checking, and encoder swapping."""

from pathlib import Path

import numpy as np
import pytest

from convaudio.config import PipelineConfig
from convaudio.errors import LicenceViolation
from convaudio.pipeline import PipelineRunner
from convaudio.stage4_acoustic.encoder_wavlm import WavLMValenceArousalEncoder
from convaudio.stage4_acoustic.interfaces import FrameEmotionEncoder
from convaudio.stage4_acoustic.train_head import train, verify_corpus_licence
from tests.fixtures.synth import generate_clean_stereo_call


def test_wavlm_encoder_protocol_and_licence() -> None:
    """Verify WavLM encoder satisfies FrameEmotionEncoder Protocol and enforces licence policy."""
    encoder = WavLMValenceArousalEncoder()
    assert isinstance(encoder, FrameEmotionEncoder)
    assert encoder.licence_code == "MIT"
    assert encoder.licence_weights == "MIT"

    # Test encoding
    wav = np.zeros(16000, dtype=np.float32)
    emotions = encoder.encode(wav, sr=16000)
    assert emotions.shape == (50, 2)
    assert emotions.dtype == np.float32


def test_acceptance_criterion_4_swap_encoder_via_config(tmp_path: Path) -> None:
    """Acceptance criterion 4: Swapping FrameEmotionEncoder requires changing ONE config value (stage4.encoder) and touching no other file."""
    wav_path = tmp_path / "call.wav"
    generate_clean_stereo_call(wav_path, duration_s=4.0, sr=16000)

    # 1. Run with stub encoder
    cfg_stub = PipelineConfig()
    cfg_stub.stage4.encoder = "stub"
    runner_stub = PipelineRunner(config=cfg_stub, stub_models=True)
    tl_stub = runner_stub.run(wav_path, run_dir=tmp_path / "run_stub")
    assert tl_stub.diagnostics.model_revisions.get("emotion_encoder") is not None

    # 2. Run with wavlm encoder (touching only config value!)
    cfg_wavlm = PipelineConfig()
    cfg_wavlm.stage4.encoder = "wavlm"
    runner_wavlm = PipelineRunner(config=cfg_wavlm, stub_models=False)
    tl_wavlm = runner_wavlm.run(wav_path, run_dir=tmp_path / "run_wavlm", force=True)
    assert tl_wavlm.diagnostics.model_revisions.get("emotion_encoder") == "efa81aa"


def test_train_head_corpus_licence_enforcement(tmp_path: Path) -> None:
    """train_head rejects disallowed corpus licence and trains when allowlisted."""
    # 1. Denied corpus (NC)
    bad_manifest = tmp_path / "bad_corpus.yaml"
    bad_manifest.write_text(
        """
corpus_name: "non_commercial_affect_dataset"
corpus_licence: "CC-BY-NC-4.0"
code_licence: "MIT"
        """.strip(),
        encoding="utf-8",
    )
    with pytest.raises(LicenceViolation):
        verify_corpus_licence(bad_manifest)

    # 2. Allowed corpus (MIT or CC-BY-4.0)
    good_manifest = tmp_path / "good_corpus.yaml"
    good_manifest.write_text(
        """
corpus_name: "commercial_allowed_affect_dataset"
corpus_licence: "CC-BY-4.0"
code_licence: "MIT"
        """.strip(),
        encoding="utf-8",
    )
    manifest_data = verify_corpus_licence(good_manifest)
    assert manifest_data["corpus_name"] == "commercial_allowed_affect_dataset"

    # Run quick training
    out_weights = tmp_path / "trained_head.pt"
    train(
        manifest_path=good_manifest,
        output_head_path=out_weights,
        epochs=1,
        batch_size=4,
    )
    assert out_weights.exists()
