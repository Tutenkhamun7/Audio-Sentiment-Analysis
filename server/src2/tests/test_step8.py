"""Tests for Step 8: ASR and forced alignment fallback (Test 13)."""

from pathlib import Path

from convaudio.config import PipelineConfig
from convaudio.pipeline import PipelineRunner
from convaudio.stage3_lexical.asr import process_stage3_lexical
from tests.fixtures.synth import generate_clean_stereo_call


def test_13_alignment_failure_falls_back_and_records_diagnostics(tmp_path: Path) -> None:
    """Test 13: alignment failure falls back and sets word_timings_source='asr_fallback' and increments diagnostics.alignment_failures."""
    wav_path = tmp_path / "call.wav"
    generate_clean_stereo_call(wav_path, duration_s=4.0, sr=16000)
    run_dir = tmp_path / "run_align_fail"

    runner = PipelineRunner(stub_models=True)
    # Run stage 1 and 2
    timeline = runner.run(audio_path=wav_path, run_dir=run_dir)

    # Force simulated alignment failure in lexical stage
    config = PipelineConfig()
    process_stage3_lexical(
        timeline=timeline,
        config=config,
        stub_models=True,
        simulate_alignment_failure=True,
    )

    assert timeline.diagnostics.alignment_failures > 0
    for turn in timeline.turns:
        assert turn.word_timings_source == "asr_fallback"
        assert turn.word_timings is not None
        assert len(turn.word_timings) > 0
