"""Tests for Step 5: Full DAG end-to-end execution and caching (Tests 17, 18)."""

import time
from pathlib import Path

from convaudio.pipeline import PipelineRunner
from convaudio.stage2_timeline.validate import validate_timeline
from tests.fixtures.synth import generate_clean_stereo_call, generate_high_overlap_mono_call


def test_17_full_dag_stub_models_fast_and_valid(tmp_path: Path) -> None:
    """Test 17: full DAG with --stub-models: no GPU, no network, schema-valid output, under 10 seconds."""
    wav_path = tmp_path / "test_call.wav"
    generate_high_overlap_mono_call(wav_path, duration_s=6.0, sr=16000)

    out_dir = tmp_path / "run_e2e"

    runner = PipelineRunner(stub_models=True)
    t0 = time.perf_counter()
    timeline = runner.run(
        audio_path=wav_path,
        run_dir=out_dir,
        num_speakers=2,
    )
    elapsed = time.perf_counter() - t0

    # Must complete in under 10 seconds
    assert elapsed < 10.0, f"Full DAG took {elapsed:.2f}s, exceeding 10s limit"

    # Schema valid output
    timeline_file = out_dir / "timeline.json"
    assert timeline_file.exists()
    validate_timeline(timeline_file)

    # Check key artefact fields
    assert timeline.schema_version in ("1.0.0", "1.1.0")
    assert timeline.frame_rate_hz > 0
    assert timeline.n_frames > 0
    assert len(timeline.turns) > 0

    # Check all turns have valid acoustic / abstain mutually exclusive fields
    for turn in timeline.turns:
        assert (turn.acoustic is None) ^ (turn.acoustic_abstain_reason is None)
        assert turn.word_timings_source in ("forced_align", "asr_fallback")


def test_18_rerunning_skips_completed_stages_and_force_reruns(tmp_path: Path) -> None:
    """Test 18: re-running convaudio run skips completed stages; --force re-runs them."""
    wav_path = tmp_path / "test_call.wav"
    generate_clean_stereo_call(wav_path, duration_s=4.0, sr=16000)
    out_dir = tmp_path / "run_resumable"

    runner = PipelineRunner(stub_models=True)

    # 1. Initial run
    runner.run(audio_path=wav_path, run_dir=out_dir)

    s1_mtime_1 = (out_dir / "stage1.json").stat().st_mtime_ns
    s2_mtime_1 = (out_dir / "stage2.json").stat().st_mtime_ns
    s3_mtime_1 = (out_dir / "stage3.json").stat().st_mtime_ns
    s4_mtime_1 = (out_dir / "stage4.json").stat().st_mtime_ns

    time.sleep(0.05)

    # 2. Re-run without --force (should skip all stages)
    runner.run(audio_path=wav_path, run_dir=out_dir, force=False)

    s1_mtime_2 = (out_dir / "stage1.json").stat().st_mtime_ns
    s2_mtime_2 = (out_dir / "stage2.json").stat().st_mtime_ns
    s3_mtime_2 = (out_dir / "stage3.json").stat().st_mtime_ns
    s4_mtime_2 = (out_dir / "stage4.json").stat().st_mtime_ns

    assert s1_mtime_1 == s1_mtime_2, "Stage 1 was re-executed instead of skipped!"
    assert s2_mtime_1 == s2_mtime_2, "Stage 2 was re-executed instead of skipped!"
    assert s3_mtime_1 == s3_mtime_2, "Stage 3 was re-executed instead of skipped!"
    assert s4_mtime_1 == s4_mtime_2, "Stage 4 was re-executed instead of skipped!"

    time.sleep(0.05)

    # 3. Re-run with --force (should re-execute stages)
    runner.run(audio_path=wav_path, run_dir=out_dir, force=True)

    s1_mtime_3 = (out_dir / "stage1.json").stat().st_mtime_ns
    assert s1_mtime_3 > s1_mtime_2, "--force did not re-execute Stage 1!"
