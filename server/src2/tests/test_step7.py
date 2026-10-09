"""Tests for Step 7: Mono separation and phantom source rejection (Tests 2, 12)."""

from pathlib import Path

import numpy as np

from convaudio.pipeline import PipelineRunner
from convaudio.stage1_separate.phantom import filter_phantom_sources
from tests.fixtures.synth import generate_low_overlap_mono_call


def test_2_mono_branch_sets_provenance_separated_and_branch_mono_separated(tmp_path: Path) -> None:
    """Test 2: mono branch sets provenance='separated' and branch='mono_separated'."""
    mono_wav = tmp_path / "mono_call.wav"
    generate_low_overlap_mono_call(mono_wav, duration_s=6.0, sr=16000)
    out_dir = tmp_path / "run_mono"

    runner = PipelineRunner(stub_models=True)
    timeline = runner.run(audio_path=mono_wav, run_dir=out_dir)

    assert timeline.audio_meta.branch == "mono_separated"
    assert len(timeline.turns) > 0
    for turn in timeline.turns:
        assert turn.provenance == "separated"


def test_12_phantom_rejection_drops_subthreshold_source_and_records_diagnostics(tmp_path: Path) -> None:
    """Test 12: phantom rejection drops a sub-threshold source and records it in diagnostics.rejected_sources."""
    # Setup 3 sources:
    # source 0: active 3.0s, normal energy
    # source 1: active 2.0s, normal energy
    # source 2: active 0.3s (sub-threshold, min is 1.0s)
    # source 3: active 2.0s but near-silent energy
    rng = np.random.default_rng(42)
    sr = 16000
    src0 = rng.standard_normal(int(3.0 * sr)).astype(np.float32) * 0.5
    src1 = rng.standard_normal(int(2.0 * sr)).astype(np.float32) * 0.5
    src2 = rng.standard_normal(int(0.3 * sr)).astype(np.float32) * 0.5
    src3 = np.zeros(int(2.0 * sr), dtype=np.float32) + 1e-6  # near-silent

    sources = [src0, src1, src2, src3]
    active_durs = [3.0, 2.0, 0.3, 2.0]

    valid_idx, rejected = filter_phantom_sources(
        sources=sources,
        active_durations_s=active_durs,
        min_speaker_speech_s=1.0,
        silent_rms_threshold=1e-4,
    )

    assert valid_idx == [0, 1]
    assert len(rejected) == 2

    # Check reason for source 2
    r_sub = next(r for r in rejected if r["source_idx"] == 2)
    assert r_sub["reason"] == "below_min_speech"
    assert r_sub["speech_s"] == 0.3

    # Check reason for source 3
    r_silent = next(r for r in rejected if r["source_idx"] == 3)
    assert r_silent["reason"] == "near_silent_energy"
