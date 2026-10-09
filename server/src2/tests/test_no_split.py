"""Tests for the option to not split audio based on speakers."""

from pathlib import Path

import torch
from typer.testing import CliRunner

from convaudio.cli import app
from convaudio.config import PipelineConfig
from convaudio.pipeline import PipelineRunner
from convaudio.stage1_separate.branch import decide_branch
from convaudio.stage2_timeline.validate import validate_timeline
from tests.fixtures.synth import generate_clean_stereo_call


def test_decide_branch_force_no_split() -> None:
    """Verify force_branch accepts 'no_split' and aliases."""
    dummy_wave = torch.randn(2, 16000)
    for alias in ("no_split", "single", "none", "nosplit"):
        branch, metrics = decide_branch(dummy_wave, force_branch=alias)
        assert branch == "no_split"
        assert metrics.get("forced") == 1.0


def test_pipeline_no_split_by_speaker(tmp_path: Path) -> None:
    """Verify full DAG execution when split_by_speaker is disabled."""
    wav_path = tmp_path / "call.wav"
    generate_clean_stereo_call(wav_path, duration_s=4.0, sr=16000)

    cfg = PipelineConfig()
    cfg.stage1.split_by_speaker = False

    runner = PipelineRunner(config=cfg, stub_models=True)
    run_dir = tmp_path / "run_no_split"
    timeline = runner.run(wav_path, run_dir=run_dir)

    # 1. Timeline properties
    assert timeline.audio_meta.branch == "no_split"
    assert len(timeline.speakers) == 1
    assert "SPEAKER_00" in timeline.speakers

    # 2. Overlap & interruptions are zero
    assert timeline.corpus_stats.overlap_ratio == 0.0
    assert timeline.corpus_stats.n_interruptions == 0

    # 3. Turns all belong to single speaker and have clean provenance
    assert len(timeline.turns) > 0
    for turn in timeline.turns:
        assert turn.speaker == "SPEAKER_00"
        assert turn.provenance == "clean"
        assert turn.overlap.ratio == 0.0
        assert turn.overlap.interrupts is None
        assert turn.overlap.interrupted_by is None

    # 4. Strict validation succeeds
    timeline_file = run_dir / "timeline.json"
    validate_timeline(timeline_file)


def test_cli_no_split_by_speaker_flag(tmp_path: Path) -> None:
    """Verify CLI accepts --no-split-by-speaker flag."""
    wav_path = tmp_path / "call.wav"
    generate_clean_stereo_call(wav_path, duration_s=3.0, sr=16000)
    out_dir = tmp_path / "run_cli_nosplit"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(wav_path),
            "--out",
            str(out_dir),
            "--no-split-by-speaker",
            "--stub-models",
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "timeline.json").exists()

    timeline_file = out_dir / "timeline.json"
    validate_timeline(timeline_file)
