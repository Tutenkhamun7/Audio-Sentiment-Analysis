"""Tests for Step 6: Markdown reporting and acceptance criterion 3."""

from pathlib import Path

from typer.testing import CliRunner

from convaudio.cli import app
from convaudio.pipeline import PipelineRunner
from convaudio.report.markdown import generate_markdown_report
from tests.fixtures.synth import generate_high_overlap_mono_call


def test_acceptance_criterion_3_report_surfaces_metrics_prominently(tmp_path: Path) -> None:
    """Acceptance criterion 3: report surfaces overlap_ratio prominently, plus interruption count, abstention breakdown, and quality split by provenance."""
    wav_path = tmp_path / "call.wav"
    generate_high_overlap_mono_call(wav_path, duration_s=6.0, sr=16000)
    run_dir = tmp_path / "run_report"

    runner = PipelineRunner(stub_models=True)
    runner.run(audio_path=wav_path, run_dir=run_dir)

    md = generate_markdown_report(run_dir / "timeline.json")

    # Assert metrics surfaced prominently
    assert "Corpus Overlap Ratio" in md
    assert "Interruption Count" in md
    assert "SHORT_TURN" in md
    assert "HIGH_OVERLAP" in md
    assert "LOW_SEP_QUALITY" in md
    assert "Mean `sep_cosine`" in md
    assert "Mean `asr_avg_logprob`" in md
    assert "clean" in md
    assert "separated" in md

    # Assert CLI command runs
    cli_runner = CliRunner()
    res = cli_runner.invoke(app, ["report", "--run-dir", str(run_dir)])
    assert res.exit_code == 0
    assert "Corpus Overlap Ratio" in res.output
