"""Typer CLI interface for convaudio."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from convaudio.errors import TimelineValidationError
from convaudio.licences import check_installed_distributions
from convaudio.stage2_timeline.validate import validate_timeline

app = typer.Typer(
    name="convaudio",
    help="Contact-centre batch audio-analysis pipeline (Stages 1-4)",
    add_completion=False,
)

licences_app = typer.Typer(
    name="licences",
    help="Licence auditing and compliance checks",
    add_completion=False,
)
app.add_typer(licences_app, name="licences")


@licences_app.command("check")
def licences_check(
    manifest_file: Optional[Path] = typer.Option(
        None, "--manifest", help="Optional extra declared weights/models manifest YAML/JSON"
    )
) -> None:
    """Scan installed distributions and declared weights manifest for licence violations."""
    extra_manifest = None
    if manifest_file and manifest_file.exists():
        import yaml

        with open(manifest_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            if isinstance(raw, list):
                extra_manifest = raw

    is_clean, violations = check_installed_distributions(extra_manifest=extra_manifest)
    if not is_clean:
        typer.secho("Licence policy violations detected:", fg=typer.colors.RED, bold=True)
        for v in violations:
            typer.echo(f"  - {v}")
        raise typer.Exit(code=1)

    typer.secho("All installed distributions and weights conform to licence policy.", fg=typer.colors.GREEN)


@app.command("validate")
def validate(
    timeline_path: Path = typer.Argument(..., help="Path to timeline.json to validate")
) -> None:
    """Validate a timeline.json file against JSON schema and domain invariants."""
    try:
        validate_timeline(timeline_path)
        typer.secho(f"Timeline at {timeline_path} is valid.", fg=typer.colors.GREEN)
    except (TimelineValidationError, FileNotFoundError) as e:
        typer.secho(f"Timeline validation failed: {e}", fg=typer.colors.RED, bold=True)
        raise typer.Exit(code=1)


@app.command("run")
def run(
    audio_path: Path = typer.Argument(..., help="Path to input audio file (mono or stereo)"),
    out: Path = typer.Option(..., "--out", help="Output directory for run artefacts"),
    num_speakers: Optional[int] = typer.Option(None, "--num-speakers", help="Expected speaker count"),
    stub_models: bool = typer.Option(False, "--stub-models", help="Use stub models for fast, offline execution"),
    force: bool = typer.Option(False, "--force", help="Force re-run stages even if cached"),
    force_branch: Optional[str] = typer.Option(None, "--force-branch", help="Force branch: 'stereo', 'mono', or 'no_split'"),
    split_by_speaker: bool = typer.Option(
        True,
        "--split-by-speaker/--no-split-by-speaker",
        help="Option to split or not split audio based on speakers. When disabled, audio is processed as a unified single speaker stream.",
    ),
    encoder: Optional[str] = typer.Option(None, "--encoder", help="Stage 4 affect encoder: 'emotion2vec', 'wavlm' or 'stub'"),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Optional path to config YAML file"),
) -> None:
    """Run full batch audio-analysis pipeline (Stages 1-4)."""
    from convaudio.config import load_config
    from convaudio.pipeline import PipelineRunner

    cfg = load_config(config_path)
    if not split_by_speaker:
        cfg.stage1.split_by_speaker = False
    if encoder:
        cfg.stage4.encoder = encoder

    runner = PipelineRunner(config=cfg, stub_models=stub_models)
    runner.run(
        audio_path=audio_path,
        run_dir=out,
        num_speakers=num_speakers,
        force=force,
        force_branch=force_branch,
    )


@app.command("stage")
def stage(
    stage_num: int = typer.Argument(..., help="Stage number to run (1, 2, 3, 4, or 5)"),
    run_dir: Path = typer.Option(..., "--run-dir", help="Run directory containing stage artefacts"),
    stub_models: bool = typer.Option(False, "--stub-models", help="Use stub models for fast, offline execution"),
    force: bool = typer.Option(False, "--force", help="Force re-run stage even if cached"),
) -> None:
    """Run an individual pipeline stage."""
    from convaudio.pipeline import PipelineRunner

    runner = PipelineRunner(stub_models=stub_models)
    runner.run_stage(stage_num=stage_num, run_dir=run_dir, force=force)


@app.command("report")
def report(
    run_dir: Path = typer.Option(..., "--run-dir", help="Directory containing completed run artefacts")
) -> None:
    """Generate Markdown diagnostic report for a completed run."""
    from convaudio.report.markdown import generate_markdown_report

    timeline_path = run_dir / "timeline.json"
    if not timeline_path.exists():
        typer.secho(f"No timeline.json found in {run_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    report_md = generate_markdown_report(timeline_path)
    report_file = run_dir / "REPORT.md"
    report_file.write_text(report_md, encoding="utf-8")
    typer.echo(report_md)


@app.command("dialogue")
def dialogue(
    target: Path = typer.Argument(..., help="Path to timeline.json or run directory"),
    merge_consecutive: bool = typer.Option(
        False, "--merge-consecutive", "-m", help="Merge consecutive turns from same speaker"
    ),
    timestamps: bool = typer.Option(
        False, "--timestamps", "-t", help="Include start/end timestamps in output"
    ),
    raw_speakers: bool = typer.Option(
        False, "--raw-speakers", help="Use raw speaker IDs instead of Speaker 1/2"
    ),
) -> None:
    """Print dialogue turns in 'Speaker X - Dialogue' format."""
    from convaudio.report.dialogue import print_dialogue

    timeline_path = target / "timeline.json" if target.is_dir() else target
    if not timeline_path.exists():
        typer.secho(f"Timeline not found: {timeline_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    print_dialogue(
        timeline_path,
        humanize_speakers=not raw_speakers,
        merge_consecutive=merge_consecutive,
        include_timestamps=timestamps,
    )


@app.command("table")
def table_cmd(
    target: Path = typer.Argument(..., help="Path to timeline.json or run directory"),
    all_turns: bool = typer.Option(False, "--all", "-a", help="Include empty non-speech turns"),
    show_probs: bool = typer.Option(True, "--probs/--no-probs", help="Show class probability breakdown"),
    show_acoustic: bool = typer.Option(False, "--acoustic", help="Include Stage 4 acoustic column"),
    max_width: int = typer.Option(100, "--max-width", "-w", help="Maximum width for text column"),
) -> None:
    """Display turns with timestamps, speakers, transcript text, and Stage 5 sentiment in a table."""
    from convaudio.report.table import print_sentiment_table

    print_sentiment_table(
        timeline_or_path=target,
        include_empty=all_turns,
        show_probs=show_probs,
        show_acoustic=show_acoustic,
        max_text_width=max_width,
    )


@app.command("export-annotation")
def export_annotation(
    runs: Path = typer.Option(..., "--runs", help="Directory containing run directories or timeline.json files"),
    n: int = typer.Option(300, "--n", help="Number of turns to sample"),
    out: Path = typer.Option(..., "--out", help="Output JSONL file path"),
) -> None:
    """Sample candidate turns for labelling using stratified informativeness weighting."""
    from convaudio.stage5_semantic.export_annotation import export_annotation_sample

    count = export_annotation_sample(runs_dir=runs, out_file=out, n_samples=n)
    typer.secho(f"Exported {count} stratified turns to {out}", fg=typer.colors.GREEN)


@app.command("agreement")
def agreement(
    a: Path = typer.Option(..., "--a", help="Path to first annotator JSONL file"),
    b: Path = typer.Option(..., "--b", help="Path to second annotator JSONL file"),
) -> None:
    """Compute Cohen's kappa agreement between two annotation files."""
    from convaudio.stage5_semantic.agreement import compute_file_agreement

    res = compute_file_agreement(a, b)
    typer.secho(f"Matched turns: {res['n_items']}", bold=True)
    typer.secho(f"Observed agreement: {res['observed_agreement']:.3f}")
    typer.secho(f"Cohen's kappa: {res['kappa']:.3f}", fg=typer.colors.GREEN if res['kappa'] >= 0.7 else typer.colors.YELLOW, bold=True)


@app.command("weak-labels")
def weak_labels_cmd(
    crm: Path = typer.Option(..., "--crm", help="Path to CRM outcomes CSV or JSON"),
    runs: Path = typer.Option(..., "--runs", help="Runs directory containing pipeline runs"),
    out: Path = typer.Option(..., "--out", help="Output weak_labels.jsonl file"),
    weight: float = typer.Option(0.3, "--weight", help="Weight for weak labels (default: 0.3)"),
) -> None:
    """Generate weak labels from a CRM outcome manifest to bootstrap training."""
    from convaudio.stage5_semantic.weak_labels import generate_weak_labels

    count = generate_weak_labels(crm_file=crm, runs_dir=runs, out_file=out, default_weight=weight)
    typer.secho(f"Generated {count} weak labels in {out}", fg=typer.colors.GREEN)


@app.command("train-semantic")
def train_semantic_cmd(
    labels: Path = typer.Option(..., "--labels", help="Path to ground truth labels.jsonl"),
    out: Path = typer.Option(..., "--out", help="Output directory for model artefacts"),
    weak_labels: Optional[Path] = typer.Option(None, "--weak-labels", help="Optional weak labels JSONL"),
    manifest: Optional[Path] = typer.Option(None, "--manifest", help="Optional dataset manifest with corpus licence"),
) -> None:
    """Train SetFit semantic classifier head with call-level splitting and stratified reporting."""
    from convaudio.stage5_semantic.train import train_semantic_model

    metrics = train_semantic_model(
        labels_file=labels,
        out_dir=out,
        weak_labels_file=weak_labels,
        manifest_file=manifest,
    )
    typer.secho(f"Trained semantic classifier saved to {out}", fg=typer.colors.GREEN)
    typer.echo(f"Metrics: {metrics}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

