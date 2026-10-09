"""Markdown diagnostic report generator for convaudio pipeline runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from convaudio.stage2_timeline.schema import ConversationTimeline


def generate_markdown_report(timeline_or_path: ConversationTimeline | Path | str) -> str:
    """Generate Markdown diagnostic report for a completed run.

    Prominently surfaces corpus overlap_ratio, interruption count, abstention breakdown,
    and mean sep_cosine / asr_avg_logprob split by provenance.
    """
    if isinstance(timeline_or_path, ConversationTimeline):
        tl = timeline_or_path
    else:
        tl = ConversationTimeline.from_json(timeline_or_path)

    stats = tl.corpus_stats
    diag = tl.diagnostics

    # Quality metrics split by provenance
    clean_cosines: list[float] = []
    sep_cosines: list[float] = []
    clean_logprobs: list[float] = []
    sep_logprobs: list[float] = []

    for turn in tl.turns:
        prov = turn.provenance.lower()
        if turn.quality.sep_cosine is not None:
            if prov == "clean":
                clean_cosines.append(turn.quality.sep_cosine)
            else:
                sep_cosines.append(turn.quality.sep_cosine)

        if turn.quality.asr_avg_logprob is not None:
            if prov == "clean":
                clean_logprobs.append(turn.quality.asr_avg_logprob)
            else:
                sep_logprobs.append(turn.quality.asr_avg_logprob)

    mean_clean_cos = float(np.mean(clean_cosines)) if clean_cosines else float("nan")
    mean_sep_cos = float(np.mean(sep_cosines)) if sep_cosines else float("nan")
    mean_clean_lp = float(np.mean(clean_logprobs)) if clean_logprobs else float("nan")
    mean_sep_lp = float(np.mean(sep_logprobs)) if sep_logprobs else float("nan")

    overlap_pct = stats.overlap_ratio * 100.0

    lines = [
        f"# Conversation Audio Analysis Report: {tl.call_id}",
        "",
        f"- **Schema Version**: `{tl.schema_version}`",
        f"- **Created (UTC)**: `{tl.created_utc}`",
        f"- **Branch Taken**: `{tl.audio_meta.branch}`",
        f"- **Input Audio Duration**: `{tl.audio_meta.duration_s:.2f}s` ({tl.audio_meta.channels} channels, original {tl.audio_meta.original_sr}Hz)",
        "",
        "## Key Call Metrics",
        "",
        "> [!IMPORTANT]",
        f"> **Corpus Overlap Ratio**: **`{stats.overlap_ratio:.4f}` ({overlap_pct:.2f}%)**",
        f"> **Interruption Count**: **`{stats.n_interruptions}`**",
        f"> **Total Speech**: `{stats.total_speech_s:.2f}s` (Overlap: `{stats.overlap_speech_s:.2f}s`)",
        f"> **Total Turns**: `{stats.n_turns}`",
        "",
        "## Acoustic Abstention Breakdown",
        "",
        f"- **Total Abstentions**: {diag.acoustic_abstentions} / {len(tl.turns)} turns",
        "",
        "| Abstention Reason | Count | Description |",
        "| :--- | :--- | :--- |",
        f"| `SHORT_TURN` | {diag.abstain_reasons.get('SHORT_TURN', 0)} | Clean speech below threshold |",
        f"| `HIGH_OVERLAP` | {diag.abstain_reasons.get('HIGH_OVERLAP', 0)} | Overlap ratio exceeds threshold |",
        f"| `LOW_SEP_QUALITY` | {diag.abstain_reasons.get('LOW_SEP_QUALITY', 0)} | Separation cosine below threshold |",
        "",
        "## Separation & ASR Quality by Provenance",
        "",
        "| Provenance | Turns | Mean `sep_cosine` | Mean `asr_avg_logprob` |",
        "| :--- | :--- | :--- | :--- |",
        f"| `clean` | {len(clean_cosines) or len(clean_logprobs)} | {mean_clean_cos:.4f} | {mean_clean_lp:.4f} |",
        f"| `separated` | {len(sep_cosines) or len(sep_logprobs)} | {mean_sep_cos:.4f} | {mean_sep_lp:.4f} |",
        "",
        "## Pipeline Diagnostics",
        "",
        f"- **Alignment Failures**: {diag.alignment_failures}",
        f"- **Rejected Phantom Sources**: {len(diag.rejected_sources)}",
    ]

    if diag.rejected_sources:
        lines.append("  - Details:")
        for r in diag.rejected_sources:
            lines.append(f"    - Source {r.get('source_idx')}: reason={r.get('reason')}, speech_s={r.get('speech_s')}")

    lines.extend([
        "",
        "### Stage Execution Durations",
        "",
        "| Stage | Duration (s) |",
        "| :--- | :--- |",
    ])
    for stage_key, dur_s in sorted(diag.stage_durations_s.items()):
        lines.append(f"| `{stage_key}` | {dur_s:.3f}s |")

    if diag.model_revisions:
        lines.extend([
            "",
            "### Loaded Model Revisions",
            "",
            "| Component | Revision SHA / Tag |",
            "| :--- | :--- |",
        ])
        for comp, rev in sorted(diag.model_revisions.items()):
            lines.append(f"| `{comp}` | `{rev}` |")

    lines.append("")
    return "\n".join(lines)
