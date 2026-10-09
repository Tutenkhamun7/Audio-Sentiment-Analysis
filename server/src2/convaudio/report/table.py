"""Terminal table reporting utility for Turn-level Sentiment (Stage 5 Semantic & Stage 4 Acoustic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from convaudio.stage2_timeline.schema import ConversationTimeline


def _load_timeline(
    timeline_or_path: ConversationTimeline | Path | str | dict[str, Any],
) -> ConversationTimeline:
    """Normalize input into a ConversationTimeline instance."""
    if isinstance(timeline_or_path, ConversationTimeline):
        return timeline_or_path

    if isinstance(timeline_or_path, dict):
        return ConversationTimeline.from_dict(timeline_or_path)

    p = Path(timeline_or_path)
    if p.is_dir():
        p = p / "timeline.json"
    if not p.exists():
        raise FileNotFoundError(f"Timeline file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ConversationTimeline.from_dict(data)


def format_sentiment_cell(turn: Any, show_probs: bool = True) -> str:
    """Format the Stage 5 semantic output column cell."""
    if hasattr(turn, "semantic"):
        sem = turn.semantic
        abs_r = turn.semantic_abstain_reason
    elif isinstance(turn, dict):
        sem = turn.get("semantic")
        abs_r = turn.get("semantic_abstain_reason")
    else:
        sem = None
        abs_r = None

    if sem is not None:
        if isinstance(sem, dict):
            label = str(sem.get("label", "UNKNOWN")).upper()
            conf = float(sem.get("conf", 0.0))
            probs_raw = sem.get("probs", {})
        else:
            label = str(getattr(sem, "label", "UNKNOWN")).upper()
            conf = float(getattr(sem, "conf", 0.0))
            probs_raw = getattr(sem, "probs", {})

        if show_probs and isinstance(probs_raw, dict) and probs_raw:
            probs_abbr = {
                "neg": round(float(probs_raw.get("negative", 0.0)), 2),
                "neu": round(float(probs_raw.get("neutral", 0.0)), 2),
                "pos": round(float(probs_raw.get("positive", 0.0)), 2),
            }
            return f"{label} (conf: {conf:.2f}, probs: {probs_abbr})"
        return f"{label} (conf: {conf:.2f})"
    elif abs_r is not None:
        return f"ABSTAIN: {abs_r}"
    return "PENDING (NULL)"


def format_acoustic_cell(turn: Any) -> str:
    """Format the Stage 4 acoustic output column cell."""
    if hasattr(turn, "acoustic"):
        ac = turn.acoustic
        abs_r = turn.acoustic_abstain_reason
    elif isinstance(turn, dict):
        ac = turn.get("acoustic")
        abs_r = turn.get("acoustic_abstain_reason")
    else:
        ac = None
        abs_r = None

    if ac is not None:
        emo: str | None = None
        if isinstance(ac, dict):
            val = float(ac.get("valence", 0.0))
            aro = float(ac.get("arousal", 0.0))
            conf = float(ac.get("conf", 0.0))
            emo = ac.get("emotion")
        else:
            val = float(getattr(ac, "valence", 0.0))
            aro = float(getattr(ac, "arousal", 0.0))
            conf = float(getattr(ac, "conf", 0.0))
            emo = getattr(ac, "emotion", None)

        if emo:
            return f"{str(emo).upper()} (val: {val:+.2f}, aro: {aro:+.2f}, conf: {conf:.2f})"
        return f"val: {val:+.2f}, aro: {aro:+.2f} (conf: {conf:.2f})"
    elif abs_r is not None:
        return f"ABSTAIN: {abs_r}"
    return "PENDING (NULL)"


def format_sentiment_table(
    timeline_or_path: ConversationTimeline | Path | str | dict[str, Any],
    include_empty: bool = True,
    show_probs: bool = True,
    show_acoustic: bool = False,
    speaker_roles: dict[str, str] | None = None,
    max_text_width: int = 100,
    ascii_only: bool = False,
) -> str:
    """Generate a formatted ASCII/Unicode table of conversation turns and sentiments."""
    tl = _load_timeline(timeline_or_path)

    headers = ["Time Range", "Speaker", "Text", "Stage 5 Semantic Output"]
    if show_acoustic:
        headers.append("Stage 4 Acoustic Output")

    rows: list[list[str]] = []

    for turn in tl.turns:
        text_str = (turn.text or "").strip()
        if not include_empty and not text_str:
            continue

        quoted_text = f'"{text_str}"' if text_str else '""'

        # Text truncation/wrapping if too long for a single line
        if len(quoted_text) > max_text_width:
            quoted_text = quoted_text[: max_text_width - 3] + '..."'

        time_range = f"{turn.start:05.2f}s - {turn.end:05.2f}s"

        spk = turn.speaker
        if speaker_roles and spk in speaker_roles:
            spk_display = speaker_roles[spk]
        else:
            spk_display = spk

        sem_output = format_sentiment_cell(turn, show_probs=show_probs)

        row = [time_range, spk_display, quoted_text, sem_output]
        if show_acoustic:
            row.append(format_acoustic_cell(turn))

        rows.append(row)

    if not rows:
        return "No turns found in timeline."

    # Compute column widths
    n_cols = len(headers)
    widths = [len(headers[i]) for i in range(n_cols)]
    for r in rows:
        for i in range(n_cols):
            widths[i] = max(widths[i], len(r[i]))

    # Add 1 padding space on each side
    col_widths = [w + 2 for w in widths]

    v_sep = "|" if ascii_only else "│"
    h_sep = "-" if ascii_only else "─"
    c_sep = "+" if ascii_only else "┼"

    # Format header
    header_cells = [f" {headers[i]:<{widths[i]}} " for i in range(n_cols)]
    header_line = v_sep.join(header_cells)

    # Format separator
    sep_cells = [h_sep * col_widths[i] for i in range(n_cols)]
    separator_line = c_sep.join(sep_cells)

    lines: list[str] = [header_line, separator_line]

    # Format data rows
    for r in rows:
        row_cells = [f" {r[i]:<{widths[i]}} " for i in range(n_cols)]
        lines.append(v_sep.join(row_cells))

    return "\n".join(lines)


def print_sentiment_table(
    timeline_or_path: ConversationTimeline | Path | str | dict[str, Any],
    include_empty: bool = True,
    show_probs: bool = True,
    show_acoustic: bool = False,
    speaker_roles: dict[str, str] | None = None,
    max_text_width: int = 100,
) -> None:
    """Print the sentiment table directly to stdout with optional rich styling."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        tl = _load_timeline(timeline_or_path)

        table = Table(
            title=f"Conversation Timeline: {tl.call_id} (schema v{tl.schema_version})",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            show_lines=False,
        )

        table.add_column("Time Range", justify="center", style="yellow", no_wrap=True)
        table.add_column("Speaker", justify="left", style="bold magenta", no_wrap=True)
        table.add_column("Text", justify="left")
        table.add_column("Stage 5 Semantic Output", justify="left")
        if show_acoustic:
            table.add_column("Stage 4 Acoustic Output", justify="left")

        for turn in tl.turns:
            text_str = (turn.text or "").strip()
            if not include_empty and not text_str:
                continue

            quoted_text = f'"{text_str}"' if text_str else '""'
            time_range = f"{turn.start:05.2f}s - {turn.end:05.2f}s"

            spk = turn.speaker
            if speaker_roles and spk in speaker_roles:
                spk_display = speaker_roles[spk]
            else:
                spk_display = spk

            sem_cell = format_sentiment_cell(turn, show_probs=show_probs)
            # Rich coloring based on label
            if "POSITIVE" in sem_cell:
                styled_sem = f"[bold green]{sem_cell}[/bold green]"
            elif "NEGATIVE" in sem_cell:
                styled_sem = f"[bold red]{sem_cell}[/bold red]"
            elif "NEUTRAL" in sem_cell:
                styled_sem = f"[bold blue]{sem_cell}[/bold blue]"
            elif "ABSTAIN" in sem_cell:
                styled_sem = f"[dim]{sem_cell}[/dim]"
            else:
                styled_sem = sem_cell

            row_items = [time_range, spk_display, quoted_text, styled_sem]
            if show_acoustic:
                ac_cell = format_acoustic_cell(turn)
                row_items.append(f"[dim]{ac_cell}[/dim]" if "ABSTAIN" in ac_cell else ac_cell)

            table.add_row(*row_items)

        console.print(table)
    except Exception:
        # Fallback to pure string formatting
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except Exception:
                pass
        try:
            print(
                format_sentiment_table(
                    timeline_or_path=timeline_or_path,
                    include_empty=include_empty,
                    show_probs=show_probs,
                    show_acoustic=show_acoustic,
                    speaker_roles=speaker_roles,
                    max_text_width=max_text_width,
                    ascii_only=False,
                )
            )
        except UnicodeEncodeError:
            print(
                format_sentiment_table(
                    timeline_or_path=timeline_or_path,
                    include_empty=include_empty,
                    show_probs=show_probs,
                    show_acoustic=show_acoustic,
                    speaker_roles=speaker_roles,
                    max_text_width=max_text_width,
                    ascii_only=True,
                )
            )
