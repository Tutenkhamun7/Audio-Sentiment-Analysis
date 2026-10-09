"""Dialogue extraction and formatting utilities from ConversationTimeline."""

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


def get_dialogue_turns(
    timeline_or_path: ConversationTimeline | Path | str | dict[str, Any],
    humanize_speakers: bool = True,
    merge_consecutive: bool = False,
    filter_empty: bool = True,
) -> list[tuple[str, str]]:
    """Extract list of (speaker_label, dialogue_text) tuples from timeline."""
    tl = _load_timeline(timeline_or_path)
    speaker_map: dict[str, str] = {}
    speaker_counter = 1

    extracted: list[tuple[str, str]] = []

    for turn in tl.turns:
        raw_text = (turn.text or "").strip()
        if filter_empty and (not raw_text or raw_text == "..."):
            continue

        raw_speaker = turn.speaker
        if humanize_speakers:
            if raw_speaker not in speaker_map:
                speaker_map[raw_speaker] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            spk_label = speaker_map[raw_speaker]
        else:
            spk_label = raw_speaker

        if merge_consecutive and extracted and extracted[-1][0] == spk_label:
            prev_spk, prev_txt = extracted[-1]
            extracted[-1] = (prev_spk, f"{prev_txt} {raw_text}".strip())
        else:
            extracted.append((spk_label, raw_text))

    return extracted


def format_dialogue(
    timeline_or_path: ConversationTimeline | Path | str | dict[str, Any],
    humanize_speakers: bool = True,
    merge_consecutive: bool = False,
    include_timestamps: bool = False,
    filter_empty: bool = True,
) -> str:
    """Format timeline into a clean dialogue script:

    Speaker 1 - Dialogue
    Speaker 2 - Dialogue
    """
    tl = _load_timeline(timeline_or_path)
    speaker_map: dict[str, str] = {}
    speaker_counter = 1

    lines: list[str] = []
    current_spk: str | None = None
    current_texts: list[str] = []
    current_start: float = 0.0
    current_end: float = 0.0

    def _flush_turn() -> None:
        if current_spk is not None and current_texts:
            merged_text = " ".join(current_texts).strip()
            if merged_text:
                if include_timestamps:
                    lines.append(
                        f"[{current_start:06.2f}s - {current_end:06.2f}s] {current_spk} - {merged_text}"
                    )
                else:
                    lines.append(f"{current_spk} - {merged_text}")

    for turn in tl.turns:
        raw_text = (turn.text or "").strip()
        if filter_empty and (not raw_text or raw_text == "..."):
            continue

        raw_speaker = turn.speaker
        if humanize_speakers:
            if raw_speaker not in speaker_map:
                speaker_map[raw_speaker] = f"Speaker {speaker_counter}"
                speaker_counter += 1
            spk_label = speaker_map[raw_speaker]
        else:
            spk_label = raw_speaker

        if merge_consecutive:
            if current_spk == spk_label:
                current_texts.append(raw_text)
                current_end = turn.end
            else:
                _flush_turn()
                current_spk = spk_label
                current_texts = [raw_text]
                current_start = turn.start
                current_end = turn.end
        else:
            if include_timestamps:
                lines.append(f"[{turn.start:06.2f}s - {turn.end:06.2f}s] {spk_label} - {raw_text}")
            else:
                lines.append(f"{spk_label} - {raw_text}")

    if merge_consecutive:
        _flush_turn()

    return "\n".join(lines)


def print_dialogue(
    timeline_or_path: ConversationTimeline | Path | str | dict[str, Any],
    humanize_speakers: bool = True,
    merge_consecutive: bool = False,
    include_timestamps: bool = False,
    filter_empty: bool = True,
) -> None:
    """Print formatted dialogue directly to stdout."""
    output = format_dialogue(
        timeline_or_path=timeline_or_path,
        humanize_speakers=humanize_speakers,
        merge_consecutive=merge_consecutive,
        include_timestamps=include_timestamps,
        filter_empty=filter_empty,
    )
    print(output)
