"""Stratified annotation turn sampling for human labelling."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from convaudio.stage2_timeline.schema import ConversationTimeline, Turn
from convaudio.stage5_semantic.window import build_context_window


def compute_turn_informativeness_weight(turn: Turn) -> float:
    """Calculate sampling priority weight to oversample informative/challenging turns.

    Prioritises:
    - Interruptions (interrupts or interrupted_by)
    - High overlap ratio (> 0.3)
    - Significant preceding pauses or negative gaps (|preceding_gap_s| > 2.0s)
    - Turns where Stage 4 acoustic classifier abstained
    """
    weight = 1.0

    if turn.overlap.interrupts is not None:
        weight += 3.0
    if turn.overlap.interrupted_by is not None:
        weight += 2.0
    if turn.overlap.ratio > 0.3:
        weight += 2.0
    if turn.overlap.preceding_gap_s is not None and abs(turn.overlap.preceding_gap_s) > 2.0:
        weight += 2.0
    if turn.acoustic_abstain_reason is not None:
        weight += 3.0

    return weight


def export_annotation_sample(
    runs_dir: Path | str,
    out_file: Path | str,
    n_samples: int = 300,
    n_context: int = 3,
    speaker_roles: dict[str, str] | None = None,
    seed: int = 42,
) -> int:
    """Sample candidate turns across runs using stratified informativeness weighting and export to JSONL."""
    rng = random.Random(seed)
    runs_path = Path(runs_dir)

    timeline_paths: list[Path] = []
    if runs_path.is_file() and runs_path.name.endswith(".json"):
        timeline_paths = [runs_path]
    else:
        timeline_paths = sorted(runs_path.rglob("timeline.json"))

    candidates: list[dict[str, Any]] = []

    for tl_path in timeline_paths:
        try:
            timeline = ConversationTimeline.from_json(tl_path)
        except Exception:
            continue

        for idx, turn in enumerate(timeline.turns):
            if turn.text is None or not turn.text.strip():
                continue

            w = compute_turn_informativeness_weight(turn)
            window_str = build_context_window(
                timeline.turns,
                target_idx=idx,
                n_context=n_context,
                speaker_roles=speaker_roles,
            )
            candidates.append({
                "turn_id": turn.turn_id,
                "call_id": timeline.call_id,
                "timeline": str(tl_path.resolve()),
                "speaker": turn.speaker,
                "text": turn.text.strip(),
                "window": window_str,
                "weight": w,
            })

    if not candidates:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text("", encoding="utf-8")
        return 0

    # Stratified sampling without replacement proportional to weights
    if len(candidates) <= n_samples:
        selected = candidates
    else:
        # Weighted sample without replacement using exponential-race algorithm
        indexed_keys = [
            (rng.random() ** (1.0 / c["weight"]), c) for c in candidates
        ]
        indexed_keys.sort(key=lambda x: x[0], reverse=True)
        selected = [item[1] for item in indexed_keys[:n_samples]]

    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for item in selected:
            record = {
                "turn_id": item["turn_id"],
                "call_id": item["call_id"],
                "timeline": item["timeline"],
                "speaker": item["speaker"],
                "text": item["text"],
                "window": item["window"],
                "label": None,
                "annotator": None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(selected)
