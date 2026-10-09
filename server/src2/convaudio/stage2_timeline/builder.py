"""Builder to construct canonical ConversationTimeline from Stage 1 outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from convaudio.stage2_timeline.overlap import detect_interruptions, overlap_ratio_for
from convaudio.stage2_timeline.schema import (
    AudioMeta,
    ConversationTimeline,
    CorpusStats,
    Diagnostics,
    SpeakerMeta,
    Turn,
)


def build_timeline(
    call_id: str,
    audio_meta: AudioMeta,
    stage1_artefacts: dict[str, Any],
    interrupt_window_s: float = 1.0,
    created_utc: str | None = None,
) -> ConversationTimeline:
    """Assemble canonical ConversationTimeline from Stage 1 separation and diarization outputs."""
    if created_utc is None:
        created_utc = datetime.now(timezone.utc).isoformat()

    frame_rate_hz = float(stage1_artefacts["frame_rate_hz"])
    overlap_mask_path = stage1_artefacts["overlap_mask_path"]
    mask = np.load(overlap_mask_path)
    n_frames = len(mask)

    # Speakers metadata
    speakers: dict[str, SpeakerMeta] = {}
    total_speech_s = 0.0
    for spk_id, spk_info in stage1_artefacts["speakers"].items():
        sm = SpeakerMeta(
            stream_path=spk_info["stream_path"],
            total_speech_s=float(spk_info["total_speech_s"]),
            embedding_key=spk_info.get("embedding_key", spk_id),
        )
        speakers[spk_id] = sm
        total_speech_s += sm.total_speech_s

    # Overlap speech stats
    overlap_frames = int(np.sum(mask))
    overlap_speech_s = float(overlap_frames / frame_rate_hz) if frame_rate_hz > 0 else 0.0
    overlap_ratio = overlap_speech_s / total_speech_s if total_speech_s > 0 else 0.0

    # Build turns
    turns: list[Turn] = stage1_artefacts["turns"]

    # Re-evaluate turn overlap metrics precisely
    for t in turns:
        ratio, clean_s, _, _ = overlap_ratio_for(
            mask, start=t.start, end=t.end, frame_rate_hz=frame_rate_hz
        )
        t.overlap.ratio = round(ratio, 4)
        t.overlap.clean_speech_s = round(clean_s, 4)

    # Detect interruptions & compute preceding gaps
    n_interruptions = detect_interruptions(turns, interrupt_window_s=interrupt_window_s)

    corpus_stats = CorpusStats(
        total_speech_s=round(total_speech_s, 4),
        overlap_speech_s=round(overlap_speech_s, 4),
        overlap_ratio=round(overlap_ratio, 4),
        n_turns=len(turns),
        n_interruptions=n_interruptions,
    )

    diagnostics = Diagnostics(
        rejected_sources=stage1_artefacts.get("rejected_sources", []),
        alignment_failures=0,
        acoustic_abstentions=0,
        abstain_reasons={"SHORT_TURN": 0, "HIGH_OVERLAP": 0, "LOW_SEP_QUALITY": 0},
        model_revisions={},
        stage_durations_s={},
    )

    return ConversationTimeline(
        schema_version="1.0.0",
        call_id=call_id,
        created_utc=created_utc,
        audio_meta=audio_meta,
        frame_rate_hz=frame_rate_hz,
        n_frames=n_frames,
        overlap_mask=str(Path(overlap_mask_path).resolve()),
        speakers=speakers,
        corpus_stats=corpus_stats,
        turns=turns,
        diagnostics=diagnostics,
        outcome=None,
    )
