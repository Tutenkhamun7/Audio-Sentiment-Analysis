"""In-memory Pyannote speaker diarization and overlap interval detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from src3.core.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class OverlapWindow:
    """Pairwise collision interval between two distinct speakers."""

    start: float
    end: float
    speaker_a: str
    speaker_b: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class RawTurn:
    """Speaker segment identified by diarization."""

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Result of in-memory diarization."""

    turns: List[RawTurn]
    speakers: List[str]
    overlaps: List[OverlapWindow]
    raw_annotation: Any


def merge_consecutive_turns(
    turns: List[RawTurn],
    max_gap_s: float = 0.8,
    min_duration_s: float = 0.25,
) -> List[RawTurn]:
    """Merge adjacent turns of the same speaker separated by small pauses, and drop micro-jitter blips."""
    if not turns:
        return []

    merged: list[RawTurn] = []
    current = RawTurn(speaker=turns[0].speaker, start=turns[0].start, end=turns[0].end)

    for nxt in turns[1:]:
        if nxt.speaker == current.speaker and (nxt.start - current.end) <= max_gap_s:
            # Extend current turn across the pause
            current.end = max(current.end, nxt.end)
        else:
            if current.duration >= min_duration_s:
                merged.append(current)
            current = RawTurn(speaker=nxt.speaker, start=nxt.start, end=nxt.end)

    if current.duration >= min_duration_s:
        merged.append(current)

    return merged


def run_diarization(
    mono_waveform: torch.Tensor,
    sr: int = 16000,
    num_speakers: Optional[int] = None,
    min_overlap_duration_s: float = 0.05,
    min_speaker_duration_s: float = 1.0,
    merge_consecutive: bool = True,
    max_merge_gap_s: float = 0.8,
    min_turn_duration_s: float = 0.25,
) -> DiarizationResult:
    """Run Pyannote diarization in RAM on a 1D tensor without disk files."""
    if mono_waveform.ndim > 1:
        mono_waveform = mono_waveform.squeeze(0)

    # 1. Obtain pipeline from registry
    registry = ModelRegistry.get_instance()
    pipeline = registry.get_diarizer()

    # Audio input dict accepted by pyannote
    audio_input = {
        "waveform": mono_waveform.unsqueeze(0).float().cpu(),
        "sample_rate": sr,
    }

    # Parameters
    kwargs: dict[str, Any] = {}
    if num_speakers is not None and num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    annotation = pipeline(audio_input, **kwargs)
    if hasattr(annotation, "speaker_diarization"):
        annotation = annotation.speaker_diarization

    # Extract tracks
    tracks = list(annotation.itertracks(yield_label=True))
    raw_turns: list[RawTurn] = []
    speaker_totals: dict[str, float] = {}

    for seg, _, spk in tracks:
        s, e = float(seg.start), float(seg.end)
        spk_str = str(spk)
        raw_turns.append(RawTurn(speaker=spk_str, start=s, end=e))
        speaker_totals[spk_str] = speaker_totals.get(spk_str, 0.0) + (e - s)

    # Filter phantom speakers with negligible speech duration
    valid_speakers = {
        spk for spk, dur in speaker_totals.items() if dur >= min_speaker_duration_s
    }
    if not valid_speakers:
        valid_speakers = set(speaker_totals.keys())

    filtered_turns = [t for t in raw_turns if t.speaker in valid_speakers]
    filtered_turns.sort(key=lambda t: (t.start, t.end))

    if merge_consecutive:
        filtered_turns = merge_consecutive_turns(
            filtered_turns,
            max_gap_s=max_merge_gap_s,
            min_duration_s=min_turn_duration_s,
        )

    # Detect pairwise overlaps
    overlaps: list[OverlapWindow] = []
    for i in range(len(tracks)):
        seg1, _, spk1 = tracks[i]
        if str(spk1) not in valid_speakers:
            continue
        for j in range(i + 1, len(tracks)):
            seg2, _, spk2 = tracks[j]
            if str(spk2) not in valid_speakers or spk1 == spk2:
                continue

            inter = seg1 & seg2
            if inter and inter.duration >= min_overlap_duration_s:
                overlaps.append(
                    OverlapWindow(
                        start=float(inter.start),
                        end=float(inter.end),
                        speaker_a=str(spk1),
                        speaker_b=str(spk2),
                    )
                )

    overlaps.sort(key=lambda o: (o.start, o.end))

    return DiarizationResult(
        turns=filtered_turns,
        speakers=sorted(list(valid_speakers)),
        overlaps=overlaps,
        raw_annotation=annotation,
    )
