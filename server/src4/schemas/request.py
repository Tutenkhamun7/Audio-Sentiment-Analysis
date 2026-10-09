"""Request models for src3 analysis endpoints."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class AnalyzeOptions(BaseModel):
    """Optional configuration overrides per analysis request."""

    num_speakers: Optional[int] = Field(
        default=None,
        description="Explicit speaker count if known in advance (e.g. 2 for agent/customer)",
    )
    force_branch: Optional[str] = Field(
        default=None,
        description="Force processing branch: 'stereo', 'mono', or 'no_split'",
    )
    enable_overlap_separation: bool = Field(
        default=True,
        description="Whether to run SepFormer on overlapping speech segments",
    )
    vad_filter: bool = Field(
        default=False,
        description="Whether to filter silence via Silero VAD during transcription",
    )
    align_words: bool = Field(
        default=True,
        description="Whether to run Wav2Vec2 CTC forced alignment for millisecond word boundaries",
    )
    predict_emotion: bool = Field(
        default=True,
        description="Whether to extract acoustic affect and textual sentiment",
    )
    merge_consecutive: bool = Field(
        default=True,
        description="Whether to merge consecutive turns of the same speaker separated by small pauses",
    )
    initial_prompt: Optional[str] = Field(
        default=None,
        description="Optional initial prompt context to guide proper nouns and terminology in Whisper",
    )
