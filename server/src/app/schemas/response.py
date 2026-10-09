"""Response DTOs for the app API service."""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class WordTiming(BaseModel):
    """Millisecond-level word timing and attribution."""

    word: str
    start: float
    end: float
    confidence: float = 1.0


class EmotionAnalysis(BaseModel):
    """Acoustic and textual sentiment attributes for a turn."""

    dominant_emotion: Optional[str] = None
    scores: Dict[str, float] = Field(default_factory=dict)
    valence: Optional[float] = None
    arousal: Optional[float] = None
    abstained: bool = False
    abstain_reason: Optional[str] = None


class SentimentAnalysis(BaseModel):
    """Semantic text sentiment classification."""

    label: str = "neutral"
    score: float = 1.0


class TurnDetail(BaseModel):
    """Complete speaker turn with text, timing, conversational dynamics, and affect."""

    turn_id: str
    speaker: str
    start: float
    end: float
    duration: float
    text: Optional[str] = None
    words: List[WordTiming] = Field(default_factory=list)
    emotion: EmotionAnalysis = Field(default_factory=EmotionAnalysis)
    sentiment: Optional[SentimentAnalysis] = None
    is_interruption: bool = False
    interrupted_by: Optional[str] = None
    interrupts: Optional[str] = None
    overlap_ratio: float = 0.0


class CallMetrics(BaseModel):
    """Aggregate statistics for the entire conversation."""

    duration_s: float
    total_speech_s: float
    overlap_s: float
    overlap_ratio: float
    interruption_count: int
    speaker_count: int
    branch_used: str


class AnalyzeResponse(BaseModel):
    """Unified API response for an audio analysis call."""

    call_id: str
    metrics: CallMetrics
    turns: List[TurnDetail]
    speakers: List[str]
    processing_time_s: float
