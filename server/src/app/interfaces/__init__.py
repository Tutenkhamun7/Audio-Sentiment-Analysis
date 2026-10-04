from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Tuple
import numpy as np

from app.schemas.common import EmotionPrediction


@dataclass
class WordTimestamp:
    """Represents a single word with temporal boundaries."""

    start: float
    end: float
    word: str


@dataclass
class RawSegment:
    """Represents a raw segment transcribed by an ASR engine."""

    start: float
    end: float
    text: str
    words: List[WordTimestamp] = field(default_factory=list)


class BaseASR(Protocol):
    """Protocol defining the interface for pluggable Speech-to-Text models."""

    def transcribe(
        self,
        file_path: str,
        beam_size: int = 5,
        language: Optional[str] = "en",
        **kwargs,
    ) -> List[RawSegment]: ...


@dataclass
class SpeakerInterval:
    """Represents an active speaker interval identified by a diarizer."""

    start: float
    end: float
    speaker: str


@dataclass
class OverlapInterval:
    """Represents a period of simultaneous speech by multiple speakers."""

    start: float
    end: float
    speakers: List[str] = field(default_factory=list)


class BaseDiarizer(Protocol):
    """Protocol defining the interface for pluggable Speaker Diarization models."""

    def diarize(self, file_path: str) -> List[SpeakerInterval]: ...

    def diarize_with_overlaps(
        self, file_path: str
    ) -> Tuple[List[SpeakerInterval], List[OverlapInterval]]: ...


class BaseSemanticScorer(Protocol):
    """Protocol defining the interface for text-based emotion scoring."""

    def score_batch(self, texts: List[str]) -> List[List[EmotionPrediction]]: ...


class BaseAcousticScorer(Protocol):
    """Protocol defining the interface for audio-waveform emotion scoring."""

    def score_chunk(self, audio_chunk: np.ndarray, sample_rate: int) -> List[EmotionPrediction]: ...
