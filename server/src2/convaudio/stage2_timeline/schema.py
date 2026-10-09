"""Dataclass schema and serialization for canonical ConversationTimeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AudioMeta:
    """Metadata for ingested raw audio."""

    path: str
    sha256: str
    original_sr: int
    channels: int
    duration_s: float
    branch: str  # "stereo_split" | "mono_separated" | "no_split"


@dataclass
class SpeakerMeta:
    """Speaker-level enrollment stream metadata."""

    stream_path: str
    total_speech_s: float
    embedding_key: str


@dataclass
class CorpusStats:
    """Overall call aggregate speech and overlap statistics."""

    total_speech_s: float
    overlap_speech_s: float
    overlap_ratio: float
    n_turns: int
    n_interruptions: int


@dataclass
class WordTiming:
    """Forced-alignment word timestamp."""

    w: str
    s: float
    e: float
    conf: float


@dataclass
class TurnOverlap:
    """Per-turn overlap and interruption metrics."""

    ratio: float
    clean_speech_s: float
    interrupts: str | None = None
    interrupted_by: str | None = None
    preceding_gap_s: float | None = None


@dataclass
class TurnQuality:
    """Quality and ASR diagnostic metrics for a turn."""

    sep_cosine: float | None = None
    asr_avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


@dataclass
class AcousticFeatures:
    """Pooled frame-level acoustic valence and arousal."""

    valence: float
    arousal: float
    conf: float
    pooled_frames: int
    masked_frames: int
    emotion: str | None = None
    emotion_scores: dict[str, float] | None = None


@dataclass
class SemanticFeatures:
    """Turn-level semantic sentiment probabilities and label."""

    probs: dict[str, float]
    label: str
    conf: float
    calibrated: bool
    model_revision: str
    adjudicated: bool = False
    adjudication_reason: str | None = None


@dataclass
class Turn:
    """Individual speaker turn."""

    turn_id: str
    speaker: str
    start: float
    end: float
    provenance: str  # "clean" | "separated"
    overlap: TurnOverlap
    quality: TurnQuality = field(default_factory=TurnQuality)
    text: str | None = None
    word_timings: list[WordTiming] | None = None
    word_timings_source: str | None = None  # "forced_align" | "asr_fallback" | None
    acoustic: AcousticFeatures | None = None
    acoustic_abstain_reason: str | None = None  # "SHORT_TURN" | "HIGH_OVERLAP" | "LOW_SEP_QUALITY" | None
    semantic: SemanticFeatures | dict[str, Any] | None = None
    semantic_abstain_reason: str | None = None  # "NO_TEXT" | "LOW_ASR_CONFIDENCE" | "LOW_SEM_CONF" | None
    fused: dict[str, Any] | None = None


@dataclass
class Diagnostics:
    """Pipeline diagnostics and execution traces."""

    rejected_sources: list[dict[str, Any]] = field(default_factory=list)
    alignment_failures: int = 0
    acoustic_abstentions: int = 0
    abstain_reasons: dict[str, int] = field(
        default_factory=lambda: {"SHORT_TURN": 0, "HIGH_OVERLAP": 0, "LOW_SEP_QUALITY": 0}
    )
    model_revisions: dict[str, str] = field(default_factory=dict)
    stage_durations_s: dict[str, float] = field(default_factory=dict)


@dataclass
class ConversationTimeline:
    """Canonical timeline representing stages 1-4 audio analysis."""

    schema_version: str
    call_id: str
    created_utc: str
    audio_meta: AudioMeta
    frame_rate_hz: float
    n_frames: int
    overlap_mask: str
    speakers: dict[str, SpeakerMeta]
    corpus_stats: CorpusStats
    turns: list[Turn]
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    outcome: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert timeline dataclass tree to dict."""
        return asdict(self)

    def to_json(self, path: Path | str | None = None, indent: int = 2) -> str:
        """Serialize timeline to JSON string and optionally save to file."""
        data = self.to_dict()
        text = json.dumps(data, indent=indent)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationTimeline:
        """Construct ConversationTimeline from dictionary."""
        audio_meta = AudioMeta(**data["audio_meta"])

        speakers = {
            k: SpeakerMeta(**v) for k, v in data.get("speakers", {}).items()
        }

        corpus_stats = CorpusStats(**data["corpus_stats"])

        turns: list[Turn] = []
        for t in data.get("turns", []):
            t_copy = dict(t)
            overlap_dict = t_copy.pop("overlap")
            overlap = TurnOverlap(**overlap_dict)

            quality_dict = t_copy.pop("quality", {})
            quality = TurnQuality(**quality_dict) if quality_dict else TurnQuality()

            acoustic_dict = t_copy.pop("acoustic", None)
            acoustic = AcousticFeatures(**acoustic_dict) if acoustic_dict is not None else None

            word_timings_raw = t_copy.pop("word_timings", None)
            word_timings = (
                [WordTiming(**w) for w in word_timings_raw]
                if word_timings_raw is not None
                else None
            )

            semantic_raw = t_copy.pop("semantic", None)
            semantic = (
                SemanticFeatures(**semantic_raw)
                if isinstance(semantic_raw, dict)
                else semantic_raw
            )
            semantic_abstain_reason = t_copy.pop("semantic_abstain_reason", None)

            turns.append(
                Turn(
                    overlap=overlap,
                    quality=quality,
                    acoustic=acoustic,
                    word_timings=word_timings,
                    semantic=semantic,
                    semantic_abstain_reason=semantic_abstain_reason,
                    **t_copy,
                )
            )

        diag_dict = data.get("diagnostics", {})
        diagnostics = Diagnostics(**diag_dict) if diag_dict else Diagnostics()

        return cls(
            schema_version=data.get("schema_version", "1.0.0"),
            call_id=data["call_id"],
            created_utc=data["created_utc"],
            audio_meta=audio_meta,
            frame_rate_hz=data["frame_rate_hz"],
            n_frames=data["n_frames"],
            overlap_mask=data["overlap_mask"],
            speakers=speakers,
            corpus_stats=corpus_stats,
            turns=turns,
            diagnostics=diagnostics,
            outcome=data.get("outcome"),
        )

    @classmethod
    def from_json(cls, path_or_str: Path | str) -> ConversationTimeline:
        """Load ConversationTimeline from JSON file path or raw string."""
        p = Path(path_or_str)
        if p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8")
        else:
            text = str(path_or_str)
        data = json.loads(text)
        return cls.from_dict(data)
