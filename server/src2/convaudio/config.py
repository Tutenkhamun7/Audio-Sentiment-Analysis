"""Configuration settings for convaudio pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Automatically discover and load .env from current directory or parents
load_dotenv(find_dotenv(usecwd=True))


class AudioConfig(BaseModel):
    """Audio ingest and preprocessing settings."""

    target_sr: int = Field(default=16000, description="Target sample rate in Hz")


class BranchConfig(BaseModel):
    """Channel branching decision thresholds."""

    stereo_corr_threshold: float = Field(
        default=0.6,
        description="Inter-channel correlation below which channels are deemed separated",
    )
    min_channel_energy_ratio: float = Field(
        default=0.05,
        description="Minimum ratio of lower-to-higher channel energy to consider both active",
    )


class Stage1Config(BaseModel):
    """Stage 1 separation & diarization parameters."""

    split_by_speaker: bool = Field(
        default=True,
        validation_alias=AliasChoices("CONVAUDIO_STAGE1__SPLIT_BY_SPEAKER", "SPLIT_BY_SPEAKER"),
        description="Option to split audio based on speakers. When False, audio is not split and treated as single speaker.",
    )
    num_speakers: int = Field(default=2, description="Target speaker count when known")
    min_speakers: int = Field(default=1, description="Minimum speakers for separation")
    max_speakers: int = Field(default=4, description="Maximum speakers for separation")
    max_speakers_per_chunk: int = Field(
        default=3, description="Maximum speakers per chunk in separation"
    )
    asr_collar_s: float = Field(
        default=0.05,
        description="ASR collar dilation in seconds for word-onset conservation",
    )
    leakage_removal: bool = Field(
        default=True, description="Enable leakage removal in separation pipeline"
    )
    min_speaker_speech_s: float = Field(
        default=1.0,
        description="Minimum total speech duration (s) to reject phantom sources",
    )
    pipeline_revision: str = Field(
        default="3.1.0",
        description="Explicit revision or git tag for pyannote separation pipeline",
    )
    separate_overlap: bool = Field(
        default=True,
        validation_alias=AliasChoices("CONVAUDIO_STAGE1__SEPARATE_OVERLAP", "SEPARATE_OVERLAP"),
        description="Enable targeted SepFormer separation on overlapping speech segments",
    )
    sepformer_model: str = Field(
        default="speechbrain/sepformer-wsj02mix",
        validation_alias=AliasChoices("CONVAUDIO_STAGE1__SEPFORMER_MODEL", "SEPFORMER_MODEL"),
        description="Pretrained SpeechBrain SepFormer model identifier or directory",
    )
    overlap_padding_s: float = Field(
        default=0.2,
        description="Context padding duration in seconds added to overlap slices",
    )
    min_overlap_duration_s: float = Field(
        default=0.05,
        description="Minimum duration in seconds to trigger SepFormer overlap separation",
    )


class Stage2Config(BaseModel):
    """Stage 2 canonical timeline settings."""

    interrupt_window_s: float = Field(
        default=1.0,
        description="Window (s) within which speaker A stops after speaker B starts to count as interruption",
    )
    default_frame_rate_hz: float = Field(
        default=50.0,
        description="Default frame rate in Hz for frame-level overlap masks",
    )


class Stage3Config(BaseModel):
    """Stage 3 lexical extraction parameters."""

    whisper_model: str = Field(
        default="large-v3-turbo",
        validation_alias=AliasChoices("CONVAUDIO_STAGE3__WHISPER_MODEL", "WHISPER_MODEL"),
        description="Whisper model name or path for faster-whisper",
    )
    vad_filter: bool = Field(
        default=True,
        validation_alias=AliasChoices("CONVAUDIO_STAGE3__VAD_FILTER", "VAD_FILTER"),
        description="Enable VAD filter in Whisper transcription",
    )
    align_model: str = Field(
        default="wav2vec2-base-960h",
        description="Wav2Vec2 model for forced alignment CTC",
    )
    align_revision: str = Field(
        default="5b9e07d",
        description="Pinned revision for alignment model",
    )


class Stage4Config(BaseModel):
    """Stage 4 acoustic extraction parameters."""

    encoder: str = Field(
        default="emotion2vec",
        validation_alias=AliasChoices("CONVAUDIO_STAGE4__ENCODER", "STAGE4_ENCODER"),
        description="Emotion encoder implementation: 'emotion2vec', 'wavlm', or 'stub'",
    )
    min_clean_speech_s: float = Field(
        default=1.0,
        description="Minimum clean speech duration in seconds below which to abstain (SHORT_TURN)",
    )
    max_overlap_ratio: float = Field(
        default=0.8,
        description="Maximum overlap ratio above which to abstain (HIGH_OVERLAP)",
    )
    min_sep_cosine: float = Field(
        default=0.5,
        description="Minimum separation cosine similarity below which to abstain (LOW_SEP_QUALITY)",
    )
    wavlm_revision: str = Field(
        default="efa81aa",
        description="Pinned revision for WavLM backbone",
    )
    emotion2vec_model: str = Field(
        default="iic/emotion2vec_plus_base",
        validation_alias=AliasChoices(
            "CONVAUDIO_STAGE4__EMOTION2VEC_MODEL", "EMOTION2VEC_MODEL"
        ),
        description="Emotion2Vec model identifier in ModelScope / local cache",
    )
    emotion2vec_revision: str = Field(
        default="master",
        description="Pinned revision for Emotion2Vec model",
    )


class Stage5Config(BaseModel):
    """Stage 5 semantic sentiment parameters."""

    classifier: str = Field(
        default="stub",
        description="Classifier implementation: 'stub', 'setfit', or 'modernbert'",
    )
    model_path: str | None = Field(
        default=None,
        description="Path to fine-tuned model / checkpoint directory",
    )
    n_context: int = Field(
        default=3,
        description="Number of preceding context turns to include in window",
    )
    speaker_roles: dict[str, str] = Field(
        default_factory=lambda: {"SPEAKER_00": "AGENT", "SPEAKER_01": "CUSTOMER"},
        description="Speaker ID to role mapping for context windows",
    )
    min_words: int = Field(
        default=1,
        description="Minimum words below which to abstain (NO_TEXT)",
    )
    min_asr_logprob: float = Field(
        default=-1.0,
        description="Minimum ASR log probability below which to abstain (LOW_ASR_CONFIDENCE)",
    )
    min_conf: float = Field(
        default=0.45,
        description="Minimum calibrated probability below which to abstain (LOW_SEM_CONF)",
    )
    require_calibration: bool = Field(
        default=True,
        description="Whether a valid calibration artefact is required or refuse",
    )
    llm_adjudication: bool = Field(
        default=False,
        description="Enable optional LLM adjudication for flagged turns",
    )
    ambiguity_margin: float = Field(
        default=0.1,
        description="Probability difference margin for flagging ambiguous turns to LLM",
    )
    max_adjudicated: int = Field(
        default=5,
        description="Maximum turns per call to send for LLM adjudication",
    )
    batch_size: int = Field(
        default=32,
        description="Inference batch size",
    )


class PipelineConfig(BaseSettings):
    """Root configuration for convaudio."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        env_prefix="CONVAUDIO_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    audio: AudioConfig = Field(default_factory=AudioConfig)
    branch: BranchConfig = Field(default_factory=BranchConfig)
    stage1: Stage1Config = Field(default_factory=Stage1Config)
    stage2: Stage2Config = Field(default_factory=Stage2Config)
    stage3: Stage3Config = Field(default_factory=Stage3Config)
    stage4: Stage4Config = Field(default_factory=Stage4Config)
    stage5: Stage5Config = Field(default_factory=Stage5Config)

    # Global environment-controlled flags
    model_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("CONVAUDIO_MODEL_DIR", "MODEL_DIR"),
        description="Local directory for cached/offline models (CONVAUDIO_MODEL_DIR)",
    )
    allow_download: bool = Field(
        default=False,
        validation_alias=AliasChoices("CONVAUDIO_ALLOW_DOWNLOAD", "ALLOW_DOWNLOAD"),
        description="Whether HuggingFace downloading is allowed (CONVAUDIO_ALLOW_DOWNLOAD)",
    )
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CONVAUDIO_HF_TOKEN", "HF_TOKEN"),
        description="Hugging Face API token (HF_TOKEN)",
    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> PipelineConfig:
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return cls()
        return cls(**data)


def load_config(config_path: Path | str | None = None) -> PipelineConfig:
    """Load configuration from optional path or default environment."""
    if config_path:
        p = Path(config_path)
        if p.exists():
            return PipelineConfig.from_yaml(p)
    return PipelineConfig()
