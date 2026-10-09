"""Configuration and device routing for src3 service."""

from __future__ import annotations

import os
from typing import Optional

import torch
from dotenv import find_dotenv, load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(find_dotenv(usecwd=True))


class Settings(BaseSettings):
    """Global configuration with granular device placement."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -------------------------------------------------------------
    # 1. Device Placement (Granular: swap between CPU & CUDA anytime)
    # -------------------------------------------------------------
    default_device: str = Field(
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        validation_alias=AliasChoices("DEFAULT_DEVICE", "DEVICE"),
        description="Default execution device when component-specific device is not set.",
    )
    diarization_device: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DIARIZATION_DEVICE"),
        description="Device for pyannote diarization ('cpu', 'cuda', 'cuda:0')",
    )
    separation_device: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("SEPARATION_DEVICE"),
        description="Device for SepFormer speech separation ('cpu', 'cuda', 'cuda:0')",
    )
    asr_device: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ASR_DEVICE", "WHISPER_DEVICE"),
        description="Device for faster-whisper ('cpu', 'cuda', 'cuda:0')",
    )
    alignment_device: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ALIGNMENT_DEVICE"),
        description="Device for Wav2Vec2 CTC forced alignment ('cpu', 'cuda', 'cuda:0')",
    )
    acoustic_device: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ACOUSTIC_DEVICE", "EMOTION_DEVICE"),
        description="Device for emotion2vec / acoustic encoder ('cpu', 'cuda', 'cuda:0')",
    )
    semantic_device: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("SEMANTIC_DEVICE"),
        description="Device for semantic LLM / text classifier ('cpu', 'cuda', 'cuda:0')",
    )

    def get_device(self, component: str) -> str:
        """Resolve the active execution device for a specific component.

        Falls back to default_device if component override is not specified.
        If 'cuda' is requested but CUDA is unavailable, safely falls back to 'cpu'.
        """
        device_str = getattr(self, f"{component}_device", None) or self.default_device
        device_str = device_str.strip().lower()
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device_str

    # -------------------------------------------------------------
    # 2. Audio Processing Parameters
    # -------------------------------------------------------------
    target_sr: int = Field(default=16000, description="Internal sample rate (Hz)")
    stereo_corr_threshold: float = Field(
        default=0.6,
        description="Correlation threshold below which 2-channel audio is treated as clean stereo",
    )
    stereo_energy_ratio_min: float = Field(
        default=0.05,
        description="Minimum energy ratio required between channels for stereo split",
    )

    # -------------------------------------------------------------
    # 3. Model Identifiers & Tokens
    # -------------------------------------------------------------
    hf_token: Optional[str] = Field(
        default_factory=lambda: os.environ.get("HF_TOKEN"),
        validation_alias=AliasChoices("HF_TOKEN"),
    )
    pyannote_model: str = Field(
        default="pyannote/speaker-diarization-community-1",
        validation_alias=AliasChoices("PYANNOTE_MODEL"),
    )
    sepformer_model: str = Field(
        default="speechbrain/sepformer-wsj02mix",
        validation_alias=AliasChoices("SEPFORMER_MODEL"),
    )
    whisper_model: str = Field(
        default="large-v3-turbo",
        validation_alias=AliasChoices("WHISPER_MODEL"),
    )
    whisper_compute_type: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("WHISPER_COMPUTE_TYPE"),
        description="Override compute type: 'float16', 'int8', etc. If None, auto-selects based on device.",
    )
    alignment_model: str = Field(
        default="wav2vec2-base-960h",
        validation_alias=AliasChoices("ALIGNMENT_MODEL"),
    )
    acoustic_encoder: str = Field(
        default="emotion2vec",
        validation_alias=AliasChoices("CONVAUDIO_STAGE4__ENCODER", "ACOUSTIC_ENCODER"),
    )
    semantic_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        validation_alias=AliasChoices("SEMANTIC_MODEL"),
    )

    # -------------------------------------------------------------
    # 4. Pipeline & Overlap Tuning
    # -------------------------------------------------------------
    enable_overlap_separation: bool = Field(
        default=True,
        validation_alias=AliasChoices("SEPARATE_OVERLAP"),
        description="Whether to run SepFormer on collision segments",
    )
    overlap_padding_s: float = Field(
        default=0.2,
        description="Padding in seconds added to overlap slices before SepFormer inference",
    )
    min_overlap_duration_s: float = Field(
        default=0.05,
        description="Minimum overlap collision duration to trigger SepFormer",
    )
    interrupt_window_s: float = Field(
        default=1.0,
        description="Window (s) in which a speaker yielding during a collision counts as interrupted",
    )
    min_speech_turn_s: float = Field(
        default=0.4,
        description="Turns shorter than this will abstain from acoustic affect prediction",
    )
    vad_filter: bool = Field(
        default=False,
        validation_alias=AliasChoices("VAD_FILTER"),
        description="VAD filter flag for faster-whisper",
    )
    asr_collar_s: float = Field(
        default=0.15,
        description="Padding in seconds added before onset and after offset of turn slices to conserve initial consonants",
    )
    max_merge_gap_s: float = Field(
        default=0.8,
        description="Maximum silence gap in seconds between consecutive same-speaker turns to merge",
    )
    min_turn_duration_s: float = Field(
        default=0.25,
        description="Minimum duration in seconds below which empty turns are discarded",
    )
    min_word_prob: float = Field(
        default=0.0,
        description="Minimum word probability threshold to prune hallucinated murmur words (default 0.0 preserves all speech)",
    )
    whisper_initial_prompt: Optional[str] = Field(
        default="Customer service call in Bangalore, Marathahalli, airport cab booking, KA 9515, SMS driver details.",
        validation_alias=AliasChoices("WHISPER_INITIAL_PROMPT"),
        description="Initial prompt context for Faster-Whisper to guide proper nouns and accents",
    )


# Global settings singleton
settings = Settings()
