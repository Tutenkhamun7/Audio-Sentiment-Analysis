"""Configuration and device routing for app service."""

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
    enable_loudness_normalization: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_LOUDNESS_NORMALIZATION"),
        description="Apply a uniform gain so quiet/far-from-mic"
    )
    target_loudness_dbfs: float = Field(
        default=-20.0,
        validation_alias = AliasChoices("TARGET_LOUDNESS_DBFS"),
        description="Target RMS Loudness (dbFS) for normalization"
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
    diarization_clustering_threshold: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("DIARIZATION_CLUSTERING_THRESHOLD"),
        description="Override pyannote clustering distance threshold. Raise (e.g. 0.75-0.8) if "
        "distinct speakers are being merged into one; lower (e.g. 0.5-0.6) if one speaker is "
        "being split into several. None uses the pipeline's pretrained default.",
    )
    diarization_min_duration_off: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("DIARIZATION_MIN_DURATION_OFF"),
        description="Override pyannote segmentation min_duration_off (s). Lower values let the "
        "pipeline react to faster speaker handovers instead of smoothing them into one turn.",
    )

    diarization_min_duration_on: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("DIARIZATION_MIN_DURATION_ON"),
        description="Override pyannote segmentation onset activation threshold(0-1).",
    )

    diarization_onset: Optional[float] = Field(
        default = None,
        validation_alias = AliasChoices("DIARIZATION_ONSET"),
        description="Override pyannote segmentation"
    )
    whisper_vad_threshold: float = Field(
        default = 0.35,
        validation_alias = AliasChoices("WHISPER_VAD_THRESHOLD"),
        description="Silero VAD speech probability threshold used vad_filter is enabled."
    )
    min_speaker_duration_s: float = Field(
        default=0.3,
        validation_alias=AliasChoices("MIN_SPEAKER_DURATION_S"),
        description="Minimum total speech duration for a diarized speaker cluster to be kept "
        "as-is. Clusters below this are reassigned (not dropped) to the nearest valid speaker.",
    )
    whisper_language: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("WHISPER_LANGUAGE"),
        description="Force a specific ISO-639-1 language code for transcription. None lets "
        "faster-whisper auto-detect language per call, needed for multilingual audio.",
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
        default="j-hartmann/emotion-english-distilroberta-base",
        validation_alias=AliasChoices("SEMANTIC_MODEL"),
    )
    enable_sarcasm_detection: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_SARCASM_DETECTION"),
        description="Detect multimodal affect incongruence (e.g. positive words with angry tone)",
    )
    acoustic_collar_s: float = Field(
        default=0.08,
        validation_alias=AliasChoices("ACOUSTIC_COLLAR_S"),
        description="Collar padding in seconds added to turn slices for emotion2vec prosody preservation",
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
    min_acoustic_confidence: float = Field(
        default=0.50,
        validation_alias=AliasChoices("MIN_ACOUSTIC_CONFIDENCE"),
        description="Minimum emotion2vec top-1 probability to trust acoustic classification. "
        "Below this threshold, the turn defaults to 'neutral' to prevent false classifications.",
    )
    min_acoustic_margin: float = Field(
        default=0.10,
        validation_alias=AliasChoices("MIN_ACOUSTIC_MARGIN"),
        description="Minimum gap between top-1 and top-2 emotion probabilities. "
        "Below this, the model is considered ambiguous and defaults to 'neutral'.",
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
    whisper_no_speech_threshold: float = Field(
        default=0.3,
        validation_alias=AliasChoices("WHISPER_NO_SPEECH_THRESHOLD"),
        description="Faster-whisper no speech_threshold."
    )
    whisper_initial_prompt: Optional[str] = Field(
        default="Customer service call in Bangalore, Marathahalli, airport cab booking, KA 9515, SMS driver details.",
        validation_alias=AliasChoices("WHISPER_INITIAL_PROMPT"),
        description="Initial prompt context for Faster-Whisper to guide proper nouns and accents",
    )
    ffmpeg_bin_path: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("FFMPEG_BIN_PATH"),
        description="Path to directory containing ffmpeg binaries/DLLs on Windows",
    )

    # -------------------------------------------------------------
    # 5. Conversational Turn Segmentation Settings
    # -------------------------------------------------------------
    max_turn_pause_s: float = Field(
        default=1.2,
        validation_alias=AliasChoices("MAX_TURN_PAUSE_S"),
        description="Pause threshold in seconds to split consecutive speech by the same speaker into a new turn",
    )
    soft_turn_duration_s: float = Field(
        default=4.0,
        validation_alias=AliasChoices("SOFT_TURN_DURATION_S"),
        description="Minimum duration before sentence-terminal punctuation triggers a turn split",
    )
    max_turn_duration_s: float = Field(
        default=8.0,
        validation_alias=AliasChoices("MAX_TURN_DURATION_S"),
        description="Hard maximum duration in seconds before a turn is forcibly split",
    )
    backchannel_threshold_s: float = Field(
        default=0.8,
        validation_alias=AliasChoices("BACKCHANNEL_THRESHOLD_S"),
        description="Duration threshold below which concurrent interjections are tagged as backchannels",
    )


# Global settings singleton
settings = Settings()


def get_settings() -> Settings:
    """Return the global Settings singleton."""
    return settings
