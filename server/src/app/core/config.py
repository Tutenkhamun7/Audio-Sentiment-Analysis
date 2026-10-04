from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from dotenv import find_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamically resolve potential .env file locations
_CORE_DIR = Path(__file__).resolve().parent
_SERVER_DIR = _CORE_DIR.parents[2]  # server directory (where .env resides)
_REPO_DIR = _SERVER_DIR.parent  # project root directory

_ENV_CANDIDATES = [
    str(_SERVER_DIR / ".env"),
    str(_REPO_DIR / ".env"),
    str(Path.cwd() / ".env"),
    find_dotenv(),
]
_RESOLVED_ENV_FILES = tuple(dict.fromkeys([p for p in _ENV_CANDIDATES if p and Path(p).is_file()]))


class Settings(BaseSettings):
    """
    Application configuration managed via Pydantic BaseSettings.
    Values can be configured through environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=_RESOLVED_ENV_FILES or ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Hardware & Device Controls ---
    device: str = Field(
        default="cuda",
        description="Global fallback device ('cuda' or 'cpu'). Used when model-specific device is not set.",
    )
    whisper_device: Optional[str] = Field(
        default=None,
        description="Target compute device for Faster-Whisper ('cuda' or 'cpu'). Defaults to `device`.",
    )
    diarization_device: Optional[str] = Field(
        default=None,
        description="Target compute device for Pyannote speaker diarization ('cuda' or 'cpu'). Defaults to `device`.",
    )
    emotion_device: str = Field(
        default="cpu",
        description="Target compute device for semantic and acoustic emotion models ('cuda' or 'cpu').",
    )
    ffmpeg_bin_path: Optional[str] = Field(
        default=None,
        description="Optional directory path containing FFmpeg DLLs/binaries on Windows.",
    )

    # --- Model Selection ---
    whisper_model: str = Field(
        default="large-v3-turbo",
        description="Whisper model size or HuggingFace repo (e.g. 'large-v3-turbo', 'medium', 'small', 'base').",
    )
    whisper_compute_type: Optional[str] = Field(
        default=None,
        description="Quantization/compute type ('float16', 'int8', 'int8_float16', 'bfloat16'). Defaults to auto-select based on device.",
    )
    pyannote_model: str = Field(
        default="pyannote/speaker-diarization-community-1",
        description="Pyannote speaker diarization pipeline name or HuggingFace repo (e.g. 'pyannote/speaker-diarization-community-1', 'pyannote/speaker-diarization-3.1').",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="HuggingFace access token for gated models like Pyannote.",
    )
    num_speakers: Optional[int] = Field(
        default=None,
        description="Exact number of speakers to detect if known in advance (None for automatic detection).",
    )
    min_speakers: Optional[int] = Field(
        default=None,
        description="Minimum number of speakers to detect (None for automatic detection).",
    )
    max_speakers: Optional[int] = Field(
        default=None,
        description="Maximum number of speakers to detect (None for automatic detection).",
    )

    # --- Semantic & Qwen / OpenRouter Selection ---
    semantic_engine: str = Field(
        default="distilroberta",
        description="Semantic scoring engine ('distilroberta', 'qwen', or 'openrouter').",
    )
    openrouter_api_key: Optional[str] = Field(
        default=None,
        description="API Key for OpenRouter or OpenAI-compatible endpoint.",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenRouter or local OpenAI-compatible endpoint (e.g. vLLM or Ollama).",
    )
    qwen_model: str = Field(
        default="qwen/qwen-2.5-72b-instruct",
        description="Qwen model identifier on OpenRouter or local endpoint (e.g. 'qwen/qwen-2.5-72b-instruct', 'qwen/qwen3.8-27b').",
    )

    # --- ASR & Segmentation Tuning ---
    whisper_beam_size: int = Field(
        default=5,
        description="Beam size for transcription decoding.",
    )
    whisper_language: Optional[str] = Field(
        default="en",
        description="Transcription language code (e.g., 'en', 'es', or None for auto-detect).",
    )
    vad_filter: bool = Field(
        default=False,
        description="Whether to enable Silero VAD filtering in Faster-Whisper.",
    )
    segment_min_duration: float = Field(
        default=2.0,
        description="Minimum duration in seconds for an utterance segment.",
    )
    segment_max_duration: float = Field(
        default=6.0,
        description="Maximum duration in seconds for an utterance segment before splitting.",
    )
    segment_max_gap: float = Field(
        default=0.5,
        description="Maximum pause gap in seconds between words to merge within the same speaker turn.",
    )

    @field_validator(
        "whisper_device",
        "diarization_device",
        "whisper_compute_type",
        "whisper_language",
        "hf_token",
        "openrouter_api_key",
        mode="before",
    )
    @classmethod
    def normalize_empty_strings(cls, v: Any) -> Any:
        """Converts empty or whitespace strings into None for clean parameter passing."""
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("num_speakers", "min_speakers", "max_speakers", mode="before")
    @classmethod
    def normalize_empty_integers(cls, v: Any) -> Any:
        """Converts empty strings into None for integer fields."""
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return int(v)

    @property
    def effective_whisper_device(self) -> str:
        return (self.whisper_device or self.device).lower()

    @property
    def effective_diarization_device(self) -> str:
        return (self.diarization_device or self.device).lower()


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton instance of Settings."""
    return Settings()
