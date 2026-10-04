from typing import TYPE_CHECKING

from app.core.config import Settings
from app.interfaces import (
    BaseAcousticScorer,
    BaseASR,
    BaseDiarizer,
    BaseSemanticScorer,
)
from app.services.alignment import AlignmentService
from app.services.affect_evaluator import AffectEvaluator

if TYPE_CHECKING:
    from app.services.emotion_engine import EmotionEngine


def create_asr_adapter(settings: Settings) -> BaseASR:
    """Instantiates the appropriate ASR adapter based on Settings."""
    if settings.whisper_model == "mock":
        from app.adapters.asr.mock import MockASRAdapter

        return MockASRAdapter()

    from app.adapters.asr.faster_whisper import FasterWhisperAdapter

    return FasterWhisperAdapter(
        model_size_or_path=settings.whisper_model,
        device=settings.effective_whisper_device,
        compute_type=settings.whisper_compute_type,
    )


def create_diarizer_adapter(settings: Settings) -> BaseDiarizer:
    """Instantiates the appropriate Diarization adapter based on Settings."""
    if settings.pyannote_model == "mock":
        from app.adapters.diarization.mock import MockDiarizerAdapter

        return MockDiarizerAdapter()

    from app.adapters.diarization.pyannote import PyannoteDiarizerAdapter

    return PyannoteDiarizerAdapter(
        model_name=settings.pyannote_model,
        hf_token=settings.hf_token,
        device=settings.effective_diarization_device,
        num_speakers=settings.num_speakers,
        min_speakers=settings.min_speakers,
        max_speakers=settings.max_speakers,
    )


def create_semantic_scorer(settings: Settings) -> BaseSemanticScorer:
    """Instantiates the semantic emotion scoring adapter (Qwen via OpenRouter or DistilRoBERTa)."""
    from app.adapters.emotion.distilroberta import DistilRobertaSemanticAdapter

    fallback = DistilRobertaSemanticAdapter(device=settings.emotion_device)
    if settings.semantic_engine.lower() in {"qwen", "openrouter"}:
        from app.adapters.emotion.qwen import QwenSemanticAdapter

        return QwenSemanticAdapter(
            model_name=settings.qwen_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            fallback_adapter=fallback,
        )

    return fallback


def create_acoustic_scorer(settings: Settings) -> BaseAcousticScorer:
    """Instantiates the acoustic emotion scoring adapter."""
    from app.adapters.emotion.emotion2vec import Emotion2VecAcousticAdapter

    return Emotion2VecAcousticAdapter(device=settings.emotion_device)


def create_alignment_service(settings: Settings) -> AlignmentService:
    """Instantiates the alignment and segmentation domain service."""
    return AlignmentService()


def create_affect_evaluator(settings: Settings) -> AffectEvaluator:
    """Instantiates the affect dynamics and conflict evaluator domain service."""
    return AffectEvaluator()


def create_emotion_engine(settings: Settings) -> "EmotionEngine":
    """Instantiates the unified emotion scoring and affect dynamics engine."""
    from app.services.emotion_engine import EmotionEngine

    return EmotionEngine(
        device=settings.emotion_device,
        semantic_scorer=create_semantic_scorer(settings),
        acoustic_scorer=create_acoustic_scorer(settings),
        affect_evaluator=create_affect_evaluator(settings),
    )
