"""Centralized Model Registry with granular device placement and lazy/eager loading."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

from src3.core.config import Settings, settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages lifecycle and device placement for all deep learning components."""

    _instance: Optional[ModelRegistry] = None

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or settings
        self._diarizer: Any = None
        self._separator: Any = None
        self._whisper: Any = None
        self._aligner: Any = None
        self._acoustic_encoder: Any = None
        self._semantic_engine: Any = None

    @classmethod
    def get_instance(cls, config: Settings | None = None) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = cls(config=config)
        return cls._instance

    # -------------------------------------------------------------
    # 1. Pyannote Diarization Pipeline & Speaker Embedder
    # -------------------------------------------------------------
    def get_diarizer(self) -> Any:
        if self._diarizer is None:
            dev = self.config.get_device("diarization")
            logger.info("Initializing Pyannote pipeline (%s) on %s...", self.config.pyannote_model, dev)
            from pyannote.audio import Pipeline

            pipeline = Pipeline.from_pretrained(
                self.config.pyannote_model,
                token=self.config.hf_token,
            )
            if pipeline is not None and dev.startswith("cuda"):
                pipeline.to(torch.device(dev))
            self._diarizer = pipeline
        return self._diarizer

    def get_speaker_embedder(self) -> Any:
        """Pyannote deep speaker embedding model for permutation matching."""
        diarizer = self.get_diarizer()
        if hasattr(diarizer, "_embedding"):
            return diarizer._embedding
        return None

    # -------------------------------------------------------------
    # 2. SpeechBrain SepFormer Speech Separator
    # -------------------------------------------------------------
    def get_separator(self) -> Any:
        if self._separator is None:
            dev = self.config.get_device("separation")
            logger.info("Initializing SepFormer (%s) on %s...", self.config.sepformer_model, dev)
            from speechbrain.inference.separation import SepformerSeparation

            run_opts = {"device": dev}
            self._separator = SepformerSeparation.from_hparams(
                source=self.config.sepformer_model,
                run_opts=run_opts,
            )
        return self._separator

    # -------------------------------------------------------------
    # 3. Faster-Whisper ASR Model
    # -------------------------------------------------------------
    def get_whisper(self) -> Any:
        if self._whisper is None:
            dev = self.config.get_device("asr")
            # Select optimal compute type based on device
            if self.config.whisper_compute_type:
                compute_type = self.config.whisper_compute_type
            elif dev.startswith("cuda"):
                compute_type = "float16"
            else:
                compute_type = "int8"

            logger.info(
                "Initializing Faster-Whisper (%s) on %s with %s...",
                self.config.whisper_model,
                dev,
                compute_type,
            )
            from faster_whisper import WhisperModel

            self._whisper = WhisperModel(
                self.config.whisper_model,
                device="cuda" if dev.startswith("cuda") else "cpu",
                compute_type=compute_type,
            )
        return self._whisper

    # -------------------------------------------------------------
    # 4. Wav2Vec2 CTC Forced Aligner
    # -------------------------------------------------------------
    def get_aligner(self) -> Any:
        if self._aligner is None:
            dev = self.config.get_device("alignment")
            logger.info("Initializing Wav2Vec2 CTC aligner on %s...", dev)
            import torchaudio

            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            model = bundle.get_model().to(torch.device(dev))
            model.eval()
            self._aligner = (model, bundle.get_labels(), bundle.sample_rate)
        return self._aligner

    # -------------------------------------------------------------
    # 5. Acoustic Emotion Encoder (emotion2vec via FunASR)
    # -------------------------------------------------------------
    def get_acoustic_encoder(self) -> Any:
        if self._acoustic_encoder is None:
            dev = self.config.get_device("acoustic")
            logger.info("Initializing Acoustic affect encoder (%s) on %s...", self.config.acoustic_encoder, dev)
            from funasr import AutoModel

            cache_base = os.path.expanduser(r"~/.cache/modelscope/models/iic--emotion2vec_plus_base/snapshots/master")
            model_target = cache_base if os.path.isdir(cache_base) else "iic/emotion2vec_plus_base"

            device_target = dev if dev.startswith("cuda") else "cpu"
            self._acoustic_encoder = AutoModel(
                model=model_target,
                disable_update=True,
                device=device_target,
            )
        return self._acoustic_encoder

    # -------------------------------------------------------------
    # 6. Semantic Sentiment Analysis Pipeline
    # -------------------------------------------------------------
    def get_semantic_classifier(self) -> Any:
        if self._semantic_engine is None:
            dev = self.config.get_device("semantic")
            logger.info("Initializing Semantic sentiment model (%s) on %s...", self.config.semantic_model, dev)
            from transformers import pipeline

            device_arg = 0 if dev.startswith("cuda") else -1
            try:
                # Prioritize local cached files to prevent HF token expiration / network 401 errors
                self._semantic_engine = pipeline(
                    "sentiment-analysis",
                    model=self.config.semantic_model,
                    device=device_arg,
                    local_files_only=True,
                    truncation=True,
                    max_length=512,
                )
            except Exception:
                self._semantic_engine = pipeline(
                    "sentiment-analysis",
                    model=self.config.semantic_model,
                    device=device_arg,
                    truncation=True,
                    max_length=512,
                )
        return self._semantic_engine

    # -------------------------------------------------------------
    # Pre-warming for FastAPI Lifespan
    # -------------------------------------------------------------
    def warm_all(self) -> None:
        """Pre-warm all neural models into memory at application startup."""
        logger.info("Pre-warming all models according to device routing...")
        try:
            self.get_diarizer()
        except Exception as e:
            logger.warning("Could not pre-warm diarizer: %s", e)

        try:
            self.get_separator()
        except Exception as e:
            logger.warning("Could not pre-warm separator: %s", e)

        try:
            self.get_whisper()
        except Exception as e:
            logger.warning("Could not pre-warm whisper: %s", e)

        try:
            self.get_aligner()
        except Exception as e:
            logger.warning("Could not pre-warm aligner: %s", e)

        try:
            self.get_acoustic_encoder()
        except Exception as e:
            logger.warning("Could not pre-warm acoustic encoder: %s", e)

        try:
            self.get_semantic_classifier()
        except Exception as e:
            logger.warning("Could not pre-warm semantic classifier: %s", e)

        logger.info("Model pre-warming complete.")
