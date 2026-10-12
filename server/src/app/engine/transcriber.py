"""In-memory Faster-Whisper ASR transcription for speaker turns."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np
import torch

from app.core.config import settings
from app.core.model_registry import ModelRegistry
from app.engine.diarizer import RawTurn
from app.schemas.response import WordTiming

logger = logging.getLogger(__name__)


def transcribe_speaker_turns(
    speaker_wav: torch.Tensor,
    turns: List[RawTurn],
    sr: int = 16000,
    vad_filter: bool = False,
    initial_prompt: Optional[str] = None,
    collar_s: float = 0.4,
    all_turns: Optional[List[RawTurn]] = None,
    min_word_prob: float = 0.35,
    language: Optional[str] = None,
) -> None:
    """Transcribe each speaker turn directly in memory without disk writes.

    Modifies turns in-place (or attaches transcribed texts & word timestamps).
    """
    if not turns:
        return

    registry = ModelRegistry.get_instance()
    whisper = registry.get_whisper()

    wav_np = speaker_wav.squeeze(0).cpu().numpy().astype(np.float32)

    vad_parameters = {"threshold": settings.whisper_vad_threshold} if vad_filter else None

    # Transcribe entire speaker stream in memory with word timestamps
    segments, info = whisper.transcribe(
        wav_np,
        language=language,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        no_speech_threshold=settings.whisper_no_speech_threshold,
        temperature=0.0,
    )
    seg_list = list(segments)

    # Flatten words
    all_words: list[Any] = []
    for s in seg_list:
        if s.words:
            all_words.extend(s.words)

    # Map words to turns based on midpoint, clamping collar to neighbor speaker bounds
    for turn in turns:
        start_bound = turn.start - collar_s
        end_bound = turn.end + collar_s

        if all_turns:
            # Floor handover: when yielding to another speaker on a turn switch
            future_other_starts = [ot.start for ot in all_turns if ot.speaker != turn.speaker and ot.start > turn.start]
            if future_other_starts:
                next_start = min(future_other_starts)
                # If next speaker started before or around turn end (overlap transition <= 0.8s), clamp
                if next_start <= turn.end and (turn.end - next_start) <= 0.8:
                    end_bound = min(end_bound, next_start)
                elif next_start > turn.end:
                    end_bound = min(end_bound, next_start)

            # Floor takeover: when taking over from another speaker on a turn switch
            prior_other_starts = [ot.start for ot in all_turns if ot.speaker != turn.speaker and ot.start < turn.start]
            if prior_other_starts:
                # Allow standard onset collar before turn.start, but do not dilate past prior speaker onset
                start_bound = max(start_bound, max(prior_other_starts))

        matched_words: list[WordTiming] = []
        for w in all_words:
            prob = float(getattr(w, "probability", 0.9))
            # Discard acoustic murmur with very low probability
            if prob < min_word_prob:
                continue

            mid = (w.start + w.end) / 2.0
            if start_bound <= mid <= end_bound:
                word_clean = w.word.strip()
                if word_clean:
                    matched_words.append(
                        WordTiming(
                            word=word_clean,
                            start=round(float(w.start), 3),
                            end=round(float(w.end), 3),
                            confidence=round(prob, 3),
                        )
                    )

        if matched_words:
            turn_text = " ".join(w.word for w in matched_words).strip()
            turn.text = turn_text
            turn.words = matched_words
        else:
            # No words spoken in this turn slice (e.g. breath, cough, silence)
            turn.text = None
            turn.words = []
            logger.warning(
                "No words transcribed for speaker=%s turn," \
                "despite diarized speech duration.",
                turn.speaker,
                turn.start,
                turn.end,
                turn.duration,
            )


def transcribe_stream_words(
    speaker_wav: torch.Tensor,
    sr: int = 16000,
    vad_filter: bool = False,
    initial_prompt: Optional[str] = None,
    min_word_prob: float = 0.0,
    language: Optional[str] = None,
) -> List[WordTiming]:
    """Transcribe a full speaker audio stream directly in RAM and extract millisecond word timings."""
    if speaker_wav.shape[-1] < int(sr * 0.1):
        return []

    registry = ModelRegistry.get_instance()
    whisper = registry.get_whisper()

    wav_np = speaker_wav.squeeze(0).cpu().numpy().astype(np.float32)
    vad_parameters = {"threshold": settings.whisper_vad_threshold} if vad_filter else None

    segments, info = whisper.transcribe(
        wav_np,
        language=language,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        condition_on_previous_text=False,
        compression_ratio_threshold=2.4,
        no_speech_threshold=settings.whisper_no_speech_threshold,
        temperature=0.0,
    )

    words: list[WordTiming] = []
    for s in segments:
        if not s.words:
            continue
        for w in s.words:
            prob = float(getattr(w, "probability", 0.9))
            if prob < min_word_prob:
                continue
            clean = w.word.strip()
            if clean:
                words.append(
                    WordTiming(
                        word=clean,
                        start=round(float(w.start), 3),
                        end=round(float(w.end), 3),
                        confidence=round(prob, 3),
                    )
                )

    return words
