"""Wav2Vec2 CTC forced alignment and turn boundary snapping."""

from __future__ import annotations

import logging
from typing import List

import torch
import torchaudio

from src3.core.model_registry import ModelRegistry
from src3.engine.diarizer import RawTurn
from src3.schemas.response import WordTiming

logger = logging.getLogger(__name__)


def align_and_snap_turn_boundaries(
    speaker_wav: torch.Tensor,
    turns: List[RawTurn],
    sr: int = 16000,
) -> None:
    """Run CTC forced alignment on transcribed turns and snap turn boundaries to spoken words."""
    if not turns:
        return

    registry = ModelRegistry.get_instance()
    try:
        model, labels, align_sr = registry.get_aligner()
    except Exception as e:
        logger.warning("Forced aligner unavailable (%s). Keeping Whisper word timestamps.", e)
        return

    dictionary = {c: i for i, c in enumerate(labels)}
    device = next(model.parameters()).device

    for turn in turns:
        text = getattr(turn, "text", None)
        if not text:
            continue

        clean_text = "".join(c.upper() for c in text if c.isalnum() or c.isspace()).strip()
        words = clean_text.split()
        if not words:
            continue

        # Extract turn slice
        s_idx = max(0, int(turn.start * sr))
        e_idx = min(speaker_wav.shape[-1], int(turn.end * sr))
        slice_wav = speaker_wav[s_idx:e_idx]

        if slice_wav.shape[-1] < int(sr * 0.1):
            continue

        # Prepare tokens for CTC
        transcript = "|".join(words)
        tokens = [dictionary[c] for c in transcript if c in dictionary]
        if not tokens:
            continue

        try:
            input_wav = slice_wav.unsqueeze(0).to(device)
            with torch.no_grad():
                emissions, _ = model(input_wav)
                emissions = torch.log_softmax(emissions, dim=-1)

            emission = emissions[0].cpu().detach()

            # Align using torchaudio forced_align
            targets = torch.tensor(tokens, dtype=torch.int32)
            alignments, scores = torchaudio.functional.forced_align(
                emission, targets, blank=0
            )

            # Convert frame alignments to word timestamps
            # emission frames are downsampled by factor of 320 (20ms per frame at 16kHz)
            frame_duration = 320.0 / float(sr)
            spanned_words: list[WordTiming] = []
            token_idx = 0

            for word in words:
                word_tokens = [dictionary[c] for c in word if c in dictionary]
                if not word_tokens:
                    continue
                w_start_frame = None
                w_end_frame = None
                for _ in word_tokens:
                    if token_idx < len(alignments):
                        frame = int(alignments[token_idx].item())
                        if w_start_frame is None or frame < w_start_frame:
                            w_start_frame = frame
                        if w_end_frame is None or frame > w_end_frame:
                            w_end_frame = frame
                        token_idx += 1

                # Skip the '|' delimiter token
                token_idx += 1

                if w_start_frame is not None and w_end_frame is not None:
                    abs_start = round(turn.start + (w_start_frame * frame_duration), 3)
                    abs_end = round(turn.start + ((w_end_frame + 1) * frame_duration), 3)
                    spanned_words.append(
                        WordTiming(
                            word=word,
                            start=abs_start,
                            end=max(abs_start + 0.05, abs_end),
                            confidence=0.95,
                        )
                    )

            if spanned_words:
                turn.words = spanned_words
                # Snap turn boundaries to exact word onsets and offsets (WhisperX technique)
                first_word_s = spanned_words[0].start
                last_word_e = spanned_words[-1].end
                turn.start = max(0.0, first_word_s - 0.05)
                turn.end = max(turn.start + 0.1, last_word_e + 0.05)
        except Exception as err:
            logger.debug("Forced alignment fallback for turn %.2f-%.2f: %s", turn.start, turn.end, err)
