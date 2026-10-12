"""Wav2Vec2 CTC forced alignment and turn boundary snapping."""

from __future__ import annotations

import logging
from typing import List

import torch
import torchaudio

from app.core.model_registry import ModelRegistry
from app.engine.diarizer import RawTurn
from app.schemas.response import WordTiming

logger = logging.getLogger(__name__)


def assign_words_to_speakers(
    words: List[WordTiming],
    diarization_turns: List[RawTurn],
    fallback_speaker: str = "SPEAKER_00",
) -> List[WordTiming]:
    """Map each word in the continuous ASR transcript to a diarized speaker.

    WhisperX intersection algorithm with floor inertia:
    1. For each word [w_start, w_end], calculate exact temporal overlap (intersection)
       with every diarization interval.
    2. The speaker with the maximum overlap duration claims the word.
    3. Floor Inertia (Hysteresis): If a word falls in a brief boundary ambiguity or pause,
       favor the current floor speaker unless another speaker clearly intersects > 50%
       of the word duration or silence gap exceeds 0.4s.
    """
    if not words:
        return []

    if not diarization_turns:
        for w in words:
            w.speaker = fallback_speaker
        return words

    current_speaker = fallback_speaker

    for i, w in enumerate(words):
        w_dur = max(0.001, w.end - w.start)
        overlap_by_spk: dict[str, float] = {}

        # 1. Exact temporal intersection
        for turn in diarization_turns:
            inter_start = max(w.start, turn.start)
            inter_end = min(w.end, turn.end)
            if inter_end > inter_start:
                dur = inter_end - inter_start
                overlap_by_spk[turn.speaker] = overlap_by_spk.get(turn.speaker, 0.0) + dur

        if overlap_by_spk:
            # Pick speaker with highest overlap
            best_spk, max_overlap = max(overlap_by_spk.items(), key=lambda kv: kv[1])
            best_ratio = max_overlap / w_dur

            # Floor inertia: if current floor speaker also has significant overlap
            # (within 0.15s or >= 35% ratio), do not flip-flop on isolated syllables
            current_overlap = overlap_by_spk.get(current_speaker, 0.0)
            if (
                current_speaker in overlap_by_spk
                and best_spk != current_speaker
                and best_ratio < 0.65
                and (current_overlap / w_dur) >= 0.35
            ):
                w.speaker = current_speaker
            else:
                w.speaker = best_spk
                current_speaker = best_spk
        else:
            # Word falls between diarization intervals (e.g. slight VAD offset or pause)
            # Find closest diarization turn within 0.5s
            closest_turn = None
            min_dist = float("inf")
            for turn in diarization_turns:
                dist = max(0.0, turn.start - w.end, w.start - turn.end)
                if dist < min_dist:
                    min_dist = dist
                    closest_turn = turn

            if closest_turn is not None and min_dist <= 0.5:
                w.speaker = closest_turn.speaker
                current_speaker = closest_turn.speaker
            else:
                # Retain current floor speaker
                w.speaker = current_speaker

    return words


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
