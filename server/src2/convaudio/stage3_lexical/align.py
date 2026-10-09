"""Forced alignment via torchaudio wav2vec2 CTC."""

from __future__ import annotations

import re

import torch
import torchaudio

from convaudio.stage2_timeline.schema import WordTiming


class AlignmentError(Exception):
    """Raised when forced alignment fails to produce valid alignments."""


def normalize_text_for_alignment(text: str) -> str:
    """Normalize text by uppercase and removing non-alphabetic/punctuation tokens."""
    # Convert to uppercase, replace apostrophe with empty, non-alpha with space
    text = text.upper()
    text = re.sub(r"[^A-Z\s]", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stub_forced_align(
    transcript: str,
    start_s: float,
    end_s: float,
    fail: bool = False,
) -> list[WordTiming]:
    """Offline deterministic stub for forced alignment."""
    if fail:
        raise AlignmentError("Simulated alignment failure")

    words = transcript.strip().split()
    if not words:
        return []

    duration = max(0.1, end_s - start_s)
    word_dur = duration / len(words)

    timings: list[WordTiming] = []
    for i, w in enumerate(words):
        w_s = start_s + i * word_dur
        w_e = w_s + word_dur
        timings.append(
            WordTiming(
                w=w,
                s=round(float(w_s), 3),
                e=round(float(w_e), 3),
                conf=0.95,
            )
        )
    return timings


def forced_align_segment(
    waveform: torch.Tensor,
    sr: int,
    transcript: str,
    model_name: str = "wav2vec2-base-960h",
    device: str = "cpu",
) -> list[WordTiming]:
    """Perform forced alignment using torchaudio wav2vec2 CTC."""
    norm_text = normalize_text_for_alignment(transcript)
    words = norm_text.split()
    if not words:
        return []

    try:
        # Load bundle
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().to(device)
        labels = bundle.get_labels()
        dictionary = {c: i for i, c in enumerate(labels)}

        # Resample if needed
        if sr != bundle.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, bundle.sample_rate)
            waveform = resampler(waveform)

        with torch.inference_mode():
            emission, _ = model(waveform.to(device))

        # Tokenize words to labels
        token_indices: list[int] = []
        for word in words:
            for char in word:
                if char in dictionary:
                    token_indices.append(dictionary[char])
            token_indices.append(dictionary.get("|", 0))  # word separator
        if token_indices and token_indices[-1] == dictionary.get("|", 0):
            token_indices.pop()

        targets = torch.tensor([token_indices], dtype=torch.int32, device=device)
        # Call forced_align
        aligned_tokens, scores = torchaudio.functional.forced_align(
            emission, targets, blank=0
        )

        # Merge tokens back to words
        frame_dur_s = waveform.shape[1] / bundle.sample_rate / emission.shape[1]
        timings: list[WordTiming] = []
        # Approximate word boundaries from tokens
        cur_w_idx = 0
        w_start = 0.0

        for frame_idx, token_id in enumerate(aligned_tokens[0]):
            tok = labels[token_id] if token_id < len(labels) else ""
            if tok == "|" and cur_w_idx < len(words):
                w_end = frame_idx * frame_dur_s
                timings.append(
                    WordTiming(
                        w=words[cur_w_idx],
                        s=round(float(w_start), 3),
                        e=round(float(w_end), 3),
                        conf=0.90,
                    )
                )
                cur_w_idx += 1
                w_start = w_end

        if cur_w_idx < len(words):
            w_end = emission.shape[1] * frame_dur_s
            timings.append(
                WordTiming(
                    w=words[cur_w_idx],
                    s=round(float(w_start), 3),
                    e=round(float(w_end), 3),
                    conf=0.90,
                )
            )

        return timings
    except Exception as e:
        raise AlignmentError(f"Forced alignment failed: {e}") from e
