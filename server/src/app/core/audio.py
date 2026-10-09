"""In-memory audio decoding, resampling, slicing, and crossfade operations."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

import soundfile as sf
import torch
import torchaudio


def load_audio_from_bytes(
    audio_bytes: bytes | BinaryIO,
    target_sr: int = 16000,
) -> tuple[torch.Tensor, int, float]:
    """Decode raw audio bytes directly in RAM into a float32 torch.Tensor.

    Returns:
        (waveform, original_sr, duration_s)
        waveform is shape [channels, samples]
    """
    if isinstance(audio_bytes, bytes):
        buffer = io.BytesIO(audio_bytes)
    else:
        buffer = audio_bytes

    data, orig_sr = sf.read(buffer, dtype="float32")
    # data is [samples] or [samples, channels]
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data.T)

    duration_s = waveform.shape[-1] / float(orig_sr)

    if orig_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)

    return waveform, target_sr, duration_s


def load_audio_from_file(
    path: Union[str, Path],
    target_sr: int = 16000,
) -> tuple[torch.Tensor, int, float]:
    """Read an audio file from disk into an in-memory float32 torch.Tensor."""
    p = Path(path)
    with open(p, "rb") as f:
        return load_audio_from_bytes(f.read(), target_sr=target_sr)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Average all channels down to 1D mono shape [1, samples] or [samples]."""
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        return waveform.mean(dim=0, keepdim=True)
    return waveform


def slice_waveform(
    waveform: torch.Tensor,
    start_s: float,
    end_s: float,
    sr: int = 16000,
) -> torch.Tensor:
    """Slice an audio segment in RAM by start and end seconds."""
    s_idx = max(0, int(start_s * sr))
    e_idx = min(waveform.shape[-1], int(end_s * sr))
    if waveform.ndim == 1:
        return waveform[s_idx:e_idx]
    return waveform[:, s_idx:e_idx]


def crossfade_splice(
    base: torch.Tensor,
    patch: torch.Tensor,
    start_idx: int,
    end_idx: int,
    fade_len: int = 160,
) -> None:
    """In-place smooth crossfade splice of patch into base[start_idx:end_idx]."""
    chunk_len = end_idx - start_idx
    if chunk_len <= 0 or patch.shape[-1] == 0:
        return

    patch_slice = patch[:chunk_len].clone()
    actual_fade = min(fade_len, chunk_len // 4)

    if actual_fade > 0:
        ramp_in = torch.linspace(0.0, 1.0, actual_fade, dtype=patch.dtype, device=patch.device)
        ramp_out = torch.linspace(1.0, 0.0, actual_fade, dtype=patch.dtype, device=patch.device)

        # Cross-fade in
        base_head = base[start_idx : start_idx + actual_fade]
        patch_slice[:actual_fade] = (
            patch_slice[:actual_fade] * ramp_in + base_head * (1.0 - ramp_in)
        )

        # Cross-fade out
        base_tail = base[end_idx - actual_fade : end_idx]
        patch_slice[-actual_fade:] = (
            patch_slice[-actual_fade:] * ramp_out + base_tail * (1.0 - ramp_out)
        )

    base[start_idx:end_idx] = patch_slice
