"""Audio file I/O, format validation, and resampling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import soundfile as sf
import torch
import torchaudio.functional as AF

from convaudio.errors import UnsupportedAudioFormat
from convaudio.stage2_timeline.schema import AudioMeta


def calculate_sha256(file_path: Path | str) -> str:
    """Calculate SHA256 hex digest of file contents."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_audio(
    file_path: Path | str,
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, AudioMeta]:
    """Decode audio via soundfile and resample to target_sr float32 via torchaudio.

    Raises UnsupportedAudioFormat on unreadable or unsupported audio formats without shelling out.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Determine file extension/format
    detected_format = path.suffix.lstrip(".").lower()
    if not detected_format:
        detected_format = "unknown"

    sha256_hash = calculate_sha256(path)

    try:
        data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        raise UnsupportedAudioFormat(
            detected_format=detected_format,
            message=f"Could not decode audio file '{path.name}' with format '{detected_format}': {e}",
        ) from e

    # sf.read returns (n_samples, channels)
    n_samples, n_channels = data.shape
    duration_s = float(n_samples) / float(orig_sr) if orig_sr > 0 else 0.0

    # Convert to torch tensor of shape (channels, samples)
    waveform = torch.from_numpy(data.T)

    # Resample if needed
    if orig_sr != target_sr:
        waveform = AF.resample(waveform, orig_freq=orig_sr, new_freq=target_sr)

    audio_meta = AudioMeta(
        path=str(path.resolve()),
        sha256=sha256_hash,
        original_sr=orig_sr,
        channels=n_channels,
        duration_s=duration_s,
        branch="",  # Populated downstream
    )

    return waveform, audio_meta


def save_audio(file_path: Path | str, waveform: torch.Tensor, sr: int = 16000) -> None:
    """Save float32 waveform tensor (shape (channels, samples) or (samples,)) to WAV."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = waveform.detach().cpu().numpy()
    if data.ndim == 2:
        data = data.T  # (samples, channels)
    sf.write(str(p), data, sr, subtype="FLOAT")
