"""Synthetic call audio generator for deterministic, offline testing."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def generate_tone_segment(
    freq_hz: float,
    duration_s: float,
    sr: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a pure sine tone with brief cosine fade-in/fade-out."""
    n_samples = int(duration_s * sr)
    t = np.linspace(0, duration_s, n_samples, endpoint=False, dtype=np.float32)
    sig = amplitude * np.sin(2 * np.pi * freq_hz * t)

    # 10ms window fade in/out
    fade_len = min(int(0.01 * sr), n_samples // 4)
    if fade_len > 0:
        window = np.sin(np.linspace(0, np.pi / 2, fade_len, dtype=np.float32)) ** 2
        sig[:fade_len] *= window
        sig[-fade_len:] *= window[::-1]

    return sig.astype(np.float32)


def generate_clean_stereo_call(
    path: Path,
    duration_s: float = 6.0,
    sr: int = 16000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate clean 2-channel stereo audio where each channel has one isolated speaker."""
    n_samples = int(duration_s * sr)
    ch0 = np.zeros(n_samples, dtype=np.float32)
    ch1 = np.zeros(n_samples, dtype=np.float32)

    # Place tones proportionally or safely within duration_s
    # Speaker 0 (ch0): active [0.1 * dur, 0.45 * dur] and [0.6 * dur, 0.9 * dur]
    t0_start = int(0.1 * duration_s * sr)
    t0_dur = 0.35 * duration_s
    sig_a1 = generate_tone_segment(440.0, t0_dur, sr=sr, amplitude=0.6)
    ch0[t0_start : t0_start + len(sig_a1)] += sig_a1

    t1_start = int(0.6 * duration_s * sr)
    t1_dur = 0.3 * duration_s
    sig_a2 = generate_tone_segment(440.0, t1_dur, sr=sr, amplitude=0.6)
    ch0[t1_start : t1_start + len(sig_a2)] += sig_a2

    # Speaker 1 (ch1): active [0.35 * dur, 0.55 * dur] (880 Hz)
    tb_start = int(0.35 * duration_s * sr)
    tb_dur = 0.2 * duration_s
    sig_b = generate_tone_segment(880.0, tb_dur, sr=sr, amplitude=0.5)
    ch1[tb_start : tb_start + len(sig_b)] += sig_b

    # Add very faint uncorrelated dither to prevent perfect 0 energy / div-by-zero
    rng = np.random.default_rng(seed=42)
    ch0 += 1e-5 * rng.standard_normal(n_samples, dtype=np.float32)
    ch1 += 1e-5 * rng.standard_normal(n_samples, dtype=np.float32)

    stereo = np.stack([ch0, ch1], axis=1)  # (n_samples, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sr, subtype="FLOAT")
    return ch0, ch1


def generate_low_overlap_mono_call(
    path: Path,
    duration_s: float = 6.0,
    sr: int = 16000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate mono mixture with sequential speakers (almost no overlap)."""
    n_samples = int(duration_s * sr)
    s0 = np.zeros(n_samples, dtype=np.float32)
    s1 = np.zeros(n_samples, dtype=np.float32)

    # Speaker 0: active [0.1 * dur, 0.45 * dur]
    t0_start = int(0.1 * duration_s * sr)
    t0_dur = 0.35 * duration_s
    sig_a = generate_tone_segment(440.0, t0_dur, sr=sr, amplitude=0.6)
    s0[t0_start : t0_start + len(sig_a)] += sig_a

    # Speaker 1: active [0.55 * dur, 0.9 * dur]
    t1_start = int(0.55 * duration_s * sr)
    t1_dur = 0.35 * duration_s
    sig_b = generate_tone_segment(880.0, t1_dur, sr=sr, amplitude=0.5)
    s1[t1_start : t1_start + len(sig_b)] += sig_b

    mono = s0 + s1
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mono, sr, subtype="FLOAT")
    return s0, s1


def generate_high_overlap_mono_call(
    path: Path,
    duration_s: float = 6.0,
    sr: int = 16000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate mono mixture with deliberate speaker interruption and high overlap.

    Speaker A starts early, Speaker B interrupts while A is active, A stops shortly after.
    """
    n_samples = int(duration_s * sr)
    s0 = np.zeros(n_samples, dtype=np.float32)
    s1 = np.zeros(n_samples, dtype=np.float32)

    # Speaker 0: active [0.1 * dur, 0.6 * dur] (e.g. 0.6s to 3.6s in 6s call)
    t0_start = int(0.1 * duration_s * sr)
    t0_dur = 0.5 * duration_s
    sig_a = generate_tone_segment(440.0, t0_dur, sr=sr, amplitude=0.6)
    s0[t0_start : t0_start + len(sig_a)] += sig_a

    # Speaker 1: active [0.4 * dur, 0.9 * dur] (e.g. 2.4s to 5.4s) -> overlaps in [0.4*dur, 0.6*dur]
    t1_start = int(0.4 * duration_s * sr)
    t1_dur = 0.5 * duration_s
    sig_b = generate_tone_segment(880.0, t1_dur, sr=sr, amplitude=0.5)
    s1[t1_start : t1_start + len(sig_b)] += sig_b

    mono = s0 + s1
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), mono, sr, subtype="FLOAT")
    return s0, s1
