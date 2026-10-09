"""Clean-stereo path: per-channel VAD segmentation and provenance assignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from convaudio.io.audio import save_audio
from convaudio.stage2_timeline.schema import Turn, TurnOverlap, TurnQuality


def _energy_vad(
    wav: np.ndarray,
    sr: int,
    frame_rate_hz: float = 50.0,
    energy_threshold_db: float = -40.0,
    min_speech_duration_s: float = 0.2,
    min_silence_duration_s: float = 0.3,
) -> Tuple[np.ndarray, list[Tuple[float, float]]]:
    """Frame-level energy VAD returning boolean mask and merged active segments."""
    frame_len = int(sr / frame_rate_hz)
    n_frames = max(1, len(wav) // frame_len)
    frames_bool = np.zeros(n_frames, dtype=bool)

    # Compute short-time energy
    for i in range(n_frames):
        chunk = wav[i * frame_len : (i + 1) * frame_len]
        rms = np.sqrt(np.mean(chunk**2) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)
        if db > energy_threshold_db:
            frames_bool[i] = True

    # Merge short silence gaps and filter short speech spikes
    segments: list[Tuple[float, float]] = []
    in_speech = False
    start_f = 0

    for i in range(n_frames):
        if frames_bool[i] and not in_speech:
            in_speech = True
            start_f = i
        elif not frames_bool[i] and in_speech:
            in_speech = False
            dur_s = (i - start_f) / frame_rate_hz
            if dur_s >= min_speech_duration_s:
                segments.append((start_f / frame_rate_hz, i / frame_rate_hz))
    if in_speech:
        dur_s = (n_frames - start_f) / frame_rate_hz
        if dur_s >= min_speech_duration_s:
            segments.append((start_f / frame_rate_hz, n_frames / frame_rate_hz))

    # Merge segments closer than min_silence_duration_s
    merged: list[Tuple[float, float]] = []
    for s, e in segments:
        if not merged:
            merged.append((s, e))
        else:
            prev_s, prev_e = merged[-1]
            if s - prev_e < min_silence_duration_s:
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))

    # Rebuild mask from merged segments
    clean_mask = np.zeros(n_frames, dtype=bool)
    for s, e in merged:
        sf = int(s * frame_rate_hz)
        ef = min(n_frames, int(np.ceil(e * frame_rate_hz)))
        clean_mask[sf:ef] = True

    return clean_mask, merged


def process_clean_stereo(
    waveform: torch.Tensor,
    sr: int,
    out_dir: Path,
    call_id: str,
    frame_rate_hz: float = 50.0,
) -> dict[str, Any]:
    """Execute clean stereo branch: splits channels, performs VAD, saves outputs.

    Provenance is set to 'clean' on all turns.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    streams_dir = out_dir / "streams"
    streams_dir.mkdir(parents=True, exist_ok=True)

    ch0 = waveform[0].cpu().numpy()
    ch1 = waveform[1].cpu().numpy()

    spk0_id = "SPEAKER_00"
    spk1_id = "SPEAKER_01"

    spk0_path = streams_dir / f"{spk0_id}.wav"
    spk1_path = streams_dir / f"{spk1_id}.wav"

    save_audio(spk0_path, waveform[0:1], sr=sr)
    save_audio(spk1_path, waveform[1:2], sr=sr)

    # VAD segmentation
    mask0, segs0 = _energy_vad(ch0, sr=sr, frame_rate_hz=frame_rate_hz)
    mask1, segs1 = _energy_vad(ch1, sr=sr, frame_rate_hz=frame_rate_hz)

    # Frame-level overlap mask: where BOTH speakers are active simultaneously
    min_len = min(len(mask0), len(mask1))
    overlap_mask = mask0[:min_len] & mask1[:min_len]
    overlap_mask_path = out_dir / "overlap_mask.npy"
    np.save(str(overlap_mask_path), overlap_mask)

    # RTTM generation
    rttm_path = out_dir / "diarization.rttm"
    rttm_lines: list[str] = []

    raw_turns: list[dict[str, Any]] = []
    for s, e in segs0:
        dur = e - s
        rttm_lines.append(f"SPEAKER {call_id} 1 {s:.3f} {dur:.3f} <NA> <NA> {spk0_id} <NA> <NA>")
        raw_turns.append({"speaker": spk0_id, "start": s, "end": e})

    for s, e in segs1:
        dur = e - s
        rttm_lines.append(f"SPEAKER {call_id} 1 {s:.3f} {dur:.3f} <NA> <NA> {spk1_id} <NA> <NA>")
        raw_turns.append({"speaker": spk1_id, "start": s, "end": e})

    # Sort turns by start time
    raw_turns.sort(key=lambda t: (t["start"], t["end"]))
    rttm_path.write_text("\n".join(rttm_lines) + "\n", encoding="utf-8")

    # Generate dummy/spectral speaker enrollment embeddings
    # Distinct fixed embeddings for SPEAKER_00 and SPEAKER_01
    rng0 = np.random.default_rng(seed=100)
    emb0 = rng0.standard_normal(192).astype(np.float32)
    emb0 = emb0 / np.linalg.norm(emb0)

    rng1 = np.random.default_rng(seed=200)
    emb1 = rng1.standard_normal(192).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)

    embeddings_path = out_dir / "embeddings.npz"
    np.savez_compressed(str(embeddings_path), **{spk0_id: emb0, spk1_id: emb1})  # type: ignore[arg-type]

    total_speech_0 = float(sum(e - s for s, e in segs0))
    total_speech_1 = float(sum(e - s for s, e in segs1))

    speakers_meta = {
        spk0_id: {
            "stream_path": str(spk0_path),
            "total_speech_s": total_speech_0,
            "embedding_key": spk0_id,
        },
        spk1_id: {
            "stream_path": str(spk1_path),
            "total_speech_s": total_speech_1,
            "embedding_key": spk1_id,
        },
    }

    # Turns with provenance="clean"
    turns: list[Turn] = []
    for idx, t in enumerate(raw_turns):
        turn_id = f"turn_{idx:03d}"
        s = t["start"]
        e = t["end"]
        # Basic overlap ratio
        dur = e - s
        sf_idx = int(s * frame_rate_hz)
        ef_idx = min(len(overlap_mask), int(np.ceil(e * frame_rate_hz)))
        turn_overlap_frames = np.sum(overlap_mask[sf_idx:ef_idx]) if ef_idx > sf_idx else 0
        overlap_s = turn_overlap_frames / frame_rate_hz
        ratio = overlap_s / dur if dur > 0 else 0.0
        clean_s = max(0.0, dur - overlap_s)

        turns.append(
            Turn(
                turn_id=turn_id,
                speaker=t["speaker"],
                start=s,
                end=e,
                provenance="clean",  # clean provenance!
                overlap=TurnOverlap(
                    ratio=float(ratio),
                    clean_speech_s=float(clean_s),
                    interrupts=None,
                    interrupted_by=None,
                    preceding_gap_s=None,
                ),
                quality=TurnQuality(sep_cosine=1.0),
            )
        )

    return {
        "branch": "stereo_split",
        "streams": {spk0_id: str(spk0_path), spk1_id: str(spk1_path)},
        "diarization_rttm": str(rttm_path),
        "overlap_mask_path": str(overlap_mask_path),
        "embeddings_path": str(embeddings_path),
        "speakers": speakers_meta,
        "turns": turns,
        "n_frames": len(overlap_mask),
        "frame_rate_hz": frame_rate_hz,
        "rejected_sources": [],
    }


def process_no_split(
    waveform: torch.Tensor,
    sr: int,
    out_dir: Path,
    call_id: str,
    frame_rate_hz: float = 50.0,
    speaker_id: str = "SPEAKER_00",
) -> dict[str, Any]:
    """Execute no-split branch: does NOT split audio by speakers.

    Treats entire audio as a single unified speaker stream. VAD segmentation is
    performed on the unified audio with provenance set to 'clean' on all turns.
    Overlap is 0.0 everywhere.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    streams_dir = out_dir / "streams"
    streams_dir.mkdir(parents=True, exist_ok=True)

    # Downmix to mono if multi-channel
    if waveform.shape[0] > 1:
        mono_wave = torch.mean(waveform, dim=0, keepdim=True)
    else:
        mono_wave = waveform

    mono_np = mono_wave[0].cpu().numpy()

    spk_path = streams_dir / f"{speaker_id}.wav"
    save_audio(spk_path, mono_wave, sr=sr)

    # VAD segmentation on unified audio
    mask, segs = _energy_vad(mono_np, sr=sr, frame_rate_hz=frame_rate_hz)

    # Frame-level overlap mask: 0 everywhere (single speaker)
    overlap_mask = np.zeros(len(mask), dtype=bool)
    overlap_mask_path = out_dir / "overlap_mask.npy"
    np.save(str(overlap_mask_path), overlap_mask)

    # RTTM generation
    rttm_path = out_dir / "diarization.rttm"
    rttm_lines: list[str] = []
    raw_turns: list[dict[str, Any]] = []

    for s, e in segs:
        dur = e - s
        rttm_lines.append(f"SPEAKER {call_id} 1 {s:.3f} {dur:.3f} <NA> <NA> {speaker_id} <NA> <NA>")
        raw_turns.append({"speaker": speaker_id, "start": s, "end": e})

    raw_turns.sort(key=lambda t: (t["start"], t["end"]))
    rttm_path.write_text("\n".join(rttm_lines) + "\n", encoding="utf-8")

    # Enrollment embedding for single speaker
    rng = np.random.default_rng(seed=100)
    emb = rng.standard_normal(192).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    embeddings_path = out_dir / "embeddings.npz"
    np.savez_compressed(str(embeddings_path), **{speaker_id: emb})

    total_speech = float(sum(e - s for s, e in segs))
    speakers_meta = {
        speaker_id: {
            "stream_path": str(spk_path),
            "total_speech_s": total_speech,
            "embedding_key": speaker_id,
        }
    }

    turns: list[Turn] = []
    for idx, t in enumerate(raw_turns):
        turn_id = f"turn_{idx:03d}"
        s = t["start"]
        e = t["end"]
        dur = e - s

        turns.append(
            Turn(
                turn_id=turn_id,
                speaker=t["speaker"],
                start=s,
                end=e,
                provenance="clean",
                overlap=TurnOverlap(
                    ratio=0.0,
                    clean_speech_s=float(dur),
                    interrupts=None,
                    interrupted_by=None,
                    preceding_gap_s=None,
                ),
                quality=TurnQuality(sep_cosine=1.0),
            )
        )

    return {
        "branch": "no_split",
        "streams": {speaker_id: str(spk_path)},
        "diarization_rttm": str(rttm_path),
        "overlap_mask_path": str(overlap_mask_path),
        "embeddings_path": str(embeddings_path),
        "speakers": speakers_meta,
        "turns": turns,
        "n_frames": len(overlap_mask),
        "frame_rate_hz": frame_rate_hz,
        "rejected_sources": [],
    }
