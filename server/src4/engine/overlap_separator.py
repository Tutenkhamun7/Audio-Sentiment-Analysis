"""In-memory targeted SepFormer overlap separation with speaker matching."""

from __future__ import annotations

import logging
from typing import Dict

import torch
import torchaudio

from src3.core.audio import crossfade_splice
from src3.core.model_registry import ModelRegistry
from src3.engine.diarizer import DiarizationResult

logger = logging.getLogger(__name__)


def extract_acoustic_profile(
    wav: torch.Tensor,
    sr: int = 16000,
    embedder: Any = None,
) -> torch.Tensor:
    """Compute a normalized speaker embedding vector for speaker matching.

    Uses Pyannote's deep neural speaker embedding (256-dim) when available,
    falling back to a 13-dim normalized MFCC timbre profile when audio is
    extremely short or when neural embedder is uninitialized.
    """
    if wav.ndim == 1:
        wav_1d = wav
    else:
        wav_1d = wav.squeeze()
    wav_1d = wav_1d.float()

    # Energy gate
    energy = wav_1d.abs()
    threshold = energy.mean() * 0.2
    active_mask = energy > threshold

    if active_mask.sum() > int(sr * 0.08):
        active_wav = wav_1d[active_mask]
    else:
        active_wav = wav_1d

    min_samples = int(sr * 0.04)
    if active_wav.shape[-1] < min_samples:
        return torch.zeros(256 if embedder is not None else 13)

    # 1. Neural speaker embedding pass (Pyannote 256-dim)
    if embedder is not None and active_wav.shape[-1] >= 400:
        try:
            dev = getattr(embedder, "device", torch.device("cpu"))
            in_wav = active_wav.unsqueeze(0).unsqueeze(0).to(dev).float()
            with torch.no_grad():
                out = embedder(in_wav)
            emb = torch.from_numpy(out).squeeze().float()
            norm = torch.linalg.norm(emb)
            if norm > 1e-6:
                return emb / norm
            return emb
        except Exception as e:
            logger.debug("Pyannote embedder extraction failed (%s), falling back to MFCC", e)

    # 2. MFCC Acoustic timbre fallback (13-dim)
    try:
        mfcc_tf = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_tf(active_wav.unsqueeze(0).float().cpu())
        profile = mfcc.squeeze(0).mean(dim=-1)
        norm = torch.linalg.norm(profile)
        if norm > 1e-6:
            profile = profile / norm
        return profile
    except Exception:
        return torch.zeros(13)


def separate_overlaps_in_memory(
    mono: torch.Tensor,
    sr: int,
    diarization: DiarizationResult,
    overlap_padding_s: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """Untangle colliding speech in RAM and return pure separated tensors per speaker."""
    total_samples = mono.shape[-1]
    mono_1d = mono.squeeze(0) if mono.ndim > 1 else mono

    # 1. Initialize speaker audio streams with their active speech masks
    speaker_streams: dict[str, torch.Tensor] = {}

    for spk in diarization.speakers:
        spk_wav = torch.zeros(total_samples, dtype=torch.float32)
        spk_turns = [t for t in diarization.turns if t.speaker == spk]

        for t in spk_turns:
            s_idx = max(0, int(t.start * sr))
            e_idx = min(total_samples, int(t.end * sr))
            if e_idx > s_idx:
                spk_wav[s_idx:e_idx] = mono_1d[s_idx:e_idx]

        speaker_streams[spk] = spk_wav

    # If no overlaps, return standard masked streams
    if not diarization.overlaps:
        return speaker_streams

    # 2. Extract clean single-speaker profiles for permutation matching
    ref_profiles: dict[str, torch.Tensor] = {}
    registry = ModelRegistry.get_instance()
    embedder: Any = None
    try:
        embedder = registry.get_speaker_embedder()
    except Exception as e:
        logger.debug("Speaker embedder unavailable for permutation matching: %s", e)

    for spk in diarization.speakers:
        spk_turns = [t for t in diarization.turns if t.speaker == spk]
        clean_pieces: list[torch.Tensor] = []
        for t in spk_turns:
            is_collision = any(
                max(t.start, ov.start) < min(t.end, ov.end)
                for ov in diarization.overlaps
                if ov.speaker_a == spk or ov.speaker_b == spk
            )
            if not is_collision:
                s_i = max(0, int(t.start * sr))
                e_i = min(total_samples, int(t.end * sr))
                if e_i > s_i:
                    clean_pieces.append(mono_1d[s_i:e_i])

        if clean_pieces:
            clean_audio = torch.cat(clean_pieces)
            ref_profiles[spk] = extract_acoustic_profile(clean_audio, sr, embedder=embedder)
        else:
            ref_profiles[spk] = extract_acoustic_profile(speaker_streams[spk], sr, embedder=embedder)

    # 3. Obtain SepFormer from registry
    try:
        separator = registry.get_separator()
    except Exception as e:
        logger.warning("SepFormer unavailable (%s). Proceeding with clean masking.", e)
        return speaker_streams

    model_sr = getattr(getattr(separator, "hparams", None), "sample_rate", 8000)
    pad_samples = int(overlap_padding_s * sr)

    # 4. Separate each collision window
    default_ref_dim = 256 if embedder is not None else 13
    for ov in diarization.overlaps:
        spk_a, spk_b = ov.speaker_a, ov.speaker_b
        if spk_a not in speaker_streams or spk_b not in speaker_streams:
            continue

        start_s, end_s = ov.start, ov.end
        slice_start_s = max(0.0, start_s - overlap_padding_s)
        slice_end_s = min(float(total_samples) / sr, end_s + overlap_padding_s)

        s_idx = int(slice_start_s * sr)
        e_idx = int(slice_end_s * sr)
        if (e_idx - s_idx) < int(sr * 0.05):
            continue

        mix_slice = mono_1d[s_idx:e_idx].unsqueeze(0)

        # Resample for SepFormer if needed
        if sr != model_sr:
            mix_input = torchaudio.functional.resample(mix_slice, sr, model_sr)
        else:
            mix_input = mix_slice

        try:
            device = next(separator.mods.parameters()).device
            mix_input = mix_input.to(device)
            with torch.no_grad():
                est_sources = separator.separate_batch(mix_input)  # [1, T, 2]
            s0_raw = est_sources[0, :, 0].cpu()
            s1_raw = est_sources[0, :, 1].cpu()
        except Exception as err:
            logger.warning("SepFormer failed on collision [%.2f - %.2f]: %s", start_s, end_s, err)
            continue

        if sr != model_sr:
            s0_pad = torchaudio.functional.resample(s0_raw.unsqueeze(0), model_sr, sr).squeeze(0)
            s1_pad = torchaudio.functional.resample(s1_raw.unsqueeze(0), model_sr, sr).squeeze(0)
        else:
            s0_pad, s1_pad = s0_raw, s1_raw

        # Core overlap region offsets
        core_s_rel = int((start_s - slice_start_s) * sr)
        core_e_rel = core_s_rel + int((end_s - start_s) * sr)
        core_e_rel = min(core_e_rel, s0_pad.shape[-1])

        # Pre-collision context energy
        lead_len = min(core_s_rel, pad_samples)
        if lead_len > 0:
            lead_e0 = float(s0_pad[core_s_rel - lead_len : core_s_rel].pow(2).mean().item())
            lead_e1 = float(s1_pad[core_s_rel - lead_len : core_s_rel].pow(2).mean().item())
        else:
            lead_e0, lead_e1 = 0.0, 0.0

        lead_a_active = any(
            t.start <= slice_start_s and t.end >= start_s
            for t in diarization.turns
            if t.speaker == spk_a
        )
        lead_b_active = any(
            t.start <= slice_start_s and t.end >= start_s
            for t in diarization.turns
            if t.speaker == spk_b
        )

        s0_core = s0_pad[core_s_rel:core_e_rel]
        s1_core = s1_pad[core_s_rel:core_e_rel]

        # Resolve permutation
        assigned_a, assigned_b = _resolve_permutation(
            s0=s0_core,
            s1=s1_core,
            ref_a=ref_profiles.get(spk_a, torch.zeros(default_ref_dim)),
            ref_b=ref_profiles.get(spk_b, torch.zeros(default_ref_dim)),
            lead_a_active=lead_a_active,
            lead_b_active=lead_b_active,
            lead_e0=lead_e0,
            lead_e1=lead_e1,
            sr=sr,
            embedder=embedder,
        )

        # Cross-fade splice back into each speaker stream
        ov_s_idx = int(start_s * sr)
        ov_e_idx = min(total_samples, ov_s_idx + len(assigned_a))
        fade_len = min(int(0.015 * sr), len(assigned_a) // 4)

        crossfade_splice(speaker_streams[spk_a], assigned_a, ov_s_idx, ov_e_idx, fade_len=fade_len)
        crossfade_splice(speaker_streams[spk_b], assigned_b, ov_s_idx, ov_e_idx, fade_len=fade_len)

    return speaker_streams


def _resolve_permutation(
    s0: torch.Tensor,
    s1: torch.Tensor,
    ref_a: torch.Tensor,
    ref_b: torch.Tensor,
    lead_a_active: bool,
    lead_b_active: bool,
    lead_e0: float,
    lead_e1: float,
    sr: int,
    embedder: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map s0 and s1 to Speaker A and Speaker B."""
    if lead_a_active and not lead_b_active and (lead_e0 + lead_e1) > 1e-6:
        if lead_e0 > lead_e1 * 1.5:
            return s0, s1
        elif lead_e1 > lead_e0 * 1.5:
            return s1, s0

    if lead_b_active and not lead_a_active and (lead_e0 + lead_e1) > 1e-6:
        if lead_e1 > lead_e0 * 1.5:
            return s0, s1
        elif lead_e0 > lead_e1 * 1.5:
            return s1, s0

    # Speaker profile matching (256-dim Pyannote neural embedding or 13-dim MFCC)
    p0 = extract_acoustic_profile(s0, sr, embedder=embedder)
    p1 = extract_acoustic_profile(s1, sr, embedder=embedder)

    # Dimensional safeguard: if shape mismatch occurs, fall back to pure 13-dim MFCC
    if p0.shape != ref_a.shape or p1.shape != ref_b.shape:
        p0 = extract_acoustic_profile(s0, sr, embedder=None)
        p1 = extract_acoustic_profile(s1, sr, embedder=None)
        if ref_a.shape != p0.shape:
            ref_a = extract_acoustic_profile(ref_a, sr, embedder=None) if ref_a.ndim > 1 else torch.zeros(13)
        if ref_b.shape != p1.shape:
            ref_b = extract_acoustic_profile(ref_b, sr, embedder=None) if ref_b.ndim > 1 else torch.zeros(13)

    sim_0_a = float(torch.dot(p0, ref_a).item()) if torch.linalg.norm(ref_a) > 0 else 0.0
    sim_1_b = float(torch.dot(p1, ref_b).item()) if torch.linalg.norm(ref_b) > 0 else 0.0
    sim_0_b = float(torch.dot(p0, ref_b).item()) if torch.linalg.norm(ref_b) > 0 else 0.0
    sim_1_a = float(torch.dot(p1, ref_a).item()) if torch.linalg.norm(ref_a) > 0 else 0.0

    score_direct = sim_0_a + sim_1_b
    score_inverted = sim_0_b + sim_1_a

    return (s0, s1) if score_direct >= score_inverted else (s1, s0)
