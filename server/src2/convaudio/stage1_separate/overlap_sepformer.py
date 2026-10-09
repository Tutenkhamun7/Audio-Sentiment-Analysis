"""Targeted overlap separation using SpeechBrain SepFormer and speaker timbral matching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torchaudio

from convaudio.config import Stage1Config

logger = logging.getLogger(__name__)


@dataclass
class OverlapSegment:
    """Represents a pairwise speaker overlap interval."""

    start: float
    end: float
    speaker_a: str
    speaker_b: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class SourceSeparator(Protocol):
    """Protocol for two-source separation backends."""

    def separate(self, mix: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Separate 1D or [1, T] mixture into two [T] sources."""
        ...


class SepformerSeparator:
    """Wrapper around SpeechBrain SepFormer for 2-speaker separation."""

    def __init__(
        self,
        model_source: str = "speechbrain/sepformer-wsj02mix",
        device: str | None = None,
    ) -> None:
        self.model_source = model_source
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._model: Any = None

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            logger.info("Loading SepFormer model from %s on %s...", self.model_source, self.device)
            from speechbrain.inference.separation import SepformerSeparation

            self._model = SepformerSeparation.from_hparams(
                source=self.model_source,
                run_opts={"device": self.device},
            )
        return self._model

    def separate(self, mix: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Separate a mono mixture slice into two sources."""
        model = self._ensure_loaded()
        if mix.ndim == 1:
            mix = mix.unsqueeze(0)

        # Expected sample rate
        model_sr = getattr(getattr(model, "hparams", None), "sample_rate", 16000)
        orig_sr = sr

        if orig_sr != model_sr:
            mix_input = torchaudio.functional.resample(mix, orig_sr, model_sr)
        else:
            mix_input = mix

        with torch.no_grad():
            est_sources = model.separate_batch(mix_input)  # [1, T, 2]

        s0 = est_sources[0, :, 0].cpu()
        s1 = est_sources[0, :, 1].cpu()

        if orig_sr != model_sr:
            s0 = torchaudio.functional.resample(s0.unsqueeze(0), model_sr, orig_sr).squeeze(0)
            s1 = torchaudio.functional.resample(s1.unsqueeze(0), model_sr, orig_sr).squeeze(0)

        # Align lengths
        target_len = mix.shape[-1]
        s0 = s0[:target_len]
        s1 = s1[:target_len]
        if s0.shape[-1] < target_len:
            s0 = torch.nn.functional.pad(s0, (0, target_len - s0.shape[-1]))
            s1 = torch.nn.functional.pad(s1, (0, target_len - s1.shape[-1]))

        return s0, s1


def extract_acoustic_profile(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Extract a timbral acoustic embedding (mean MFCC) for speaker matching."""
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav.float()

    # Energy gate to drop silence/noise
    abs_wav = wav.abs()
    threshold = abs_wav.mean() * 0.2
    active_mask = (abs_wav > threshold).squeeze(0)

    if active_mask.sum() > int(sr * 0.08):  # at least 80ms active
        active_wav = wav[:, active_mask]
    else:
        active_wav = wav

    if active_wav.shape[-1] < int(sr * 0.04):
        return torch.zeros(13)

    try:
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
        )
        mfcc = mfcc_transform(active_wav)
        profile = mfcc.squeeze(0).mean(dim=-1)
        norm = torch.linalg.norm(profile)
        if norm > 1e-6:
            profile = profile / norm
        return profile
    except Exception:
        return torch.zeros(13)


def find_pairwise_overlaps(
    diarization: Any,
    min_duration_s: float = 0.05,
) -> list[OverlapSegment]:
    """Extract pairwise overlapping speech regions from a pyannote diarization object."""
    overlaps: list[OverlapSegment] = []
    tracks = list(diarization.itertracks(yield_label=True))

    for i in range(len(tracks)):
        seg1, _, spk1 = tracks[i]
        for j in range(i + 1, len(tracks)):
            seg2, _, spk2 = tracks[j]
            if spk1 == spk2:
                continue

            inter = seg1 & seg2
            if inter and inter.duration >= min_duration_s:
                overlaps.append(
                    OverlapSegment(
                        start=float(inter.start),
                        end=float(inter.end),
                        speaker_a=str(spk1),
                        speaker_b=str(spk2),
                    )
                )

    # Sort chronologically
    overlaps.sort(key=lambda o: (o.start, o.end))
    return overlaps


def crossfade_splice(
    base: torch.Tensor,
    patch: torch.Tensor,
    start_idx: int,
    end_idx: int,
    fade_len: int = 160,
) -> None:
    """Splice patch into base[start_idx:end_idx] with boundary cross-fading."""
    chunk_len = end_idx - start_idx
    if chunk_len <= 0 or patch.shape[-1] == 0:
        return

    patch_slice = patch[:chunk_len].clone()
    actual_fade = min(fade_len, chunk_len // 4)

    if actual_fade > 0:
        ramp_in = torch.linspace(0.0, 1.0, actual_fade)
        ramp_out = torch.linspace(1.0, 0.0, actual_fade)

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


def assign_separated_tracks(
    s0: torch.Tensor,
    s1: torch.Tensor,
    ref_a: torch.Tensor,
    ref_b: torch.Tensor,
    lead_a_active: bool,
    lead_b_active: bool,
    lead_s0_energy: float,
    lead_s1_energy: float,
    sr: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign separated tracks (s0, s1) to (speaker_a, speaker_b) using timbre & context."""
    # 1. Lead-in context heuristic: if one speaker spoke uniquely during pre-overlap context
    if lead_a_active and not lead_b_active and (lead_s0_energy + lead_s1_energy) > 1e-6:
        if lead_s0_energy > lead_s1_energy * 1.5:
            return s0, s1
        elif lead_s1_energy > lead_s0_energy * 1.5:
            return s1, s0

    if lead_b_active and not lead_a_active and (lead_s0_energy + lead_s1_energy) > 1e-6:
        if lead_s1_energy > lead_s0_energy * 1.5:
            return s0, s1
        elif lead_s0_energy > lead_s1_energy * 1.5:
            return s1, s0

    # 2. Timbral / MFCC profile cosine matching
    prof0 = extract_acoustic_profile(s0, sr)
    prof1 = extract_acoustic_profile(s1, sr)

    sim_0_a = float(torch.dot(prof0, ref_a).item()) if torch.linalg.norm(ref_a) > 0 else 0.0
    sim_1_b = float(torch.dot(prof1, ref_b).item()) if torch.linalg.norm(ref_b) > 0 else 0.0
    sim_0_b = float(torch.dot(prof0, ref_b).item()) if torch.linalg.norm(ref_b) > 0 else 0.0
    sim_1_a = float(torch.dot(prof1, ref_a).item()) if torch.linalg.norm(ref_a) > 0 else 0.0

    score_direct = sim_0_a + sim_1_b
    score_inverted = sim_0_b + sim_1_a

    if score_direct >= score_inverted:
        return s0, s1
    return s1, s0


def separate_and_stitch_overlaps(
    mono: torch.Tensor,
    sr: int,
    diarization: Any,
    labels: list[str],
    config: Stage1Config,
    separator: SourceSeparator | None = None,
) -> dict[str, torch.Tensor]:
    """Build separated audio streams for each speaker by targeting overlap intervals with SepFormer."""
    total_samples = mono.shape[-1]
    mono_1d = mono.squeeze(0) if mono.ndim > 1 else mono

    # Step 1: Initialize speaker streams with active speech according to diarization
    speaker_waves: dict[str, torch.Tensor] = {}

    # Find overlap intervals
    min_dur = getattr(config, "min_overlap_duration_s", 0.05)
    overlaps = find_pairwise_overlaps(diarization, min_duration_s=min_dur)

    # Build an interval set for overlaps to identify clean single-speaker regions
    for spk in labels:
        timeline = diarization.label_timeline(spk)
        spk_wav = torch.zeros(total_samples, dtype=torch.float32)

        for seg in timeline:
            s_idx = max(0, int(seg.start * sr))
            e_idx = min(total_samples, int(seg.end * sr))
            if e_idx > s_idx:
                spk_wav[s_idx:e_idx] = mono_1d[s_idx:e_idx]

        speaker_waves[spk] = spk_wav

    # Step 2: If overlap separation is disabled or no overlaps, return standard diarized streams
    if not getattr(config, "separate_overlap", True) or not overlaps:
        return speaker_waves

    # Step 3: Extract clean reference acoustic profiles for each speaker
    ref_profiles: dict[str, torch.Tensor] = {}
    for spk in labels:
        timeline = diarization.label_timeline(spk)
        clean_pieces: list[torch.Tensor] = []
        for seg in timeline:
            seg_start, seg_end = float(seg.start), float(seg.end)
            # Check overlap collision
            is_overlapped = any(
                max(seg_start, ov.start) < min(seg_end, ov.end)
                for ov in overlaps
                if ov.speaker_a == spk or ov.speaker_b == spk
            )
            if not is_overlapped:
                s_i = max(0, int(seg_start * sr))
                e_i = min(total_samples, int(seg_end * sr))
                if e_i > s_i:
                    clean_pieces.append(mono_1d[s_i:e_i])

        if clean_pieces:
            clean_cat = torch.cat(clean_pieces)
            ref_profiles[spk] = extract_acoustic_profile(clean_cat, sr)
        else:
            # Fallback to entire active timeline
            spk_active = speaker_waves[spk]
            ref_profiles[spk] = extract_acoustic_profile(spk_active, sr)

    # Step 4: Ensure separator is ready
    pad_s = getattr(config, "overlap_padding_s", 0.2)
    pad_samples = int(pad_s * sr)

    if separator is None:
        try:
            model_id = getattr(config, "sepformer_model", "speechbrain/sepformer-wham16k")
            separator = SepformerSeparator(model_source=model_id)
        except Exception as e:
            logger.warning(
                "Could not initialize SepformerSeparator (%s). Keeping standard masking.", e
            )
            return speaker_waves

    # Step 5: Process each overlap segment
    for ov in overlaps:
        spk_a, spk_b = ov.speaker_a, ov.speaker_b
        if spk_a not in speaker_waves or spk_b not in speaker_waves:
            continue

        start_s = ov.start
        end_s = ov.end

        # Padded slice bounds
        slice_start_s = max(0.0, start_s - pad_s)
        slice_end_s = min(float(total_samples) / sr, end_s + pad_s)

        slice_start_idx = int(slice_start_s * sr)
        slice_end_idx = int(slice_end_s * sr)

        if slice_end_idx - slice_start_idx < int(sr * 0.05):
            continue

        mix_slice = mono_1d[slice_start_idx:slice_end_idx]

        try:
            s0_pad, s1_pad = separator.separate(mix_slice, sr=sr)
        except Exception as err:
            logger.warning("SepFormer separation failed on chunk %.2f-%.2f: %s", start_s, end_s, err)
            continue

        # Overlap offset within padded slice
        core_start_rel = int((start_s - slice_start_s) * sr)
        core_end_rel = core_start_rel + int((end_s - start_s) * sr)
        core_end_rel = min(core_end_rel, s0_pad.shape[-1])

        # Pre-overlap lead context analysis
        lead_len = min(core_start_rel, pad_samples)
        if lead_len > 0:
            lead_s0 = s0_pad[core_start_rel - lead_len : core_start_rel]
            lead_s1 = s1_pad[core_start_rel - lead_len : core_start_rel]
            lead_e0 = float(lead_s0.pow(2).mean().item())
            lead_e1 = float(lead_s1.pow(2).mean().item())
        else:
            lead_e0, lead_e1 = 0.0, 0.0

        lead_a_active = (start_s - slice_start_s) > 0.05 and any(
            seg.start <= slice_start_s and seg.end >= start_s
            for seg in diarization.label_timeline(spk_a)
        )
        lead_b_active = (start_s - slice_start_s) > 0.05 and any(
            seg.start <= slice_start_s and seg.end >= start_s
            for seg in diarization.label_timeline(spk_b)
        )

        # Core overlap slices
        s0_core = s0_pad[core_start_rel:core_end_rel]
        s1_core = s1_pad[core_start_rel:core_end_rel]

        # Resolve permutation
        assigned_a, assigned_b = assign_separated_tracks(
            s0=s0_core,
            s1=s1_core,
            ref_a=ref_profiles.get(spk_a, torch.zeros(13)),
            ref_b=ref_profiles.get(spk_b, torch.zeros(13)),
            lead_a_active=lead_a_active,
            lead_b_active=lead_b_active,
            lead_s0_energy=lead_e0,
            lead_s1_energy=lead_e1,
            sr=sr,
        )

        # Splice with crossfade into each speaker's stream
        ov_start_idx = int(start_s * sr)
        ov_end_idx = min(total_samples, ov_start_idx + len(assigned_a))

        fade_len = min(int(0.015 * sr), len(assigned_a) // 4)
        crossfade_splice(speaker_waves[spk_a], assigned_a, ov_start_idx, ov_end_idx, fade_len=fade_len)
        crossfade_splice(speaker_waves[spk_b], assigned_b, ov_start_idx, ov_end_idx, fade_len=fade_len)

    return speaker_waves
