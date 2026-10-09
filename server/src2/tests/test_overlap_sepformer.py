"""Unit tests for SepFormer targeted overlap separation and speaker assignment."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from convaudio.config import Stage1Config
from convaudio.stage1_separate.overlap_sepformer import (
    assign_separated_tracks,
    crossfade_splice,
    extract_acoustic_profile,
    find_pairwise_overlaps,
    separate_and_stitch_overlaps,
)


class DummySegment:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def __and__(self, other: DummySegment) -> DummySegment | None:
        s = max(self.start, other.start)
        e = min(self.end, other.end)
        if e > s:
            return DummySegment(s, e)
        return None


class DummyTimeline:
    def __init__(self, segments: list[DummySegment]) -> None:
        self.segments = segments

    def __iter__(self):
        return iter(self.segments)

    def duration(self) -> float:
        return sum(s.duration for s in self.segments)


class DummyAnnotation:
    """Mock Pyannote Annotation object."""

    def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
        # (start, end, speaker)
        self._tracks = tracks

    def itertracks(self, yield_label: bool = True):
        for s, e, spk in self._tracks:
            seg = DummySegment(s, e)
            if yield_label:
                yield seg, "track", spk
            else:
                yield seg, "track"

    def label_timeline(self, label: str) -> DummyTimeline:
        segs = [DummySegment(s, e) for s, e, spk in self._tracks if spk == label]
        return DummyTimeline(segs)

    def labels(self) -> list[str]:
        seen = []
        for _, _, spk in self._tracks:
            if spk not in seen:
                seen.append(spk)
        return seen


class MockSepformerSeparator:
    """Mock 2-source separator that decomposes into low-frequency and high-frequency components."""

    def __init__(self, invert_permutation: bool = False) -> None:
        self.invert_permutation = invert_permutation

    def separate(self, mix: torch.Tensor, sr: int) -> tuple[torch.Tensor, torch.Tensor]:
        if mix.ndim > 1:
            mix = mix.squeeze(0)

        # Create 2 synthetic separated sources
        # source_a: low-pass / smooth
        # source_b: high-pass / diff
        t = torch.linspace(0, 1, mix.shape[-1])
        s_a = mix * 0.7 + 0.1 * torch.sin(2 * np.pi * 300 * t)
        s_b = mix * 0.3 + 0.1 * torch.sin(2 * np.pi * 1200 * t)

        if self.invert_permutation:
            return s_b, s_a
        return s_a, s_b


def test_find_pairwise_overlaps():
    # Speaker 0: [0.0, 3.0]
    # Speaker 1: [2.0, 5.0]
    # Overlap: [2.0, 3.0]
    annotation = DummyAnnotation([
        (0.0, 3.0, "SPEAKER_00"),
        (2.0, 5.0, "SPEAKER_01"),
    ])

    overlaps = find_pairwise_overlaps(annotation, min_duration_s=0.1)
    assert len(overlaps) == 1
    assert pytest.approx(overlaps[0].start, 0.01) == 2.0
    assert pytest.approx(overlaps[0].end, 0.01) == 3.0
    assert set([overlaps[0].speaker_a, overlaps[0].speaker_b]) == {"SPEAKER_00", "SPEAKER_01"}


def test_no_overlaps_found():
    annotation = DummyAnnotation([
        (0.0, 2.0, "SPEAKER_00"),
        (2.5, 5.0, "SPEAKER_01"),
    ])
    overlaps = find_pairwise_overlaps(annotation, min_duration_s=0.1)
    assert len(overlaps) == 0


def test_extract_acoustic_profile():
    sr = 16000
    t = torch.linspace(0, 1.0, sr)
    tone_low = torch.sin(2 * np.pi * 200 * t)
    tone_high = torch.sin(2 * np.pi * 2000 * t)

    prof_low = extract_acoustic_profile(tone_low, sr)
    prof_high = extract_acoustic_profile(tone_high, sr)

    assert prof_low.shape == (13,)
    assert prof_high.shape == (13,)
    # Norm should be 1.0
    assert pytest.approx(torch.linalg.norm(prof_low).item(), 0.01) == 1.0
    assert pytest.approx(torch.linalg.norm(prof_high).item(), 0.01) == 1.0
    # Profiles should be distinct
    cosine_sim = torch.dot(prof_low, prof_high).item()
    assert cosine_sim < 0.95


def test_crossfade_splice():
    base = torch.zeros(1000)
    patch = torch.ones(400)
    crossfade_splice(base, patch, start_idx=200, end_idx=600, fade_len=20)

    # Core should be 1.0
    assert base[190] == 0.0
    assert base[250] == 1.0
    assert base[550] == 1.0
    assert base[610] == 0.0
    # Boundary ramps
    assert 0.0 < base[205].item() < 1.0
    assert 0.0 < base[595].item() < 1.0


def test_assign_separated_tracks_permutation():
    sr = 16000
    t = torch.linspace(0, 0.5, sr // 2)
    tone_a = torch.sin(2 * np.pi * 250 * t)
    tone_b = torch.sin(2 * np.pi * 1500 * t)

    ref_a = extract_acoustic_profile(tone_a, sr)
    ref_b = extract_acoustic_profile(tone_b, sr)

    # Inverted outputs: s0 is tone_b, s1 is tone_a
    assigned_a, assigned_b = assign_separated_tracks(
        s0=tone_b,
        s1=tone_a,
        ref_a=ref_a,
        ref_b=ref_b,
        lead_a_active=False,
        lead_b_active=False,
        lead_s0_energy=0.0,
        lead_s1_energy=0.0,
        sr=sr,
    )

    # assigned_a should match tone_a
    assert torch.allclose(assigned_a, tone_a)
    assert torch.allclose(assigned_b, tone_b)


def test_separate_and_stitch_end_to_end():
    sr = 16000
    dur_s = 5.0
    n_samples = int(dur_s * sr)
    t = torch.linspace(0, dur_s, n_samples)

    # Speaker 00 active [0, 3] with 300Hz tone
    # Speaker 01 active [2, 5] with 1200Hz tone
    # Overlap is [2, 3]
    wav_a = torch.sin(2 * np.pi * 300 * t)
    wav_b = torch.sin(2 * np.pi * 1200 * t)

    mono = torch.zeros(n_samples)
    mono[0 : int(3.0 * sr)] += wav_a[0 : int(3.0 * sr)]
    mono[int(2.0 * sr) : n_samples] += wav_b[int(2.0 * sr) : n_samples]

    annotation = DummyAnnotation([
        (0.0, 3.0, "SPEAKER_00"),
        (2.0, 5.0, "SPEAKER_01"),
    ])

    config = Stage1Config(
        separate_overlap=True,
        overlap_padding_s=0.2,
        min_overlap_duration_s=0.05,
    )

    mock_separator = MockSepformerSeparator(invert_permutation=False)

    streams = separate_and_stitch_overlaps(
        mono=mono,
        sr=sr,
        diarization=annotation,
        labels=["SPEAKER_00", "SPEAKER_01"],
        config=config,
        separator=mock_separator,
    )

    assert "SPEAKER_00" in streams
    assert "SPEAKER_01" in streams

    spk0_out = streams["SPEAKER_00"]
    spk1_out = streams["SPEAKER_01"]

    assert spk0_out.shape == (n_samples,)
    assert spk1_out.shape == (n_samples,)

    # Clean pre-overlap region [0, 1.8s] for Speaker 0 should be non-zero and match mono
    assert spk0_out[int(0.5 * sr) : int(1.5 * sr)].abs().mean() > 0.1
    # Speaker 1 should be completely silent in [0, 1.8s]
    assert spk1_out[int(0.5 * sr) : int(1.5 * sr)].abs().mean() == 0.0

    # Clean post-overlap region [3.2, 4.8s] for Speaker 1 should be non-zero and match mono
    assert spk1_out[int(3.5 * sr) : int(4.5 * sr)].abs().mean() > 0.1
    # Speaker 0 should be completely silent in [3.2, 4.8s]
    assert spk0_out[int(3.5 * sr) : int(4.5 * sr)].abs().mean() == 0.0


def test_disabled_overlap_separation():
    sr = 16000
    n_samples = int(4.0 * sr)
    mono = torch.ones(n_samples)

    annotation = DummyAnnotation([
        (0.0, 2.5, "SPEAKER_00"),
        (2.0, 4.0, "SPEAKER_01"),
    ])

    config = Stage1Config(separate_overlap=False)

    streams = separate_and_stitch_overlaps(
        mono=mono,
        sr=sr,
        diarization=annotation,
        labels=["SPEAKER_00", "SPEAKER_01"],
        config=config,
    )

    # In overlap [2.0, 2.5], both receive unseparated mono
    idx = int(2.2 * sr)
    assert streams["SPEAKER_00"][idx] == 1.0
    assert streams["SPEAKER_01"][idx] == 1.0
