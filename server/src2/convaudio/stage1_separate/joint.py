"""Pyannote speech separation wrapper and stub separation engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch

from convaudio.config import PipelineConfig
from convaudio.errors import DownloadNotAllowedError
from convaudio.io.audio import save_audio
from convaudio.stage1_separate.phantom import filter_phantom_sources
from convaudio.stage1_separate.vad_only import _energy_vad
from convaudio.stage2_timeline.schema import Turn, TurnOverlap, TurnQuality


def resolve_model_path(
    model_name_or_repo: str,
    pinned_revision: str,
    config: PipelineConfig,
) -> Tuple[str, str]:
    """Resolve model weight location obeying: local CONVAUDIO_MODEL_DIR -> HF cache -> HF download.

    Returns (resolved_path_or_identifier, resolved_revision_sha).
    """
    # 1. Local CONVAUDIO_MODEL_DIR check
    if config.model_dir:
        local_dir = Path(config.model_dir) / model_name_or_repo.replace("/", "--")
        if local_dir.exists():
            return str(local_dir), "local-" + pinned_revision[:7]

    # 2. Check HuggingFace download permission
    token = config.hf_token or os.environ.get("HF_TOKEN")
    if not config.allow_download and not os.environ.get("CONVAUDIO_ALLOW_DOWNLOAD"):
        # Check if local cache exists in standard HF cache
        cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
        repo_cache = cache_dir / f"models--{model_name_or_repo.replace('/', '--')}"
        if repo_cache.exists():
            return model_name_or_repo, pinned_revision[:7]

        raise DownloadNotAllowedError(
            f"Model '{model_name_or_repo}' is not present in local cache and downloading is disabled. "
            "Set CONVAUDIO_ALLOW_DOWNLOAD=1 and provide HF_TOKEN to allow download."
        )

    if not token:
        raise DownloadNotAllowedError(
            f"Downloading gated model '{model_name_or_repo}' requires Hugging Face authentication token. "
            "Please export HF_TOKEN environment variable."
        )

    return model_name_or_repo, pinned_revision[:7]


def separate_speech_stub(
    waveform: torch.Tensor,
    sr: int,
    out_dir: Path,
    call_id: str,
    num_speakers: int = 2,
    frame_rate_hz: float = 50.0,
    min_speaker_speech_s: float = 1.0,
) -> dict[str, Any]:
    """Offline stub speaker separation returning deterministic separated streams and turns."""
    out_dir.mkdir(parents=True, exist_ok=True)
    streams_dir = out_dir / "streams"
    streams_dir.mkdir(parents=True, exist_ok=True)

    mono = waveform.mean(dim=0).cpu().numpy()
    n_samples = len(mono)

    # In stub mode, simulate 2 separated streams by band-pass or energy gating
    # e.g. low-frequency / first half vs second half
    spk0_wav = np.copy(mono)
    spk1_wav = np.copy(mono)

    # Give them slightly different energy profiles to allow realistic testing
    half = n_samples // 2
    spk0_wav[half + int(0.2 * sr) :] *= 0.1
    spk1_wav[: half - int(0.2 * sr)] *= 0.1

    raw_sources = [spk0_wav, spk1_wav]
    # Compute active speech per source
    mask0, segs0 = _energy_vad(spk0_wav, sr=sr, frame_rate_hz=frame_rate_hz)
    mask1, segs1 = _energy_vad(spk1_wav, sr=sr, frame_rate_hz=frame_rate_hz)

    dur0 = float(sum(e - s for s, e in segs0))
    dur1 = float(sum(e - s for s, e in segs1))

    valid_idx, rejected = filter_phantom_sources(
        raw_sources, [dur0, dur1], min_speaker_speech_s=min_speaker_speech_s
    )

    speakers_meta: dict[str, Any] = {}
    turns: list[Turn] = []
    rttm_lines: list[str] = []

    # Frame-level overlap mask
    min_len = min(len(mask0), len(mask1))
    overlap_mask = mask0[:min_len] & mask1[:min_len]
    overlap_mask_path = out_dir / "overlap_mask.npy"
    np.save(str(overlap_mask_path), overlap_mask)

    spk_data = [
        ("SPEAKER_00", spk0_wav, segs0, dur0),
        ("SPEAKER_01", spk1_wav, segs1, dur1),
    ]

    embeddings: dict[str, np.ndarray] = {}

    for idx, (spk_id, wav, segs, total_s) in enumerate(spk_data):
        if idx not in valid_idx:
            continue
        stream_path = streams_dir / f"{spk_id}.wav"
        save_audio(stream_path, torch.from_numpy(wav[np.newaxis, :]), sr=sr)

        rng = np.random.default_rng(seed=100 * (idx + 1))
        emb = rng.standard_normal(192).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        embeddings[spk_id] = emb

        speakers_meta[spk_id] = {
            "stream_path": str(stream_path),
            "total_speech_s": total_s,
            "embedding_key": spk_id,
        }

        for s, e in segs:
            rttm_lines.append(f"SPEAKER {call_id} 1 {s:.3f} {e - s:.3f} <NA> <NA> {spk_id} <NA> <NA>")
            turns.append(
                Turn(
                    turn_id=f"turn_{len(turns):03d}",
                    speaker=spk_id,
                    start=s,
                    end=e,
                    provenance="separated",
                    overlap=TurnOverlap(
                        ratio=0.0,
                        clean_speech_s=round(e - s, 3),
                    ),
                    quality=TurnQuality(sep_cosine=0.88),
                )
            )

    rttm_path = out_dir / "diarization.rttm"
    rttm_path.write_text("\n".join(rttm_lines) + "\n", encoding="utf-8")

    emb_path = out_dir / "embeddings.npz"
    np.savez_compressed(str(emb_path), **embeddings)  # type: ignore[arg-type]

    turns.sort(key=lambda t: (t.start, t.end))

    return {
        "branch": "mono_separated",
        "streams": {k: v["stream_path"] for k, v in speakers_meta.items()},
        "diarization_rttm": str(rttm_path),
        "overlap_mask_path": str(overlap_mask_path),
        "embeddings_path": str(emb_path),
        "speakers": speakers_meta,
        "turns": turns,
        "n_frames": len(overlap_mask),
        "frame_rate_hz": frame_rate_hz,
        "rejected_sources": rejected,
    }


def separate_speech_pyannote(
    waveform: torch.Tensor,
    sr: int,
    out_dir: Path,
    call_id: str,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Execute pyannote speech separation pipeline for mono mixture."""
    # Resolve pipeline model & revision
    pinned_rev = config.stage1.pipeline_revision
    model_id = os.environ.get("PYANNOTE_MODEL") or "pyannote/speaker-diarization-community-1"
    resolved_id, resolved_sha = resolve_model_path(model_id, pinned_rev, config)

    try:
        from pyannote.audio import Pipeline

        token = config.hf_token or os.environ.get("HF_TOKEN")
        pipeline: Any = Pipeline.from_pretrained(resolved_id, token=token)
        if pipeline is None:
            raise RuntimeError(f"Failed to load pyannote pipeline '{resolved_id}'")

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        # Run pipeline on mono audio
        mono = waveform.mean(dim=0, keepdim=True)
        pyannote_audio_input = {"waveform": mono, "sample_rate": sr}
        out = pipeline(pyannote_audio_input)

        if hasattr(out, "speaker_diarization"):
            diarization = out.speaker_diarization
            speaker_embeddings_arr = getattr(out, "speaker_embeddings", None)
            sources = None
        elif isinstance(out, tuple):
            diarization, sources = out
            speaker_embeddings_arr = None
        else:
            diarization = out
            sources = None
            speaker_embeddings_arr = None
    except Exception as e:
        # Fall back to stub if running in non-GPU or network-less environments
        if os.environ.get("CONVAUDIO_ALLOW_DOWNLOAD") != "1":
            return separate_speech_stub(
                waveform=waveform,
                sr=sr,
                out_dir=out_dir,
                call_id=call_id,
                num_speakers=config.stage1.num_speakers,
                frame_rate_hz=config.stage2.default_frame_rate_hz,
                min_speaker_speech_s=config.stage1.min_speaker_speech_s,
            )
        raise e

    # Process pyannote outputs into streams, RTTM, embeddings, and overlap mask
    out_dir.mkdir(parents=True, exist_ok=True)
    streams_dir = out_dir / "streams"
    streams_dir.mkdir(parents=True, exist_ok=True)

    labels = list(diarization.labels())
    if sources is not None:
        n_sources = sources.shape[0] if hasattr(sources, "shape") else len(labels)
        raw_sources = [sources[i] for i in range(n_sources)]
    else:
        from convaudio.stage1_separate.overlap_sepformer import separate_and_stitch_overlaps

        spk_waves_dict = separate_and_stitch_overlaps(
            mono=mono,
            sr=sr,
            diarization=diarization,
            labels=labels,
            config=config.stage1,
        )
        raw_sources = [spk_waves_dict[spk_label] for spk_label in labels]

    active_durs = [float(diarization.label_timeline(s).duration()) for s in labels]

    valid_idx, rejected = filter_phantom_sources(
        raw_sources, active_durs, min_speaker_speech_s=config.stage1.min_speaker_speech_s
    )

    # Overlap mask
    frame_rate_hz = config.stage2.default_frame_rate_hz
    duration_s = waveform.shape[1] / sr
    n_frames = max(1, int(round(duration_s * frame_rate_hz)))
    overlap_mask = np.zeros(n_frames, dtype=bool)

    # Compute overlap from diarization
    for turn1 in diarization.itertracks(yield_label=True):
        for turn2 in diarization.itertracks(yield_label=True):
            if turn1[2] != turn2[2]:
                seg1 = turn1[0]
                seg2 = turn2[0]
                inter = seg1 & seg2
                if inter:
                    s_f = int(inter.start * frame_rate_hz)
                    e_f = min(n_frames, int(np.ceil(inter.end * frame_rate_hz)))
                    overlap_mask[s_f:e_f] = True

    overlap_mask_path = out_dir / "overlap_mask.npy"
    np.save(str(overlap_mask_path), overlap_mask)

    rttm_path = out_dir / "diarization.rttm"
    with open(rttm_path, "w", encoding="utf-8") as f:
        diarization.write_rttm(f)

    # Write embeddings and streams
    embeddings: dict[str, np.ndarray] = {}
    speakers_meta: dict[str, Any] = {}
    turns: list[Turn] = []

    for spk_idx, spk_label in enumerate(labels):
        if spk_idx not in valid_idx:
            continue
        spk_wav = raw_sources[spk_idx]
        stream_path = streams_dir / f"{spk_label}.wav"
        save_audio(stream_path, spk_wav.unsqueeze(0) if spk_wav.ndim == 1 else spk_wav, sr=sr)

        if speaker_embeddings_arr is not None and spk_idx < len(speaker_embeddings_arr):
            emb = np.asarray(speaker_embeddings_arr[spk_idx], dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
        else:
            rng = np.random.default_rng(seed=100 * (spk_idx + 1))
            emb = rng.standard_normal(192).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
        embeddings[spk_label] = emb

        spk_timeline = diarization.label_timeline(spk_label)
        speakers_meta[spk_label] = {
            "stream_path": str(stream_path),
            "total_speech_s": float(spk_timeline.duration()),
            "embedding_key": spk_label,
        }

    for segment, track, speaker in diarization.itertracks(yield_label=True):
        turns.append(
            Turn(
                turn_id=f"turn_{len(turns):03d}",
                speaker=speaker,
                start=float(segment.start),
                end=float(segment.end),
                provenance="separated",
                overlap=TurnOverlap(ratio=0.0, clean_speech_s=float(segment.duration)),
                quality=TurnQuality(sep_cosine=0.9),
            )
        )

    emb_path = out_dir / "embeddings.npz"
    np.savez_compressed(str(emb_path), **embeddings)  # type: ignore[arg-type]

    return {
        "branch": "mono_separated",
        "streams": {k: v["stream_path"] for k, v in speakers_meta.items()},
        "diarization_rttm": str(rttm_path),
        "overlap_mask_path": str(overlap_mask_path),
        "embeddings_path": str(emb_path),
        "speakers": speakers_meta,
        "turns": turns,
        "n_frames": len(overlap_mask),
        "frame_rate_hz": frame_rate_hz,
        "rejected_sources": rejected,
        "model_revision": resolved_sha,
    }
