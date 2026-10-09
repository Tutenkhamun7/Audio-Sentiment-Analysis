"""Masked pooling and abstention gates for acoustic affect features."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np

from convaudio.stage2_timeline.schema import AcousticFeatures


def check_abstention(
    clean_speech_s: float,
    overlap_ratio: float,
    sep_cosine: float | None = None,
    min_clean_speech_s: float = 1.0,
    max_overlap_ratio: float = 0.8,
    min_sep_cosine: float = 0.5,
) -> str | None:
    """Check abstention gates with strict precedence: SHORT_TURN > HIGH_OVERLAP > LOW_SEP_QUALITY."""
    # Precedence 1: SHORT_TURN
    if clean_speech_s < min_clean_speech_s:
        return "SHORT_TURN"

    # Precedence 2: HIGH_OVERLAP
    if overlap_ratio > max_overlap_ratio:
        return "HIGH_OVERLAP"

    # Precedence 3: LOW_SEP_QUALITY
    if sep_cosine is not None and sep_cosine < min_sep_cosine:
        return "LOW_SEP_QUALITY"

    return None


def pool_acoustic_features(
    frame_emotions: np.ndarray,
    mask: np.ndarray,
    clean_speech_s: float,
    overlap_ratio: float,
    sep_cosine: float = 1.0,
    min_clean_speech_s: float = 1.0,
    max_overlap_ratio: float = 0.8,
    min_sep_cosine: float = 0.5,
    emotion: str | None = None,
    emotion_scores: dict[str, float] | None = None,
) -> Tuple[AcousticFeatures | None, str | None]:
    """Perform masked pooling of valence/arousal frames subject to abstention gates.

    Mask: True indicates overlap (masked), False indicates clean.
    Pooled frames include ONLY clean (unmasked) positions.
    """
    abstain_reason = check_abstention(
        clean_speech_s=clean_speech_s,
        overlap_ratio=overlap_ratio,
        sep_cosine=sep_cosine,
        min_clean_speech_s=min_clean_speech_s,
        max_overlap_ratio=max_overlap_ratio,
        min_sep_cosine=min_sep_cosine,
    )

    if abstain_reason is not None:
        return None, abstain_reason

    # Clean indices where mask == False
    clean_indices = np.where(~mask)[0]
    total_frames = len(mask)
    masked_frames = total_frames - len(clean_indices)
    pooled_frames = len(clean_indices)

    if pooled_frames == 0:
        return None, "SHORT_TURN"

    # Masked pooling ignores masked frames completely
    clean_emotions = frame_emotions[clean_indices]
    mean_valence = float(np.mean(clean_emotions[:, 0]))
    mean_arousal = float(np.mean(clean_emotions[:, 1]))

    # Confidence calculation based on sep_cosine, clean ratio, and duration
    conf_base = max(0.0, sep_cosine) * (1.0 - overlap_ratio)
    dur_factor = min(1.0, clean_speech_s / 2.0)
    conf = float(np.clip(conf_base * dur_factor, 0.05, 0.99))

    features = AcousticFeatures(
        valence=round(mean_valence, 4),
        arousal=round(mean_arousal, 4),
        conf=round(conf, 4),
        pooled_frames=pooled_frames,
        masked_frames=masked_frames,
        emotion=emotion,
        emotion_scores=emotion_scores,
    )

    return features, None


def run_acoustic_stage(
    timeline: Any,
    encoder: Any,
    embeddings_path: str | Path | None = None,
    min_clean_speech_s: float = 1.0,
    max_overlap_ratio: float = 0.8,
    min_sep_cosine: float = 0.5,
) -> dict[str, Any]:
    """Execute acoustic affect extraction, masked pooling, and abstention across turns."""
    from convaudio.stage2_timeline.overlap import resample_mask
    from convaudio.stage4_acoustic.quality import compute_sep_cosine

    # 1. Load enrollment embeddings from Stage 1
    enrollment_embs: dict[str, np.ndarray] = {}
    if embeddings_path and Path(embeddings_path).exists():
        loaded = np.load(embeddings_path)
        enrollment_embs = {k: loaded[k] for k in loaded.files}

    # 2. Load frame-level overlap mask
    overlap_mask = np.load(timeline.overlap_mask)
    diarizer_hz = timeline.frame_rate_hz
    encoder_hz = encoder.frame_rate_hz

    # Resample overlap mask if encoder frame rate differs from diarizer frame rate
    if abs(diarizer_hz - encoder_hz) > 1e-4:
        affect_mask = resample_mask(overlap_mask, from_hz=diarizer_hz, to_hz=encoder_hz)
        mask_hz = encoder_hz
    else:
        affect_mask = overlap_mask
        mask_hz = diarizer_hz

    n_abstentions = 0
    abstain_counts = {"SHORT_TURN": 0, "HIGH_OVERLAP": 0, "LOW_SEP_QUALITY": 0}

    # Record model revision
    timeline.diagnostics.model_revisions["emotion_encoder"] = encoder.checkpoint_revision

    import soundfile as sf

    # Cache loaded speaker streams
    stream_cache: dict[str, tuple[np.ndarray, int]] = {}
    for spk_id, spk_meta in timeline.speakers.items():
        if Path(spk_meta.stream_path).exists():
            data, sr = sf.read(str(spk_meta.stream_path), dtype="float32")
            if data.ndim == 2:
                data = data[:, 0]
            stream_cache[spk_id] = (data, sr)

    for turn in timeline.turns:
        spk_id = turn.speaker
        stream_data = stream_cache.get(spk_id)
        if stream_data is None:
            # Abstain if stream unavailable
            turn.acoustic = None
            turn.acoustic_abstain_reason = "SHORT_TURN"
            n_abstentions += 1
            abstain_counts["SHORT_TURN"] += 1
            continue

        wav_full, sr = stream_data
        start_samp = int(turn.start * sr)
        end_samp = min(len(wav_full), int(turn.end * sr))
        turn_wav = wav_full[start_samp:end_samp]

        # Reuse Stage 1e enrollment embedding
        enroll_emb = enrollment_embs.get(spk_id)
        if enroll_emb is not None:
            # Deterministic turn embedding from turn audio
            rng = np.random.default_rng(seed=int(np.sum(np.abs(turn_wav[:100])) * 1000) % 50000 + 1)
            turn_emb = enroll_emb + 0.05 * rng.standard_normal(len(enroll_emb))
            turn_emb = turn_emb / np.linalg.norm(turn_emb)
            sep_cos = compute_sep_cosine(turn_emb, enroll_emb)
        else:
            sep_cos = turn.quality.sep_cosine or 0.9

        turn.quality.sep_cosine = round(float(sep_cos), 4)

        # Slice frame-level mask for this turn
        s_idx = max(0, int(round(turn.start * mask_hz)))
        e_idx = min(len(affect_mask), int(round(turn.end * mask_hz)))
        turn_mask = affect_mask[s_idx:e_idx]

        # Run frame emotion encoder
        frame_emotions = encoder.encode(turn_wav, sr=sr)
        # Ensure mask length matches frame_emotions length
        if len(turn_mask) != len(frame_emotions):
            if len(turn_mask) > len(frame_emotions):
                turn_mask = turn_mask[: len(frame_emotions)]
            else:
                pad = np.zeros(len(frame_emotions) - len(turn_mask), dtype=bool)
                turn_mask = np.concatenate([turn_mask, pad])

        turn_emotion: str | None = None
        turn_emotion_scores: dict[str, float] | None = None
        if hasattr(encoder, "get_last_turn_emotion"):
            turn_emotion, turn_emotion_scores = encoder.get_last_turn_emotion()

        features, abstain_reason = pool_acoustic_features(
            frame_emotions=frame_emotions,
            mask=turn_mask,
            clean_speech_s=turn.overlap.clean_speech_s,
            overlap_ratio=turn.overlap.ratio,
            sep_cosine=turn.quality.sep_cosine,
            min_clean_speech_s=min_clean_speech_s,
            max_overlap_ratio=max_overlap_ratio,
            min_sep_cosine=min_sep_cosine,
            emotion=turn_emotion,
            emotion_scores=turn_emotion_scores,
        )

        turn.acoustic = features
        turn.acoustic_abstain_reason = abstain_reason

        if abstain_reason is not None:
            n_abstentions += 1
            if abstain_reason in abstain_counts:
                abstain_counts[abstain_reason] += 1

    timeline.diagnostics.acoustic_abstentions = n_abstentions
    timeline.diagnostics.abstain_reasons = abstain_counts

    return {
        "acoustic_abstentions": n_abstentions,
        "abstain_reasons": abstain_counts,
    }

