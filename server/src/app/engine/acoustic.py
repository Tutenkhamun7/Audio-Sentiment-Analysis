"""In-memory acoustic emotion extraction with masked pooling and abstention gates."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch

from app.core.model_registry import ModelRegistry
from app.engine.diarizer import RawTurn
from app.schemas.response import EmotionAnalysis

logger = logging.getLogger(__name__)

# Standard Valence/Arousal coordinate mapping for emotion2vec labels
EMOTION_COORDINATES: dict[str, tuple[float, float]] = {
    "angry": (-0.65, 0.70),
    "disgusted": (-0.60, 0.40),
    "fearful": (-0.60, 0.65),
    "happy": (0.75, 0.55),
    "neutral": (0.00, 0.00),
    "other": (0.00, 0.00),
    "sad": (-0.70, -0.40),
    "surprised": (0.20, 0.60),
}


def compute_dimensional_affect(scores: dict[str, float]) -> tuple[float, float]:
    """Calculate continuous valence and arousal as the expectation across the affect distribution.

    Prevents discrete argmax collapse where neutral plurality erases secondary affective intensity.
    """
    if not scores:
        return 0.0, 0.0

    total_prob = sum(scores.get(k, 0.0) for k in EMOTION_COORDINATES if k in scores)
    if total_prob <= 1e-6:
        return 0.0, 0.0

    val = sum(scores.get(k, 0.0) * EMOTION_COORDINATES[k][0] for k in EMOTION_COORDINATES if k in scores) / total_prob
    aro = sum(scores.get(k, 0.0) * EMOTION_COORDINATES[k][1] for k in EMOTION_COORDINATES if k in scores) / total_prob
    return round(float(val), 3), round(float(aro), 3)


def extract_turn_emotions(
    speaker_wav: torch.Tensor,
    turns: List[RawTurn],
    sr: int = 16000,
    min_speech_turn_s: float = 0.4,
    collar_s: float = 0.08,
    min_acoustic_confidence: float = 0.50,
    min_top_margin: float = 0.10,
) -> None:
    """Predict acoustic affect on turns in RAM with strict quality abstention rules."""
    if not turns:
        return

    registry = ModelRegistry.get_instance()
    try:
        pipe = registry.get_acoustic_encoder()
    except Exception as e:
        logger.warning("Acoustic encoder unavailable (%s). Emotion skipped.", e)
        for t in turns:
            t.emotion = EmotionAnalysis(abstained=True, abstain_reason="MODEL_UNAVAILABLE")
        return

    total_samples = speaker_wav.shape[-1]
    wav_1d = speaker_wav.squeeze(0).cpu()

    for turn in turns:
        dur = turn.duration
        # Abstention Rule 1: SHORT_TURN
        if dur < min_speech_turn_s:
            turn.emotion = EmotionAnalysis(
                abstained=True,
                abstain_reason="SHORT_TURN",
                dominant_emotion="neutral",
            )
            continue

        s_idx = max(0, int((turn.start - collar_s) * sr))
        e_idx = min(total_samples, int((turn.end + collar_s) * sr))
        turn_audio = wav_1d[s_idx:e_idx]

        # Check energy
        if turn_audio.abs().mean() < 1e-4:
            turn.emotion = EmotionAnalysis(
                abstained=True,
                abstain_reason="LOW_ENERGY",
                dominant_emotion="neutral",
            )
            continue

        turn_np = turn_audio.numpy().astype(np.float32)

        try:
            # FunASR AutoModel inference
            if hasattr(pipe, "generate"):
                res = pipe.generate(turn_np, output_dir=None, granularity="utterance", disable_pbar=True)
            else:
                res = pipe(turn_np)

            if isinstance(res, list) and len(res) > 0:
                item = res[0]
            else:
                item = res

            scores: dict[str, float] = {}
            dominant = "neutral"

            if isinstance(item, dict):
                # Parse scores and labels (stripping multilingual prefixes like '生气/angry')
                labels = item.get("labels", [])
                score_vals = item.get("scores", [])
                if labels and score_vals and len(labels) == len(score_vals):
                    for lbl, sc in zip(labels, score_vals):
                        clean_lbl = str(lbl).split("/")[-1].strip().lower()
                        if clean_lbl and clean_lbl != "<unk>":
                            scores[clean_lbl] = round(float(sc), 4)

                    if scores:
                        sorted_emos = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                        top_label, top_score = sorted_emos[0]
                        second_score = sorted_emos[1][1] if len(sorted_emos) > 1 else 0.0
                        margin = top_score - second_score

                        # Gate: If top score is below confidence threshold or top-1/top-2 margin is too narrow,
                        # the acoustic model is uncertain or the utterance is emotionally flat/neutral.
                        if top_score < min_acoustic_confidence or margin < min_top_margin:
                            dominant = "neutral"
                        else:
                            dominant = top_label
                elif "key" in item:
                    dominant = str(item["key"]).lower()

            # Estimate continuous valence & arousal as distributional expectation
            valence, arousal = compute_dimensional_affect(scores)
            if not scores:
                valence, arousal = EMOTION_COORDINATES.get(dominant, (0.0, 0.0))

            # Abstention Rule: Extreme uncertainty across entire affect distribution (<0.35 max score)
            top_prob = max(scores.values()) if scores else 1.0
            if scores and top_prob < 0.35:
                turn.emotion = EmotionAnalysis(
                    dominant_emotion="neutral",
                    scores=scores,
                    valence=valence,
                    arousal=arousal,
                    abstained=True,
                    abstain_reason="LOW_ACOUSTIC_CONF",
                )
                continue

            turn.emotion = EmotionAnalysis(
                dominant_emotion=dominant,
                scores=scores,
                valence=valence,
                arousal=arousal,
                abstained=False,
            )
        except Exception as err:
            logger.debug("Emotion inference failed on turn %.2f-%.2f: %s", turn.start, turn.end, err)
            turn.emotion = EmotionAnalysis(
                dominant_emotion="neutral",
                abstained=True,
                abstain_reason="INFERENCE_ERROR",
            )
