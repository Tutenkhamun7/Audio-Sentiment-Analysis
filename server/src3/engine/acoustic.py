"""In-memory acoustic emotion extraction with masked pooling and abstention gates."""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch

from src3.core.model_registry import ModelRegistry
from src3.engine.diarizer import RawTurn
from src3.schemas.response import EmotionAnalysis

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


def extract_turn_emotions(
    speaker_wav: torch.Tensor,
    turns: List[RawTurn],
    sr: int = 16000,
    min_speech_turn_s: float = 0.4,
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

        s_idx = max(0, int(turn.start * sr))
        e_idx = min(total_samples, int(turn.end * sr))
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
                        dominant = max(scores.items(), key=lambda kv: kv[1])[0]
                elif "key" in item:
                    dominant = str(item["key"]).lower()

            # Estimate valence & arousal
            valence, arousal = EMOTION_COORDINATES.get(dominant, (0.0, 0.0))

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
