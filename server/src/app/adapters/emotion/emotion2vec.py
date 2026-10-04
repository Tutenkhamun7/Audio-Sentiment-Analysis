from typing import List
import numpy as np
from funasr import AutoModel

from app.interfaces import BaseAcousticScorer
from app.schemas.common import EmotionPrediction, UnifiedEmotion


class Emotion2VecAcousticAdapter(BaseAcousticScorer):
    """FunASR emotion2vec+ acoustic model adapter."""

    def __init__(
        self,
        model_name: str = "iic/emotion2vec_plus_base",
        device: str = "cpu",
        top_k: int = 3,
    ):
        self.device = device
        self.top_k = top_k
        self.model = AutoModel(
            model=model_name,
            device=self.device,
            disable_update=True,
        )

    @staticmethod
    def _map_label(raw_label: str) -> UnifiedEmotion:
        cleaned = raw_label.split("/")[-1].strip().lower()
        mapping = {
            "angry": UnifiedEmotion.ANGRY,
            "disgust": UnifiedEmotion.DISGUST,
            "disgusted": UnifiedEmotion.DISGUST,
            "fear": UnifiedEmotion.FEARFUL,
            "fearful": UnifiedEmotion.FEARFUL,
            "happy": UnifiedEmotion.HAPPY,
            "neutral": UnifiedEmotion.NEUTRAL,
            "sad": UnifiedEmotion.SAD,
            "surprise": UnifiedEmotion.SURPRISED,
            "surprised": UnifiedEmotion.SURPRISED,
            "other": UnifiedEmotion.AMBIGUOUS,
            "<unk>": UnifiedEmotion.AMBIGUOUS,
            "unknown": UnifiedEmotion.AMBIGUOUS,
        }
        return mapping.get(cleaned, UnifiedEmotion.AMBIGUOUS)

    def score_chunk(self, audio_chunk: np.ndarray, sample_rate: int) -> List[EmotionPrediction]:
        # Defensively ensure audio_chunk is float32 to prevent torchaudio resampler dtype mismatch
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

        # Require at least 0.5s of audio for valid inference
        if len(audio_chunk) < (sample_rate * 0.5):
            return []

        raw_scores = self.model.generate(
            input=audio_chunk,
            fs=sample_rate,
            granularity="utterance",
        )

        preds: List[EmotionPrediction] = []
        if raw_scores and isinstance(raw_scores, list):
            labels = raw_scores[0].get("labels", [])
            scores = raw_scores[0].get("scores", [])
            sorted_pairs = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
            for label, score in sorted_pairs[: self.top_k]:
                preds.append(
                    EmotionPrediction(
                        label=self._map_label(label),
                        original_label=label,
                        score=round(float(score), 4),
                    )
                )

        return preds
