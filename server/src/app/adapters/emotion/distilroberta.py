from typing import Any, Dict, List, cast
from transformers import pipeline

from app.interfaces import BaseSemanticScorer
from app.schemas.common import EmotionPrediction, UnifiedEmotion


class DistilRobertaSemanticAdapter(BaseSemanticScorer):
    """HuggingFace text-classification adapter using DistilRoBERTa."""

    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        device: str = "cpu",
        top_k: int = 3,
    ):
        self.device = device
        self.top_k = top_k
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=top_k,
            device=-1 if self.device == "cpu" else 0,
        )

    @staticmethod
    def _map_label(raw_label: str) -> UnifiedEmotion:
        mapping = {
            "anger": UnifiedEmotion.ANGRY,
            "disgust": UnifiedEmotion.DISGUST,
            "fear": UnifiedEmotion.FEARFUL,
            "joy": UnifiedEmotion.HAPPY,
            "neutral": UnifiedEmotion.NEUTRAL,
            "sadness": UnifiedEmotion.SAD,
            "surprise": UnifiedEmotion.SURPRISED,
        }
        return mapping.get(raw_label.lower(), UnifiedEmotion.AMBIGUOUS)

    def score_batch(self, texts: List[str]) -> List[List[EmotionPrediction]]:
        """Scores a batch of text strings and returns standardized ranked predictions."""
        if not texts:
            return []

        cleaned_batch = [t.strip() or " " for t in texts]
        batch_predictions: Any = self.classifier(cleaned_batch)

        results: List[List[EmotionPrediction]] = []
        for raw_text, scores_list in zip(texts, batch_predictions):
            preds: List[EmotionPrediction] = []
            if raw_text.strip():
                for score_dict in cast(List[Dict[str, Any]], scores_list):
                    preds.append(
                        EmotionPrediction(
                            label=self._map_label(score_dict["label"]),
                            original_label=score_dict["label"],
                            score=round(float(score_dict["score"]), 4),
                        )
                    )
            results.append(preds)

        return results
