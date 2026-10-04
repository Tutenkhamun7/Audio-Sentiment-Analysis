from typing import List, Optional
import numpy as np

from app.interfaces import BaseAcousticScorer, BaseSemanticScorer
from app.schemas.common import EmotionPrediction, UnifiedEmotion


class MockSemanticScorer(BaseSemanticScorer):
    """Mock semantic scorer for rapid testing."""

    def __init__(self, default_predictions: Optional[List[EmotionPrediction]] = None):
        self.default_predictions = default_predictions or [
            EmotionPrediction(label=UnifiedEmotion.HAPPY, original_label="joy", score=0.90),
            EmotionPrediction(label=UnifiedEmotion.NEUTRAL, original_label="neutral", score=0.08),
        ]

    def score_batch(self, texts: List[str]) -> List[List[EmotionPrediction]]:
        return [self.default_predictions for _ in texts]


class MockAcousticScorer(BaseAcousticScorer):
    """Mock acoustic scorer for rapid testing."""

    def __init__(self, default_predictions: Optional[List[EmotionPrediction]] = None):
        self.default_predictions = default_predictions or [
            EmotionPrediction(label=UnifiedEmotion.HAPPY, original_label="happy", score=0.88),
            EmotionPrediction(label=UnifiedEmotion.NEUTRAL, original_label="neutral", score=0.10),
        ]

    def score_chunk(self, audio_chunk: np.ndarray, sample_rate: int) -> List[EmotionPrediction]:
        return self.default_predictions
