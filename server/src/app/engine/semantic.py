"""Textual sentiment analysis for transcribed turns."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from app.core.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Heuristic fallback keywords
_POSITIVE_WORDS = {
    "thank", "thanks", "great", "excellent", "super", "lovely", "welcome",
    "glad", "pleasure", "appreciate", "good", "perfect", "awesome", "wonderful", "sure",
}
_NEGATIVE_WORDS = {
    "ditched", "urgent", "urgently", "late", "sorry", "problem", "delay",
    "angry", "bad", "terrible", "worst", "fail", "wrong", "cancel", "issue", "frustrated",
}


def _heuristic_sentiment(text: str) -> Tuple[str, float]:
    lower = text.lower()
    pos_count = sum(1 for w in _POSITIVE_WORDS if w in lower)
    neg_count = sum(1 for w in _NEGATIVE_WORDS if w in lower)

    if pos_count > neg_count:
        return "positive", min(0.95, 0.6 + 0.1 * pos_count)
    elif neg_count > pos_count:
        return "negative", min(0.95, 0.6 + 0.1 * neg_count)
    return "neutral", 0.8


def analyze_text_sentiment(text: Optional[str]) -> Tuple[str, float]:
    """Classify sentiment of turn text into (label, confidence).

    Labels: 'positive', 'negative', 'neutral'.
    Uses cardiffnlp RoBERTa classifier with fast heuristic fallback.
    """
    if not text or not text.strip():
        return "neutral", 1.0

    try:
        classifier = ModelRegistry.get_instance().get_semantic_classifier()
        res = classifier(text.strip())
        if res and isinstance(res, list):
            item = res[0]
            label = str(item.get("label", "neutral")).lower()
            score = round(float(item.get("score", 1.0)), 4)
            return label, score
    except Exception as e:
        logger.debug("Transformer sentiment inference failed (%s), using keyword fallback", e)

    return _heuristic_sentiment(text)


def analyze_texts_batch(texts: List[str]) -> List[Tuple[str, float]]:
    """Batch-process multiple turn texts for high-throughput GPU inference."""
    if not texts:
        return []

    valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    results: List[Tuple[str, float]] = [("neutral", 1.0)] * len(texts)

    if not valid_indices:
        return results

    try:
        classifier = ModelRegistry.get_instance().get_semantic_classifier()
        valid_texts = [texts[i].strip() for i in valid_indices]
        batch_res = classifier(valid_texts, batch_size=16)

        for idx, res in zip(valid_indices, batch_res):
            label = str(res.get("label", "neutral")).lower()
            score = round(float(res.get("score", 1.0)), 4)
            results[idx] = (label, score)
        return results
    except Exception as e:
        logger.debug("Batch semantic inference failed (%s), falling back to heuristic", e)
        for i in valid_indices:
            results[i] = _heuristic_sentiment(texts[i])
        return results
