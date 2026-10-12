"""Textual affect and emotion analysis for conversational turns."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.core.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Canonical mapping bridging DistilRoBERTa/DeBERTa/Twitter-RoBERTa labels to emotion2vec taxonomy
CANONICAL_EMOTION_MAP: dict[str, str] = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgusted",
    "disgusted": "disgusted",
    "fear": "fearful",
    "fearful": "fearful",
    "joy": "happy",
    "happy": "happy",
    "sadness": "sad",
    "sad": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
    "neutral": "neutral",
    # Backward compatibility with older 3-class sentiment models
    "positive": "happy",
    "negative": "sad",
}

# Heuristic fallback keywords
_POSITIVE_WORDS = {
    "thank", "thanks", "great", "excellent", "super", "lovely", "welcome",
    "glad", "pleasure", "appreciate", "good", "perfect", "awesome", "wonderful", "sure",
}
_NEGATIVE_WORDS = {
    "ditched", "urgent", "urgently", "late", "sorry", "problem", "delay",
    "angry", "bad", "terrible", "worst", "fail", "wrong", "cancel", "issue", "frustrated", "horrible",
}


def _heuristic_affect(text: str) -> Tuple[str, float, Dict[str, float]]:
    lower = text.lower()
    pos_count = sum(1 for w in _POSITIVE_WORDS if w in lower)
    neg_count = sum(1 for w in _NEGATIVE_WORDS if w in lower)

    scores: Dict[str, float] = {
        "angry": 0.05,
        "disgusted": 0.05,
        "fearful": 0.05,
        "happy": 0.05,
        "neutral": 0.65,
        "sad": 0.05,
        "surprised": 0.10,
    }

    if pos_count > neg_count:
        sc = min(0.95, 0.6 + 0.1 * pos_count)
        scores["happy"] = round(sc, 4)
        scores["neutral"] = round(max(0.01, 1.0 - sc), 4)
        return "happy", sc, scores
    elif neg_count > pos_count:
        sc = min(0.95, 0.6 + 0.1 * neg_count)
        scores["angry"] = round(sc * 0.6, 4)
        scores["sad"] = round(sc * 0.4, 4)
        scores["neutral"] = round(max(0.01, 1.0 - sc), 4)
        dominant = "angry" if ("angry" in lower or "terrible" in lower or "horrible" in lower) else "sad"
        return dominant, sc, scores

    return "neutral", 0.75, scores


def format_context_prompt(text: str, prior_text: Optional[str] = None) -> str:
    """Format dialogue utterance with prior turn context within token budget."""
    clean_curr = text.strip()
    if not prior_text or not prior_text.strip():
        return clean_curr
    clean_prior = prior_text.strip()
    return f"Context: {clean_prior} | Utterance: {clean_curr}"


def _parse_pipeline_item(item: Any) -> Tuple[str, float, Dict[str, float]]:
    """Parse transformer pipeline output (top_k=None list of dicts or single dict)."""
    scores: Dict[str, float] = {}

    if isinstance(item, list):
        for entry in item:
            if isinstance(entry, dict):
                raw_lbl = str(entry.get("label", "")).lower()
                sc = float(entry.get("score", 0.0))
                canonical = CANONICAL_EMOTION_MAP.get(raw_lbl, raw_lbl)
                scores[canonical] = round(sc, 4)
    elif isinstance(item, dict):
        raw_lbl = str(item.get("label", "neutral")).lower()
        sc = float(item.get("score", 1.0))
        canonical = CANONICAL_EMOTION_MAP.get(raw_lbl, raw_lbl)
        scores[canonical] = round(sc, 4)

    if not scores:
        return "neutral", 1.0, {"neutral": 1.0}

    dominant = max(scores.items(), key=lambda kv: kv[1])[0]
    conf = scores[dominant]
    return dominant, conf, scores


def analyze_text_sentiment(text: Optional[str], prior_text: Optional[str] = None) -> Tuple[str, float]:
    """Classify sentiment/emotion of turn text into (label, confidence)."""
    dominant, conf, _ = analyze_text_affect(text, prior_text=prior_text)
    return dominant, conf


def analyze_text_affect(text: Optional[str], prior_text: Optional[str] = None) -> Tuple[str, float, Dict[str, float]]:
    """Classify affect of turn text into (label, confidence, full_scores_dict)."""
    if not text or not text.strip():
        return "neutral", 1.0, {"neutral": 1.0}

    try:
        classifier = ModelRegistry.get_instance().get_semantic_classifier()
        prompt = format_context_prompt(text, prior_text)
        res = classifier(prompt, truncation=True, max_length=512)
        if res:
            item = res[0] if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list) else res
            return _parse_pipeline_item(item)
    except Exception as e:
        logger.debug("Transformer sentiment inference failed (%s), using keyword fallback", e)

    return _heuristic_affect(text)


def analyze_texts_batch(
    texts: List[str],
    prior_texts: Optional[List[Optional[str]]] = None,
) -> List[Tuple[str, float, Dict[str, float]]]:
    """Batch-process multiple turn texts conditioned on dialogue context."""
    if not texts:
        return []

    priors = prior_texts if prior_texts is not None else [None] * len(texts)
    valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    results: List[Tuple[str, float, Dict[str, float]]] = [("neutral", 1.0, {"neutral": 1.0})] * len(texts)

    if not valid_indices:
        return results

    try:
        classifier = ModelRegistry.get_instance().get_semantic_classifier()
        formatted_inputs = [
            format_context_prompt(texts[i], priors[i] if i < len(priors) else None)
            for i in valid_indices
        ]
        batch_res = classifier(formatted_inputs, batch_size=16, truncation=True, max_length=512)

        for idx, res in zip(valid_indices, batch_res):
            results[idx] = _parse_pipeline_item(res)
        return results
    except Exception as e:
        logger.debug("Batch semantic inference failed (%s), falling back to heuristic", e)
        for i in valid_indices:
            results[i] = _heuristic_affect(texts[i])
        return results
