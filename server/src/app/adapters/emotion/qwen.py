import json
import logging
import re
from typing import Any, Dict, List, Optional
import httpx

from app.interfaces import BaseSemanticScorer
from app.schemas.common import EmotionPrediction, UnifiedEmotion

logger = logging.getLogger(__name__)


class QwenSemanticAdapter(BaseSemanticScorer):
    """
    Semantic emotion scoring adapter powered by Qwen (via OpenRouter or OpenAI-compatible endpoint).
    Provides deep contextual dialogue emotion analysis with awareness of sarcasm and conversational flow.
    Falls back gracefully to a local classifier (e.g. DistilRoBERTa) if offline or API is unavailable.
    """

    LABEL_MAPPING: Dict[str, UnifiedEmotion] = {
        "happy": UnifiedEmotion.HAPPY,
        "joy": UnifiedEmotion.HAPPY,
        "pleased": UnifiedEmotion.HAPPY,
        "sad": UnifiedEmotion.SAD,
        "sadness": UnifiedEmotion.SAD,
        "angry": UnifiedEmotion.ANGRY,
        "anger": UnifiedEmotion.ANGRY,
        "frustrated": UnifiedEmotion.ANGRY,
        "annoyed": UnifiedEmotion.ANGRY,
        "fearful": UnifiedEmotion.FEARFUL,
        "fear": UnifiedEmotion.FEARFUL,
        "anxious": UnifiedEmotion.FEARFUL,
        "disgust": UnifiedEmotion.DISGUST,
        "disgusted": UnifiedEmotion.DISGUST,
        "surprised": UnifiedEmotion.SURPRISED,
        "surprise": UnifiedEmotion.SURPRISED,
        "neutral": UnifiedEmotion.NEUTRAL,
        "ambiguous": UnifiedEmotion.AMBIGUOUS,
    }

    def __init__(
        self,
        model_name: str = "qwen/qwen-2.5-72b-instruct",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 30.0,
        fallback_adapter: Optional[BaseSemanticScorer] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.fallback_adapter = fallback_adapter

    @classmethod
    def _map_label(cls, raw_label: str) -> UnifiedEmotion:
        cleaned = raw_label.strip().lower()
        return cls.LABEL_MAPPING.get(cleaned, UnifiedEmotion.AMBIGUOUS)

    def _build_prompt(self, indexed_turns: List[Dict[str, Any]]) -> str:
        turns_json = json.dumps(indexed_turns, indent=2, ensure_ascii=False)
        return (
            "You are an expert conversational emotion and sentiment analyst for telephone dialogues.\n"
            "Analyze the following chronological sequence of spoken dialogue turns from an audio recording.\n"
            "Evaluate each turn IN CONTEXT of the preceding dialogue flow (detecting subtle sarcasm, "
            "passive-aggression, polite frustration, relief, or customer reassurance).\n\n"
            "Allowed emotions: NEUTRAL, HAPPY, SAD, ANGRY, FEARFUL, DISGUST, SURPRISED.\n\n"
            "Input turns:\n"
            f"{turns_json}\n\n"
            "Respond ONLY with a valid JSON array of objects, one per turn, in this exact format:\n"
            "[\n"
            '  {"turn_id": 0, "emotion": "NEUTRAL", "score": 0.85},\n'
            '  {"turn_id": 1, "emotion": "HAPPY", "score": 0.90}\n'
            "]"
        )

    def _call_llm(self, prompt: str) -> str:
        """Dispatches the prompt to the OpenAI-compatible chat completion endpoint."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # OpenRouter-specific attribution headers
        if "openrouter" in self.base_url.lower():
            headers["HTTP-Referer"] = "https://github.com/AudioSentimentalAnalysis"
            headers["X-Title"] = "Audio Sentiment Analysis Server"

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise, objective conversational emotion classifier. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
            if "qwen" in self.model_name.lower()
            else None,
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    def _parse_llm_response(self, raw_content: str, expected_count: int) -> Dict[int, Dict]:
        """Safely extracts JSON array from model response."""
        content = raw_content.strip()
        # Strip potential markdown fences
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        try:
            parsed = json.loads(content)
            # Some models wrap array in a top-level key like {"turns": [...]}
            if isinstance(parsed, dict):
                for val in parsed.values():
                    if isinstance(val, list):
                        parsed = val
                        break
            if isinstance(parsed, list):
                return {
                    item.get("turn_id", idx): item
                    for idx, item in enumerate(parsed)
                    if isinstance(item, dict)
                }
        except Exception as exc:
            logger.warning(
                f"[QwenSemanticAdapter] Failed to parse LLM JSON: {exc}. Content: {raw_content[:200]}"
            )

        return {}

    def score_batch(self, texts: List[str]) -> List[List[EmotionPrediction]]:
        """
        Scores a batch of chronological dialogue turns using Qwen with conversational context.
        Falls back to fallback_adapter if API key is missing or request fails.
        """
        if not texts:
            return []

        # If no API key is configured and endpoint requires one, trigger fallback directly
        if not self.api_key and "openrouter" in self.base_url.lower():
            logger.info(
                "[QwenSemanticAdapter] No OPENROUTER_API_KEY configured. Using fallback adapter."
            )
            if self.fallback_adapter:
                return self.fallback_adapter.score_batch(texts)
            return [
                [
                    EmotionPrediction(
                        label=UnifiedEmotion.NEUTRAL, original_label="neutral", score=1.0
                    )
                ]
                for _ in texts
            ]

        indexed_turns = [{"turn_id": idx, "text": t.strip() or " "} for idx, t in enumerate(texts)]

        try:
            prompt = self._build_prompt(indexed_turns)
            raw_response = self._call_llm(prompt)
            parsed_results = self._parse_llm_response(raw_response, len(texts))

            results: List[List[EmotionPrediction]] = []
            for idx in range(len(texts)):
                turn_eval = parsed_results.get(idx)
                if turn_eval:
                    label_str = str(turn_eval.get("emotion", "NEUTRAL"))
                    score_val = float(turn_eval.get("score", 0.85))
                    unified = self._map_label(label_str)
                    results.append(
                        [
                            EmotionPrediction(
                                label=unified,
                                original_label=label_str,
                                score=round(score_val, 4),
                            )
                        ]
                    )
                else:
                    # If LLM missed a specific index, use fallback for that turn
                    if self.fallback_adapter:
                        fallback_pred = self.fallback_adapter.score_batch([texts[idx]])[0]
                        results.append(fallback_pred)
                    else:
                        results.append(
                            [
                                EmotionPrediction(
                                    label=UnifiedEmotion.NEUTRAL,
                                    original_label="neutral",
                                    score=0.70,
                                )
                            ]
                        )

            return results

        except Exception as exc:
            logger.warning(
                f"[QwenSemanticAdapter] OpenRouter/Qwen API call failed ({exc}). Falling back to local adapter."
            )
            if self.fallback_adapter:
                return self.fallback_adapter.score_batch(texts)
            return [
                [
                    EmotionPrediction(
                        label=UnifiedEmotion.NEUTRAL, original_label="fallback_neutral", score=1.0
                    )
                ]
                for _ in texts
            ]
