from typing import List, Optional, Tuple
from app.schemas.common import EmotionPrediction, UnifiedEmotion


class AffectEvaluator:
    """
    Pure domain service responsible for:
      1. Evaluating confidence margin to detect blended or ambiguous affect.
      2. Comparing semantic and acoustic valence to detect incongruence (sarcasm, emotional masking).
    """

    def __init__(
        self,
        ambiguity_margin: float = 0.15,
        ambiguity_max_top_score: float = 0.65,
        min_conflict_score: float = 0.45,
    ):
        self.ambiguity_margin = ambiguity_margin
        self.ambiguity_max_top_score = ambiguity_max_top_score
        self.min_conflict_score = min_conflict_score

        self.positives = {UnifiedEmotion.HAPPY}
        self.negatives = {
            UnifiedEmotion.ANGRY,
            UnifiedEmotion.DISGUST,
            UnifiedEmotion.SAD,
            UnifiedEmotion.FEARFUL,
        }

    def disambiguate_acoustic(
        self,
        semantic_preds: List[EmotionPrediction],
        acoustic_preds: List[EmotionPrediction],
        text: str = "",
    ) -> List[EmotionPrediction]:
        """
        Refines acoustic emotion predictions using semantic context:
        1. High Arousal Disambiguation (Excitement vs Anger):
           Excited speech and angry speech share identical acoustic features (high pitch, high energy).
           If the acoustic model has a near-tie between ANGRY and HAPPY, but the semantic text
           is strongly HAPPY (e.g. 'Oh, excellent, that is lovely!'), the acoustic arousal represents
           enthusiasm, so HAPPY is promoted to #1.
        2. Question-Intonation Normalization:
           Polite questions with rising inflection ('Would that still work for you?') frequently
           trigger a false ANGRY tag in short slices. If the text is a question (ends with '?') and
           is semantically NEUTRAL, demote false ANGRY to NEUTRAL.
        """
        if not acoustic_preds or not semantic_preds:
            return acoustic_preds

        sem_top = semantic_preds[0]
        ac_top = acoustic_preds[0]
        cleaned_text = text.strip()

        # 1. High-Arousal Near-Tie Disambiguation (HAPPY vs ANGRY)
        if len(acoustic_preds) >= 2:
            second_ac = acoustic_preds[1]
            if (
                ac_top.label == UnifiedEmotion.ANGRY
                and second_ac.label == UnifiedEmotion.HAPPY
                and abs(ac_top.score - second_ac.score) <= self.ambiguity_margin
            ):
                if sem_top.label == UnifiedEmotion.HAPPY and sem_top.score >= 0.45:
                    calibrated = list(acoustic_preds)
                    calibrated[0], calibrated[1] = calibrated[1], calibrated[0]
                    return calibrated

        # 2. Question-Intonation Normalization (e.g., "Would that still work for you?")
        if cleaned_text.endswith("?") and ac_top.label == UnifiedEmotion.ANGRY:
            if sem_top.label in {UnifiedEmotion.NEUTRAL, UnifiedEmotion.HAPPY}:
                calibrated = list(acoustic_preds)
                neutral_pred = next(
                    (p for p in calibrated if p.label == UnifiedEmotion.NEUTRAL), None
                )
                if neutral_pred:
                    neutral_idx = calibrated.index(neutral_pred)
                    calibrated[0], calibrated[neutral_idx] = calibrated[neutral_idx], calibrated[0]
                else:
                    calibrated[0] = EmotionPrediction(
                        label=UnifiedEmotion.NEUTRAL,
                        original_label="neutral_question",
                        score=ac_top.score,
                    )
                return calibrated

        return acoustic_preds

    def evaluate(
        self,
        semantic_preds: List[EmotionPrediction],
        acoustic_preds: List[EmotionPrediction],
        is_interruption: bool = False,
        interrupted_by: Optional[str] = None,
        overlap_duration: float = 0.0,
        text: str = "",
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        Evaluates predictions for a single timeline event.
        Returns:
            (is_conflict, is_ambiguous, conflict_detail)
        """
        # Ensure acoustic predictions are disambiguated with text context
        calibrated_acoustic = self.disambiguate_acoustic(semantic_preds, acoustic_preds, text=text)
        is_conflict = False
        is_ambiguous = False
        details: List[str] = []

        # 1. Ambiguity / Blended Affect Check (primarily on acoustic affect, fallback to semantic)
        preds_to_check = calibrated_acoustic if calibrated_acoustic else semantic_preds
        if len(preds_to_check) >= 2:
            top_score = preds_to_check[0].score
            second_score = preds_to_check[1].score
            if (
                top_score - second_score
            ) < self.ambiguity_margin and top_score < self.ambiguity_max_top_score:
                is_ambiguous = True
                details.append(
                    f"Blended affect: {preds_to_check[0].label.value} ({top_score:.2f}) vs "
                    f"{preds_to_check[1].label.value} ({second_score:.2f})"
                )

        # 2. Text vs Tone Conflict (Valence Discrepancy)
        if semantic_preds and calibrated_acoustic:
            sem_top = semantic_preds[0]
            ac_top = calibrated_acoustic[0]

            # Sarcasm / Masking: Happy words, Negative voice
            if (
                sem_top.label in self.positives
                and ac_top.label in self.negatives
                and sem_top.score >= self.min_conflict_score
                and ac_top.score >= self.min_conflict_score - 0.05
            ):
                is_conflict = True
                details.append(
                    f"Valence conflict (Possible sarcasm/masking): Text is {sem_top.label.value} ({sem_top.score:.2f}) "
                    f"but Tone is {ac_top.label.value} ({ac_top.score:.2f})"
                )
            # Understated / Playful affect: Negative words, Happy/Upbeat voice
            elif (
                sem_top.label in self.negatives
                and ac_top.label in self.positives
                and sem_top.score >= self.min_conflict_score
                and ac_top.score >= self.min_conflict_score - 0.05
            ):
                is_conflict = True
                details.append(
                    f"Valence conflict (Playful/Discrepant tone): Text is {sem_top.label.value} ({sem_top.score:.2f}) "
                    f"but Tone is {ac_top.label.value} ({ac_top.score:.2f})"
                )

        # 3. Interruption and Cross-Talk Friction Context
        if is_interruption and overlap_duration >= 0.8:
            has_neg = any(
                p.label in self.negatives for p in (semantic_preds[:1] + acoustic_preds[:1])
            )
            if has_neg:
                details.append(
                    f"Hostile talk-over interruption ({overlap_duration:.2f}s cross-talk)"
                )
            else:
                details.append(
                    f"Conversational talk-over interruption ({overlap_duration:.2f}s cross-talk)"
                )
        elif interrupted_by and overlap_duration >= 0.8:
            details.append(f"Interrupted by {interrupted_by} ({overlap_duration:.2f}s cross-talk)")

        conflict_detail = "; ".join(details) if details else None
        return is_conflict, is_ambiguous, conflict_detail
