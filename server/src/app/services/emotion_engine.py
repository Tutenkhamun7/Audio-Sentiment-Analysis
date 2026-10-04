import concurrent.futures
from typing import Dict, List, Optional
import numpy as np
import soundfile as sf

from app.core.config import get_settings
from app.core.factory import create_acoustic_scorer, create_semantic_scorer
from app.interfaces import BaseAcousticScorer, BaseSemanticScorer
from app.schemas.common import EmotionPrediction, TimelineEvent, UnifiedEmotion
from app.services.affect_evaluator import AffectEvaluator


class EmotionEngine:
    """
    Unified engine for Multimodal Semantic and Acoustic Emotion Scoring,
    cross-modal affect evaluation, and conversational affective inertia smoothing.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        context_padding_sec: float = 0.35,
        semantic_scorer: Optional[BaseSemanticScorer] = None,
        acoustic_scorer: Optional[BaseAcousticScorer] = None,
        affect_evaluator: Optional[AffectEvaluator] = None,
    ):
        settings = get_settings()
        self.device = device or settings.emotion_device
        self.context_padding_sec = context_padding_sec
        self.semantic_scorer = semantic_scorer or create_semantic_scorer(settings)
        self.acoustic_scorer = acoustic_scorer or create_acoustic_scorer(settings)
        self.affect_evaluator = affect_evaluator or AffectEvaluator()

        # Aliases for backwards compatibility
        self.text_adapter = self.semantic_scorer
        self.audio_adapter = self.acoustic_scorer

    @staticmethod
    def apply_affective_inertia(events: List[TimelineEvent]) -> List[TimelineEvent]:
        """
        Applies conversational temporal smoothing (Affective Inertia) per speaker.
        In human dialogue, emotional states exhibit inertia and do not flash into
        transient extreme negative affect (e.g. ANGRY) for < 1.8s when surrounded
        by non-negative turns of the same speaker and the semantic text is calm.
        """
        if len(events) < 2:
            return events

        speaker_events: Dict[str, List[int]] = {}
        for idx, ev in enumerate(events):
            speaker_events.setdefault(ev.speaker, []).append(idx)

        non_negative_labels = {UnifiedEmotion.NEUTRAL, UnifiedEmotion.HAPPY}
        negative_arousal = {UnifiedEmotion.ANGRY, UnifiedEmotion.FEARFUL, UnifiedEmotion.DISGUST}

        for speaker, indices in speaker_events.items():
            for i, curr_idx in enumerate(indices):
                curr = events[curr_idx]
                duration = curr.end_time - curr.start_time

                # Only inspect short transient utterances
                if duration > 1.8:
                    continue

                if not curr.acoustic_emotion:
                    continue

                curr_ac_top = curr.acoustic_emotion[0]
                if curr_ac_top.label not in negative_arousal:
                    continue

                curr_sem_top = (
                    curr.semantic_emotion[0].label
                    if curr.semantic_emotion
                    else UnifiedEmotion.NEUTRAL
                )
                if curr_sem_top not in non_negative_labels:
                    # Semantic text itself is negative, so acoustic negativity is valid
                    continue

                # Check preceding and succeeding turns of the SAME speaker
                prev_ev = events[indices[i - 1]] if i > 0 else None
                next_ev = events[indices[i + 1]] if i + 1 < len(indices) else None

                prev_is_calm = prev_ev is None or (
                    curr.start_time - prev_ev.end_time <= 15.0
                    and prev_ev.acoustic_emotion
                    and prev_ev.acoustic_emotion[0].label in non_negative_labels
                )
                next_is_calm = next_ev is None or (
                    next_ev.start_time - curr.end_time <= 15.0
                    and next_ev.acoustic_emotion
                    and next_ev.acoustic_emotion[0].label in non_negative_labels
                )

                if prev_is_calm and next_is_calm:
                    # Transient acoustic spike surrounded by calm turns and calm text.
                    # Promote NEUTRAL or fallback to calm affect.
                    calibrated = list(curr.acoustic_emotion)
                    neutral_pred = next(
                        (p for p in calibrated if p.label in non_negative_labels), None
                    )
                    if neutral_pred:
                        n_idx = calibrated.index(neutral_pred)
                        calibrated[0], calibrated[n_idx] = calibrated[n_idx], calibrated[0]
                    else:
                        calibrated[0] = EmotionPrediction(
                            label=UnifiedEmotion.NEUTRAL,
                            original_label="affective_inertia_smoothing",
                            score=curr_ac_top.score,
                        )
                    curr.acoustic_emotion = calibrated
                    # Clear conflict if it was falsely triggered by this transient spike
                    curr.is_conflict = False
                    curr.conflict_detail = None

        return events

    def process(self, file_path: str, diarized_segments: List[Dict]) -> List[TimelineEvent]:
        """
        Takes the audio file and speaker segments, scores them in parallel,
        evaluates affect dynamics, and returns TimelineEvent items.
        """
        if not diarized_segments:
            return []

        audio_data, sample_rate = sf.read(file_path, dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        total_frames = len(audio_data)
        pad_frames = int(self.context_padding_sec * sample_rate)

        def run_semantic() -> List[List[EmotionPrediction]]:
            texts = [seg.get("text", "").strip() or " " for seg in diarized_segments]
            return self.semantic_scorer.score_batch(texts)

        def run_acoustic() -> List[List[EmotionPrediction]]:
            results = []
            for seg in diarized_segments:
                start_frame = max(0, int(seg["start"] * sample_rate) - pad_frames)
                end_frame = min(total_frames, int(seg["end"] * sample_rate) + pad_frames)
                chunk = audio_data[start_frame:end_frame]
                results.append(self.acoustic_scorer.score_chunk(chunk, sample_rate))
            return results

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_sem = executor.submit(run_semantic)
            fut_ac = executor.submit(run_acoustic)
            semantic_results = fut_sem.result()
            acoustic_results = fut_ac.result()

        timeline_events = []
        for idx, (segment, text_preds, audio_preds) in enumerate(
            zip(diarized_segments, semantic_results, acoustic_results)
        ):
            seg_text = segment.get("text", "").strip()
            calibrated_audio = self.affect_evaluator.disambiguate_acoustic(
                text_preds, audio_preds, text=seg_text
            )
            is_conflict, is_ambiguous, conflict_detail = self.affect_evaluator.evaluate(
                text_preds,
                calibrated_audio,
                is_interruption=segment.get("is_interruption", False),
                interrupted_by=segment.get("interrupted_by"),
                overlap_duration=segment.get("overlap_duration", 0.0),
                text=seg_text,
            )
            event = TimelineEvent(
                segment_id=idx,
                speaker=segment["speaker"],
                start_time=round(segment["start"], 2),
                end_time=round(segment["end"], 2),
                text=seg_text,
                semantic_emotion=text_preds,
                acoustic_emotion=calibrated_audio,
                is_conflict=is_conflict,
                is_ambiguous=is_ambiguous,
                conflict_detail=conflict_detail,
                is_interruption=segment.get("is_interruption", False),
                interrupted_by=segment.get("interrupted_by"),
                overlap_duration=segment.get("overlap_duration", 0.0),
            )
            timeline_events.append(event)

        # Apply conversational affective inertia to smooth transient single-turn acoustic spikes
        timeline_events = self.apply_affective_inertia(timeline_events)

        return timeline_events
