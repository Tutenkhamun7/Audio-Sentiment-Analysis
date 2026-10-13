"""End-to-end in-memory audio analysis orchestrator."""

from __future__ import annotations

import gc
import logging
import time
from typing import Dict, List, Optional
import uuid

import torch

from app.core.audio import to_mono
from app.core.config import settings
from app.engine.acoustic import (
    EMOTION_COORDINATES,
    compute_dimensional_affect,
    extract_turn_emotions,
)
from app.engine.aligner import align_and_snap_turn_boundaries, assign_words_to_speakers
from app.engine.branch import decide_branch
from app.engine.diarizer import DiarizationResult, RawTurn, run_diarization
from app.engine.overlap_separator import separate_overlaps_in_memory
from app.engine.segmenter import (
    assemble_interleaved_dialogue,
    group_aligned_words_into_turns,
    group_words_into_turns,
    reattribute_turns_by_timbre,
)
from app.engine.semantic import analyze_texts_batch
from app.engine.transcriber import transcribe_stream_words
from app.schemas.request import AnalyzeOptions
from app.schemas.response import (
    AnalyzeResponse,
    CallMetrics,
    EmotionAnalysis,
    SentimentAnalysis,
    TurnDetail,
    WordTiming,
)

logger = logging.getLogger(__name__)


class AudioAnalysisOrchestrator:
    """Coordinates in-memory diarization, overlap separation, ASR, and emotion extraction."""

    def __init__(self) -> None:
        pass

    def analyze(
        self,
        waveform: torch.Tensor,
        sr: int = 16000,
        options: Optional[AnalyzeOptions] = None,
        call_id: Optional[str] = None,
    ) -> AnalyzeResponse:
        """Run complete audio analysis pipeline in RAM without writing temporary WAV files."""
        t0 = time.perf_counter()
        call_id = call_id or f"call_{uuid.uuid4().hex[:8]}"
        opts = options or AnalyzeOptions()

        total_samples = waveform.shape[-1]
        duration_s = float(total_samples) / float(sr)

        # ---------------------------------------------------------
        # Step 1: Branch Decision (Instant bypass if clean stereo)
        # ---------------------------------------------------------
        branch, _ = decide_branch(
            waveform,
            corr_threshold=settings.stereo_corr_threshold,
            min_energy_ratio=settings.stereo_energy_ratio_min,
            force_branch=opts.force_branch,
        )

        speaker_streams: Dict[str, torch.Tensor] = {}
        overlaps_list = []

        if branch == "stereo_split":
            # Left channel = Speaker 0, Right channel = Speaker 1
            spk0 = waveform[0]
            spk1 = waveform[1]
            speaker_streams["SPEAKER_00"] = spk0
            speaker_streams["SPEAKER_01"] = spk1
            speakers = ["SPEAKER_00", "SPEAKER_01"]
        elif branch == "no_split":
            mono = to_mono(waveform).squeeze(0)
            speaker_streams["SPEAKER_00"] = mono
            speakers = ["SPEAKER_00"]
        else:
            # Mono separated branch
            mono = to_mono(waveform)
            # -----------------------------------------------------
            # Step 2: Diarization
            # -----------------------------------------------------
            diar_res: DiarizationResult = run_diarization(
                mono_waveform=mono,
                sr=sr,
                num_speakers=opts.num_speakers,
                min_overlap_duration_s=settings.min_overlap_duration_s,
                min_speaker_duration_s=settings.min_speaker_duration_s,
                merge_consecutive=opts.merge_consecutive,
                max_merge_gap_s=settings.max_merge_gap_s,
                min_turn_duration_s=settings.min_turn_duration_s,
            )
            speakers = diar_res.speakers
            overlaps_list = diar_res.overlaps

            # -----------------------------------------------------
            # Step 3: Targeted Overlap Separation (SepFormer)
            # -----------------------------------------------------
            if opts.enable_overlap_separation and settings.enable_overlap_separation:
                speaker_streams = separate_overlaps_in_memory(
                    mono=mono,
                    sr=sr,
                    diarization=diar_res,
                    overlap_padding_s=settings.overlap_padding_s,
                )
            else:
                for spk in speakers:
                    spk_wav = torch.zeros(total_samples, dtype=torch.float32)
                    for t in diar_res.turns:
                        if t.speaker == spk:
                            s_idx = max(0, int(t.start * sr))
                            e_idx = min(total_samples, int(t.end * sr))
                            spk_wav[s_idx:e_idx] = mono.squeeze(0)[s_idx:e_idx]
                    speaker_streams[spk] = spk_wav

        # ---------------------------------------------------------
        # Step 4: Transcription & Conversational Turn Segmentation
        # ---------------------------------------------------------
        # Flush PyTorch cached allocations from diarization & separation
        # to ensure CTranslate2 Faster-Whisper has maximum clean VRAM headroom.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        prompt = opts.initial_prompt or settings.whisper_initial_prompt
        valid_raw_turns: List[RawTurn] = []

        if branch == "stereo_split":
            # Native 2-channel stereo: each channel is an independent physical microphone stream
            per_speaker_turns: Dict[str, List[RawTurn]] = {}
            for spk in speakers:
                spk_audio = speaker_streams.get(spk, torch.zeros(total_samples))
                spk_words = transcribe_stream_words(
                    speaker_wav=spk_audio,
                    sr=sr,
                    vad_filter=opts.vad_filter,
                    initial_prompt=prompt,
                    min_word_prob=settings.min_word_prob,
                    language=settings.whisper_language,
                )
                spk_turns = group_words_into_turns(
                    words=spk_words,
                    speaker=spk,
                    max_pause_s=settings.max_turn_pause_s,
                    soft_turn_duration_s=settings.soft_turn_duration_s,
                    max_turn_duration_s=settings.max_turn_duration_s,
                )
                if opts.align_words and spk_turns:
                    align_and_snap_turn_boundaries(speaker_wav=spk_audio, turns=spk_turns, sr=sr)
                if opts.predict_emotion and spk_turns:
                    extract_turn_emotions(
                        speaker_wav=spk_audio,
                        turns=spk_turns,
                        sr=sr,
                        min_speech_turn_s=settings.min_speech_turn_s,
                        collar_s=settings.acoustic_collar_s,
                        min_acoustic_confidence=settings.min_acoustic_confidence,
                        min_top_margin=settings.min_acoustic_margin,
                    )
                per_speaker_turns[spk] = spk_turns

            valid_raw_turns = assemble_interleaved_dialogue(
                per_speaker_turns=per_speaker_turns,
                backchannel_duration_s=settings.backchannel_threshold_s,
            )
        else:
            # Mono Continuous ASR + Word-Level Diarization Intersection (WhisperX Paradigm)
            # Transcribe raw continuous mono audio ONCE so no syllables or words are ever silenced or chopped by maskers
            mono_continuous = mono.squeeze(0) if mono.ndim > 1 else mono
            raw_words = transcribe_stream_words(
                speaker_wav=mono_continuous,
                sr=sr,
                vad_filter=opts.vad_filter,
                initial_prompt=prompt,
                min_word_prob=settings.min_word_prob,
                language=settings.whisper_language,
            )

            # Map words to diarization turns via temporal intersection, acoustic energy disambiguation, and floor inertia
            diar_turns_for_mapping = diar_res.turns if (branch == "mono_separated" and 'diar_res' in locals()) else []
            primary_speaker = speakers[0] if speakers else "SPEAKER_00"
            assigned_words = assign_words_to_speakers(
                words=raw_words,
                diarization_turns=diar_turns_for_mapping,
                fallback_speaker=primary_speaker,
                speaker_streams=speaker_streams if (branch == "mono_separated" and opts.enable_overlap_separation) else None,
                sr=sr,
            )

            # Group words per speaker to preserve coherent conversational turns without 1-word shredding during overlaps
            per_speaker_turns: Dict[str, List[RawTurn]] = {}
            for spk in speakers:
                spk_words = [w for w in assigned_words if w.speaker == spk]
                if spk_words:
                    per_speaker_turns[spk] = group_words_into_turns(
                        words=spk_words,
                        speaker=spk,
                        max_pause_s=settings.max_turn_pause_s,
                        soft_turn_duration_s=settings.soft_turn_duration_s,
                        max_turn_duration_s=settings.max_turn_duration_s,
                    )

            valid_raw_turns = assemble_interleaved_dialogue(
                per_speaker_turns=per_speaker_turns,
                backchannel_duration_s=settings.backchannel_threshold_s,
            )

            # Fallback for no_split mode with non-speech/tones where zero words were recognized
            if not valid_raw_turns and branch == "no_split" and duration_s >= settings.min_turn_duration_s:
                fallback_turn = RawTurn(speaker=primary_speaker, start=0.0, end=round(duration_s, 3))
                valid_raw_turns = [fallback_turn]

            # CTC alignment and boundary snapping on mono continuous audio
            if opts.align_words and valid_raw_turns:
                align_and_snap_turn_boundaries(
                    speaker_wav=mono_continuous,
                    turns=valid_raw_turns,
                    sr=sr,
                )

            # Acoustic Emotion Extraction on turn slices (using isolated speaker streams when available)
            if opts.predict_emotion and valid_raw_turns:
                if speaker_streams and branch == "mono_separated":
                    for spk, stream in speaker_streams.items():
                        spk_turns = [t for t in valid_raw_turns if t.speaker == spk]
                        if spk_turns:
                            extract_turn_emotions(
                                speaker_wav=stream,
                                turns=spk_turns,
                                sr=sr,
                                min_speech_turn_s=settings.min_speech_turn_s,
                                collar_s=settings.acoustic_collar_s,
                                min_acoustic_confidence=settings.min_acoustic_confidence,
                                min_top_margin=settings.min_acoustic_margin,
                            )
                else:
                    extract_turn_emotions(
                        speaker_wav=mono_continuous,
                        turns=valid_raw_turns,
                        sr=sr,
                        min_speech_turn_s=settings.min_speech_turn_s,
                        collar_s=settings.acoustic_collar_s,
                        min_acoustic_confidence=settings.min_acoustic_confidence,
                        min_top_margin=settings.min_acoustic_margin,
                    )

            # Tag acoustic backchannels
            valid_raw_turns = assemble_interleaved_dialogue(
                per_speaker_turns=valid_raw_turns,
                backchannel_duration_s=settings.backchannel_threshold_s,
            )

            # Timbral speaker verification for turns near rapid handovers
            if len(speakers) > 1 and branch == "mono_separated":
                valid_raw_turns = reattribute_turns_by_timbre(
                    turns=valid_raw_turns,
                    mono_wav=mono,
                    sr=sr,
                )

        final_turns: List[TurnDetail] = []
        interruption_count = 0
        total_overlap_duration = sum(o.duration for o in overlaps_list)

        # Batch analyze textual affects across all turns with preceding conversational context
        turn_texts = [getattr(t, "text", "") or "" for t in valid_raw_turns]
        prior_texts = [
            getattr(valid_raw_turns[i - 1], "text", None) if i > 0 else None
            for i in range(len(valid_raw_turns))
        ]
        turn_sentiments = analyze_texts_batch(turn_texts, prior_texts=prior_texts)

        for idx, t in enumerate(valid_raw_turns):
            # Check interruption dynamic
            is_interrupt = False
            interrupted_by = None
            interrupts = None

            if idx > 0 and len(final_turns) > 0:
                prev_turn = valid_raw_turns[idx - 1]
                # If current turn starts before previous turn ends
                if t.start < prev_turn.end and t.speaker != prev_turn.speaker:
                    # If previous speaker stopped within interrupt_window_s after current speaker started
                    # AND current turn is not a simple backchannel
                    if (prev_turn.end - t.start) <= settings.interrupt_window_s and getattr(t, "turn_type", "speech") != "backchannel":
                        is_interrupt = True
                        interrupts = prev_turn.speaker
                        interruption_count += 1
                        final_turns[-1].interrupted_by = t.speaker

            turn_words = getattr(t, "words", []) or []
            turn_text = getattr(t, "text", None)
            turn_emotion = getattr(t, "emotion", None) or EmotionAnalysis()

            s_lbl, s_conf, s_scores = (
                turn_sentiments[idx]
                if idx < len(turn_sentiments)
                else ("neutral", 1.0, {"neutral": 1.0})
            )
            turn_sentiment = SentimentAnalysis(label=s_lbl, score=s_conf, scores=s_scores)

            # -------------------------------------------------------------
            # Multimodal Fusion & Sarcasm Incongruence Engine
            # -------------------------------------------------------------
            is_sarcastic = False
            incongruence_score = 0.0

            # Case A: Acoustic emotion was abstained or unavailable -> text affect fallback
            if turn_emotion.abstained or not turn_emotion.dominant_emotion:
                turn_emotion.dominant_emotion = s_lbl
                turn_emotion.scores = dict(s_scores)
                val, aro = compute_dimensional_affect(s_scores)
                turn_emotion.valence = val
                turn_emotion.arousal = aro
            else:
                a_lbl = turn_emotion.dominant_emotion
                a_conf = turn_emotion.scores.get(a_lbl, 0.5)

                # Case B: Sarcasm / Affect Incongruence Detection
                # Text is positive/happy, but vocal tone is angry, disgusted, or has strong negative valence
                text_is_positive = (s_lbl == "happy" or s_scores.get("happy", 0.0) >= 0.40)
                voice_is_negative = (
                    a_lbl in ("angry", "disgusted", "sad")
                    or (turn_emotion.valence is not None and turn_emotion.valence <= -0.25)
                )

                if settings.enable_sarcasm_detection and text_is_positive and voice_is_negative:
                    is_sarcastic = True
                    incongruence_score = round(float(s_conf * a_conf), 4)
                    turn_emotion.is_sarcastic = True
                    turn_emotion.incongruence_score = incongruence_score
                    # Vocal affect exposes true underlying emotional intent
                    turn_emotion.dominant_emotion = a_lbl
                    val, aro = compute_dimensional_affect(turn_emotion.scores)
                    turn_emotion.valence = val
                    turn_emotion.arousal = aro
                elif opts.predict_emotion:
                    # Case C: Joint Multimodal Late Fusion with Confidence Calibration
                    # If semantic classifier is confident in a non-neutral emotion (e.g. "angry" >= 0.70)
                    # while acoustic model defaults to generic neutral baseline, discount the uncalibrated acoustic prior.
                    if s_lbl != "neutral" and s_conf >= 0.70 and a_lbl == "neutral":
                        alpha = 0.20
                    elif t.duration < 1.0:
                        alpha = 0.35
                    else:
                        alpha = 0.55

                    fused_scores: dict[str, float] = {}
                    all_keys = set(turn_emotion.scores.keys()).union(set(s_scores.keys()))
                    for k in all_keys:
                        a_p = turn_emotion.scores.get(k, 0.0)
                        s_p = s_scores.get(k, 0.0)
                        fused_scores[k] = round(alpha * a_p + (1.0 - alpha) * s_p, 4)

                    if fused_scores:
                        turn_emotion.scores = fused_scores
                        best_emotion = max(fused_scores.items(), key=lambda kv: kv[1])[0]
                        turn_emotion.dominant_emotion = best_emotion
                        val, aro = compute_dimensional_affect(fused_scores)
                        turn_emotion.valence = val
                        turn_emotion.arousal = aro

            final_turns.append(
                TurnDetail(
                    turn_id=f"turn_{idx:03d}",
                    speaker=t.speaker,
                    start=round(t.start, 3),
                    end=round(t.end, 3),
                    duration=round(t.duration, 3),
                    text=turn_text,
                    words=turn_words,
                    emotion=turn_emotion,
                    sentiment=turn_sentiment,
                    is_interruption=is_interrupt,
                    interrupted_by=interrupted_by,
                    interrupts=interrupts,
                    overlap_ratio=round(total_overlap_duration / max(0.1, duration_s), 4),
                    turn_type=getattr(t, "turn_type", "speech"),
                    is_sarcastic=is_sarcastic,
                )
            )

        total_speech_s = sum(t.duration for t in final_turns)

        metrics = CallMetrics(
            duration_s=round(duration_s, 2),
            total_speech_s=round(total_speech_s, 2),
            overlap_s=round(total_overlap_duration, 2),
            overlap_ratio=round(total_overlap_duration / max(0.1, duration_s), 4),
            interruption_count=interruption_count,
            speaker_count=len(speakers),
            branch_used=branch,
        )

        elapsed = round(time.perf_counter() - t0, 3)

        # Free cached memory allocations to keep VRAM clean for subsequent requests
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return AnalyzeResponse(
            call_id=call_id,
            metrics=metrics,
            turns=final_turns,
            speakers=speakers,
            processing_time_s=elapsed,
        )
