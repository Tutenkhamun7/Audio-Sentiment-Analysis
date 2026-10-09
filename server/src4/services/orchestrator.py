"""End-to-end in-memory audio analysis orchestrator."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional
import uuid

import torch

from src3.core.audio import to_mono
from src3.core.config import settings
from src3.engine.acoustic import extract_turn_emotions
from src3.engine.aligner import align_and_snap_turn_boundaries
from src3.engine.branch import decide_branch
from src3.engine.diarizer import DiarizationResult, RawTurn, run_diarization
from src3.engine.overlap_separator import separate_overlaps_in_memory
from src3.engine.semantic import analyze_text_sentiment, analyze_texts_batch
from src3.engine.transcriber import transcribe_speaker_turns
from src3.schemas.request import AnalyzeOptions
from src3.schemas.response import (
    AnalyzeResponse,
    CallMetrics,
    EmotionAnalysis,
    SentimentAnalysis,
    TurnDetail,
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
        all_raw_turns: List[RawTurn] = []
        overlaps_list = []

        if branch == "stereo_split":
            # Left channel = Speaker 0, Right channel = Speaker 1
            spk0 = waveform[0]
            spk1 = waveform[1]
            speaker_streams["SPEAKER_00"] = spk0
            speaker_streams["SPEAKER_01"] = spk1
            # Run simple VAD / full timeline
            all_raw_turns.append(RawTurn(speaker="SPEAKER_00", start=0.0, end=duration_s))
            all_raw_turns.append(RawTurn(speaker="SPEAKER_01", start=0.0, end=duration_s))
            speakers = ["SPEAKER_00", "SPEAKER_01"]
        elif branch == "no_split":
            mono = to_mono(waveform).squeeze(0)
            speaker_streams["SPEAKER_00"] = mono
            all_raw_turns.append(RawTurn(speaker="SPEAKER_00", start=0.0, end=duration_s))
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
                merge_consecutive=opts.merge_consecutive,
                max_merge_gap_s=settings.max_merge_gap_s,
                min_turn_duration_s=settings.min_turn_duration_s,
            )
            speakers = diar_res.speakers
            all_raw_turns = diar_res.turns
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
        # Step 4: Transcription & Forced Alignment (Per Speaker)
        # ---------------------------------------------------------
        prompt = opts.initial_prompt or settings.whisper_initial_prompt
        for spk in speakers:
            spk_audio = speaker_streams.get(spk, torch.zeros(total_samples))
            spk_turns = [t for t in all_raw_turns if t.speaker == spk]

            # 4a. Transcribe via Faster-Whisper
            transcribe_speaker_turns(
                speaker_wav=spk_audio,
                turns=spk_turns,
                sr=sr,
                vad_filter=opts.vad_filter,
                initial_prompt=prompt,
                collar_s=settings.asr_collar_s,
                all_turns=all_raw_turns,
                min_word_prob=settings.min_word_prob,
            )

            # 4b. Align & Snap turn boundaries via CTC Wav2Vec2
            if opts.align_words:
                align_and_snap_turn_boundaries(
                    speaker_wav=spk_audio,
                    turns=spk_turns,
                    sr=sr,
                )

            # 4c. Acoustic Emotion (emotion2vec)
            if opts.predict_emotion:
                extract_turn_emotions(
                    speaker_wav=spk_audio,
                    turns=spk_turns,
                    sr=sr,
                    min_speech_turn_s=settings.min_speech_turn_s,
                )

        # Filter out micro-turns without speech
        valid_raw_turns = []
        for t in all_raw_turns:
            has_text = bool(getattr(t, "text", None) and getattr(t, "text", "").strip())
            # In no_split mode keep turns with sufficient duration (even for tones/silence)
            if branch == "no_split":
                if has_text or t.duration >= settings.min_turn_duration_s:
                    valid_raw_turns.append(t)
            else:
                # In diarized modes, keep turns that produced recognized speech
                if has_text:
                    valid_raw_turns.append(t)
        valid_raw_turns.sort(key=lambda t: (t.start, t.end))

        final_turns: List[TurnDetail] = []
        interruption_count = 0
        total_overlap_duration = sum(o.duration for o in overlaps_list)

        # Batch analyze textual sentiments across all turns
        turn_texts = [getattr(t, "text", "") or "" for t in valid_raw_turns]
        turn_sentiments = analyze_texts_batch(turn_texts)

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
                    if (prev_turn.end - t.start) <= settings.interrupt_window_s:
                        is_interrupt = True
                        interrupts = prev_turn.speaker
                        interruption_count += 1
                        final_turns[-1].interrupted_by = t.speaker

            turn_words = getattr(t, "words", [])
            turn_text = getattr(t, "text", None)
            turn_emotion = getattr(t, "emotion", EmotionAnalysis())

            s_lbl, s_conf = turn_sentiments[idx] if idx < len(turn_sentiments) else ("neutral", 1.0)
            turn_sentiment = SentimentAnalysis(label=s_lbl, score=s_conf)

            # Contextual affect augmentation: if acoustic emotion was abstained or empty,
            # provide conversational sentiment fallback
            if not turn_emotion.dominant_emotion:
                turn_emotion.dominant_emotion = s_lbl
                if s_lbl not in turn_emotion.scores:
                    turn_emotion.scores[s_lbl] = s_conf

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

        return AnalyzeResponse(
            call_id=call_id,
            metrics=metrics,
            turns=final_turns,
            speakers=speakers,
            processing_time_s=elapsed,
        )
