import asyncio
from collections import Counter
import os
from typing import Dict, Optional

from app.core.config import Settings, get_settings
from app.core.factory import (
    create_acoustic_scorer,
    create_affect_evaluator,
    create_alignment_service,
    create_asr_adapter,
    create_diarizer_adapter,
    create_emotion_engine,
    create_semantic_scorer,
)
from app.interfaces import (
    BaseAcousticScorer,
    BaseASR,
    BaseDiarizer,
    BaseSemanticScorer,
)
from app.schemas.common import (
    AgentSynthesis,
    AudioAnalysisResult,
    AudioResponse,
    UnifiedEmotion,
)
from app.services.affect_evaluator import AffectEvaluator
from app.services.alignment import AlignmentService
from app.services.emotion_engine import EmotionEngine


class PipelineOrchestrator:
    """
    Coordinates the execution flow across pluggable ASR, Diarization,
    and Emotion scoring adapters and domain services.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        asr_adapter: Optional[BaseASR] = None,
        diarizer_adapter: Optional[BaseDiarizer] = None,
        alignment_service: Optional[AlignmentService] = None,
        affect_evaluator: Optional[AffectEvaluator] = None,
        emotion_engine: Optional[EmotionEngine] = None,
        semantic_scorer: Optional[BaseSemanticScorer] = None,
        acoustic_scorer: Optional[BaseAcousticScorer] = None,
    ):
        self.settings = settings or get_settings()

        # Wire pluggable adapters and domain services
        self.asr_adapter = asr_adapter or create_asr_adapter(self.settings)
        self.diarizer_adapter = diarizer_adapter or create_diarizer_adapter(self.settings)
        self.alignment_service = alignment_service or create_alignment_service(self.settings)
        self.affect_evaluator = affect_evaluator or create_affect_evaluator(self.settings)

        # Wire unified EmotionEngine
        if emotion_engine is not None:
            self.emotion_engine = emotion_engine
        elif semantic_scorer or acoustic_scorer:
            self.emotion_engine = EmotionEngine(
                device=self.settings.emotion_device,
                semantic_scorer=semantic_scorer or create_semantic_scorer(self.settings),
                acoustic_scorer=acoustic_scorer or create_acoustic_scorer(self.settings),
                affect_evaluator=self.affect_evaluator,
            )
        else:
            self.emotion_engine = create_emotion_engine(self.settings)

        # Backward compatibility accessors
        self.semantic_scorer = self.emotion_engine.semantic_scorer
        self.acoustic_scorer = self.emotion_engine.acoustic_scorer

    def _process_synchronous(
        self, file_path: str, split_by_speaker: Optional[bool] = None
    ) -> AudioResponse:
        """The blocking synchronous pipeline running the ML and domain logic."""

        # 1. Speech-to-Text with word timestamps
        raw_segments = self.asr_adapter.transcribe(
            file_path,
            beam_size=self.settings.whisper_beam_size,
            language=self.settings.whisper_language,
            vad_filter=self.settings.vad_filter,
        )

        # 2. Speaker Diarization intervals & Overlapping Speech
        should_split = (
            split_by_speaker
            if split_by_speaker is not None
            else self.settings.split_by_speaker
        )
        if should_split:
            if hasattr(self.diarizer_adapter, "diarize_with_overlaps"):
                speaker_intervals, overlap_intervals = self.diarizer_adapter.diarize_with_overlaps(
                    file_path
                )
            else:
                speaker_intervals = self.diarizer_adapter.diarize(file_path)
                overlap_intervals = []
        else:
            from app.interfaces import SpeakerInterval

            speaker_intervals = [
                SpeakerInterval(start=0.0, end=999999.0, speaker="SPEAKER_00")
            ]
            overlap_intervals = []

        # 3. Domain Alignment & Sentence/Duration Chunking with Overlap Enrichment
        diarized_segments, speakers = self.alignment_service.process(
            raw_segments,
            speaker_intervals,
            overlap_intervals=overlap_intervals,
            min_duration=self.settings.segment_min_duration,
            max_duration=self.settings.segment_max_duration,
            max_gap=self.settings.segment_max_gap,
        )

        if not diarized_segments:
            empty_result = AudioAnalysisResult(
                overall_transcript="",
                timeline=[],
                agent_context=AgentSynthesis(
                    summary="No speech detected in audio.",
                    escalation_detected=False,
                    primary_speaker_sentiments={},
                    flagged_anomalies=[],
                    interruption_count=0,
                    total_overtalk_seconds=0.0,
                    overtalk_ratio=0.0,
                ),
            )
            return AudioResponse(
                file_name=os.path.basename(file_path),
                speakers_detected=[],
                analysis=empty_result,
            )

        # 4. Multimodal Emotion Scoring & Affect Dynamics
        timeline_events = self.emotion_engine.process(file_path, diarized_segments)

        # 5. Consolidated Transcript
        overall_transcript = "\n".join(
            f"[{event.start_time:05.2f}s - {event.end_time:05.2f}s] {event.speaker}: {event.text}"
            for event in timeline_events
        )

        # 6. High-Level Synthesis & Interruption Metrics
        flagged_anomalies = [
            f"[{event.start_time:05.2f}s - {event.end_time:05.2f}s] {event.speaker}: {event.conflict_detail}"
            for event in timeline_events
            if event.is_conflict or (event.conflict_detail and "Hostile" in event.conflict_detail)
        ]

        total_overlap_sec = round(sum(ov.end - ov.start for ov in overlap_intervals), 2)
        total_speech_sec = sum(e.end_time - e.start_time for e in timeline_events)
        overtalk_ratio = (
            round(total_overlap_sec / total_speech_sec, 4) if total_speech_sec > 0 else 0.0
        )
        interruption_count = sum(1 for e in timeline_events if e.is_interruption)

        escalation = any(
            (
                event.acoustic_emotion
                and event.acoustic_emotion[0].label == UnifiedEmotion.ANGRY
                and event.acoustic_emotion[0].score >= 0.50
            )
            or (
                event.is_interruption
                and any(
                    p.label in {UnifiedEmotion.ANGRY, UnifiedEmotion.DISGUST}
                    for p in event.semantic_emotion[:1] + event.acoustic_emotion[:1]
                )
            )
            for event in timeline_events
        )

        speaker_sentiments: Dict[str, str] = {}
        for spk in speakers:
            spk_events = [e for e in timeline_events if e.speaker == spk]
            labels = [e.acoustic_emotion[0].label.value for e in spk_events if e.acoustic_emotion]
            if labels:
                speaker_sentiments[spk] = Counter(labels).most_common(1)[0][0]

        agent_context = AgentSynthesis(
            summary=f"Analyzed {len(timeline_events)} segments across {len(speakers)} speaker(s). "
            f"Detected {interruption_count} interruption(s) totaling {total_overlap_sec}s over-talk.",
            escalation_detected=escalation,
            primary_speaker_sentiments=speaker_sentiments,
            flagged_anomalies=flagged_anomalies,
            interruption_count=interruption_count,
            total_overtalk_seconds=total_overlap_sec,
            overtalk_ratio=overtalk_ratio,
        )

        analysis_result = AudioAnalysisResult(
            overall_transcript=overall_transcript,
            timeline=timeline_events,
            agent_context=agent_context,
        )

        return AudioResponse(
            file_name=os.path.basename(file_path),
            speakers_detected=speakers,
            analysis=analysis_result,
        )

    async def process_audio_async(
        self, file_path: str, split_by_speaker: Optional[bool] = None
    ) -> AudioResponse:
        """Asynchronously dispatches the synchronous pipeline to a worker thread."""
        return await asyncio.to_thread(self._process_synchronous, file_path, split_by_speaker)


if __name__ == "__main__":
    import argparse

    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Ensure FFmpeg DLL path is configured on Windows
    FFMPEG_BIN_PATH = (
        r"C:\Users\Anvay\AppData\Local\Microsoft\WinGet\Packages"
        r"\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
        r"\ffmpeg-9.0.2-full_build-shared\bin"
    )
    if os.name == "nt" and os.path.exists(FFMPEG_BIN_PATH):
        try:
            os.add_dll_directory(FFMPEG_BIN_PATH)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Test the Pipeline Orchestrator directly.")
    parser.add_argument(
        "--audio",
        type=str,
        default=r"C:\Users\Anvay\Downloads\test.wav",
        help="Absolute path to a real test .wav file",
    )
    args = parser.parse_args()

    async def run_test():
        if not os.path.exists(args.audio):
            print(f"ERROR: Could not find file at {os.path.abspath(args.audio)}")
            return

        print("=" * 80)
        print(" RUNNING PIPELINE ORCHESTRATOR DIRECTLY")
        print("=" * 80)
        print(f"[*] Audio File : {args.audio}")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator.process_audio_async(args.audio)

        print("\n" + "=" * 80)
        print(" PIPELINE RESULTS SUMMARY")
        print("=" * 80)
        print(f"File Name              : {result.file_name}")
        print(f"Speakers Detected      : {', '.join(result.speakers_detected)}")
        print(f"Total Timeline Events  : {len(result.analysis.timeline)}")

        if result.analysis.agent_context:
            ctx = result.analysis.agent_context
            print(f"Summary                : {ctx.summary}")
            print(f"Escalation Detected    : {'YES [ALERT]' if ctx.escalation_detected else 'No'}")
            print(f"Interruption Count     : {ctx.interruption_count}")
            print(f"Total Overtalk Seconds : {ctx.total_overtalk_seconds:.2f}s")
            print(f"Overtalk Ratio         : {ctx.overtalk_ratio * 100:.2f}%")
            print(f"Speaker Sentiments     : {ctx.primary_speaker_sentiments}")

            if ctx.flagged_anomalies:
                print("\nFlagged Anomalies / Friction Events:")
                for anomaly in ctx.flagged_anomalies:
                    print(f"  - {anomaly}")

        print("\n" + "=" * 80)
        print(" CHRONOLOGICAL TIMELINE (With Overlap & Interruption Tracking)")
        print("=" * 80)
        for ev in result.analysis.timeline:
            sem = ev.semantic_emotion[0].label.value if ev.semantic_emotion else "N/A"
            ac = ev.acoustic_emotion[0].label.value if ev.acoustic_emotion else "N/A"
            overlap_tag = ""
            if ev.is_interruption:
                overlap_tag += f" [INTERRUPT: {ev.overlap_duration:.2f}s]"
            elif ev.interrupted_by:
                overlap_tag += f" [CUT OFF by {ev.interrupted_by}]"
            elif ev.overlap_duration > 0:
                overlap_tag += f" [OVERLAP: {ev.overlap_duration:.2f}s]"

            print(
                f"[{ev.start_time:06.2f}s - {ev.end_time:06.2f}s] "
                f'{ev.speaker:<11} | Text: {sem:<7} | Tone: {ac:<7}{overlap_tag} | "{ev.text}"'
            )

    asyncio.run(run_test())
