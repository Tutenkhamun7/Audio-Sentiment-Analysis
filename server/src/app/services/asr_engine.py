import os
from typing import Dict, List, Optional, Tuple

from app.adapters.asr.faster_whisper import FasterWhisperAdapter
from app.adapters.diarization.pyannote import PyannoteDiarizerAdapter
from app.core.config import get_settings
from app.interfaces import RawSegment, SpeakerInterval
from app.services.alignment import AlignmentService


class ASREngine:
    """
    Unified engine for Speech-to-Text and Speaker Diarization.
    Delegates to modular FasterWhisperAdapter, PyannoteDiarizerAdapter, and AlignmentService.
    """

    def __init__(
        self,
        whisper_model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        hf_token: Optional[str] = None,
        cpu_threads: Optional[int] = None,
        diarization_device: Optional[str] = None,
        pyannote_model: Optional[str] = None,
    ):
        settings = get_settings()

        self.device = device or settings.effective_whisper_device
        self.diarization_device = diarization_device or settings.effective_diarization_device
        self.pyannote_model = pyannote_model or settings.pyannote_model
        self.hf_token = hf_token or settings.hf_token or os.getenv("HF_TOKEN")
        self.compute_type = compute_type or settings.whisper_compute_type
        model_name = whisper_model_size or settings.whisper_model

        # Adapters & Services
        self.asr_adapter = FasterWhisperAdapter(
            model_size_or_path=model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=cpu_threads,
        )
        self.diarizer_adapter = PyannoteDiarizerAdapter(
            model_name=self.pyannote_model,
            hf_token=self.hf_token,
            device=self.diarization_device,
        )
        self.alignment_service = AlignmentService()

    def _diarize_audio(self, file_path: str) -> List[Dict]:
        """Runs diarization and returns list of interval dicts for backward compatibility."""
        intervals = self.diarizer_adapter.diarize(file_path)
        return [{"start": i.start, "end": i.end, "speaker": i.speaker} for i in intervals]

    def _align_words_to_speakers(
        self,
        raw_segments: List[RawSegment],
        speaker_intervals: List[Dict],
        max_duration: float = 6.0,
        min_split_duration: float = 2.0,
    ) -> List[Dict]:
        """Backward-compatible alignment helper."""
        typed_intervals = [
            SpeakerInterval(start=i["start"], end=i["end"], speaker=i["speaker"])
            for i in speaker_intervals
        ]
        return self.alignment_service.align_words_to_speakers(
            raw_segments,
            typed_intervals,
            max_duration=max_duration,
            min_split_duration=min_split_duration,
        )

    def _merge_segments_by_speaker(
        self,
        turns: List[Dict],
        min_duration: float = 2.0,
        max_gap: float = 0.5,
        max_duration: float = 6.0,
    ) -> List[Dict]:
        """Backward-compatible merge helper."""
        return self.alignment_service.merge_segments_by_speaker(
            turns,
            min_duration=min_duration,
            max_gap=max_gap,
            max_duration=max_duration,
        )

    def process(
        self,
        file_path: str,
        beam_size: int = 5,
        language: Optional[str] = "en",
        min_duration: float = 2.0,
        max_duration: float = 6.0,
        max_gap: float = 0.5,
    ) -> Tuple[List[Dict], List[str]]:
        """Transcribes and diarizes the audio file into speaker-bounded utterance segments."""
        raw_segments = self.asr_adapter.transcribe(
            file_path,
            beam_size=beam_size,
            language=language,
        )
        speaker_intervals = self.diarizer_adapter.diarize(file_path)
        return self.alignment_service.process(
            raw_segments,
            speaker_intervals,
            min_duration=min_duration,
            max_duration=max_duration,
            max_gap=max_gap,
        )


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Test ASREngine facade.")
    default_sample = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "sample", "Annoyance _Anger.mp3"
        )
    )
    parser.add_argument("audio_path", nargs="?", default=default_sample)
    args = parser.parse_args()

    engine = ASREngine()
    start_time = time.time()
    turns, speakers = engine.process(args.audio_path)
    print(f"Processed in {time.time() - start_time:.2f}s")
    print(f"Speakers: {speakers}")
    for t in turns:
        print(f"[{t['start']:.2f}s - {t['end']:.2f}s] {t['speaker']}: {t['text']}")
