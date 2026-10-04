import os
from typing import List, Optional, cast
import ctranslate2
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, Word

from app.interfaces import BaseASR, RawSegment, WordTimestamp


class FasterWhisperAdapter(BaseASR):
    """Faster-Whisper implementation of BaseASR supporting CUDA & CPU inference."""

    def __init__(
        self,
        model_size_or_path: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: Optional[str] = None,
        cpu_threads: Optional[int] = None,
    ):
        self.device = device
        self.cpu_threads = cpu_threads or min(8, os.cpu_count() or 4)

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        # Verify CUDA capability and fall back gracefully if missing
        active_device = self.device
        active_compute = self.compute_type
        if active_device == "cuda":
            if ctranslate2.get_cuda_device_count() == 0:
                print(
                    "[FasterWhisperAdapter] Warning: CUDA requested but no devices found. "
                    "Falling back to CPU."
                )
                active_device = "cpu"
                active_compute = "int8"

        try:
            self.model = WhisperModel(
                model_size_or_path=model_size_or_path,
                device=active_device,
                compute_type=active_compute,
                cpu_threads=self.cpu_threads,
                num_workers=1,
            )
            print(
                f"[FasterWhisperAdapter] Model ('{model_size_or_path}') loaded on {active_device} ({active_compute})"
            )
        except Exception as exc:
            if active_device == "cuda":
                print(f"[FasterWhisperAdapter] Failed to load on CUDA: {exc}. Falling back to CPU.")
                self.model = WhisperModel(
                    model_size_or_path=model_size_or_path,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self.cpu_threads,
                    num_workers=1,
                )
            else:
                raise exc

    def transcribe(
        self,
        file_path: str,
        beam_size: int = 5,
        language: Optional[str] = "en",
        vad_filter: bool = False,
        **kwargs,
    ) -> List[RawSegment]:
        """Transcribes the audio file and returns standard RawSegment objects."""
        # Conversational speech VAD tuning: 500ms min silence gap, 400ms safety pad, 150ms noise floor
        default_vad_params = dict(
            threshold=0.5,
            min_silence_duration_ms=500,
            speech_pad_ms=400,
            min_speech_duration_ms=150,
        )
        vad_params = kwargs.pop("vad_parameters", default_vad_params if vad_filter else None)

        segments_gen, _ = self.model.transcribe(
            file_path,
            beam_size=beam_size,
            best_of=beam_size if beam_size > 1 else 1,
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=vad_filter,
            vad_parameters=vad_params,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            **kwargs,
        )

        raw_segments: List[RawSegment] = []
        for s in segments_gen:
            seg = cast(Segment, s)
            words: List[WordTimestamp] = []
            if seg.words is not None:
                for w in seg.words:
                    wd = cast(Word, w)
                    words.append(WordTimestamp(start=wd.start, end=wd.end, word=wd.word))
            raw_segments.append(
                RawSegment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
            )

        return raw_segments
