import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
from pyannote.audio import Pipeline

from app.interfaces import BaseDiarizer, OverlapInterval, SpeakerInterval

logger = logging.getLogger("server.diarization")

# Ensure FFmpeg DLL directory is registered on Windows for torchcodec/torchaudio
FFMPEG_BIN_PATH = os.getenv(
    "FFMPEG_BIN_PATH",
    r"C:\Users\Anvay\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build-shared\bin",
)

if os.name == "nt" and os.path.exists(FFMPEG_BIN_PATH):
    try:
        os.add_dll_directory(FFMPEG_BIN_PATH)
        print(f"[System] Successfully registered FFmpeg DLLs from {FFMPEG_BIN_PATH}")
    except Exception as e:
        print(f"[System] Failed to register FFmpeg DLLs: {e}")


class PyannoteDiarizerAdapter(BaseDiarizer):
    """Pyannote speaker diarization implementation of BaseDiarizer."""

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-community-1",
        hf_token: Optional[str] = None,
        device: str = "cuda",
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.pipeline: Optional[Pipeline] = None

        if self.hf_token:
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.hf_token,
            )
            if self.pipeline:
                target_device = self.device
                if "cuda" in target_device and not torch.cuda.is_available():
                    print(
                        f"[PyannoteDiarizerAdapter] Warning: CUDA requested ('{target_device}') "
                        f"but torch.cuda.is_available() is False. Falling back to CPU."
                    )
                    target_device = "cpu"
                try:
                    self.pipeline.to(torch.device(target_device))
                    print(f"[PyannoteDiarizerAdapter] Loaded ('{model_name}') on {target_device}")
                except Exception as exc:
                    print(
                        f"[PyannoteDiarizerAdapter] Error sending to {target_device}: {exc}. Falling back to CPU."
                    )
                    self.pipeline.to(torch.device("cpu"))
        else:
            msg = "[PyannoteDiarizerAdapter] Warning: HF_TOKEN not provided. Diarization disabled; all utterances will be attributed to a single speaker."
            print(msg)
            logger.warning(msg)

    def diarize_with_overlaps(
        self, file_path: str
    ) -> Tuple[List[SpeakerInterval], List[OverlapInterval]]:
        """Runs diarization on the audio file and returns both speaker turns and overlapping speech intervals."""
        if not self.pipeline:
            msg = "[PyannoteDiarizerAdapter] Diarization skipped: pipeline not initialized (HF_TOKEN missing or invalid)."
            print(msg)
            logger.warning(msg)
            return [], []

        # Construct clustering constraints dynamically (None = natural automatic clustering)
        diarize_kwargs: Dict[str, Any] = {}
        if self.num_speakers is not None and self.num_speakers > 0:
            diarize_kwargs["num_speakers"] = self.num_speakers
        else:
            if self.min_speakers is not None and self.min_speakers > 0:
                diarize_kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers is not None and self.max_speakers > 0:
                diarize_kwargs["max_speakers"] = self.max_speakers

        try:
            diarization: Any = self.pipeline(file_path, **diarize_kwargs)
        except Exception as exc:
            print(f"[PyannoteDiarizerAdapter] Diarization failed on '{file_path}': {exc}")
            return [], []

        intervals: List[SpeakerInterval] = []
        overlap_intervals: List[OverlapInterval] = []

        # Pyannote output inspection (Annotation object or DiarizeOutput dataclass)
        speaker_annotation = getattr(diarization, "speaker_diarization", None)
        if speaker_annotation is None:
            # Check if output is itself an Annotation
            if hasattr(diarization, "itertracks"):
                speaker_annotation = diarization

        if speaker_annotation is not None:
            # 1. Extract chronological speaker intervals
            if hasattr(speaker_annotation, "itertracks"):
                for turn, _, speaker in speaker_annotation.itertracks(yield_label=True):
                    # Filter out zero-length or sub-50ms artifacts
                    if (turn.end - turn.start) >= 0.05:
                        intervals.append(
                            SpeakerInterval(
                                start=float(turn.start), end=float(turn.end), speaker=str(speaker)
                            )
                        )
            else:
                for turn, speaker in speaker_annotation:
                    if (turn.end - turn.start) >= 0.05:
                        intervals.append(
                            SpeakerInterval(
                                start=float(turn.start), end=float(turn.end), speaker=str(speaker)
                            )
                        )

            # 2. Extract multi-speaker overlapping segments natively
            if hasattr(speaker_annotation, "get_overlap"):
                try:
                    overlaps = speaker_annotation.get_overlap()
                    for seg in overlaps:
                        if (seg.end - seg.start) >= 0.05:
                            active_spks = set()
                            if hasattr(speaker_annotation, "itertracks"):
                                for turn, _, spk in speaker_annotation.itertracks(yield_label=True):
                                    if max(turn.start, seg.start) < min(turn.end, seg.end):
                                        active_spks.add(str(spk))
                            overlap_intervals.append(
                                OverlapInterval(
                                    start=round(float(seg.start), 2),
                                    end=round(float(seg.end), 2),
                                    speakers=sorted(list(active_spks)),
                                )
                            )
                except Exception as ov_err:
                    print(f"[PyannoteDiarizerAdapter] Overlap extraction warning: {ov_err}")

        # Sort chronologically by start timestamp
        intervals.sort(key=lambda x: (x.start, x.end))
        overlap_intervals.sort(key=lambda x: (x.start, x.end))
        return intervals, overlap_intervals

    def diarize(self, file_path: str) -> List[SpeakerInterval]:
        """Runs diarization on the audio file and returns chronological intervals."""
        intervals, _ = self.diarize_with_overlaps(file_path)
        return intervals
