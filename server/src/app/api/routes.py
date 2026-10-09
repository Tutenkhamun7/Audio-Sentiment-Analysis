"""FastAPI API endpoints for app service."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import torch

from app.core.audio import load_audio_from_bytes
from app.core.config import settings
from app.schemas.request import AnalyzeOptions
from app.schemas.response import AnalyzeResponse
from app.services.orchestrator import AudioAnalysisOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file (wav, mp3, m4a, flac)"),
    num_speakers: Optional[int] = Form(None, description="Expected speaker count"),
    force_branch: Optional[str] = Form(None, description="Force 'stereo', 'mono', or 'no_split'"),
    enable_overlap_separation: bool = Form(True, description="Enable SepFormer overlap separation"),
    vad_filter: bool = Form(False, description="Filter silence with VAD in transcription"),
    align_words: bool = Form(True, description="Run Wav2Vec2 CTC forced alignment for millisecond words"),
    predict_emotion: bool = Form(True, description="Extract acoustic emotion and text sentiment"),
    merge_consecutive: bool = Form(True, description="Merge consecutive turns of the same speaker"),
    initial_prompt: Optional[str] = Form(None, description="Initial prompt context for Whisper vocabulary/accents"),
) -> AnalyzeResponse:
    """Analyze uploaded conversation audio in RAM with zero disk file generation."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Decode in RAM
        waveform, orig_sr, duration_s = load_audio_from_bytes(content, target_sr=settings.target_sr)
    except Exception as e:
        logger.error("Failed to decode audio: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted audio file: {e}")

    options = AnalyzeOptions(
        num_speakers=num_speakers,
        force_branch=force_branch,
        enable_overlap_separation=enable_overlap_separation,
        vad_filter=vad_filter,
        align_words=align_words,
        predict_emotion=predict_emotion,
        merge_consecutive=merge_consecutive,
        initial_prompt=initial_prompt,
    )

    orchestrator = AudioAnalysisOrchestrator()
    try:
        result = orchestrator.analyze(
            waveform=waveform,
            sr=settings.target_sr,
            options=options,
            call_id=file.filename or "call",
        )
        return result
    except Exception as err:
        logger.exception("Analysis pipeline failed: %s", err)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(err)}")


@router.get("/health")
def health_check() -> dict[str, object]:
    """Health status and current hardware/device routing."""
    cuda_available = torch.cuda.is_available()
    vram_info = None
    if cuda_available:
        device_idx = torch.cuda.current_device()
        vram_info = {
            "device_name": torch.cuda.get_device_name(device_idx),
            "allocated_mb": round(torch.cuda.memory_allocated(device_idx) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(device_idx) / (1024 * 1024), 2),
        }

    return {
        "status": "ok",
        "service": "app-audio-sentiment-service",
        "cuda_available": cuda_available,
        "default_device": settings.default_device,
        "device_routing": {
            "diarization": settings.get_device("diarization"),
            "separation": settings.get_device("separation"),
            "asr": settings.get_device("asr"),
            "alignment": settings.get_device("alignment"),
            "acoustic": settings.get_device("acoustic"),
            "semantic": settings.get_device("semantic"),
        },
        "vram": vram_info,
    }
