import asyncio
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Optional
import uuid

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.schemas.common import AudioResponse
from app.services.orchestrator import PipelineOrchestrator

logger = logging.getLogger("server.api.analyze")

router = APIRouter(tags=["Audio Analysis"])

SUPPORTED_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
    ".webm",
    ".aac",
    ".wma",
    ".opus",
)


@router.post(
    "/api/v1/audio/analyze",
    response_model=AudioResponse,
    summary="Analyze an uploaded audio file for sentiment and emotion",
)
@router.post(
    "/analyze",
    response_model=AudioResponse,
    include_in_schema=False,
)
async def analyze_audio(
    request: Request,
    file: UploadFile = File(...),
) -> AudioResponse:
    """
    Receives an audio file, processes it via ASR and Emotion engines asynchronously,
    and returns synchronized timeline events with semantic and acoustic sentiment.
    Uses a hybrid gatekeeper to normalize WebM and compressed audio to 16kHz mono WAV.
    """
    orchestrator: Optional[PipelineOrchestrator] = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline models are still initializing. Please try again shortly.",
        )

    filename = file.filename or "recording.wav"
    lower_name = filename.lower()

    # 1. Accept WebM alongside standard audio formats
    if not lower_name.endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    server_dir = Path(__file__).resolve().parents[3]
    temp_dir = server_dir / "temp_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Unique IDs prevent filename collisions under concurrent requests
    unique_prefix = uuid.uuid4().hex
    raw_file_path = str(temp_dir / f"{unique_prefix}_{filename}")
    wav_file_path = None

    try:
        # Save raw uploaded file
        with open(raw_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. Universal Gatekeeper: Normalize all audio to 16kHz Mono 16-bit PCM WAV
        wav_file_path = str(temp_dir / f"{unique_prefix}_transcoded.wav")
        logger.info("Normalizing audio input to 16kHz Mono 16-bit PCM WAV via FFmpeg...")

        await asyncio.to_thread(
            subprocess.run,
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_file_path,
                "-ar",
                "16000",  # Resample to 16kHz (required for accurate Pyannote & emotion2vec+)
                "-ac",
                "1",  # Force Mono
                "-c:a",
                "pcm_s16le",  # Standard 16-bit WAV encoding
                "-loglevel",
                "error",  # Hide FFmpeg terminal spam
                wav_file_path,
            ],
            check=True,
        )

        # 3. Pass the normalized file to the ML pipeline
        response = await orchestrator.process_audio_async(wav_file_path)
        response.file_name = filename
        return response

    except subprocess.CalledProcessError as cpe:
        logger.error(f"FFmpeg transcoding failed for '{filename}': {cpe}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="FFmpeg could not decode audio. File might be corrupted or invalid.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error analyzing audio file '{filename}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {exc}",
        )
    finally:
        # 4. Clean up both the raw upload and the converted file
        for p in (raw_file_path, wav_file_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
