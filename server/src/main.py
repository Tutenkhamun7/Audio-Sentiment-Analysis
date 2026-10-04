import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.core.config import get_settings
from app.services.orchestrator import PipelineOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager that manages startup and shutdown lifecycle.
    Loads models into memory once at startup so they persist across requests.
    """
    settings = get_settings()

    ffmpeg_bin_path = settings.ffmpeg_bin_path or os.getenv("FFMPEG_BIN_PATH")
    if os.name == "nt" and ffmpeg_bin_path and os.path.exists(ffmpeg_bin_path):
        try:
            os.add_dll_directory(ffmpeg_bin_path)
            if ffmpeg_bin_path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{ffmpeg_bin_path};{os.environ.get('PATH', '')}"
            logger.info(f"Registered FFmpeg DLL directory: {ffmpeg_bin_path}")
        except Exception as exc:
            logger.warning(f"Failed to register FFmpeg DLL directory: {exc}")

    hf_status = (
        "configured" if settings.hf_token else "MISSING (Pyannote Diarization will be disabled!)"
    )
    logger.info(
        f"Initializing Pipeline Orchestrator with settings: "
        f"Whisper='{settings.whisper_model}' (device={settings.effective_whisper_device}), "
        f"Pyannote='{settings.pyannote_model}' (device={settings.effective_diarization_device}, HF_TOKEN={hf_status}), "
        f"Emotion (device={settings.emotion_device})"
    )
    try:
        orchestrator = PipelineOrchestrator(settings=settings)
        app.state.orchestrator = orchestrator
        logger.info("Pipeline Orchestrator initialized successfully. Models ready.")
    except Exception as exc:
        logger.error(f"Failed to initialize PipelineOrchestrator: {exc}", exc_info=True)
        raise exc

    yield

    logger.info("Shutting down server and releasing orchestrator resources...")
    app.state.orchestrator = None


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title="Audio Sentiment & Emotion Analysis API",
        version="0.2.0",
        description="Multimodal Speech-to-Text and Emotion Scoring pipeline.",
        lifespan=lifespan,
    )

    # Enable CORS for the web client
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register modular API routers
    app.include_router(api_router)

    return app


app = create_app()


def main():
    import uvicorn

    src_dir = os.path.dirname(os.path.abspath(__file__))
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, app_dir=src_dir)


if __name__ == "__main__":
    main()
