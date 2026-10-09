"""Entrypoint for the Audio Sentiment & Emotion Analysis FastAPI server."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure 'src' directory is in Python path for direct script execution
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Register Windows FFmpeg DLL directory if configured
from app.core.config import settings

ffmpeg_bin = getattr(settings, "ffmpeg_bin_path", None) or os.getenv("FFMPEG_BIN_PATH")
if os.name == "nt" and ffmpeg_bin and os.path.exists(ffmpeg_bin):
    try:
        os.add_dll_directory(ffmpeg_bin)
        if ffmpeg_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{ffmpeg_bin};{os.environ.get('PATH', '')}"
    except Exception as exc:
        logging.warning("Failed to register FFmpeg DLL directory: %s", exc)

from app.api.app import app, create_app

__all__ = ["app", "create_app", "main"]


def main() -> None:
    """Run the FastAPI server via Uvicorn."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() in ("true", "1")

    print(f"\n[*] Starting Audio Sentiment Analysis Server on http://{host}:{port} ...\n")
    uvicorn.run("main:app", host=host, port=port, reload=reload, app_dir=str(SRC_DIR))


if __name__ == "__main__":
    main()
