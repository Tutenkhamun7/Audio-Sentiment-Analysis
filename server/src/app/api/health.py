from typing import Any, Dict
from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/")
async def root(request: Request) -> Dict[str, Any]:
    """Root status check."""
    is_ready = getattr(request.app.state, "orchestrator", None) is not None
    return {
        "name": "Audio Sentiment Analysis API",
        "version": "0.2.0",
        "status": "online",
        "models_loaded": is_ready,
        "docs_url": "/docs",
    }


@router.get("/health")
@router.get("/api/v1/audio/health")
async def health(request: Request) -> Dict[str, Any]:
    """Health check endpoint indicating model readiness."""
    is_ready = getattr(request.app.state, "orchestrator", None) is not None
    return {
        "status": "healthy" if is_ready else "starting",
        "models_loaded": is_ready,
    }
