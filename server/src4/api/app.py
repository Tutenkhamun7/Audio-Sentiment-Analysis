"""FastAPI Application factory and lifespan configuration for src3."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src3.api.routes import router
from src3.core.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src3")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager that warms models into VRAM/RAM on startup."""
    logger.info("Initializing src3 service: warming models into memory...")
    try:
        registry = ModelRegistry.get_instance()
        # Pre-warm models so the first HTTP request experiences zero cold-start delay
        registry.warm_all()
    except Exception as e:
        logger.warning("Model warming during startup encountered an error: %s", e)

    yield

    logger.info("Shutting down src3 service...")


def create_app() -> FastAPI:
    """Application factory for src3 FastAPI backend."""
    app = FastAPI(
        title="Audio Sentiment & Diarization Engine (src3)",
        version="1.0.0",
        description="Production in-memory audio sentiment analysis service with targeted overlap separation.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app


app = create_app()
