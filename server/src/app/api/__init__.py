from fastapi import APIRouter

from app.api.analyze import router as analyze_router
from app.api.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(analyze_router)

__all__ = ["api_router", "analyze_router", "health_router"]
