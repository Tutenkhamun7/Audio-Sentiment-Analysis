"""FastAPI application and route definitions for app."""

from app.api.app import app, create_app
from app.api.routes import router

__all__ = ["app", "create_app", "router"]
