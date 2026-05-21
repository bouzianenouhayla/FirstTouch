from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import router

_BASE = Path(__file__).parent


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with routes and static files mounted.
    """
    app = FastAPI(title="FirstTouch", version="0.1.0")
    app.mount("/static", StaticFiles(directory=_BASE / "static"), name="static")
    app.include_router(router)
    return app
