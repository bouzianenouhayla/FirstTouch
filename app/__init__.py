from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import structlog
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from .logging_config import configure_logging
from .routes import router

_BASE = Path(__file__).parent
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    configure_logging()
    logger.info("startup", version=app.version)
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance with routes, middleware, and static files mounted.
    """
    app = FastAPI(title="FirstTouch", version="0.1.0", lifespan=_lifespan)

    @app.middleware("http")
    async def _bind_request_id(request: Request, call_next):
        structlog.contextvars.clear_contextvars()
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.mount("/static", StaticFiles(directory=_BASE / "static"), name="static")
    app.include_router(router)
    return app
