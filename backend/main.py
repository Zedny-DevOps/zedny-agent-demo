"""FastAPI entry — wires routes, middleware, lifespan, and rate limiting."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from config.settings import settings
from services import modal_client
from utils.logger import JSONLoggingMiddleware, configure_logging, log

from .deps import limiter
from .routes import chat as chat_routes
from .routes import health as health_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info(
        "startup",
        modal_app=settings.modal_app_name,
        modal_class=settings.modal_class_name,
        rpm_limit=settings.rate_limit_per_minute,
    )
    # Fire-and-forget warmup so the first real request doesn't pay cold start.
    import asyncio

    asyncio.create_task(modal_client.warmup())
    yield
    log.info("shutdown")


app = FastAPI(
    title="Gemma-4 Chat API",
    version="0.1.0",
    description="Multimodal chat over Gemma-4-31B-it on Modal A100-80GB",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(JSONLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(health_routes.router)
app.include_router(chat_routes.router)
