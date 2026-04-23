from __future__ import annotations

import logging
import sys
import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from config.settings import settings


def configure_logging() -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.backend_log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.backend_log_level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()


class JSONLoggingMiddleware(BaseHTTPMiddleware):
    """Emit one JSON log line per request with method, path, status, latency."""

    async def dispatch(self, request: Request, call_next) -> Response:
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            log.error(
                "request",
                method=request.method,
                path=request.url.path,
                status=500,
                latency_ms=latency_ms,
                error=str(exc),
            )
            raise

        latency_ms = int((time.perf_counter() - t0) * 1000)
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=latency_ms,
            client=request.client.host if request.client else None,
        )
        return response
