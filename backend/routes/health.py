from __future__ import annotations

from fastapi import APIRouter

from utils.metrics import tracker

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/metrics")
async def metrics() -> dict:
    return tracker.snapshot()
