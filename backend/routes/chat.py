"""Chat endpoints — text and multimodal, blocking and streaming."""
from __future__ import annotations

import base64
import json
import time
from typing import AsyncIterator

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from config.settings import settings
from schemas.chat import ChatRequest, ChatResponse, GenerationParams, Message, Metrics, StreamEvent
from services import modal_client
from utils.cache import cache
from utils.cost import compute_cost
from utils.logger import log
from utils.metrics import tracker

from ..deps import limiter

router = APIRouter(prefix="", tags=["chat"])
_RATE = f"{settings.rate_limit_per_minute}/minute"


# ── helpers ──────────────────────────────────────────────────────────────────
def _gen_kwargs(params: GenerationParams) -> dict:
    return {
        "max_new_tokens": params.max_new_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "enable_thinking": params.enable_thinking,
    }


def _messages_dump(messages: list[Message]) -> list[dict]:
    return [m.model_dump() for m in messages]


def _parse_messages_form(raw: str) -> list[Message]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid 'messages' JSON: {exc}")
    if not isinstance(data, list) or not data:
        raise HTTPException(status_code=422, detail="'messages' must be a non-empty list")
    return [Message(**m) for m in data]


def _parse_params_form(raw: str | None) -> GenerationParams:
    if not raw:
        return GenerationParams()
    try:
        return GenerationParams(**json.loads(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid 'params' JSON: {exc}")


def _sse(event: StreamEvent) -> bytes:
    return f"data: {event.model_dump_json(exclude_none=True)}\n\n".encode("utf-8")


async def _stream_to_sse(
    upstream: AsyncIterator[dict],
    t0: float,
) -> AsyncIterator[bytes]:
    """Translate Modal stream events into SSE bytes; emit final 'done' with metrics."""
    try:
        async for evt in upstream:
            etype = evt.get("type")
            if etype == "token":
                yield _sse(StreamEvent(type="token", text=evt.get("text", "")))
            elif etype == "done":
                latency_ms = int((time.perf_counter() - t0) * 1000)
                in_tok = int(evt.get("input_tokens", 0))
                out_tok = int(evt.get("output_tokens", 0))
                cost = compute_cost(in_tok, out_tok)
                metrics = Metrics(
                    latency_ms=latency_ms,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=cost,
                )
                tracker.record(latency_ms, in_tok, out_tok, cost)
                log.info(
                    "chat_stream_done",
                    latency_ms=latency_ms,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=cost,
                )
                yield _sse(StreamEvent(type="done", metrics=metrics))
    except Exception as exc:
        log.error("chat_stream_error", error=str(exc))
        yield _sse(StreamEvent(type="error", error=str(exc)))


# ── /chat ────────────────────────────────────────────────────────────────────
@router.post("/chat", response_model=ChatResponse)
@limiter.limit(_RATE)
async def chat(
    request: Request,
    body: ChatRequest,
    enable_cache: bool = Query(False, description="Use LRU cache on identical (messages, params)"),
) -> ChatResponse:
    t0 = time.perf_counter()
    messages = _messages_dump(body.messages)
    params = body.params.model_dump()

    if enable_cache:
        cached = cache.get(messages, params)
        if cached is not None:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            metrics = Metrics(
                latency_ms=latency_ms,
                input_tokens=cached["input_tokens"],
                output_tokens=cached["output_tokens"],
                cost_usd=0.0,
                cache_hit=True,
            )
            tracker.record(latency_ms, 0, 0, 0.0, cache_hit=True)
            return ChatResponse(
                message=Message(role="assistant", content=cached["text"]),
                metrics=metrics,
            )

    try:
        result = await modal_client.chat(messages=messages, **_gen_kwargs(body.params))
    except Exception as exc:
        log.error("modal_call_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Inference backend error: {exc}") from exc

    latency_ms = int((time.perf_counter() - t0) * 1000)
    in_tok = int(result["input_tokens"])
    out_tok = int(result["output_tokens"])
    cost = compute_cost(in_tok, out_tok)
    metrics = Metrics(latency_ms=latency_ms, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
    tracker.record(latency_ms, in_tok, out_tok, cost)

    if enable_cache:
        cache.set(messages, params, result)

    log.info(
        "chat_done",
        latency_ms=latency_ms,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
    )
    return ChatResponse(
        message=Message(role="assistant", content=result["text"]),
        metrics=metrics,
    )


# ── /chat/stream ─────────────────────────────────────────────────────────────
@router.post("/chat/stream")
@limiter.limit(_RATE)
async def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    t0 = time.perf_counter()
    messages = _messages_dump(body.messages)
    upstream = modal_client.chat_stream(messages=messages, **_gen_kwargs(body.params))
    return StreamingResponse(_stream_to_sse(upstream, t0), media_type="text/event-stream")


# ── /chat-with-image ─────────────────────────────────────────────────────────
@router.post("/chat-with-image", response_model=ChatResponse)
@limiter.limit(_RATE)
async def chat_with_image(
    request: Request,
    messages: str = Form(..., description="JSON array of {role, content}"),
    params: str | None = Form(None, description="Optional JSON GenerationParams"),
    image: UploadFile = File(...),
) -> ChatResponse:
    t0 = time.perf_counter()
    parsed_messages = _parse_messages_form(messages)
    parsed_params = _parse_params_form(params)

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Empty image upload")
    img_b64 = base64.b64encode(raw).decode("ascii")

    try:
        result = await modal_client.chat_multimodal(
            messages=_messages_dump(parsed_messages),
            images_b64=[img_b64],
            **_gen_kwargs(parsed_params),
        )
    except Exception as exc:
        log.error("modal_multimodal_failed", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Inference backend error: {exc}") from exc

    latency_ms = int((time.perf_counter() - t0) * 1000)
    in_tok = int(result["input_tokens"])
    out_tok = int(result["output_tokens"])
    cost = compute_cost(in_tok, out_tok)
    metrics = Metrics(latency_ms=latency_ms, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost)
    tracker.record(latency_ms, in_tok, out_tok, cost)
    log.info(
        "chat_with_image_done",
        latency_ms=latency_ms,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=cost,
    )
    return ChatResponse(
        message=Message(role="assistant", content=result["text"]),
        metrics=metrics,
    )


# ── /chat-with-image/stream ──────────────────────────────────────────────────
@router.post("/chat-with-image/stream")
@limiter.limit(_RATE)
async def chat_with_image_stream(
    request: Request,
    messages: str = Form(...),
    params: str | None = Form(None),
    image: UploadFile = File(...),
) -> StreamingResponse:
    t0 = time.perf_counter()
    parsed_messages = _parse_messages_form(messages)
    parsed_params = _parse_params_form(params)

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Empty image upload")
    img_b64 = base64.b64encode(raw).decode("ascii")

    upstream = modal_client.chat_multimodal_stream(
        messages=_messages_dump(parsed_messages),
        images_b64=[img_b64],
        **_gen_kwargs(parsed_params),
    )
    return StreamingResponse(_stream_to_sse(upstream, t0), media_type="text/event-stream")
