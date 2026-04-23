"""Async wrapper around the deployed Modal class.

Resolves the class once at import time (module-level singleton) and exposes
async functions that the FastAPI handlers can `await` without blocking the
event loop.

The Modal app must be deployed first:
    modal deploy services/modal_app.py
"""
from __future__ import annotations

from typing import AsyncIterator

import modal

from config.settings import settings

# Resolve the deployed class once. Cheap; just creates a remote handle.
_Cls = modal.Cls.from_name(settings.modal_app_name, settings.modal_class_name)
_instance = _Cls()


async def chat(messages: list[dict], **gen_kwargs) -> dict:
    """Non-streaming text chat. Returns {text, input_tokens, output_tokens}."""
    return await _instance.chat.remote.aio(messages=messages, **gen_kwargs)


async def chat_stream(messages: list[dict], **gen_kwargs) -> AsyncIterator[dict]:
    """Streaming text chat. Yields {type:'token',text} ... then {type:'done',...}."""
    async for evt in _instance.chat_stream.remote_gen.aio(messages=messages, **gen_kwargs):
        yield evt


async def chat_multimodal(
    messages: list[dict],
    images_b64: list[str],
    **gen_kwargs,
) -> dict:
    return await _instance.chat_multimodal.remote.aio(
        messages=messages,
        images_b64=images_b64,
        **gen_kwargs,
    )


async def chat_multimodal_stream(
    messages: list[dict],
    images_b64: list[str],
    **gen_kwargs,
) -> AsyncIterator[dict]:
    async for evt in _instance.chat_multimodal_stream.remote_gen.aio(
        messages=messages,
        images_b64=images_b64,
        **gen_kwargs,
    ):
        yield evt


async def warmup() -> None:
    """Tiny call to wake up the container before the first real user request."""
    try:
        await _instance.chat.remote.aio(
            messages=[{"role": "user", "content": "ok"}],
            max_new_tokens=1,
        )
    except Exception:
        # Warmup failures shouldn't crash boot — they'll surface on first real call.
        pass
