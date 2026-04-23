from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class GenerationParams(BaseModel):
    max_new_tokens: int = Field(1024, ge=1, le=4096)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(64, ge=0, le=200)
    enable_thinking: bool = False


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    params: GenerationParams = Field(default_factory=GenerationParams)


class Metrics(BaseModel):
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cache_hit: bool = False


class ChatResponse(BaseModel):
    message: Message
    metrics: Metrics


class StreamEvent(BaseModel):
    type: Literal["token", "done", "error"]
    text: str | None = None
    metrics: Metrics | None = None
    error: str | None = None
