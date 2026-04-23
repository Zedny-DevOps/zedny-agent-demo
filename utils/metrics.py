from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Counters:
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
    cache_hits: int = 0
    request_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=10_000))


class MetricsTracker:
    """In-memory aggregate counters + rolling RPM. Process-local; not durable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c = _Counters()

    def record(
        self,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cache_hit: bool = False,
    ) -> None:
        with self._lock:
            self._c.total_requests += 1
            self._c.total_input_tokens += input_tokens
            self._c.total_output_tokens += output_tokens
            self._c.total_cost_usd += cost_usd
            self._c.total_latency_ms += latency_ms
            if cache_hit:
                self._c.cache_hits += 1
            self._c.request_timestamps.append(time.time())

    def snapshot(self) -> dict:
        with self._lock:
            now = time.time()
            rpm = sum(1 for t in self._c.request_timestamps if now - t <= 60.0)
            n = self._c.total_requests
            avg_latency = (self._c.total_latency_ms / n) if n else 0.0
            return {
                "total_requests": n,
                "rpm_rolling_60s": rpm,
                "avg_latency_ms": round(avg_latency, 2),
                "total_input_tokens": self._c.total_input_tokens,
                "total_output_tokens": self._c.total_output_tokens,
                "total_cost_usd": round(self._c.total_cost_usd, 6),
                "cache_hits": self._c.cache_hits,
            }


tracker = MetricsTracker()
