from __future__ import annotations

from config.settings import settings


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    p = settings.pricing
    cost = (input_tokens / 1000.0) * p.input_per_1k + (output_tokens / 1000.0) * p.output_per_1k
    return round(cost, 6)
