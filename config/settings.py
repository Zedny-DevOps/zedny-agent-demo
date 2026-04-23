from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Pricing(BaseModel):
    input_per_1k: float = 0.30
    output_per_1k: float = 0.60


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    modal_app_name: str = "gemma4-chat"
    modal_class_name: str = "Gemma4Chat"

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_log_level: str = "INFO"

    rate_limit_per_minute: int = 60
    cache_max_entries: int = 256

    pricing_path: str = "config/pricing.yaml"
    backend_url: str = "http://localhost:8000"

    pricing: Pricing = Field(default_factory=Pricing)

    def load_pricing(self) -> None:
        p = Path(self.pricing_path)
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            self.pricing = Pricing(**data)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.load_pricing()
    return s


settings = get_settings()
