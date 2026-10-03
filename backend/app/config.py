"""Centralised, typed configuration loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- App ----
    app_name: str = "Spiritual RAG"
    environment: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = ["*"]

    # ---- Security ----
    admin_api_key: SecretStr | None = None  # protects /admin endpoints
    public_api_key: SecretStr | None = None  # optional: protects chat/search
    rate_limit_per_minute: int = 30

    # ---- Gemini ----
    gemini_api_key: SecretStr
    generation_model: str = "gemini-3.8-flash"
    # Tried in order when the primary model is overloaded (503) or rate-limited (429).
    generation_fallback_models: list[str] = ["gemini-3.6-flash", "gemini-3.5-flash"]
    first_token_timeout_s: float = 5.0  # switch to the next model if it hasn't started answering
    rewrite_model: str = "gemini-3.5-flash-lite"
    ocr_model: str = "gemini-3.8-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    generation_thinking_level: Literal["minimal", "low", "medium", "high"] | None = "low"
    rewrite_thinking_level: Literal["minimal", "low", "medium", "high"] | None = "minimal"
    ocr_thinking_level: Literal["minimal", "low", "medium", "high"] | None = "low"
    generation_temperature: float = 0.3
    max_output_tokens: int = 2048
    llm_timeout_s: float = 60.0
    rewrite_timeout_s: float = 6.0

    # ---- Vector store ----
    qdrant_url: str | None = None  # e.g. http://localhost:6333 ; if unset, embedded local mode
    qdrant_api_key: SecretStr | None = None
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant"
    collection_name: str = "scriptures"

    # ---- Ingestion ----
    data_dir: Path = PROJECT_ROOT / "data"
    ocr_mode: Literal["auto", "always", "never"] = "auto"
    ocr_dpi: int = 150
    ocr_concurrency: int = 8
    embed_batch_size: int = 64
    embed_concurrency: int = 4
    chunk_size_chars: int = 1100
    chunk_overlap_chars: int = 220
    max_upload_mb: int = 200

    # ---- Retrieval ----
    top_k: int = 6
    prefetch_k: int = 40
    bm25_avg_doc_len: float = 140.0
    max_history_turns: int = 6

    # ---- Caching ----
    answer_cache_ttl_s: int = 3600
    answer_cache_size: int = 2048
    embedding_cache_size: int = 8192

    @property
    def ocr_cache_dir(self) -> Path:
        return self.data_dir / "cache" / "ocr"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "books.json"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "raw"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
