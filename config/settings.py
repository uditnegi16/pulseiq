"""Typed configuration loaded from environment / .env.

Single source of truth for every credential and tunable in the project.
Nothing else in the codebase reads os.environ directly -- import `settings`
from here instead. That is what makes the "no hardcoded secrets" claim
enforceable rather than aspirational.

Usage:
    from config.settings import settings
    client = Groq(api_key=settings.groq_api_key)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All configuration, validated at import time."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- app ---------------------------------------------------------------
    app_env: str = "local"
    log_level: str = "INFO"

    # --- LLM providers -----------------------------------------------------
    # SecretStr means these never appear in logs, tracebacks, or repr().
    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.1-8b-instant"
    nvidia_nim_api_key: SecretStr | None = None
    nvidia_nim_model: str | None = None

    # --- notifications -----------------------------------------------------
    slack_webhook_url: SecretStr | None = None

    # --- mongodb (raw scraped documents) -----------------------------------
    mongodb_uri: SecretStr | None = None
    mongodb_db: str = "pulseiq"
    mongodb_raw_collection: str = "raw_scrapes"

    # --- relational store (cleaned data) -----------------------------------
    database_url: str = "sqlite:///./pulseiq.db"

    # --- cache -------------------------------------------------------------
    redis_url: SecretStr | None = None
    cache_ttl_seconds: int = 3600

    # --- mlflow ------------------------------------------------------------
    # MLflow 3.15 put the filesystem store into maintenance mode -- "file:./mlruns"
    # now raises unless MLFLOW_ALLOW_FILE_STORE=true. SQLite is the recommended
    # local backend and swaps to Postgres/S3 in Phase 7 by changing this string.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "pulseiq"

    # --- scraper -----------------------------------------------------------
    scrape_headless: bool = True
    scrape_max_retries: int = Field(default=3, ge=1, le=10)
    scrape_backoff_seconds: float = Field(default=2.0, ge=0)
    scrape_delay_seconds: float = Field(default=5.0, ge=0)
    scrape_user_agent: str = "Mozilla/5.0 (compatible; PulseIQResearchBot/0.1)"

    # --- api / frontend ----------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_base_url: str = "http://localhost:8000"

    # --- paths -------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"

    @field_validator(
        "groq_api_key",
        "nvidia_nim_api_key",
        "slack_webhook_url",
        "mongodb_uri",
        "redis_url",
        mode="before",
    )
    @classmethod
    def blank_is_unset(cls, v: object) -> object:
        """Treat an empty or whitespace-only value as absent.

        `.env.example` ships every optional key blank so the file documents what
        exists. Copied to `.env`, `REDIS_URL=` is present-but-empty, which
        pydantic reads as the string "" -- truthy enough to look configured, so
        the client is handed "" and fails with a confusing scheme error instead
        of cleanly reporting "not configured".

        NOTE: this validator was accidentally defined twice at one point. Python
        keeps only the last definition, which silently disabled it for every
        field. One definition only.
        """
        if isinstance(v, str) and not v.strip():
            return None
        return v

    def require(self, field: str) -> str:
        """Fetch a secret that must be present, with a useful error if it isn't.

        Fails loudly at the point of use rather than sending `None` to an API
        and debugging a 401 later.
        """
        value = getattr(self, field, None)
        if value is None:
            raise RuntimeError(
                f"Missing required setting '{field.upper()}'. "
                f"Add it to your .env file (see .env.example)."
            )
        return value.get_secret_value() if isinstance(value, SecretStr) else str(value)


@lru_cache
def get_settings() -> Settings:
    """Cached accessor -- .env is read once per process."""
    return Settings()


settings = get_settings()
