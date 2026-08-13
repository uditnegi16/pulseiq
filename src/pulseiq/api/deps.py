"""FastAPI dependencies: database sessions, model loading, cache.

The sentiment model is loaded once per process and reused. Loading DistilBERT
takes several seconds -- acceptable at startup, unacceptable per request. It is
loaded lazily rather than at import so the API starts even when the adapter is
absent, and only the /sentiment endpoint fails.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from config.settings import settings
from pulseiq.llm.cache import Cache, get_cache

logger = logging.getLogger(__name__)

_sentiment_model: tuple[Any, Any] | None = None
_sentiment_error: str | None = None


def get_db() -> Iterator[Any]:
    """Yield a database session, closed afterwards."""
    from pulseiq.storage.relational import get_engine, init_db, session_scope

    engine = get_engine()
    init_db(engine)
    with session_scope(engine) as session:
        yield session


def get_cache_dep() -> Cache:
    return get_cache()


def get_sentiment_model() -> tuple[Any, Any]:
    """Load the fine-tuned model once per process.

    Raises RuntimeError with the original reason if loading failed. The failure
    is cached too: retrying a missing-adapter load on every request would add
    seconds of latency to a request that cannot succeed.
    """
    global _sentiment_model, _sentiment_error

    if _sentiment_model is not None:
        return _sentiment_model
    if _sentiment_error is not None:
        raise RuntimeError(_sentiment_error)

    try:
        from pulseiq.training.sentiment.predict import load_model

        model, tokenizer, _ = load_model()
        _sentiment_model = (model, tokenizer)
        logger.info("sentiment model loaded")
        return _sentiment_model
    except Exception as exc:  # noqa: BLE001
        _sentiment_error = str(exc)
        logger.warning("sentiment model unavailable: %s", exc)
        raise RuntimeError(_sentiment_error) from exc


def reset_model_cache() -> None:
    """Drop cached model state. Used by test fixtures."""
    global _sentiment_model, _sentiment_error
    _sentiment_model = None
    _sentiment_error = None


def cache_ttl() -> int:
    return settings.cache_ttl_seconds
