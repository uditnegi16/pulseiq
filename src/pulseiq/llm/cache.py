"""Redis-backed cache with an in-memory fallback.

The fallback matters: without it every developer needs a Redis instance before
the API will start, and the project's "runs with zero credentials" property
disappears for the sake of a cache.

`get_cache()` returns Redis when REDIS_URL is set and a bounded in-process dict
otherwise. Same interface either way, no branching at the call site.

The in-memory version is explicitly NOT production-suitable and says so: it is
per-process, so a multi-worker deployment gets one cache per worker, and it dies
with the process. Fine for local development, wrong for anything else.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Protocol

from config.settings import settings

logger = logging.getLogger(__name__)


class Cache(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    def delete(self, key: str) -> None: ...
    @property
    def backend(self) -> str: ...


def make_key(namespace: str, *parts: Any) -> str:
    """Build a deterministic cache key.

    Long or unicode inputs (review text, accented product names) are hashed
    rather than embedded: Redis keys have a length limit, and an un-hashed key
    would leak user content to anything that lists keys.
    """
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"pulseiq:{namespace}:{digest}"


class InMemoryCache:
    """Bounded dict cache with TTL. Development fallback only.

    Bounded because an unbounded process-local cache is a memory leak with a
    friendly name. Eviction is oldest-first -- not LRU, but adequate for a
    fallback nobody should run in production.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._store: dict[str, tuple[Any, float | None]] = {}
        self.max_entries = max_entries

    @property
    def backend(self) -> str:
        return "memory"

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        if len(self._store) >= self.max_entries:
            del self._store[next(iter(self._store))]
        self._store[key] = (value, time.time() + ttl if ttl else None)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


class RedisCache:
    """Redis/Upstash cache. Values JSON-encoded.

    Every operation swallows connection errors and logs them. A cache is an
    optimisation: if it is unreachable the request should still be served, just
    slower. Propagating a Redis timeout as a 500 would make the cache a
    liability rather than a benefit.
    """

    def __init__(self, url: str, *, socket_timeout: float = 2.0) -> None:
        import redis

        self._client = redis.from_url(
            url,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            decode_responses=True,
        )

    @property
    def backend(self) -> str:
        return "redis"

    def get(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:  # noqa: BLE001 - a cache miss is always survivable
            logger.warning("cache get failed (%s); treating as a miss", exc)
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        try:
            payload = json.dumps(value)
            if ttl:
                self._client.setex(key, ttl, payload)
            else:
                self._client.set(key, payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache set failed (%s); continuing uncached", exc)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache delete failed (%s)", exc)

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False


_cache: Cache | None = None


def get_cache(*, force_memory: bool = False) -> Cache:
    """Return the process-wide cache. Redis when configured, memory otherwise."""
    global _cache
    if _cache is not None and not force_memory:
        return _cache

    if force_memory or settings.redis_url is None:
        if not force_memory:
            logger.info("REDIS_URL not set -- using in-memory cache (development only)")
        _cache = InMemoryCache()
        return _cache

    try:
        cache = RedisCache(settings.require("redis_url"))
        if cache.ping():
            logger.info("connected to Redis")
            _cache = cache
            return _cache
        logger.warning("Redis unreachable -- falling back to in-memory cache")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis init failed (%s) -- falling back to in-memory cache", exc)

    _cache = InMemoryCache()
    return _cache


def reset_cache() -> None:
    """Drop the cached instance. Used by test fixtures."""
    global _cache
    _cache = None
