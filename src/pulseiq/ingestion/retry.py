"""Retry, backoff, and rate limiting for network-facing work.

The original scraper retried with a fixed `time.sleep(5)` inside a while loop,
retried on *every* exception including programming errors, and had no delay
between products at all. This module fixes all three.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ScrapeError(Exception):
    """Raised when a page cannot be retrieved or parsed after all retries."""


class TransientScrapeError(ScrapeError):
    """A failure worth retrying: timeout, stale element, empty page.

    Deliberately distinct from ScrapeError so that permanent failures (404,
    selector genuinely absent) fail fast instead of burning three retries.
    """


def backoff_delay(attempt: int, base: float, *, jitter: bool = True, cap: float = 60.0) -> float:
    """Exponential backoff with full jitter.

    attempt is 0-indexed: 0 -> base, 1 -> 2*base, 2 -> 4*base, capped.

    Jitter matters when scraping several products: without it, every retry
    fires simultaneously and looks exactly like an attack.
    """
    delay = min(base * (2**attempt), cap)
    if jitter:
        delay = random.uniform(0, delay)
    return delay


def retry_on_transient(
    max_attempts: int = 3,
    base_delay: float = 2.0,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a function on TransientScrapeError only.

    `sleep` is injectable so tests run instantly instead of actually waiting.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except TransientScrapeError as exc:
                    last_error = exc
                    if attempt == max_attempts - 1:
                        break
                    delay = backoff_delay(attempt, base_delay)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s -- retrying in %.1fs",
                        func.__name__,
                        attempt + 1,
                        max_attempts,
                        exc,
                        delay,
                    )
                    sleep(delay)
            raise ScrapeError(
                f"{func.__name__} failed after {max_attempts} attempts: {last_error}"
            ) from last_error

        return wrapper

    return decorator


class RateLimiter:
    """Enforce a minimum interval between calls.

    Politeness, and self-preservation: hammering a site is the fastest way to
    get an IP banned mid-project. Documented in the README as a deliberate
    choice rather than an oversight.
    """

    def __init__(
        self,
        min_interval: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last_call: float | None = None

    def wait(self) -> float:
        """Block until the minimum interval has elapsed. Returns seconds waited."""
        now = self._clock()
        if self._last_call is None:
            self._last_call = now
            return 0.0

        elapsed = now - self._last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            self._sleep(remaining)
            self._last_call = self._clock()
            return remaining

        self._last_call = now
        return 0.0
