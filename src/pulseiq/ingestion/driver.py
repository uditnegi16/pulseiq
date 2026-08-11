"""Selenium WebDriver factory.

Isolated from scraping logic so the browser can be swapped, mocked, or removed
entirely without touching parsing code. Nothing here parses HTML.

Ported and hardened from the original scarpe.py:53 `get_driver`, which leaked a
driver on every call (no quit, no context manager) and hardcoded its options.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webdriver import WebDriver

from config.settings import settings

logger = logging.getLogger(__name__)


def build_options(
    *,
    headless: bool | None = None,
    user_agent: str | None = None,
    window_size: tuple[int, int] = (1920, 1080),
) -> Options:
    """Assemble Chrome options. Pure -- no browser is launched here, so this is
    testable without Chrome installed."""
    headless = settings.scrape_headless if headless is None else headless
    user_agent = user_agent or settings.scrape_user_agent

    options = Options()
    if headless:
        options.add_argument("--headless=new")

    # Required in containers/CI; harmless locally.
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    options.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
    options.add_argument(f"--user-agent={user_agent}")

    # Images are the bulk of page weight and we never read them.
    options.add_argument("--blink-settings=imagesEnabled=false")

    # Reduces the most obvious automation fingerprint.
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    return options


@contextmanager
def get_driver(
    *,
    headless: bool | None = None,
    page_load_timeout: int = 30,
) -> Iterator[WebDriver]:
    """Yield a WebDriver, guaranteed to be quit afterwards.

    Always use as a context manager:

        with get_driver() as driver:
            driver.get(url)

    The original code returned a bare driver, so every exception leaked a Chrome
    process. A long scraping run would exhaust system memory.
    """
    options = build_options(headless=headless)
    driver: WebDriver | None = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(page_load_timeout)
        logger.debug("chrome driver started (headless=%s)", headless)
        yield driver
    finally:
        if driver is not None:
            try:
                driver.quit()
                logger.debug("chrome driver closed")
            except Exception:  # noqa: BLE001 - never mask the original error
                logger.warning("driver.quit() failed", exc_info=True)
