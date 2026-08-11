"""Config-driven product scraper.

Architecture note -- the single most important thing in this module:

    fetch_html()   does I/O, needs a browser, untestable in CI
    parse_product() is pure: HTML string in, dict out, fully unit-tested

The original scarpe.py fused these together, so no part of the scraping logic
could be tested without launching Chrome and hitting Amazon. Splitting them
means the parsing rules -- where all the actual bugs live -- are covered by
fast, offline tests.

Selectors live in targets.yaml, not in code. Retargeting to a different site is
a config change, not a rewrite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from bs4 import BeautifulSoup

from config.settings import settings
from pulseiq.ingestion.parsers import (
    clean_review_text,
    extract_discount_pct,
    extract_price,
    extract_rating,
)
from pulseiq.ingestion.retry import RateLimiter, TransientScrapeError, retry_on_transient

logger = logging.getLogger(__name__)

DEFAULT_TARGETS = Path(__file__).parent / "targets.yaml"


@dataclass(frozen=True)
class SiteConfig:
    """CSS selectors and metadata for one target site."""

    name: str
    base_url: str
    selectors: dict[str, str]
    ready_selector: str | None = None
    products: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> SiteConfig:
        missing = {"base_url", "selectors"} - raw.keys()
        if missing:
            raise ValueError(f"site '{name}' missing required keys: {sorted(missing)}")
        if "selling_price" not in raw["selectors"]:
            raise ValueError(f"site '{name}' must define a 'selling_price' selector")
        return cls(
            name=name,
            base_url=raw["base_url"],
            selectors=raw["selectors"],
            ready_selector=raw.get("ready_selector"),
            products=raw.get("products", []),
        )


def load_targets(path: Path | str = DEFAULT_TARGETS) -> dict[str, SiteConfig]:
    """Load and validate targets.yaml. Raises on malformed config at startup
    rather than halfway through a scraping run."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"targets file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sites = raw.get("sites", {})
    if not sites:
        raise ValueError(f"no sites defined in {path}")

    return {name: SiteConfig.from_dict(name, cfg) for name, cfg in sites.items()}


def _select_text(soup: BeautifulSoup, selector: str | None) -> str | None:
    """First match's text for a CSS selector, or None. Never raises."""
    if not selector:
        return None
    element = soup.select_one(selector)
    if element is None:
        return None
    text = element.get_text(strip=True)
    return text or None


def parse_product(
    html: str,
    config: SiteConfig,
    *,
    product_name: str | None = None,
    product_url: str | None = None,
    observed_on: date | None = None,
) -> dict[str, Any]:
    """Parse a product page into a raw dict. PURE -- no I/O, no browser.

    Output feeds straight into validation.validate_price_snapshots(), so keys
    match what that function expects. Missing fields become None rather than
    raising: a page with no discount badge is normal, not an error.
    """
    soup = BeautifulSoup(html, "html.parser")
    sel = config.selectors

    name = product_name or _select_text(soup, sel.get("product_name"))
    selling = extract_price(_select_text(soup, sel.get("selling_price")))
    original = extract_price(_select_text(soup, sel.get("original_price")))
    discount = extract_discount_pct(_select_text(soup, sel.get("discount")))
    rating = extract_rating(_select_text(soup, sel.get("rating")))

    reviews: list[str] = []
    if review_sel := sel.get("reviews"):
        for element in soup.select(review_sel):
            if text := clean_review_text(element.get_text(strip=True)):
                reviews.append(text)

    return {
        "product_name": name,
        "product_url": product_url,
        "source": config.name,
        "price": selling,
        "original_price": original,
        "discount": discount,
        "rating": rating,
        "reviews": reviews,
        "date": (observed_on or date.today()).isoformat(),
    }


@retry_on_transient(max_attempts=3, base_delay=2.0)
def fetch_html(driver: Any, url: str, config: SiteConfig, *, timeout: int = 15) -> str:
    """Load a URL and return its HTML. Does I/O -- not covered by unit tests.

    Raises TransientScrapeError (retryable) when the page loads but the expected
    content never appears, which is what a soft block or a slow render looks
    like.
    """
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.wait import WebDriverWait

    try:
        driver.get(url)
    except WebDriverException as exc:
        raise TransientScrapeError(f"navigation failed for {url}: {exc}") from exc

    if config.ready_selector:
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, config.ready_selector))
            )
        except TimeoutException as exc:
            raise TransientScrapeError(
                f"ready_selector {config.ready_selector!r} never appeared on {url}"
            ) from exc

    html = driver.page_source
    if not html or len(html) < 500:
        raise TransientScrapeError(f"suspiciously small page ({len(html or '')} bytes): {url}")
    return html


def scrape_site(driver: Any, config: SiteConfig) -> list[dict[str, Any]]:
    """Scrape every product listed for a site, rate-limited between requests.

    Failures are logged and skipped -- one dead product URL should not abort a
    run that has already collected 19 others.
    """
    limiter = RateLimiter(settings.scrape_delay_seconds)
    results: list[dict[str, Any]] = []

    for entry in config.products:
        url = entry.get("url")
        if not url:
            logger.warning("skipping entry with no url: %r", entry)
            continue

        limiter.wait()
        try:
            html = fetch_html(driver, url, config)
            record = parse_product(html, config, product_name=entry.get("name"), product_url=url)
            results.append(record)
            logger.info("scraped %s", record.get("product_name") or url)
        except Exception:  # noqa: BLE001 - one bad URL must not kill the run
            logger.exception("failed to scrape %s", url)

    logger.info("scraped %d/%d products from %s", len(results), len(config.products), config.name)
    return results
