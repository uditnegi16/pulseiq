"""Unit tests for the scraper: config loading, HTML parsing, retry, rate limit.

Not one of these launches a browser or touches the network. The browser-facing
half (fetch_html, scrape_site) is covered by integration tests instead.
"""

from datetime import date
from pathlib import Path

import pytest

from pulseiq.ingestion.retry import (
    RateLimiter,
    ScrapeError,
    TransientScrapeError,
    backoff_delay,
    retry_on_transient,
)
from pulseiq.ingestion.scraper import SiteConfig, load_targets, parse_product

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def config() -> SiteConfig:
    return SiteConfig(
        name="test_site",
        base_url="https://example.test",
        selectors={
            "product_name": "div.product_main > h1",
            "selling_price": "div.product_main p.price_color",
            "original_price": "div.product_main p.mrp",
            "discount": "div.product_main span.savings",
            "rating": "div.product_main p.star-rating",
            "reviews": "#product_description ~ p",
        },
    )


@pytest.fixture
def sample_html() -> str:
    return (FIXTURES / "sample_product.html").read_text(encoding="utf-8")


@pytest.fixture
def empty_html() -> str:
    return (FIXTURES / "empty_product.html").read_text(encoding="utf-8")


class TestSiteConfig:
    def test_requires_selling_price_selector(self):
        with pytest.raises(ValueError, match="selling_price"):
            SiteConfig.from_dict("bad", {"base_url": "x", "selectors": {"rating": "y"}})

    def test_requires_base_url_and_selectors(self):
        with pytest.raises(ValueError, match="missing required keys"):
            SiteConfig.from_dict("bad", {"selectors": {"selling_price": "x"}})

    def test_builds_from_valid_dict(self):
        cfg = SiteConfig.from_dict(
            "good",
            {"base_url": "https://x.test", "selectors": {"selling_price": ".p"}},
        )
        assert cfg.name == "good"
        assert cfg.products == []


class TestLoadTargets:
    def test_shipped_targets_file_is_valid(self):
        """Guards against a YAML typo shipping unnoticed -- this file is edited
        by hand every time a site is added."""
        sites = load_targets()
        assert "books_toscrape" in sites
        assert sites["books_toscrape"].products

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_targets("does/not/exist.yaml")

    def test_empty_config_raises(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("sites: {}", encoding="utf-8")
        with pytest.raises(ValueError, match="no sites defined"):
            load_targets(path)


class TestParseProduct:
    def test_extracts_all_fields(self, sample_html, config):
        result = parse_product(sample_html, config)
        assert result["product_name"] == "Apple AirPods Pro (2nd Generation)"
        assert result["price"] == pytest.approx(51.77)
        assert result["original_price"] == pytest.approx(72.50)
        assert result["discount"] == pytest.approx(29.0)
        assert result["rating"] == pytest.approx(4.3)
        assert result["source"] == "test_site"

    def test_collects_reviews_and_drops_empties(self, sample_html, config):
        result = parse_product(sample_html, config)
        assert len(result["reviews"]) == 2
        assert result["reviews"][0].startswith("Amazing sound quality")

    def test_normalises_review_whitespace(self, sample_html, config):
        result = parse_product(sample_html, config)
        assert "    " not in result["reviews"][1]

    def test_missing_fields_become_none_not_errors(self, empty_html, config):
        """A page missing a discount badge is normal, not a failure."""
        result = parse_product(empty_html, config)
        assert result["product_name"] == "Ghost Item"
        assert result["price"] is None
        assert result["discount"] is None
        assert result["reviews"] == []

    def test_explicit_name_overrides_scraped(self, sample_html, config):
        result = parse_product(sample_html, config, product_name="Override")
        assert result["product_name"] == "Override"

    def test_defaults_to_today(self, sample_html, config):
        assert parse_product(sample_html, config)["date"] == date.today().isoformat()

    def test_accepts_explicit_date(self, sample_html, config):
        result = parse_product(sample_html, config, observed_on=date(2026, 1, 15))
        assert result["date"] == "2026-01-15"

    def test_output_feeds_validation_layer(self, sample_html, config):
        """The contract that matters: scraper output must validate cleanly."""
        from pulseiq.ingestion.validation import validate_price_snapshots

        raw = parse_product(sample_html, config)
        records, report = validate_price_snapshots([raw])
        assert report.valid == 1
        assert records[0].selling_price == pytest.approx(51.77)

    def test_garbage_html_does_not_raise(self, config):
        result = parse_product("<html><body>nope</body></html>", config)
        assert result["price"] is None


class TestBackoffDelay:
    def test_grows_exponentially_without_jitter(self):
        assert backoff_delay(0, 2.0, jitter=False) == 2.0
        assert backoff_delay(1, 2.0, jitter=False) == 4.0
        assert backoff_delay(2, 2.0, jitter=False) == 8.0

    def test_respects_cap(self):
        assert backoff_delay(10, 2.0, jitter=False, cap=30.0) == 30.0

    def test_jitter_stays_within_bounds(self):
        for _ in range(50):
            assert 0 <= backoff_delay(2, 2.0, jitter=True) <= 8.0


class TestRetryOnTransient:
    def test_returns_on_first_success(self):
        calls = []

        @retry_on_transient(max_attempts=3, sleep=lambda _: None)
        def works():
            calls.append(1)
            return "ok"

        assert works() == "ok"
        assert len(calls) == 1

    def test_retries_then_succeeds(self):
        calls = []

        @retry_on_transient(max_attempts=3, sleep=lambda _: None)
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise TransientScrapeError("not yet")
            return "ok"

        assert flaky() == "ok"
        assert len(calls) == 3

    def test_raises_after_exhausting_attempts(self):
        calls = []

        @retry_on_transient(max_attempts=3, sleep=lambda _: None)
        def always_fails():
            calls.append(1)
            raise TransientScrapeError("down")

        with pytest.raises(ScrapeError, match="after 3 attempts"):
            always_fails()
        assert len(calls) == 3

    def test_does_not_retry_programming_errors(self):
        """A ValueError is a bug, not a flaky network. Retrying it three times
        just delays the traceback."""
        calls = []

        @retry_on_transient(max_attempts=3, sleep=lambda _: None)
        def bug():
            calls.append(1)
            raise ValueError("typo in selector")

        with pytest.raises(ValueError):
            bug()
        assert len(calls) == 1

    def test_preserves_function_metadata(self):
        @retry_on_transient(sleep=lambda _: None)
        def documented():
            """Docstring survives."""

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "Docstring survives."


class TestRateLimiter:
    def test_first_call_does_not_wait(self):
        limiter = RateLimiter(5.0, clock=lambda: 100.0, sleep=lambda _: None)
        assert limiter.wait() == 0.0

    def test_waits_remaining_interval(self):
        times = iter([100.0, 102.0, 105.0])
        slept: list[float] = []
        limiter = RateLimiter(5.0, clock=lambda: next(times), sleep=slept.append)
        limiter.wait()
        waited = limiter.wait()
        assert waited == pytest.approx(3.0)
        assert slept == [pytest.approx(3.0)]

    def test_no_wait_when_interval_already_elapsed(self):
        times = iter([100.0, 110.0])
        slept: list[float] = []
        limiter = RateLimiter(5.0, clock=lambda: next(times), sleep=slept.append)
        limiter.wait()
        assert limiter.wait() == 0.0
        assert slept == []
