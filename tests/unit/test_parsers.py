"""Unit tests for pulseiq.ingestion.parsers.

No browser, no network, no database -- these run in milliseconds and are the
CI gate that stays green from Phase 1 onward.
"""

import pytest

from pulseiq.ingestion.parsers import (
    clean_review_text,
    compute_discount_pct,
    extract_discount_pct,
    extract_price,
    extract_rating,
)


class TestExtractPrice:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("₹24,999.00", 24999.0),
            ("₹24,999", 24999.0),
            ("Rs. 1,299.50", 1299.5),
            ("$1,049.99", 1049.99),
            ("1299", 1299.0),
            ("  ₹ 999 ", 999.0),
            ("₹1,29,999", 129999.0),  # Indian lakh grouping
            ("INR 45999.00", 45999.0),
        ],
    )
    def test_parses_real_price_formats(self, raw, expected):
        assert extract_price(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "   ", None, "out of stock", "N/A", "--"])
    def test_returns_none_for_unparseable(self, raw):
        assert extract_price(raw) is None

    def test_decimal_point_is_not_swallowed(self):
        """Regression: the original extract_price stripped non-digits, turning
        1,299.50 into 129950 -- a 100x error."""
        assert extract_price("₹1,299.50") == 1299.5

    def test_none_is_distinct_from_zero(self):
        """Regression: the original returned 0 on failure, making a free item
        indistinguishable from a parse error."""
        assert extract_price("garbage") is None
        assert extract_price("₹0") == 0.0


class TestExtractDiscountPct:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("-15%", 15.0),
            ("15% off", 15.0),
            ("(20% off)", 20.0),
            ("Save 30%", 30.0),
            ("-7.5%", 7.5),
            ("0%", 0.0),
            ("100%", 100.0),
        ],
    )
    def test_parses_discount_badges(self, raw, expected):
        assert extract_discount_pct(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", None, "no deal", "150%", "-5", "abc%"])
    def test_rejects_missing_or_impossible(self, raw):
        assert extract_discount_pct(raw) is None


class TestExtractRating:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("4.3 out of 5 stars", 4.3),
            ("4.3 OUT OF 5 STARS", 4.3),
            ("5.0 out of 5 stars", 5.0),
            ("4.3", 4.3),
            ("5", 5.0),
            ("  4.0  ", 4.0),
            ("8 out of 10 stars", 4.0),  # rescaled to /5
        ],
    )
    def test_parses_ratings(self, raw, expected):
        assert extract_rating(raw) == pytest.approx(expected)

    def test_rejects_malformed_rating_from_original_csv(self):
        """reviews.csv row 2 contains '4.3.0'. Truncating it to 4.3 would be a
        silent data-quality failure; we reject it instead."""
        assert extract_rating("4.3.0") is None

    @pytest.mark.parametrize("raw", ["", None, "no rating", "6.0", "-1"])
    def test_rejects_out_of_range_and_junk(self, raw):
        assert extract_rating(raw) is None


class TestComputeDiscountPct:
    def test_typical_discount(self):
        assert compute_discount_pct(8000.0, 10000.0) == 20.0

    def test_no_discount(self):
        assert compute_discount_pct(10000.0, 10000.0) == 0.0

    def test_price_above_original_clamps_to_zero(self):
        assert compute_discount_pct(12000.0, 10000.0) == 0.0

    def test_rounds_to_two_places(self):
        assert compute_discount_pct(6666.0, 10000.0) == 33.34

    @pytest.mark.parametrize(
        "selling,original",
        [(None, 10000.0), (8000.0, None), (None, None), (8000.0, 0.0), (-1.0, 100.0)],
    )
    def test_returns_none_on_bad_input(self, selling, original):
        assert compute_discount_pct(selling, original) is None

    def test_no_division_by_zero(self):
        """Guards against inf/NaN entering the dataset."""
        assert compute_discount_pct(100.0, 0.0) is None


class TestCleanReviewText:
    def test_collapses_whitespace(self):
        assert clean_review_text("  Great\n\n sound   quality  ") == "Great sound quality"

    @pytest.mark.parametrize("raw", ["", "   ", "\n\t", None])
    def test_empty_becomes_none(self, raw):
        assert clean_review_text(raw) is None
