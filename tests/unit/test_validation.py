"""Unit tests for the validation layer and record schemas."""

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from pulseiq.ingestion.validation import (
    ValidationReport,
    validate_price_snapshots,
    validate_reviews,
)
from pulseiq.storage.schemas import PriceSnapshot, Review, Sentiment

TODAY = date.today()


def price_row(**overrides):
    row = {
        "product_name": "Apple AirPods Pro (2nd Generation)",
        "price": "₹24,999.00",
        "original_price": "₹29,999.00",
        "date": TODAY.isoformat(),
    }
    row.update(overrides)
    return row


def review_row(**overrides):
    row = {
        "product_name": "Apple AirPods Pro (2nd Generation)",
        "review": "Amazing sound quality and noise cancellation.",
        "rating": "4.5",
        "date": TODAY.isoformat(),
    }
    row.update(overrides)
    return row


class TestSentiment:
    @pytest.mark.parametrize(
        "rating,expected",
        [
            (1.0, Sentiment.NEGATIVE),
            (2.0, Sentiment.NEGATIVE),
            (3.0, Sentiment.NEUTRAL),
            (4.0, Sentiment.POSITIVE),
            (5.0, Sentiment.POSITIVE),
        ],
    )
    def test_derives_label_from_rating(self, rating, expected):
        assert Sentiment.from_rating(rating) == expected


class TestPriceSnapshot:
    def test_derives_discount_when_absent(self):
        snap = PriceSnapshot(
            product_name="Widget",
            selling_price=8000.0,
            original_price=10000.0,
            observed_on=TODAY,
        )
        assert snap.discount_pct == 20.0

    def test_keeps_explicit_discount(self):
        snap = PriceSnapshot(
            product_name="Widget",
            selling_price=8000.0,
            original_price=10000.0,
            discount_pct=18.0,
            observed_on=TODAY,
        )
        assert snap.discount_pct == 18.0

    def test_rejects_future_date(self):
        """A future observation date breaks any time-based train/test split."""
        with pytest.raises(ValidationError):
            PriceSnapshot(
                product_name="Widget",
                selling_price=100.0,
                observed_on=TODAY + timedelta(days=1),
            )

    def test_rejects_selling_above_original(self):
        with pytest.raises(ValidationError):
            PriceSnapshot(
                product_name="Widget",
                selling_price=15000.0,
                original_price=10000.0,
                observed_on=TODAY,
            )

    def test_rejects_negative_price(self):
        with pytest.raises(ValidationError):
            PriceSnapshot(product_name="W", selling_price=-1.0, observed_on=TODAY)

    def test_dedupe_key_is_case_insensitive(self):
        a = PriceSnapshot(product_name="Widget", selling_price=1.0, observed_on=TODAY)
        b = PriceSnapshot(product_name="WIDGET", selling_price=2.0, observed_on=TODAY)
        assert a.dedupe_key == b.dedupe_key


class TestReview:
    def test_derives_sentiment(self):
        r = Review(product_name="W", review_text="Terrible battery life", rating=1.0)
        assert r.sentiment == Sentiment.NEGATIVE

    def test_rejects_trivial_text(self):
        with pytest.raises(ValidationError):
            Review(product_name="W", review_text="ok")

    def test_rejects_out_of_range_rating(self):
        with pytest.raises(ValidationError):
            Review(product_name="W", review_text="Good enough", rating=9.0)


class TestValidatePriceSnapshots:
    def test_happy_path(self):
        records, report = validate_price_snapshots([price_row()])
        assert report.valid == 1
        assert report.rejected == 0
        assert records[0].selling_price == 24999.0
        assert records[0].discount_pct == pytest.approx(16.67, abs=0.01)

    def test_counts_rejection_reasons(self):
        rows = [
            price_row(),
            price_row(product_name=""),
            price_row(price="out of stock"),
            price_row(date="not-a-date"),
        ]
        _, report = validate_price_snapshots(rows)
        assert report.total == 4
        assert report.valid == 1
        assert report.rejections["missing_product_name"] == 1
        assert report.rejections["unparseable_selling_price"] == 1
        assert report.rejections["missing_or_unparseable_date"] == 1

    def test_deduplicates_same_product_same_day(self):
        records, report = validate_price_snapshots([price_row(), price_row()])
        assert len(records) == 1
        assert report.duplicates == 1

    def test_dedupe_can_be_disabled(self):
        records, _ = validate_price_snapshots([price_row(), price_row()], deduplicate=False)
        assert len(records) == 2

    def test_accepts_multiple_date_formats(self):
        rows = [
            price_row(date=TODAY.strftime("%Y-%m-%d")),
            price_row(product_name="B", date=TODAY.strftime("%d/%m/%Y")),
        ]
        records, report = validate_price_snapshots(rows)
        assert report.valid == 2
        assert all(r.observed_on == TODAY for r in records)

    def test_bare_numeric_discount_is_accepted(self):
        records, _ = validate_price_snapshots([price_row(original_price=None, discount="15")])
        assert records[0].discount_pct == 15.0

    def test_empty_input(self):
        records, report = validate_price_snapshots([])
        assert records == []
        assert report.total == 0
        assert report.pass_rate == 0.0


class TestValidateReviews:
    def test_happy_path(self):
        records, report = validate_reviews([review_row()])
        assert report.valid == 1
        assert records[0].sentiment == Sentiment.POSITIVE

    def test_rejects_corrupt_rating_from_original_csv(self):
        """'4.3.0' is real data from reviews.csv row 2."""
        _, report = validate_reviews([review_row(rating="4.3.0")])
        assert report.valid == 0
        assert report.rejections["unparseable_rating"] == 1

    def test_filters_one_word_reviews(self):
        _, report = validate_reviews([review_row(review="Good")])
        assert report.rejections["review_too_short"] == 1

    def test_deduplicates_identical_text(self):
        records, report = validate_reviews([review_row(), review_row()])
        assert len(records) == 1
        assert report.duplicates == 1

    def test_review_without_rating_has_no_sentiment(self):
        records, _ = validate_reviews([review_row(rating=None)])
        assert records[0].sentiment is None

    def test_whitespace_is_normalised(self):
        records, _ = validate_reviews([review_row(review="  Great   sound\n\nquality  ")])
        assert records[0].review_text == "Great sound quality"


class TestValidationReport:
    def test_pass_rate_and_summary(self):
        report = ValidationReport(total=10, valid=7)
        report.reject("bad_price")
        report.reject("bad_price")
        report.reject("bad_date")
        assert report.rejected == 3
        assert report.pass_rate == 0.7
        assert "bad_price: 2" in report.summary()

    def test_pass_rate_no_division_by_zero(self):
        assert ValidationReport().pass_rate == 0.0
