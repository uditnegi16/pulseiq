"""Canonical record shapes for the whole project.

These models are the contract between layers: the scraper produces them,
validation enforces them, MongoDB stores them, training consumes them, and the
API returns them. Defining them once here is what stops "price" being a string
in one module and a float in another.

SQLAlchemy table definitions live in storage/relational.py and mirror these.
"""

from __future__ import annotations

from datetime import UTC, datetime
from datetime import date as Date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Sentiment(StrEnum):
    """Sentiment label. Derived from star rating during dataset construction.

    NEGATIVE for 1-2 stars, POSITIVE for 4-5, NEUTRAL for 3. Note this is a
    *proxy* label -- a 5-star review can still contain complaints. Documented
    as a known limitation in docs/decision-log.md.
    """

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"

    @classmethod
    def from_rating(cls, rating: float) -> Sentiment:
        if rating <= 2.0:
            return cls.NEGATIVE
        if rating >= 4.0:
            return cls.POSITIVE
        return cls.NEUTRAL


class PriceSnapshot(BaseModel):
    """One product's pricing at one point in time. The unit of the forecasting
    dataset: a series of these per product becomes the time series."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    product_name: str = Field(min_length=1, max_length=500)
    product_url: str | None = None
    source: str = Field(default="unknown", description="Which site/dataset this came from")

    selling_price: float = Field(ge=0)
    original_price: float | None = Field(default=None, ge=0)
    discount_pct: float | None = Field(default=None, ge=0, le=100)
    rating: float | None = Field(default=None, ge=0, le=5)

    observed_on: Date
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("observed_on")
    @classmethod
    def not_in_the_future(cls, v: Date) -> Date:
        """A price observed tomorrow is a clock bug or a bad parse, and it would
        silently corrupt any time-based train/test split."""
        if v > datetime.now(UTC).date():
            raise ValueError(f"observed_on {v} is in the future")
        return v

    @model_validator(mode="after")
    def fill_and_check_discount(self) -> PriceSnapshot:
        """Derive discount from the prices when absent; reject impossible pairs."""
        if self.original_price is not None:
            if self.selling_price > self.original_price * 1.01:  # 1% float tolerance
                raise ValueError(
                    f"selling_price {self.selling_price} exceeds "
                    f"original_price {self.original_price}"
                )
            if self.discount_pct is None and self.original_price > 0:
                derived = (1.0 - self.selling_price / self.original_price) * 100.0
                object.__setattr__(self, "discount_pct", round(max(derived, 0.0), 2))
        return self

    @property
    def dedupe_key(self) -> tuple[str, Date]:
        """One observation per product per day. A second scrape the same day is
        a duplicate, not new information."""
        return (self.product_name.lower(), self.observed_on)


class Review(BaseModel):
    """One customer review. The unit of the sentiment fine-tuning dataset."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    product_name: str = Field(min_length=1, max_length=500)
    review_text: str = Field(min_length=1)
    rating: float | None = Field(default=None, ge=0, le=5)
    sentiment: Sentiment | None = None
    verified_purchase: bool | None = None
    source: str = Field(default="unknown")

    observed_on: Date | None = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("review_text")
    @classmethod
    def reject_trivial_text(cls, v: str) -> str:
        """Reviews shorter than 3 characters carry no signal and only add noise
        to the fine-tuning set."""
        if len(v.strip()) < 3:
            raise ValueError("review_text too short to be useful")
        return v

    @model_validator(mode="after")
    def derive_sentiment(self) -> Review:
        if self.sentiment is None and self.rating is not None:
            object.__setattr__(self, "sentiment", Sentiment.from_rating(self.rating))
        return self

    @property
    def dedupe_key(self) -> tuple[str, str]:
        """Same product + same text = same review, however many times scraped."""
        return (self.product_name.lower(), self.review_text.strip().lower())
