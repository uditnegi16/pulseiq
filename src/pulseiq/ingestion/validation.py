"""Data validation layer: raw scraped dicts -> validated records + a report.

Every row that enters the project passes through here. Rows are never silently
dropped: each rejection is counted by reason, so `ValidationReport` tells you
*why* 300 of 1000 rows disappeared instead of leaving you to guess.

That report is also the input to drift monitoring later -- a sudden jump in one
rejection reason usually means the site changed its markup, not that the data
changed.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from pulseiq.ingestion.parsers import (
    clean_review_text,
    extract_discount_pct,
    extract_price,
    extract_rating,
)
from pulseiq.storage.schemas import PriceSnapshot, Review

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Outcome of validating a batch. Log this after every ingestion run."""

    total: int = 0
    valid: int = 0
    duplicates: int = 0
    rejections: Counter[str] = field(default_factory=Counter)

    @property
    def rejected(self) -> int:
        return sum(self.rejections.values())

    @property
    def pass_rate(self) -> float:
        return self.valid / self.total if self.total else 0.0

    def reject(self, reason: str) -> None:
        self.rejections[reason] += 1

    def summary(self) -> str:
        lines = [
            f"rows={self.total} valid={self.valid} "
            f"rejected={self.rejected} duplicates={self.duplicates} "
            f"pass_rate={self.pass_rate:.1%}"
        ]
        for reason, count in self.rejections.most_common():
            lines.append(f"  - {reason}: {count}")
        return "\n".join(lines)


def _coerce_date(value: Any) -> Date | None:
    """Accept date, datetime, or ISO-ish string. Return None if unparseable."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, Date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
    return None


def validate_price_snapshots(
    rows: Iterable[dict[str, Any]],
    *,
    source: str = "unknown",
    deduplicate: bool = True,
) -> tuple[list[PriceSnapshot], ValidationReport]:
    """Validate raw price rows into PriceSnapshot records.

    Expects loose keys -- 'price'/'selling_price', 'discount'/'discount_pct',
    'date'/'observed_on' -- because scrapers and CSVs disagree on naming.
    """
    report = ValidationReport()
    seen: set[tuple[str, Date]] = set()
    out: list[PriceSnapshot] = []

    for row in rows:
        report.total += 1

        name = (row.get("product_name") or row.get("name") or "").strip()
        if not name:
            report.reject("missing_product_name")
            continue

        raw_price = row.get("selling_price", row.get("price"))
        selling = extract_price(str(raw_price)) if raw_price is not None else None
        if selling is None:
            report.reject("unparseable_selling_price")
            continue

        raw_original = row.get("original_price", row.get("mrp"))
        original = extract_price(str(raw_original)) if raw_original is not None else None

        raw_discount = row.get("discount_pct", row.get("discount"))
        discount = extract_discount_pct(str(raw_discount)) if raw_discount is not None else None
        if discount is None and raw_discount is not None:
            # A bare number like "15" has no % sign but is still a percentage.
            numeric = extract_price(str(raw_discount))
            discount = numeric if numeric is not None and 0 <= numeric <= 100 else None

        rating = extract_rating(str(row["rating"])) if row.get("rating") is not None else None

        observed = _coerce_date(row.get("observed_on", row.get("date")))
        if observed is None:
            report.reject("missing_or_unparseable_date")
            continue

        try:
            record = PriceSnapshot(
                product_name=name,
                product_url=row.get("product_url") or row.get("url"),
                source=row.get("source", source),
                selling_price=selling,
                original_price=original,
                discount_pct=discount,
                rating=rating,
                observed_on=observed,
            )
        except ValidationError as exc:
            reason = exc.errors()[0].get("msg", "schema_violation")
            report.reject(f"schema:{reason[:60]}")
            continue

        if deduplicate:
            if record.dedupe_key in seen:
                report.duplicates += 1
                continue
            seen.add(record.dedupe_key)

        out.append(record)
        report.valid += 1

    logger.info("price validation:\n%s", report.summary())
    return out, report


def validate_reviews(
    rows: Iterable[dict[str, Any]],
    *,
    source: str = "unknown",
    deduplicate: bool = True,
    min_words: int = 2,
) -> tuple[list[Review], ValidationReport]:
    """Validate raw review rows into Review records.

    `min_words` filters one-word reviews ("Good", "Ok"), which are technically
    valid but contribute almost nothing to a fine-tuning set.
    """
    report = ValidationReport()
    seen: set[tuple[str, str]] = set()
    out: list[Review] = []

    for row in rows:
        report.total += 1

        name = (row.get("product_name") or row.get("name") or "").strip()
        if not name:
            report.reject("missing_product_name")
            continue

        text = clean_review_text(row.get("review_text") or row.get("review") or row.get("text"))
        if text is None:
            report.reject("empty_review_text")
            continue
        if len(text.split()) < min_words:
            report.reject("review_too_short")
            continue

        rating = extract_rating(str(row["rating"])) if row.get("rating") is not None else None
        if row.get("rating") is not None and rating is None:
            report.reject("unparseable_rating")
            continue

        try:
            record = Review(
                product_name=name,
                review_text=text,
                rating=rating,
                verified_purchase=row.get("verified_purchase"),
                source=row.get("source", source),
                observed_on=_coerce_date(row.get("observed_on", row.get("date"))),
            )
        except ValidationError as exc:
            reason = exc.errors()[0].get("msg", "schema_violation")
            report.reject(f"schema:{reason[:60]}")
            continue

        if deduplicate:
            if record.dedupe_key in seen:
                report.duplicates += 1
                continue
            seen.add(record.dedupe_key)

        out.append(record)
        report.valid += 1

    logger.info("review validation:\n%s", report.summary())
    return out, report
