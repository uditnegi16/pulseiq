"""Repository layer: the only place that reads or writes the relational store.

Two properties everything here guarantees:

  * **Idempotent writes.** Running the same ingestion twice inserts nothing the
    second time. Scrapers get re-run constantly -- after a crash, after a fix,
    on a cron that overlaps -- and a pipeline that duplicates rows on re-run is
    a pipeline you cannot trust.
  * **DataFrame reads.** Training code wants pandas, not ORM objects. The
    conversion happens once, here.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from pulseiq.storage.relational import PriceSnapshotRow, ReviewRow
from pulseiq.storage.schemas import PriceSnapshot, Review

logger = logging.getLogger(__name__)


@dataclass
class WriteResult:
    """Outcome of a save. Log this -- `skipped` being high on a fresh run means
    your scraper is re-collecting data you already have."""

    inserted: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.inserted + self.skipped

    def __str__(self) -> str:
        return f"inserted={self.inserted} skipped_existing={self.skipped}"


def _product_key(name: str) -> str:
    return name.strip().lower()


def _review_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def save_price_snapshots(session: Session, records: list[PriceSnapshot]) -> WriteResult:
    """Insert snapshots, skipping any (product, day) already stored.

    Existing keys are fetched in one query rather than one per row -- a
    per-row exists() check turns a 1000-row insert into 1000 round trips.
    """
    result = WriteResult()
    if not records:
        return result

    keys = {(_product_key(r.product_name), r.observed_on) for r in records}
    existing = set(
        session.execute(
            select(PriceSnapshotRow.product_key, PriceSnapshotRow.observed_on).where(
                PriceSnapshotRow.product_key.in_({k for k, _ in keys})
            )
        ).all()
    )

    for record in records:
        key = (_product_key(record.product_name), record.observed_on)
        if key in existing:
            result.skipped += 1
            continue

        session.add(
            PriceSnapshotRow(
                product_name=record.product_name,
                product_key=key[0],
                product_url=record.product_url,
                source=record.source,
                selling_price=record.selling_price,
                original_price=record.original_price,
                discount_pct=record.discount_pct,
                rating=record.rating,
                observed_on=record.observed_on,
                ingested_at=record.ingested_at.replace(tzinfo=None),
            )
        )
        existing.add(key)  # guards duplicates within this same batch
        result.inserted += 1

    session.flush()
    logger.info("price snapshots: %s", result)
    return result


def save_reviews(session: Session, records: list[Review]) -> WriteResult:
    """Insert reviews, skipping any (product, text) already stored."""
    result = WriteResult()
    if not records:
        return result

    product_keys = {_product_key(r.product_name) for r in records}
    existing = set(
        session.execute(
            select(ReviewRow.product_key, ReviewRow.review_hash).where(
                ReviewRow.product_key.in_(product_keys)
            )
        ).all()
    )

    for record in records:
        key = (_product_key(record.product_name), _review_hash(record.review_text))
        if key in existing:
            result.skipped += 1
            continue

        session.add(
            ReviewRow(
                product_name=record.product_name,
                product_key=key[0],
                review_text=record.review_text,
                review_hash=key[1],
                rating=record.rating,
                sentiment=record.sentiment.value if record.sentiment else None,
                verified_purchase=record.verified_purchase,
                source=record.source,
                observed_on=record.observed_on,
                ingested_at=record.ingested_at.replace(tzinfo=None),
            )
        )
        existing.add(key)
        result.inserted += 1

    session.flush()
    logger.info("reviews: %s", result)
    return result


def load_price_history(
    session: Session, product_name: str | None = None, *, min_observations: int = 0
) -> pd.DataFrame:
    """Load price history as a DataFrame, sorted by product then date.

    Chronological order is not cosmetic -- Phase 2 splits train/test by time,
    and an unsorted frame produces a split that leaks the future into training.

    `min_observations` drops products with too few points to forecast.
    """
    stmt = select(PriceSnapshotRow)
    if product_name:
        stmt = stmt.where(PriceSnapshotRow.product_key == _product_key(product_name))

    rows = session.execute(stmt).scalars().all()
    if not rows:
        return pd.DataFrame(
            columns=[
                "product_name",
                "source",
                "selling_price",
                "original_price",
                "discount_pct",
                "rating",
                "observed_on",
            ]
        )

    frame = pd.DataFrame(
        [
            {
                "product_name": r.product_name,
                "source": r.source,
                "selling_price": r.selling_price,
                "original_price": r.original_price,
                "discount_pct": r.discount_pct,
                "rating": r.rating,
                "observed_on": r.observed_on,
            }
            for r in rows
        ]
    )
    frame["observed_on"] = pd.to_datetime(frame["observed_on"])
    frame = frame.sort_values(["product_name", "observed_on"]).reset_index(drop=True)

    if min_observations > 0:
        counts = frame.groupby("product_name")["observed_on"].transform("size")
        dropped = frame.loc[counts < min_observations, "product_name"].nunique()
        if dropped:
            logger.info("dropped %d products with <%d observations", dropped, min_observations)
        frame = frame[counts >= min_observations].reset_index(drop=True)

    return frame


def load_reviews(
    session: Session, product_name: str | None = None, *, labelled_only: bool = False
) -> pd.DataFrame:
    """Load reviews as a DataFrame. `labelled_only` drops rows without a
    sentiment label, which is what the fine-tuning set needs."""
    stmt = select(ReviewRow)
    if product_name:
        stmt = stmt.where(ReviewRow.product_key == _product_key(product_name))
    if labelled_only:
        stmt = stmt.where(ReviewRow.sentiment.is_not(None))

    rows = session.execute(stmt).scalars().all()
    if not rows:
        return pd.DataFrame(
            columns=["product_name", "review_text", "rating", "sentiment", "source", "observed_on"]
        )

    return pd.DataFrame(
        [
            {
                "product_name": r.product_name,
                "review_text": r.review_text,
                "rating": r.rating,
                "sentiment": r.sentiment,
                "source": r.source,
                "observed_on": r.observed_on,
            }
            for r in rows
        ]
    )


def count_rows(session: Session) -> dict[str, int]:
    """Row counts per table. Used by run_ingest to report state after a run."""
    from sqlalchemy import func

    return {
        "price_snapshots": session.execute(
            select(func.count()).select_from(PriceSnapshotRow)
        ).scalar_one(),
        "reviews": session.execute(select(func.count()).select_from(ReviewRow)).scalar_one(),
    }
