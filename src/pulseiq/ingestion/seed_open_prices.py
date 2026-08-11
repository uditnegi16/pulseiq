"""Open Prices (Open Food Facts) loader - real price history under an open licence.

WHY THIS DATASET
----------------
The forecasting task needs price observations for the same product over time.
Scraping cannot produce history retroactively: one run yields one snapshot, so
starting today gives zero training rows today. Commercial price-history APIs
(Keepa et al.) forbid redistribution, which would make this project's results
impossible for anyone else to reproduce.

Open Prices is crowdsourced product price data published under the Open Database
License. It has what almost no free dataset has: an explicit ground-truth
discount label (`price_is_discounted` / `price_without_discount`) captured from
receipts and shelf tags -- not inferred, not simulated.

LICENCE - ODbL (https://opendatacommons.org/licenses/odbl/1.0/)
--------------------------------------------------------------
Two obligations, both handled in code:

1. ATTRIBUTION. Every row is tagged source="open_prices". See ATTRIBUTION below
   and docs/decision-log.md.
2. SHARE-ALIKE. If ODbL data is *combined* with another database, the combined
   result must also be released as open data. This loader therefore never
   merges Open Prices rows with scraped rows: they land in the same table but
   are always distinguishable by `source`, and any export must filter on it.
   Keeping the provenance column is the mechanism that keeps share-alike
   contained.

DATA SHAPE
----------
Series identity is (product_code, location_id) - the same barcode costs
different amounts in different shops, so a per-store series is the honest unit
for forecasting. Pooling stores would average away the very signal we want.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

SOURCE = "open_prices"

ATTRIBUTION = (
    "Price data from Open Prices (Open Food Facts), "
    "https://prices.openfoodfacts.org - licensed under ODbL 1.0."
)

# Daily-refreshed Parquet export published on data.gouv.fr.
PARQUET_URL = "https://www.data.gouv.fr/api/1/datasets/r/49716ed5-aacf-4692-8b2d-3cc6d15bf1d1"

# Only the columns we use. Reading a subset keeps a 28MB file cheap to load and
# means an upstream schema addition cannot silently change our behaviour.
USED_COLUMNS = [
    "product_code",
    "price",
    "price_is_discounted",
    "price_without_discount",
    "currency",
    "date",
    "location_id",
]


def read_open_prices(source: str | Path = PARQUET_URL) -> pd.DataFrame:
    """Read the Open Prices Parquet export.

    Accepts a URL, a local path, or an fsspec URI (``s3://bucket/key.parquet``).
    pandas delegates to fsspec, so the local-to-S3 move in Phase 7 is a string
    change, not a code change.
    """
    logger.info("reading Open Prices from %s", source)
    frame = pd.read_parquet(source, columns=USED_COLUMNS)
    logger.info("read %d rows", len(frame))
    return frame


def _as_float(value: Any) -> float | None:
    """Coerce a numeric-ish value to float, or None if it isn't one.

    Open Prices stores monetary columns as Parquet DECIMAL, which pandas
    surfaces as `decimal.Decimal`. Decimal does not interoperate with float
    arithmetic -- `float / Decimal` raises TypeError -- so every price is
    normalised here before any maths happens. Also absorbs None, NaN and
    stray strings.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError, ArithmeticError):
        return None


def compute_discount(row_price: Any, without_discount: Any, is_discounted: bool) -> float:
    """Ground-truth discount percentage for one observation.

    Not discounted -> 0.0. Discounted with a valid reference -> the real
    percentage. Discounted but the reference is missing or nonsensical -> 0.0,
    because inventing a number here would put noise straight into the target.
    """
    if not is_discounted:
        return 0.0

    price = _as_float(row_price)
    reference = _as_float(without_discount)

    if price is None or reference is None or reference <= 0:
        return 0.0
    if price > reference:
        return 0.0
    return round((1.0 - price / reference) * 100.0, 2)


def build_series_id(product_code: Any, location_id: Any) -> str | None:
    """Series identity: one barcode at one shop.

    Returns None for rows without a barcode (Open Prices also holds loose
    produce like fruit, which has no stable identity to track over time).
    """
    if product_code is None or pd.isna(product_code) or str(product_code).strip() in {"", "None"}:
        return None
    if location_id is None or pd.isna(location_id):
        return None
    return f"{str(product_code).strip()}@{int(location_id)}"


def transform_open_prices(
    frame: pd.DataFrame,
    *,
    currency: str | None = "EUR",
    min_observations: int = 8,
    max_series: int | None = None,
) -> list[dict[str, Any]]:
    """Convert the Open Prices export into rows for validate_price_snapshots().

    PURE: no I/O, no network. Fully unit-tested against a synthetic frame that
    matches the published schema.

    Filtering rationale:
      * one currency only -- mixing EUR and USD into one price series is
        meaningless without FX conversion, and conversion would need a second
        dated dataset
      * min_observations -- a barcode seen twice cannot be forecast; dropping it
        here is louder and cheaper than discovering it during training
      * max_series -- cap for quick local runs
    """
    if frame.empty:
        return []

    work = frame.copy()

    if currency is not None and "currency" in work.columns:
        before = len(work)
        work = work[work["currency"] == currency]
        logger.info("currency filter %s: %d -> %d rows", currency, before, len(work))

    work["series_id"] = [
        build_series_id(code, loc)
        for code, loc in zip(work["product_code"], work["location_id"], strict=False)
    ]
    work = work[work["series_id"].notna()]

    # DECIMAL -> float on both monetary columns before any arithmetic.
    work["price"] = work["price"].map(_as_float)
    work["price_without_discount"] = work["price_without_discount"].map(_as_float)
    work = work[work["price"].notna() & (work["price"] > 0)]

    work["observed_on"] = pd.to_datetime(work["date"], errors="coerce")
    work = work[work["observed_on"].notna()]

    # Drop future-dated rows before they reach schema validation, which would
    # otherwise reject them one by one and inflate the rejection counters.
    today = pd.Timestamp.today().normalize()
    work = work[work["observed_on"] <= today]

    if work.empty:
        logger.warning("no rows survived filtering")
        return []

    counts = work.groupby("series_id")["series_id"].transform("size")
    work = work[counts >= min_observations]
    if work.empty:
        logger.warning(
            "no series has >=%d observations; lower min_observations or widen the currency filter",
            min_observations,
        )
        return []

    if max_series is not None:
        keep = work.groupby("series_id").size().sort_values(ascending=False).head(max_series).index
        work = work[work["series_id"].isin(keep)]

    work = work.sort_values(["series_id", "observed_on"])

    rows: list[dict[str, Any]] = []
    for record in work.itertuples(index=False):
        discount = compute_discount(
            record.price,
            getattr(record, "price_without_discount", None),
            bool(getattr(record, "price_is_discounted", False)),
        )
        reference = _as_float(record.price_without_discount)
        original = reference if discount > 0 and reference is not None else None
        rows.append(
            {
                "product_name": record.series_id,
                "source": SOURCE,
                "price": float(record.price),
                "original_price": original,
                "discount": discount,
                "date": record.observed_on.date().isoformat(),
            }
        )

    logger.info("prepared %d rows across %d series", len(rows), work["series_id"].nunique())
    return rows


def summarise(rows: list[dict[str, Any]]) -> str:
    """One-line description of a prepared batch, for logs and the phase log."""
    if not rows:
        return "no rows"
    frame = pd.DataFrame(rows)
    discounted = (frame["discount"] > 0).mean()
    return (
        f"{len(frame)} rows | {frame['product_name'].nunique()} series | "
        f"{frame['date'].min()}..{frame['date'].max()} | "
        f"discounted {discounted:.1%} | "
        f"median obs/series {frame.groupby('product_name').size().median():.0f}"
    )
