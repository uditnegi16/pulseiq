"""Ingestion CLI: scrape -> validate -> raw (Mongo) -> clean (SQL).

The orchestration layer. Contains no parsing, no validation rules, and no SQL --
it wires together modules that are each tested independently.

Usage (from the project root, venv active):

    python -m pulseiq.ingestion.run_ingest --site books_toscrape
    python -m pulseiq.ingestion.run_ingest --site books_toscrape --no-mongo
    python -m pulseiq.ingestion.run_ingest --site books_toscrape --dry-run
    python -m pulseiq.ingestion.run_ingest --from-csv data/raw/competitor_data.csv

--dry-run scrapes and validates but writes nothing, which is how you check a
selector change before it touches the database.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from config.settings import settings
from pulseiq.ingestion.scraper import load_targets, scrape_site
from pulseiq.ingestion.validation import validate_price_snapshots, validate_reviews
from pulseiq.storage import mongo
from pulseiq.storage.relational import get_engine, init_db, session_scope
from pulseiq.storage.repository import (
    count_rows,
    save_price_snapshots,
    save_reviews,
)

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def explode_reviews(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten each product's `reviews` list into one row per review.

    Pure and separately tested: the scraper returns reviews nested under a
    product, but the sentiment dataset needs one row per review.
    """
    rows: list[dict[str, Any]] = []
    for record in raw_records:
        for text in record.get("reviews") or []:
            rows.append(
                {
                    "product_name": record.get("product_name"),
                    "review_text": text,
                    "rating": record.get("rating"),
                    "source": record.get("source"),
                    "date": record.get("date"),
                }
            )
    return rows


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read a CSV into raw dicts. Lets the legacy CSVs enter the same
    validation path as scraped data, instead of a separate code path."""
    import csv

    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def ingest(
    *,
    site: str | None = None,
    csv_path: Path | None = None,
    use_mongo: bool = True,
    dry_run: bool = False,
    headless: bool | None = None,
) -> int:
    """Run one ingestion pass. Returns a process exit code."""
    run_id = uuid.uuid4().hex[:12]
    logger.info("ingestion run %s starting", run_id)

    # --- 1. acquire raw records -------------------------------------------
    if csv_path is not None:
        if not csv_path.exists():
            logger.error("csv not found: %s", csv_path)
            return 1
        raw_records = load_csv_rows(csv_path)
        source = csv_path.stem
        logger.info("loaded %d rows from %s", len(raw_records), csv_path)
    else:
        targets = load_targets()
        if site not in targets:
            logger.error("unknown site %r. available: %s", site, sorted(targets))
            return 1
        config = targets[site]
        if not config.products:
            logger.error("site %r has no products configured in targets.yaml", site)
            return 1

        from pulseiq.ingestion.driver import get_driver

        source = config.name
        with get_driver(headless=headless) as driver:
            raw_records = scrape_site(driver, config)

    if not raw_records:
        logger.error("no records acquired -- nothing to do")
        return 1

    # --- 2. validate -------------------------------------------------------
    prices, price_report = validate_price_snapshots(raw_records, source=source)
    review_rows = explode_reviews(raw_records) or raw_records
    reviews, review_report = validate_reviews(review_rows, source=source)

    print("\n--- validation ---")
    print("prices :", price_report.summary())
    print("reviews:", review_report.summary())

    if dry_run:
        logger.info("dry run -- nothing written")
        return 0

    # --- 3. raw documents -> mongo (best effort) ---------------------------
    if use_mongo and mongo.is_configured():
        try:
            inserted = mongo.insert_raw_documents(raw_records, source=source, run_id=run_id)
            logger.info("mongo: %d raw documents stored", inserted)
        except Exception:  # noqa: BLE001 - Mongo is the audit trail, not the pipeline
            logger.exception("mongo write failed -- continuing to relational store")
    elif use_mongo:
        logger.warning("MONGODB_URI not set -- skipping raw document store")

    # --- 4. cleaned records -> relational ----------------------------------
    engine = get_engine()
    init_db(engine)
    with session_scope(engine) as session:
        price_result = save_price_snapshots(session, prices)
        review_result = save_reviews(session, reviews)
        counts = count_rows(session)

    print("\n--- written ---")
    print("prices :", price_result)
    print("reviews:", review_result)
    print("totals :", counts)

    logger.info("ingestion run %s complete", run_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pulseiq.ingestion.run_ingest",
        description="Scrape or load product data, validate it, and persist it.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--site", help="site key from ingestion/targets.yaml")
    group.add_argument("--from-csv", type=Path, help="load rows from a CSV instead")

    parser.add_argument("--no-mongo", action="store_true", help="skip the raw document store")
    parser.add_argument("--dry-run", action="store_true", help="validate only, write nothing")
    parser.add_argument(
        "--show-browser", action="store_true", help="run Chrome visibly (debugging)"
    )
    parser.add_argument("--log-level", default=settings.log_level)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    return ingest(
        site=args.site,
        csv_path=args.from_csv,
        use_mongo=not args.no_mongo,
        dry_run=args.dry_run,
        headless=False if args.show_browser else None,
    )


if __name__ == "__main__":
    sys.exit(main())
