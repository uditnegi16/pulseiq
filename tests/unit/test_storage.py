"""Tests for the storage layer and ingestion orchestration.

The relational half runs against in-memory SQLite -- real SQL, real constraints,
real upsert behaviour, zero external services. Mongo is tested with an injected
fake collection, since its value here is the document envelope, not pymongo.
"""

from datetime import date, timedelta

import pytest

from pulseiq.ingestion.run_ingest import explode_reviews, load_csv_rows
from pulseiq.storage import mongo
from pulseiq.storage.relational import get_engine, init_db, session_scope
from pulseiq.storage.repository import (
    count_rows,
    load_price_history,
    load_reviews,
    save_price_snapshots,
    save_reviews,
)
from pulseiq.storage.schemas import PriceSnapshot, Review, Sentiment

TODAY = date.today()


@pytest.fixture
def session():
    """Fresh in-memory database per test. Fast, isolated, no cleanup."""
    engine = get_engine("sqlite://")
    init_db(engine)
    with session_scope(engine) as s:
        yield s


def snap(name="Widget", price=100.0, days_ago=0, **kw):
    return PriceSnapshot(
        product_name=name,
        selling_price=price,
        observed_on=TODAY - timedelta(days=days_ago),
        **kw,
    )


def review(text="Great product, works well", name="Widget", rating=5.0, **kw):
    return Review(product_name=name, review_text=text, rating=rating, **kw)


class TestSavePriceSnapshots:
    def test_inserts_new_records(self, session):
        result = save_price_snapshots(session, [snap(), snap(days_ago=1)])
        assert result.inserted == 2
        assert result.skipped == 0

    def test_empty_input_is_a_noop(self, session):
        assert save_price_snapshots(session, []).total == 0

    def test_skips_duplicate_product_day_across_calls(self, session):
        save_price_snapshots(session, [snap(price=100.0)])
        result = save_price_snapshots(session, [snap(price=999.0)])
        assert result.inserted == 0
        assert result.skipped == 1

    def test_skips_duplicates_within_one_batch(self, session):
        """Two scrapes of the same product on the same day in one run."""
        result = save_price_snapshots(session, [snap(), snap()])
        assert result.inserted == 1
        assert result.skipped == 1

    def test_dedupe_is_case_insensitive(self, session):
        save_price_snapshots(session, [snap(name="Widget")])
        result = save_price_snapshots(session, [snap(name="WIDGET")])
        assert result.skipped == 1

    def test_same_product_different_days_both_kept(self, session):
        result = save_price_snapshots(
            session, [snap(days_ago=0), snap(days_ago=1), snap(days_ago=2)]
        )
        assert result.inserted == 3

    def test_ingestion_is_idempotent(self, session):
        """Re-running the pipeline must not multiply rows -- the property that
        makes a scheduled daily scrape safe to retry."""
        records = [snap(days_ago=i) for i in range(5)]
        save_price_snapshots(session, records)
        save_price_snapshots(session, records)
        save_price_snapshots(session, records)
        assert count_rows(session)["price_snapshots"] == 5


class TestSaveReviews:
    def test_inserts_and_dedupes_by_text(self, session):
        assert save_reviews(session, [review()]).inserted == 1
        assert save_reviews(session, [review()]).skipped == 1

    def test_different_text_both_kept(self, session):
        result = save_reviews(
            session,
            [review(text="Excellent build quality"), review(text="Poor battery life")],
        )
        assert result.inserted == 2

    def test_sentiment_is_persisted(self, session):
        save_reviews(session, [review(text="Absolutely terrible", rating=1.0)])
        frame = load_reviews(session)
        assert frame.iloc[0]["sentiment"] == Sentiment.NEGATIVE.value


class TestLoadPriceHistory:
    def test_returns_chronological_order(self, session):
        """Phase 2 splits train/test by time. An unsorted frame leaks the
        future into training -- this is the guard against that."""
        save_price_snapshots(session, [snap(days_ago=0), snap(days_ago=5), snap(days_ago=2)])
        frame = load_price_history(session)
        dates = list(frame["observed_on"])
        assert dates == sorted(dates)

    def test_empty_table_returns_empty_frame_with_columns(self, session):
        frame = load_price_history(session)
        assert frame.empty
        assert "selling_price" in frame.columns

    def test_filters_by_product(self, session):
        save_price_snapshots(session, [snap(name="A"), snap(name="B")])
        frame = load_price_history(session, product_name="A")
        assert len(frame) == 1
        assert frame.iloc[0]["product_name"] == "A"

    def test_product_filter_is_case_insensitive(self, session):
        save_price_snapshots(session, [snap(name="Widget")])
        assert len(load_price_history(session, product_name="widget")) == 1

    def test_min_observations_drops_short_series(self, session):
        """A product with 2 price points cannot be forecast; dropping it early
        beats an ARIMA fit that silently produces noise."""
        save_price_snapshots(
            session,
            [snap(name="Long", days_ago=i) for i in range(6)]
            + [snap(name="Short", days_ago=i) for i in range(2)],
        )
        frame = load_price_history(session, min_observations=5)
        assert set(frame["product_name"]) == {"Long"}


class TestCountRows:
    def test_counts_both_tables(self, session):
        save_price_snapshots(session, [snap()])
        save_reviews(session, [review()])
        counts = count_rows(session)
        assert counts["price_snapshots"] == 1
        assert counts["reviews"] == 1

    def test_zero_on_empty(self, session):
        assert count_rows(session)["price_snapshots"] == 0


class TestMongoHelpers:
    def test_envelope_wraps_payload_with_metadata(self):
        doc = mongo.build_raw_document({"price": 100}, source="site_x", run_id="abc123")
        assert doc["payload"] == {"price": 100}
        assert doc["source"] == "site_x"
        assert doc["run_id"] == "abc123"
        assert doc["ingested_at"].tzinfo is not None

    def test_insert_uses_injected_collection(self):
        class FakeCollection:
            def __init__(self):
                self.docs = []

            def insert_many(self, docs):
                self.docs.extend(docs)
                return type("R", (), {"inserted_ids": list(range(len(docs)))})()

        fake = FakeCollection()
        count = mongo.insert_raw_documents(
            [{"a": 1}, {"b": 2}], source="s", run_id="r", collection=fake
        )
        assert count == 2
        assert len(fake.docs) == 2

    def test_insert_empty_does_not_touch_collection(self):
        count = mongo.insert_raw_documents([], source="s", collection=None)
        assert count == 0

    def test_unconfigured_client_raises_actionable_error(self, monkeypatch):
        monkeypatch.setattr(mongo, "is_configured", lambda: False)
        with pytest.raises(mongo.MongoNotConfigured, match="MONGODB_URI"):
            mongo.get_client()

    def test_ping_returns_false_instead_of_raising(self):
        class Broken:
            @property
            def admin(self):
                raise ConnectionError("no route to host")

        assert mongo.ping(Broken()) is False


class TestExplodeReviews:
    def test_one_row_per_review(self):
        raw = [
            {
                "product_name": "Widget",
                "rating": 4.5,
                "source": "site",
                "date": TODAY.isoformat(),
                "reviews": ["First review here", "Second review here"],
            }
        ]
        rows = explode_reviews(raw)
        assert len(rows) == 2
        assert rows[0]["review_text"] == "First review here"
        assert rows[1]["product_name"] == "Widget"

    def test_product_with_no_reviews_yields_nothing(self):
        assert explode_reviews([{"product_name": "W", "reviews": []}]) == []

    def test_missing_reviews_key_is_safe(self):
        assert explode_reviews([{"product_name": "W"}]) == []


class TestLoadCsvRows:
    def test_reads_header_and_rows(self, tmp_path):
        path = tmp_path / "sample.csv"
        path.write_text("product_name,price,date\nWidget,100,2026-01-01\n", encoding="utf-8")
        rows = load_csv_rows(path)
        assert rows == [{"product_name": "Widget", "price": "100", "date": "2026-01-01"}]

    def test_strips_utf8_bom(self, tmp_path):
        """Excel writes a BOM, which turns the first column name into
        '\\ufeffproduct_name' and breaks every downstream key lookup."""
        path = tmp_path / "bom.csv"
        path.write_bytes("product_name,price\nWidget,100\n".encode("utf-8-sig"))
        rows = load_csv_rows(path)
        assert "product_name" in rows[0]


class TestEndToEndValidationToStorage:
    def test_scraped_shape_survives_the_whole_pipeline(self, session):
        """The contract across four modules: parse output -> validation ->
        repository -> DataFrame, with nothing lost or reordered."""
        from pulseiq.ingestion.validation import validate_price_snapshots

        raw = [
            {
                "product_name": "Apple AirPods Pro",
                "price": "â‚¹24,999.00",
                "original_price": "â‚¹29,999.00",
                "date": (TODAY - timedelta(days=d)).isoformat(),
            }
            for d in range(3)
        ]
        records, report = validate_price_snapshots(raw)
        assert report.valid == 3

        save_price_snapshots(session, records)
        frame = load_price_history(session, product_name="Apple AirPods Pro")

        assert len(frame) == 3
        assert frame.iloc[0]["selling_price"] == pytest.approx(24999.0)
        assert frame.iloc[0]["discount_pct"] == pytest.approx(16.67, abs=0.01)
        assert list(frame["observed_on"]) == sorted(frame["observed_on"])


class TestHealthcheck:
    """The health check must never raise -- it exists to report, not to fail."""

    def test_masks_credentials_in_uris(self):
        from pulseiq.storage.healthcheck import _mask

        masked = _mask("mongodb+srv://EXAMPLE_USER:EXAMPLE_PASSWORD@example.invalid/db")
        assert "EXAMPLE_PASSWORD" not in masked
        assert "EXAMPLE_USER" not in masked
        assert "example.invalid" in masked

    def test_mask_handles_uris_without_credentials(self):
        from pulseiq.storage.healthcheck import _mask

        assert _mask("sqlite:///./pulseiq.db") == "sqlite:///./pulseiq.db"

    def test_mask_handles_none(self):
        from pulseiq.storage.healthcheck import _mask

        assert _mask(None) == "(not set)"

    def test_unconfigured_mongo_skips_rather_than_fails(self):
        """Mongo is optional by design; absence is not an error."""
        from pulseiq.storage import healthcheck, mongo

        original = mongo.is_configured
        try:
            mongo.is_configured = lambda: False
            status, detail = healthcheck.check_mongo()
            assert status == healthcheck.SKIP
            assert "MONGODB_URI" in detail
        finally:
            mongo.is_configured = original

    def test_file_store_mlflow_uri_is_reported_as_a_failure(self):
        """MLflow 3.15 removed the file store; catching it here beats catching
        it after a completed training run."""
        from config.settings import settings
        from pulseiq.storage import healthcheck

        original = settings.mlflow_tracking_uri
        try:
            object.__setattr__(settings, "mlflow_tracking_uri", "file:./mlruns")
            status, detail = healthcheck.check_mlflow()
            assert status == healthcheck.FAIL
            assert "sqlite" in detail
        finally:
            object.__setattr__(settings, "mlflow_tracking_uri", original)


class TestOpenPricesIngestionPath:
    def test_price_only_source_does_not_report_review_failures(self):
        """Price rows have no review text. Reporting one rejection per row was
        expected behaviour rendered as hundreds of failures."""
        from pulseiq.ingestion.run_ingest import explode_reviews

        price_rows = [
            {"product_name": "P1", "price": 10.0, "date": "2026-01-01"},
            {"product_name": "P2", "price": 20.0, "date": "2026-01-01"},
        ]
        assert explode_reviews(price_rows) == []
