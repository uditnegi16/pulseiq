"""Integration tests: the full pipeline, real components, no fakes.

WHY THESE EXIST SEPARATELY FROM THE UNIT TESTS
----------------------------------------------
Every unit test in this project passes. Several real bugs still reached working
code, and they all shared a shape: the individual pieces were correct and the
*joins* between them were not.

  E-002  a fixture built from documentation used floats; the real Parquet used
         Decimal, and `float / Decimal` raises
  E-003  `split_per_product` produced splits its own leakage check rejected,
         because the check compared dates globally while the split was per-product
  E-013  a test asserted a 503 that only occurred when the model happened to be
         absent from the filesystem

A unit test with a fake on both sides of a boundary cannot catch a mismatch at
that boundary. These tests use real SQLite, real Parquet round-trips, and the
real FastAPI app, and they assert that data survives every hand-off with its
meaning intact.

Marked `integration` so they can be excluded from a fast inner loop:

    pytest -m "not integration"
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """An isolated project workspace: temp database, temp reports, clean caches."""
    from config.settings import settings
    from pulseiq.api import deps
    from pulseiq.llm.cache import reset_cache

    monkeypatch.setattr(settings, "database_url", f"sqlite:///{tmp_path / 'pipeline.db'}")
    monkeypatch.setattr(settings, "redis_url", None)
    reset_cache()
    deps.reset_model_cache()

    (tmp_path / "reports").mkdir()
    yield tmp_path

    reset_cache()
    deps.reset_model_cache()


@pytest.fixture
def open_prices_parquet(tmp_path):
    """A Parquet file matching the real Open Prices schema, DECIMAL columns included.

    DECIMAL is the whole point: E-002 was caused by a float fixture standing in
    for a decimal128 column. Writing through pyarrow with an explicit schema means
    pandas hands back `decimal.Decimal` exactly as the real export does.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pulseiq.ingestion.seed_open_prices import USED_COLUMNS

    rng = np.random.default_rng(17)
    today = date.today()
    rows = []

    for series in range(12):
        base = float(rng.uniform(2.0, 30.0))
        # 20 observations at irregular intervals, mirroring the real data's
        # median 17-day gap.
        offset = 0
        for _ in range(20):
            offset += int(rng.integers(5, 40))
            discounted = rng.random() < 0.15
            price = base * (0.8 if discounted else 1.0) * float(rng.uniform(0.98, 1.02))
            rows.append(
                {
                    "product_code": f"30{series:011d}",
                    "price": Decimal(str(round(price, 2))),
                    "price_is_discounted": bool(discounted),
                    "price_without_discount": Decimal(str(round(base, 2))) if discounted else None,
                    "currency": "EUR",
                    "date": today - timedelta(days=800 - offset),
                    "location_id": int(rng.integers(100, 103)),
                }
            )

    frame = pd.DataFrame(rows, columns=USED_COLUMNS)
    schema = pa.schema(
        [
            ("product_code", pa.string()),
            ("price", pa.decimal128(10, 2)),
            ("price_is_discounted", pa.bool_()),
            ("price_without_discount", pa.decimal128(10, 2)),
            ("currency", pa.string()),
            ("date", pa.date32()),
            ("location_id", pa.int64()),
        ]
    )
    path = tmp_path / "open_prices.parquet"
    pq.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False), path)
    return path


# --- ingestion --------------------------------------------------------------


class TestIngestionPipeline:
    def test_parquet_to_database_end_to_end(self, workspace, open_prices_parquet):
        """Parquet (DECIMAL) -> transform -> validate -> SQL -> DataFrame.

        Four modules, three type conversions. The regression guard for E-002.
        """
        from pulseiq.ingestion.seed_open_prices import read_open_prices, transform_open_prices
        from pulseiq.ingestion.validation import validate_price_snapshots
        from pulseiq.storage.relational import get_engine, init_db, session_scope
        from pulseiq.storage.repository import load_price_history, save_price_snapshots

        raw = read_open_prices(open_prices_parquet)
        assert isinstance(raw["price"].iloc[0], Decimal), "fixture must use DECIMAL, not float"

        rows = transform_open_prices(raw, min_observations=8)
        assert rows, "transform produced nothing"
        assert all(isinstance(r["price"], float) for r in rows), "Decimal not normalised"

        records, report = validate_price_snapshots(rows)
        assert report.valid > 0
        assert report.rejected == 0, f"unexpected rejections: {report.rejections}"

        engine = get_engine()
        init_db(engine)
        with session_scope(engine) as session:
            result = save_price_snapshots(session, records)
            frame = load_price_history(session)

        assert result.inserted == len(records)
        assert len(frame) == len(records)
        assert set(frame["source"]) == {"open_prices"}

    def test_reingestion_is_idempotent(self, workspace, open_prices_parquet):
        """The property that makes a scheduled scrape safe to retry."""
        from pulseiq.ingestion.seed_open_prices import read_open_prices, transform_open_prices
        from pulseiq.ingestion.validation import validate_price_snapshots
        from pulseiq.storage.relational import get_engine, init_db, session_scope
        from pulseiq.storage.repository import count_rows, save_price_snapshots

        rows = transform_open_prices(read_open_prices(open_prices_parquet), min_observations=8)
        records, _ = validate_price_snapshots(rows)

        engine = get_engine()
        init_db(engine)

        with session_scope(engine) as session:
            first = save_price_snapshots(session, records)
        with session_scope(engine) as session:
            second = save_price_snapshots(session, records)
            total = count_rows(session)["price_snapshots"]

        assert first.inserted == len(records)
        assert second.inserted == 0
        assert second.skipped == len(records)
        assert total == len(records)

    def test_csv_and_parquet_share_the_same_validation_path(self, workspace, tmp_path):
        """Two sources, one validation layer -- no parallel code path to drift."""
        # Written with the csv module rather than a formatted string: the price
        # "₹1,299.50" contains a comma, which must be quoted or it splits the
        # column and the date lands in the wrong field.
        import csv as csv_module

        from pulseiq.ingestion.run_ingest import load_csv_rows
        from pulseiq.ingestion.validation import validate_price_snapshots

        csv_path = tmp_path / "prices.csv"
        today = date.today().isoformat()
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv_module.writer(handle)
            writer.writerow(["product_name", "price", "date"])
            writer.writerow(["Widget", "₹1,299.50", today])
            writer.writerow(["Gadget", "999", today])
            writer.writerow(["Broken", "out of stock", today])

        records, report = validate_price_snapshots(load_csv_rows(csv_path))
        assert report.total == 3
        assert report.valid == 2
        assert report.rejections["unparseable_selling_price"] == 1
        # The decimal point survives: 1,299.50 must not become 129950 (E-001).
        assert any(r.selling_price == pytest.approx(1299.50) for r in records)


# --- training ---------------------------------------------------------------


class TestTrainingPipeline:
    def test_database_to_leak_free_evaluation(self, workspace, open_prices_parquet):
        """DB -> resample -> split -> models -> metrics, with leakage asserted.

        The regression guard for E-003: `split_per_product` producing partitions
        its own leakage check rejected.
        """
        from pulseiq.evaluation.harness import evaluate_models
        from pulseiq.features.resample import resample_panel
        from pulseiq.features.splits import split_per_product
        from pulseiq.ingestion.seed_open_prices import read_open_prices, transform_open_prices
        from pulseiq.ingestion.validation import validate_price_snapshots
        from pulseiq.storage.relational import get_engine, init_db, session_scope
        from pulseiq.storage.repository import load_price_history, save_price_snapshots
        from pulseiq.training.forecasting.baseline import Mean, NaiveLast

        rows = transform_open_prices(read_open_prices(open_prices_parquet), min_observations=8)
        records, _ = validate_price_snapshots(rows)

        engine = get_engine()
        init_db(engine)
        with session_scope(engine) as session:
            save_price_snapshots(session, records)
            frame = load_price_history(session, min_observations=8)

        grid = resample_panel(frame, freq="MS", min_observed=6, max_fill_periods=3)
        assert not grid.empty
        assert "is_imputed" in grid.columns

        split = split_per_product(grid, test_size=0.25, min_observations=6)
        split.assert_no_leakage()

        report = evaluate_models(split, [NaiveLast, Mean], min_train=4)
        assert report.results
        assert set(report.to_frame()["model"]) == {"naive_last", "mean"}
        assert all(r.mae >= 0 for r in report.results)

    def test_imputed_rows_never_reach_scoring(self, workspace, open_prices_parquet):
        """A forward-filled row is a copy of its predecessor, so a naive model
        predicts it exactly. Scoring on imputed rows rewards the model that
        learns least."""
        from pulseiq.evaluation.harness import evaluate_models
        from pulseiq.features.resample import resample_panel
        from pulseiq.features.splits import split_per_product
        from pulseiq.ingestion.seed_open_prices import read_open_prices, transform_open_prices
        from pulseiq.ingestion.validation import validate_price_snapshots
        from pulseiq.training.forecasting.baseline import NaiveLast

        rows = transform_open_prices(read_open_prices(open_prices_parquet), min_observations=8)
        records, _ = validate_price_snapshots(rows)
        frame = pd.DataFrame(
            [
                {
                    "product_name": r.product_name,
                    "observed_on": r.observed_on,
                    "selling_price": r.selling_price,
                }
                for r in records
            ]
        )

        grid = resample_panel(frame, freq="MS", min_observed=6, max_fill_periods=3)
        split = split_per_product(grid, test_size=0.25, min_observations=6)

        scored = evaluate_models(split, [NaiveLast], min_train=4, score_observed_only=True)
        unscored = evaluate_models(split, [NaiveLast], min_train=4, score_observed_only=False)

        assert grid["is_imputed"].any(), "fixture produced no imputed rows to exclude"
        scored_rows = sum(r.n_test for r in scored.results)
        unscored_rows = sum(r.n_test for r in unscored.results)
        assert scored_rows < unscored_rows

    def test_horizon_curve_error_grows_with_distance(self, workspace):
        """Sanity property of any correct evaluation: forecasting further ahead
        is harder. An evaluation not showing this is suspect."""
        from pulseiq.evaluation.horizon import evaluate_horizons
        from pulseiq.features.resample import resample_panel
        from pulseiq.training.forecasting.baseline import NaiveLast

        rng = np.random.default_rng(3)
        frames = []
        for series in range(3):
            price = 10.0 + series
            values = []
            for _ in range(60):
                price *= 1 + rng.normal(0, 0.02)
                values.append(round(price, 2))
            frames.append(
                pd.DataFrame(
                    {
                        "product_name": f"P{series}",
                        "observed_on": pd.date_range("2021-01-01", periods=60, freq="MS"),
                        "selling_price": values,
                    }
                )
            )

        grid = resample_panel(pd.concat(frames, ignore_index=True), min_observed=12)
        report = evaluate_horizons(grid, [NaiveLast], horizons=(1, 12), n_splits=2)
        curve = report.curve().set_index("horizon")["mae"]
        assert curve[12] > curve[1]


# --- serving ----------------------------------------------------------------


class TestServingPipeline:
    def test_ingested_data_is_immediately_forecastable_over_http(
        self, workspace, open_prices_parquet
    ):
        """The complete journey: Parquet on disk to JSON over HTTP."""
        from pulseiq.api.main import create_app
        from pulseiq.ingestion.seed_open_prices import read_open_prices, transform_open_prices
        from pulseiq.ingestion.validation import validate_price_snapshots
        from pulseiq.storage.relational import get_engine, init_db, session_scope
        from pulseiq.storage.repository import save_price_snapshots

        rows = transform_open_prices(read_open_prices(open_prices_parquet), min_observations=8)
        records, _ = validate_price_snapshots(rows)
        engine = get_engine()
        init_db(engine)
        with session_scope(engine) as session:
            save_price_snapshots(session, records)

        with TestClient(create_app()) as client:
            products = client.get("/forecast/products", params={"min_observations": 8}).json()
            assert products, "ingested data did not surface through the API"

            name = products[0]["product_name"]
            response = client.post("/forecast", json={"product_name": name, "horizon": 3})
            assert response.status_code == 200

            payload = response.json()
            assert len(payload["forecast"]) == 3
            assert payload["last_observed_price"] > 0
            # The measured finding must travel with the prediction (D-019).
            assert "no model beats" in payload["baseline_note"].lower()

    def test_cache_returns_identical_content_not_merely_a_flag(
        self, workspace, open_prices_parquet
    ):
        """A cache that returns a different answer on the second call is worse
        than no cache."""
        from pulseiq.api.main import create_app
        from pulseiq.ingestion.seed_open_prices import read_open_prices, transform_open_prices
        from pulseiq.ingestion.validation import validate_price_snapshots
        from pulseiq.storage.relational import get_engine, init_db, session_scope
        from pulseiq.storage.repository import save_price_snapshots

        rows = transform_open_prices(read_open_prices(open_prices_parquet), min_observations=8)
        records, _ = validate_price_snapshots(rows)
        engine = get_engine()
        init_db(engine)
        with session_scope(engine) as session:
            save_price_snapshots(session, records)

        with TestClient(create_app()) as client:
            name = client.get("/forecast/products").json()[0]["product_name"]
            body = {"product_name": name, "horizon": 3, "model": "naive_last"}

            first = client.post("/forecast", json=body).json()
            second = client.post("/forecast", json=body).json()

            assert first["cached"] is False
            assert second["cached"] is True
            assert first["forecast"] == second["forecast"]
            assert first["last_observed_price"] == second["last_observed_price"]

    def test_health_reports_degraded_rather_than_dead(self, workspace):
        """Forecasting works without Mongo, Redis, an LLM key or the adapter.
        Reporting a hard failure would remove the service from a load balancer
        for something it can operate without."""
        from pulseiq.api.main import create_app

        with TestClient(create_app()) as client:
            payload = client.get("/health").json()

        assert payload["status"] in {"ok", "degraded"}
        assert {c["name"] for c in payload["components"]} == {
            "database",
            "cache",
            "sentiment_model",
            "llm",
        }

    def test_api_survives_an_empty_database(self, workspace):
        """A fresh clone with no ingested data must not produce 500s."""
        from pulseiq.api.main import create_app

        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/forecast/products").json() == []
            assert client.post("/forecast", json={"product_name": "anything"}).status_code == 404


# --- evaluation gate --------------------------------------------------------


class TestGateIntegration:
    def test_gate_reads_reports_the_training_run_actually_writes(self, workspace):
        """The gate and the training code must agree on file names and metric
        keys. A silent mismatch means the gate skips every check and reports
        success forever."""
        from pulseiq.evaluation.regression_gate import Status, run_gate

        reports = workspace / "reports"
        (reports / "sentiment_finetuned.json").write_text(
            json.dumps(
                {
                    "accuracy": 0.9413,
                    "macro_f1": 0.9413,
                    "recall": 0.9333,
                    "precision": 0.9485,
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {"model": "naive_last", "horizon": h, "mae": m}
                for h, m in [(1, 0.0), (3, 0.010), (6, 0.033), (12, 0.042)]
            ]
        ).to_csv(reports / "horizon_curve.csv", index=False)

        report, exit_code = run_gate(reports_dir=reports)

        assert exit_code == 0
        assert not report.failed
        assert not report.skipped, (
            f"gate skipped checks it should have run: "
            f"{[c.name for c in report.skipped]} -- file names or metric keys "
            f"have drifted between the training code and thresholds.yaml"
        )
        assert all(c.status is Status.PASS for c in report.checks)

    def test_gate_blocks_a_regression(self, workspace):
        from pulseiq.evaluation.regression_gate import run_gate

        reports = workspace / "reports"
        (reports / "sentiment_finetuned.json").write_text(
            json.dumps({"accuracy": 0.80, "macro_f1": 0.80, "recall": 0.75}), encoding="utf-8"
        )

        _, exit_code = run_gate(reports_dir=reports)
        assert exit_code == 1
