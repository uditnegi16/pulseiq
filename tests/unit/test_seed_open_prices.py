"""Tests for the Open Prices loader.

No network. A synthetic frame matching the published Open Prices schema stands
in for the real Parquet export, so the transform logic is covered offline.
"""

from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from pulseiq.ingestion.seed_open_prices import (
    ATTRIBUTION,
    SOURCE,
    USED_COLUMNS,
    _as_float,
    build_series_id,
    compute_discount,
    summarise,
    transform_open_prices,
)

TODAY = date.today()


def open_prices_frame(n_series=3, n_obs=10, currency="EUR", *, decimal=True):
    """Build a frame with the real Open Prices column names and dtypes.

    Monetary columns default to `decimal.Decimal`, which is what pandas actually
    yields when reading the Parquet DECIMAL columns. An earlier float-based
    fixture let a `float / Decimal` TypeError reach production data.
    """
    money = (lambda v: Decimal(str(round(v, 2)))) if decimal else float
    rows = []
    for s in range(n_series):
        for i in range(n_obs):
            discounted = i % 4 == 0
            base = 10.0 + s
            rows.append(
                {
                    "product_code": f"300000000{s:03d}",
                    "price": money(base * 0.8) if discounted else money(base),
                    "price_is_discounted": discounted,
                    "price_without_discount": money(base) if discounted else None,
                    "currency": currency,
                    "date": TODAY - timedelta(days=n_obs - i),
                    "location_id": 100 + s,
                }
            )
    return pd.DataFrame(rows, columns=USED_COLUMNS)


class TestComputeDiscount:
    def test_not_discounted_is_zero(self):
        assert compute_discount(10.0, None, False) == 0.0

    def test_real_discount(self):
        assert compute_discount(8.0, 10.0, True) == 20.0

    def test_rounds_to_two_places(self):
        assert compute_discount(6.666, 10.0, True) == 33.34

    @pytest.mark.parametrize("without", [None, np.nan, 0.0, -5.0])
    def test_missing_or_invalid_reference_is_zero_not_invented(self, without):
        """Better a truthful 0 than a fabricated percentage in the target."""
        assert compute_discount(8.0, without, True) == 0.0

    def test_price_above_reference_is_zero(self):
        assert compute_discount(12.0, 10.0, True) == 0.0


class TestBuildSeriesId:
    def test_combines_barcode_and_location(self):
        assert build_series_id("3000000000123", 42) == "3000000000123@42"

    @pytest.mark.parametrize("code", [None, np.nan, "", "  ", "None"])
    def test_rejects_missing_barcode(self, code):
        """Loose produce (fruit, veg) has no barcode and no stable identity."""
        assert build_series_id(code, 42) is None

    def test_rejects_missing_location(self):
        assert build_series_id("3000000000123", None) is None

    def test_same_barcode_different_shops_are_different_series(self):
        assert build_series_id("300", 1) != build_series_id("300", 2)


class TestTransformOpenPrices:
    def test_happy_path(self):
        rows = transform_open_prices(open_prices_frame(), min_observations=5)
        assert len(rows) == 30
        assert {r["source"] for r in rows} == {SOURCE}

    def test_empty_frame(self):
        assert transform_open_prices(pd.DataFrame(columns=USED_COLUMNS)) == []

    def test_output_shape_matches_validation_layer(self):
        """The contract that matters: this must flow into the existing pipeline."""
        from pulseiq.ingestion.validation import validate_price_snapshots

        rows = transform_open_prices(open_prices_frame(), min_observations=5)
        _, report = validate_price_snapshots(rows)
        assert report.valid == len(rows)
        assert report.rejected == 0

    def test_ground_truth_discount_is_preserved(self):
        rows = transform_open_prices(open_prices_frame(), min_observations=5)
        discounted = [r for r in rows if r["discount"] > 0]
        assert discounted
        assert all(r["discount"] == pytest.approx(20.0) for r in discounted)

    def test_undiscounted_rows_have_no_original_price(self):
        rows = transform_open_prices(open_prices_frame(), min_observations=5)
        plain = [r for r in rows if r["discount"] == 0]
        assert all(r["original_price"] is None for r in plain)

    def test_currency_filter(self):
        frame = pd.concat(
            [
                open_prices_frame(n_series=2, currency="EUR"),
                open_prices_frame(n_series=2, currency="USD"),
            ],
            ignore_index=True,
        )
        rows = transform_open_prices(frame, currency="EUR", min_observations=5)
        assert len(rows) == 20

    def test_currency_filter_can_be_disabled(self):
        frame = pd.concat(
            [
                open_prices_frame(n_series=1, currency="EUR"),
                open_prices_frame(n_series=1, currency="USD"),
            ],
            ignore_index=True,
        )
        rows = transform_open_prices(frame, currency=None, min_observations=5)
        assert len(rows) == 20

    def test_drops_series_below_min_observations(self):
        """A barcode seen 3 times cannot be forecast."""
        short = open_prices_frame(n_series=1, n_obs=3)
        long = open_prices_frame(n_series=1, n_obs=20)
        long["product_code"] = "999999999999"
        rows = transform_open_prices(
            pd.concat([short, long], ignore_index=True), min_observations=8
        )
        assert {r["product_name"] for r in rows} == {"999999999999@100"}

    def test_returns_empty_when_nothing_qualifies(self):
        rows = transform_open_prices(open_prices_frame(n_obs=2), min_observations=8)
        assert rows == []

    def test_max_series_caps_output(self):
        rows = transform_open_prices(
            open_prices_frame(n_series=5), min_observations=5, max_series=2
        )
        assert len({r["product_name"] for r in rows}) == 2

    def test_rows_without_barcode_are_dropped(self):
        frame = open_prices_frame(n_series=2)
        frame.loc[frame.index[:5], "product_code"] = None
        rows = transform_open_prices(frame, min_observations=1)
        assert len(rows) == 15

    def test_drops_unparseable_prices(self):
        frame = open_prices_frame(n_series=1, n_obs=12)
        frame.loc[frame.index[:2], "price"] = None
        rows = transform_open_prices(frame, min_observations=5)
        assert len(rows) == 10

    def test_drops_future_dates(self):
        """Guarded here so schema validation isn't spammed with rejections."""
        frame = open_prices_frame(n_series=1, n_obs=12)
        frame.loc[frame.index[:3], "date"] = TODAY + timedelta(days=30)
        rows = transform_open_prices(frame, min_observations=5)
        assert len(rows) == 9
        assert all(r["date"] <= TODAY.isoformat() for r in rows)

    def test_output_is_chronological_per_series(self):
        rows = transform_open_prices(open_prices_frame(n_series=2), min_observations=5)
        frame = pd.DataFrame(rows)
        for _, group in frame.groupby("product_name"):
            assert list(group["date"]) == sorted(group["date"])


class TestEndToEndIntoStorage:
    def test_open_prices_rows_persist_and_reload(self):
        """Full path: Parquet shape -> transform -> validate -> DB -> DataFrame."""
        from pulseiq.ingestion.validation import validate_price_snapshots
        from pulseiq.storage.relational import get_engine, init_db, session_scope
        from pulseiq.storage.repository import load_price_history, save_price_snapshots

        rows = transform_open_prices(open_prices_frame(n_series=2, n_obs=15), min_observations=8)
        records, report = validate_price_snapshots(rows)
        assert report.valid > 0

        engine = get_engine("sqlite://")
        init_db(engine)
        with session_scope(engine) as session:
            save_price_snapshots(session, records)
            frame = load_price_history(session, min_observations=8)

        assert len(frame) == len(records)
        assert set(frame["source"]) == {SOURCE}
        # Provenance survives the round trip -- this is what keeps ODbL
        # share-alike contained to filterable rows.
        assert "open_prices" in set(frame["source"])


class TestAttribution:
    def test_attribution_names_source_and_licence(self):
        assert "Open Prices" in ATTRIBUTION
        assert "ODbL" in ATTRIBUTION


class TestSummarise:
    def test_empty(self):
        assert summarise([]) == "no rows"

    def test_reports_counts_and_discount_rate(self):
        rows = transform_open_prices(open_prices_frame(), min_observations=5)
        text = summarise(rows)
        assert "30 rows" in text
        assert "3 series" in text
        assert "discounted" in text


class TestDecimalHandling:
    """Open Prices ships monetary columns as Parquet DECIMAL, not float.

    `float / Decimal` raises TypeError, so these guard the real dtype rather
    than the convenient one.
    """

    def test_compute_discount_accepts_decimals(self):
        assert compute_discount(Decimal("8.00"), Decimal("10.00"), True) == 20.0

    def test_compute_discount_mixed_decimal_and_float(self):
        assert compute_discount(8.0, Decimal("10.00"), True) == 20.0
        assert compute_discount(Decimal("8.00"), 10.0, True) == 20.0

    def test_as_float_handles_every_shape_parquet_produces(self):
        assert _as_float(Decimal("3.50")) == 3.5
        assert _as_float(2.5) == 2.5
        assert _as_float("4.25") == 4.25
        assert _as_float(None) is None
        assert _as_float(np.nan) is None
        assert _as_float("not a price") is None

    def test_transform_survives_decimal_columns(self):
        rows = transform_open_prices(open_prices_frame(decimal=True), min_observations=5)
        assert len(rows) == 30
        assert all(isinstance(r["price"], float) for r in rows)
        assert all(
            r["original_price"] is None or isinstance(r["original_price"], float) for r in rows
        )

    def test_decimal_and_float_paths_agree(self):
        """The fixture change must not alter results, only dtypes."""
        dec = transform_open_prices(open_prices_frame(decimal=True), min_observations=5)
        flt = transform_open_prices(open_prices_frame(decimal=False), min_observations=5)
        assert [r["discount"] for r in dec] == [r["discount"] for r in flt]
        assert [r["price"] for r in dec] == pytest.approx([r["price"] for r in flt])
