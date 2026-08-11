"""Tests for splitting and feature engineering.

Most of these are leakage tests. They exist because leakage is silent: it makes
metrics look better, so nothing fails and nobody notices until an interviewer
asks how the split was done.
"""

import numpy as np
import pandas as pd
import pytest

from pulseiq.features.build_features import (
    add_calendar_features,
    add_lag_features,
    add_price_change_features,
    add_rolling_features,
    build_feature_frame,
    define_discount_target,
    feature_columns,
)
from pulseiq.features.splits import (
    InsufficientDataError,
    rolling_origin_splits,
    split_by_date,
    split_by_fraction,
    split_per_product,
)


def series(name="Widget", n=30, start="2026-01-01", prices=None):
    dates = pd.date_range(start, periods=n, freq="D")
    if prices is None:
        prices = [1000.0 - i * 5 for i in range(n)]
    return pd.DataFrame(
        {
            "product_name": name,
            "observed_on": dates,
            "selling_price": prices[:n],
        }
    )


def multi_product(n=30):
    return pd.concat(
        [series("A", n=n), series("B", n=n, prices=[500.0 + i for i in range(n)])],
        ignore_index=True,
    )


class TestSplitByDate:
    def test_train_strictly_before_cutoff(self):
        split = split_by_date(series(n=20), "2026-01-15")
        assert split.train["observed_on"].max() < pd.Timestamp("2026-01-15")
        assert split.test["observed_on"].min() >= pd.Timestamp("2026-01-15")
        split.assert_no_leakage()

    def test_cutoff_outside_range_raises(self):
        with pytest.raises(InsufficientDataError, match="yields train"):
            split_by_date(series(n=20), "2030-01-01")

    def test_cutoff_before_all_data_raises(self):
        with pytest.raises(InsufficientDataError):
            split_by_date(series(n=20), "2020-01-01")


class TestSplitByFraction:
    def test_no_leakage(self):
        split_by_fraction(series(n=50), 0.2).assert_no_leakage()

    def test_test_set_is_the_later_data(self):
        split = split_by_fraction(series(n=50), 0.2)
        assert split.train["observed_on"].max() < split.test["observed_on"].min()

    def test_rejects_invalid_test_size(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="test_size"):
                split_by_fraction(series(n=20), bad)

    def test_single_row_raises(self):
        with pytest.raises(InsufficientDataError):
            split_by_fraction(series(n=1))

    def test_handles_identical_timestamps(self):
        """Degenerate input must still yield a usable partition."""
        frame = pd.DataFrame(
            {
                "product_name": "W",
                "observed_on": [pd.Timestamp("2026-01-01")] * 10,
                "selling_price": range(10),
            }
        )
        split = split_by_fraction(frame, 0.3)
        assert len(split.train) > 0
        assert len(split.test) > 0

    def test_unsorted_input_is_sorted_first(self):
        frame = series(n=30).sample(frac=1.0, random_state=0)
        split_by_fraction(frame, 0.2).assert_no_leakage()


class TestSplitPerProduct:
    def test_no_leakage_within_each_product(self):
        split_per_product(multi_product(n=30), 0.2).assert_no_leakage()

    def test_both_products_present_in_both_partitions(self):
        split = split_per_product(multi_product(n=30), 0.2)
        assert set(split.train["product_name"]) == {"A", "B"}
        assert set(split.test["product_name"]) == {"A", "B"}

    def test_drops_products_below_min_observations(self):
        frame = pd.concat([series("Long", n=30), series("Short", n=3)], ignore_index=True)
        split = split_per_product(frame, 0.2, min_observations=8)
        assert set(split.train["product_name"]) == {"Long"}

    def test_raises_when_nothing_qualifies(self):
        """Your 45-row CSV across several products will hit exactly this."""
        frame = pd.concat([series("A", n=3), series("B", n=4)], ignore_index=True)
        with pytest.raises(InsufficientDataError, match="Collect more history"):
            split_per_product(frame, 0.2, min_observations=8)

    def test_falls_back_when_no_group_column(self):
        frame = series(n=30).drop(columns=["product_name"])
        split = split_per_product(frame, 0.2)
        assert len(split.train) > 0


class TestRollingOriginSplits:
    def test_produces_requested_folds(self):
        splits = rolling_origin_splits(series(n=60), n_splits=3, horizon_days=7)
        assert len(splits) == 3

    def test_every_fold_is_leak_free(self):
        for split in rolling_origin_splits(series(n=60), n_splits=3, horizon_days=7):
            split.assert_no_leakage()

    def test_training_window_expands(self):
        splits = rolling_origin_splits(series(n=60), n_splits=3, horizon_days=7)
        sizes = [len(s.train) for s in splits]
        assert sizes == sorted(sizes)
        assert sizes[0] < sizes[-1]

    def test_short_series_raises_with_actionable_message(self):
        with pytest.raises(InsufficientDataError, match="need at least"):
            rolling_origin_splits(series(n=10), n_splits=3, horizon_days=7)


class TestDefineDiscountTarget:
    def test_reference_price_excludes_today(self):
        """The core leakage guard: today's price must not set today's reference.

        Prices rise monotonically, so if today were included the reference would
        equal today's price and every discount would be exactly 0.
        """
        frame = series(n=10, prices=[100.0 + i * 10 for i in range(10)])
        out = define_discount_target(frame, min_periods=1)
        assert out["reference_price"].iloc[1] == 100.0
        assert out["reference_price"].iloc[2] == 110.0

    def test_first_row_reference_is_nan(self):
        out = define_discount_target(series(n=10), min_periods=1)
        assert pd.isna(out["reference_price"].iloc[0])

    def test_discount_is_never_negative(self):
        frame = series(n=20, prices=[100.0 + i * 10 for i in range(20)])
        out = define_discount_target(frame, min_periods=1)
        assert (out["discount_from_reference"].dropna() >= 0).all()

    def test_computes_expected_discount(self):
        frame = series(n=5, prices=[1000.0, 1000.0, 1000.0, 800.0, 800.0])
        out = define_discount_target(frame, min_periods=1)
        assert out["discount_from_reference"].iloc[3] == pytest.approx(20.0)


class TestAddLagFeatures:
    def test_lags_are_previous_values(self):
        out = add_lag_features(series(n=10, prices=list(range(10))), lags=(1, 2))
        assert out["selling_price_lag1"].iloc[3] == 2
        assert out["selling_price_lag2"].iloc[3] == 1

    def test_lags_do_not_cross_product_boundaries(self):
        """An ungrouped shift pulls product A's last row into product B's first."""
        out = add_lag_features(multi_product(n=10), lags=(1,))
        first_b = out[out["product_name"] == "B"].iloc[0]
        assert pd.isna(first_b["selling_price_lag1"])

    def test_leading_rows_are_nan(self):
        out = add_lag_features(series(n=10), lags=(3,))
        assert out["selling_price_lag3"].iloc[:3].isna().all()


class TestAddRollingFeatures:
    def test_rolling_mean_excludes_current_row(self):
        """Without the shift, a rolling mean of the target contains the answer.
        This is the most convincing leaky feature you can accidentally build."""
        frame = series(n=6, prices=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        out = add_rolling_features(frame, windows=(3,))
        # Row 3 sees rows 0-2 only: mean(10, 20, 30) = 20
        assert out["selling_price_roll3_mean"].iloc[3] == pytest.approx(20.0)

    def test_first_row_has_no_rolling_value(self):
        out = add_rolling_features(series(n=10), windows=(3,))
        assert pd.isna(out["selling_price_roll3_mean"].iloc[0])

    def test_rolling_does_not_cross_products(self):
        out = add_rolling_features(multi_product(n=10), windows=(3,))
        first_b = out[out["product_name"] == "B"].iloc[0]
        assert pd.isna(first_b["selling_price_roll3_mean"])


class TestAddCalendarFeatures:
    def test_weekend_flag(self):
        frame = pd.DataFrame(
            {
                "product_name": "W",
                "observed_on": pd.to_datetime(["2026-01-03", "2026-01-05"]),
                "selling_price": [1.0, 2.0],
            }
        )
        out = add_calendar_features(frame)
        assert out["is_weekend"].tolist() == [1, 0]  # Sat, Mon

    def test_month_boundaries(self):
        frame = pd.DataFrame(
            {
                "product_name": "W",
                "observed_on": pd.to_datetime(["2026-01-01", "2026-01-31"]),
                "selling_price": [1.0, 2.0],
            }
        )
        out = add_calendar_features(frame)
        assert out["is_month_start"].tolist() == [1, 0]
        assert out["is_month_end"].tolist() == [0, 1]


class TestAddPriceChangeFeatures:
    def test_days_since_change_resets_on_change(self):
        frame = series(n=6, prices=[100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
        out = add_price_change_features(frame)
        assert out["days_since_price_change"].tolist() == [0, 1, 2, 0, 1, 2]

    def test_diff_is_within_product(self):
        out = add_price_change_features(multi_product(n=10))
        first_b = out[out["product_name"] == "B"].iloc[0]
        assert pd.isna(first_b["price_diff_1"])


class TestBuildFeatureFrame:
    def test_produces_complete_rows(self):
        out = build_feature_frame(series(n=40))
        assert not out.empty
        lag_cols = [c for c in out.columns if "_lag" in c]
        assert not out[lag_cols].isna().any().any()

    def test_drops_early_incomplete_history(self):
        frame = series(n=40)
        out = build_feature_frame(frame)
        assert len(out) < len(frame)

    def test_empty_input_returns_empty(self):
        assert build_feature_frame(pd.DataFrame()).empty

    def test_handles_multiple_products(self):
        out = build_feature_frame(multi_product(n=40))
        assert set(out["product_name"]) == {"A", "B"}


class TestFeatureColumns:
    def test_excludes_target_and_derived_columns(self):
        """Guard against a leaky column being added later and silently used."""
        out = build_feature_frame(series(n=40))
        cols = feature_columns(out)
        for banned in (
            "selling_price",
            "discount_pct",
            "discount_from_reference",
            "reference_price",
        ):
            assert banned not in cols

    def test_includes_lags_rolls_and_calendar(self):
        cols = feature_columns(build_feature_frame(series(n=40)))
        assert "selling_price_lag1" in cols
        assert "selling_price_roll3_mean" in cols
        assert "day_of_week" in cols

    def test_all_selected_columns_are_numeric(self):
        out = build_feature_frame(series(n=40))
        selected = out[feature_columns(out)]
        assert all(np.issubdtype(dt, np.number) for dt in selected.dtypes)


class TestLeakageEndToEnd:
    def test_full_pipeline_split_is_leak_free(self):
        """Features then split -- the order used in training. Building features
        across the whole frame is safe only because every feature is trailing."""
        featured = build_feature_frame(multi_product(n=60))
        split = split_per_product(featured, 0.2)
        split.assert_no_leakage()
        assert len(split.train) > len(split.test)

    def test_no_test_row_predates_any_train_row_of_same_product(self):
        featured = build_feature_frame(multi_product(n=60))
        split = split_per_product(featured, 0.2)
        for product in split.test["product_name"].unique():
            train_max = split.train.loc[split.train["product_name"] == product, "observed_on"].max()
            test_min = split.test.loc[split.test["product_name"] == product, "observed_on"].min()
            assert train_max < test_min
