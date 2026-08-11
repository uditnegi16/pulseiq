"""Tests for resampling, metrics and baseline forecasters.

The resampling tests exist mostly to protect one invariant: imputed rows must
never reach a scoring function. That mistake makes every model look better and
the dumbest model look best.
"""

import numpy as np
import pandas as pd
import pytest

from pulseiq.evaluation.metrics import (
    EmptyEvaluationError,
    ForecastMetrics,
    evaluate_forecast,
    mae,
    mape,
    mase,
    smape,
)
from pulseiq.features.resample import (
    observed_only,
    resample_panel,
    resample_series,
)
from pulseiq.training.forecasting.baseline import (
    Drift,
    Mean,
    MovingAverage,
    NaiveLast,
    NotFittedError,
    SeasonalNaive,
    default_baselines,
)


def irregular_series(name="Widget", dates=None, prices=None):
    dates = dates or ["2026-01-05", "2026-01-20", "2026-03-02", "2026-06-15"]
    prices = prices or [10.0, 11.0, 12.0, 13.0]
    return pd.DataFrame(
        {
            "product_name": name,
            "observed_on": pd.to_datetime(dates),
            "selling_price": prices,
        }
    )


def monthly_series(name="Widget", n=24, start="2024-01-01", price=10.0):
    return pd.DataFrame(
        {
            "product_name": name,
            "observed_on": pd.date_range(start, periods=n, freq="MS"),
            "selling_price": [price + i * 0.1 for i in range(n)],
        }
    )


class TestResampleSeries:
    def test_puts_observations_on_a_monthly_grid(self):
        out = resample_series(irregular_series())
        assert (out["observed_on"].dt.day == 1).all()

    def test_marks_gap_months_as_imputed(self):
        out = resample_series(irregular_series())
        by_month = dict(
            zip(
                out["observed_on"].dt.strftime("%Y-%m"),
                out["is_imputed"],
                strict=False,
            )
        )
        assert by_month["2026-01"] is np.False_ or by_month["2026-01"] is False
        assert by_month["2026-02"]  # nothing observed in February
        assert not by_month["2026-03"]

    def test_multiple_observations_in_a_month_are_aggregated(self):
        """Two January prices collapse to one grid point."""
        out = resample_series(irregular_series())
        january = out[out["observed_on"] == pd.Timestamp("2026-01-01")]
        assert len(january) == 1
        assert january["selling_price"].iloc[0] == 10.5  # median of 10 and 11

    def test_median_resists_a_typo(self):
        """Crowdsourced data contains mistyped prices; a mean would follow them."""
        frame = irregular_series(
            dates=["2026-01-05", "2026-01-10", "2026-01-15"],
            prices=[10.0, 10.0, 999.0],
        )
        out = resample_series(frame, aggregate="median")
        assert out["selling_price"].iloc[0] == 10.0

    def test_fill_limit_leaves_long_gaps_unfilled(self):
        """A price is not assumed to hold for a year without evidence."""
        frame = irregular_series(dates=["2024-01-05", "2026-01-05"], prices=[10.0, 20.0])
        out = resample_series(frame, max_fill_periods=3)
        assert len(out) < 25

    def test_empty_input(self):
        out = resample_series(
            pd.DataFrame(columns=["product_name", "observed_on", "selling_price"])
        )
        assert out.empty
        assert "is_imputed" in out.columns

    def test_preserves_product_name(self):
        out = resample_series(irregular_series(name="Specific"))
        assert set(out["product_name"]) == {"Specific"}


class TestResamplePanel:
    def test_resamples_each_product_independently(self):
        frame = pd.concat([monthly_series("A"), monthly_series("B")], ignore_index=True)
        out = resample_panel(frame, min_observed=8)
        assert set(out["product_name"]) == {"A", "B"}

    def test_drops_series_with_too_few_real_observations(self):
        """A series can have 30 grid rows and only 4 facts in it."""
        sparse = irregular_series("Sparse", dates=["2024-01-05", "2024-02-05"], prices=[1.0, 2.0])
        out = resample_panel(
            pd.concat([monthly_series("Dense"), sparse], ignore_index=True), min_observed=8
        )
        assert set(out["product_name"]) == {"Dense"}

    def test_returns_empty_frame_when_nothing_qualifies(self):
        out = resample_panel(irregular_series(), min_observed=50)
        assert out.empty
        assert "is_imputed" in out.columns

    def test_output_is_sorted_by_product_then_date(self):
        frame = pd.concat([monthly_series("B"), monthly_series("A")], ignore_index=True)
        out = resample_panel(frame, min_observed=8)
        assert out.equals(out.sort_values(["product_name", "observed_on"]).reset_index(drop=True))

    def test_no_series_bleed_between_products(self):
        a = monthly_series("A", n=12, price=10.0)
        b = monthly_series("B", n=12, price=500.0)
        out = resample_panel(pd.concat([a, b], ignore_index=True), min_observed=8)
        assert out[out["product_name"] == "A"]["selling_price"].max() < 100


class TestObservedOnly:
    def test_removes_imputed_rows(self):
        out = resample_panel(monthly_series(n=24), min_observed=8)
        out.loc[out.index[:3], "is_imputed"] = True
        assert len(observed_only(out)) == len(out) - 3

    def test_passthrough_when_column_absent(self):
        frame = pd.DataFrame({"a": [1, 2]})
        assert len(observed_only(frame)) == 2

    def test_scoring_on_imputed_rows_flatters_the_naive_model(self):
        """The reason observed_only() exists, demonstrated.

        Forward-filled rows are copies of their predecessor, so NaiveLast
        predicts them exactly and its error collapses.
        """
        frame = irregular_series(dates=["2026-01-05", "2026-05-10"], prices=[10.0, 40.0])
        grid = resample_series(frame, max_fill_periods=6)
        with_imputed = grid["selling_price"].to_numpy()
        naive_pred = np.roll(with_imputed, 1)[1:]
        actual = with_imputed[1:]

        error_all = mae(actual, naive_pred)
        clean = observed_only(grid)
        assert len(clean) < len(grid)
        assert error_all < 30.0  # flattered by the repeated values


class TestForecastMetrics:
    def test_perfect_prediction(self):
        m = evaluate_forecast([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert m.mae == 0.0
        assert m.rmse == 0.0
        assert m.mape == 0.0

    def test_known_values(self):
        m = evaluate_forecast([10.0, 20.0], [12.0, 18.0])
        assert m.mae == 2.0
        assert m.rmse == 2.0
        assert m.mape == pytest.approx(15.0)
        assert m.n == 2

    def test_rmse_exceeds_mae_when_errors_are_uneven(self):
        m = evaluate_forecast([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 10.0])
        assert m.rmse > m.mae

    def test_empty_input_raises_rather_than_returning_zero(self):
        """A silent 0.0 is how a broken evaluation gets reported as perfect."""
        with pytest.raises(EmptyEvaluationError):
            evaluate_forecast([], [])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate_forecast([1.0, 2.0], [1.0])

    def test_nan_pairs_are_dropped(self):
        m = evaluate_forecast([1.0, np.nan, 3.0], [1.0, 2.0, 3.0])
        assert m.n == 2
        assert m.mae == 0.0

    def test_all_nan_raises(self):
        with pytest.raises(EmptyEvaluationError):
            evaluate_forecast([np.nan, np.nan], [1.0, 2.0])

    def test_metrics_serialise(self):
        d = evaluate_forecast([1.0, 2.0], [1.1, 2.1]).as_dict()
        assert set(d) == {"mae", "rmse", "mape", "smape", "n"}

    def test_str_is_readable(self):
        assert "MAE=" in str(evaluate_forecast([1.0, 2.0], [1.0, 2.0]))


class TestMape:
    def test_returns_none_when_all_actuals_are_zero(self):
        """Not a huge number. A division artefact is not a metric."""
        assert mape([0.0, 0.0], [1.0, 2.0]) is None

    def test_skips_zero_actuals_but_uses_the_rest(self):
        assert mape([0.0, 10.0], [1.0, 11.0]) == pytest.approx(10.0)

    def test_evaluate_forecast_carries_none_through(self):
        assert evaluate_forecast([0.0, 0.0], [1.0, 1.0]).mape is None


class TestSmape:
    def test_bounded_at_200_percent(self):
        assert smape([1.0], [-1.0]) <= 200.0

    def test_defined_when_actual_is_zero(self):
        assert smape([0.0], [5.0]) == pytest.approx(200.0)

    def test_symmetric(self):
        assert smape([10.0], [12.0]) == pytest.approx(smape([12.0], [10.0]))


class TestMase:
    def test_below_one_beats_naive(self):
        train = [1.0, 3.0, 5.0, 7.0, 9.0]  # naive error 2.0 per step
        assert mase([10.0], [10.5], train) == pytest.approx(0.25)

    def test_above_one_is_worse_than_doing_nothing(self):
        assert mase([10.0], [20.0], [1.0, 3.0, 5.0, 7.0]) > 1.0

    def test_flat_training_series_returns_none(self):
        assert mase([10.0], [11.0], [5.0, 5.0, 5.0]) is None

    def test_too_short_training_series_returns_none(self):
        assert mase([10.0], [11.0], [5.0]) is None


class TestBaselines:
    def test_naive_last_repeats_final_value(self):
        assert NaiveLast().fit_predict([1.0, 2.0, 7.0], 3).tolist() == [7.0, 7.0, 7.0]

    def test_moving_average_uses_only_the_window(self):
        assert MovingAverage(3).fit_predict([1.0, 2.0, 3.0, 4.0, 5.0], 1)[0] == 4.0

    def test_mean_uses_full_history(self):
        assert Mean().fit_predict([1.0, 2.0, 3.0], 1)[0] == 2.0

    def test_drift_extrapolates_the_trend(self):
        assert Drift().fit_predict([10.0, 11.0, 12.0], 2).tolist() == [13.0, 14.0]

    def test_drift_on_flat_series_is_flat(self):
        assert Drift().fit_predict([5.0, 5.0, 5.0], 2).tolist() == [5.0, 5.0]

    def test_seasonal_naive_repeats_the_season(self):
        y = [1.0, 2.0, 3.0, 4.0]
        assert SeasonalNaive(period=4).fit_predict(y, 4).tolist() == y

    def test_seasonal_naive_falls_back_on_short_history(self):
        """Most series here are under three years; failing would drop them all."""
        assert SeasonalNaive(period=12).fit_predict([1.0, 2.0], 2).tolist() == [2.0, 2.0]

    @pytest.mark.parametrize(
        "model_factory", [NaiveLast, Mean, Drift, MovingAverage, SeasonalNaive]
    )
    def test_predict_before_fit_raises(self, model_factory):
        with pytest.raises(NotFittedError):
            model_factory().predict(1)

    def test_fit_on_empty_series_raises(self):
        with pytest.raises(ValueError, match="empty series"):
            NaiveLast().fit([])

    def test_nan_values_are_dropped_before_fitting(self):
        assert NaiveLast().fit_predict([1.0, np.nan, 5.0], 1)[0] == 5.0

    def test_horizon_must_be_positive(self):
        with pytest.raises(ValueError, match="horizon"):
            NaiveLast().fit([1.0]).predict(0)

    @pytest.mark.parametrize("horizon", [1, 3, 12])
    def test_output_length_matches_horizon(self, horizon):
        for model in default_baselines():
            assert len(model.fit_predict([1.0, 2.0, 3.0, 4.0, 5.0], horizon)) == horizon

    def test_default_baselines_are_fresh_instances(self):
        """Shared state would leak one product's history into the next."""
        first = default_baselines()[0].fit([1.0, 2.0, 99.0])
        second = default_baselines()[0]
        with pytest.raises(NotFittedError):
            second.predict(1)
        assert first.predict(1)[0] == 99.0

    def test_all_baselines_have_distinct_names(self):
        names = [m.name for m in default_baselines()]
        assert len(names) == len(set(names))


class TestEndToEndBaselineEvaluation:
    def test_resample_split_score_pipeline(self):
        """Irregular observations -> monthly grid -> chronological split ->
        scored on observed rows only."""
        from pulseiq.features.splits import split_by_fraction

        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        keep = sorted(rng.choice(200, 60, replace=False))
        frame = pd.DataFrame(
            {
                "product_name": "W",
                "observed_on": dates[keep],
                "selling_price": 20 + np.cumsum(rng.normal(0, 0.3, 60)),
            }
        )

        grid = resample_panel(frame, min_observed=6)
        assert not grid.empty

        split = split_by_fraction(grid, 0.25)
        split.assert_no_leakage()

        test_observed = observed_only(split.test)
        if len(test_observed) == 0:
            pytest.skip("no observed rows in the test window")

        results: dict[str, ForecastMetrics] = {}
        for model in default_baselines():
            preds = model.fit_predict(split.train["selling_price"], len(test_observed))
            results[model.name] = evaluate_forecast(test_observed["selling_price"], preds)

        assert len(results) == 5
        assert all(m.mae >= 0 for m in results.values())
        assert all(m.n == len(test_observed) for m in results.values())
