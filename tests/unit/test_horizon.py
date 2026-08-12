"""Tests for horizon-curve evaluation."""

import numpy as np
import pandas as pd

from pulseiq.evaluation.horizon import (
    HorizonReport,
    HorizonResult,
    evaluate_horizons,
)
from pulseiq.features.resample import resample_panel
from pulseiq.training.forecasting.baseline import Mean, MovingAverage, NaiveLast


def monthly_panel(n_series=4, n=48, seed=3):
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_series):
        price = 10 + s
        values = []
        for _ in range(n):
            if rng.random() < 0.1:
                price *= rng.uniform(0.9, 1.1)
            values.append(round(price * rng.uniform(0.99, 1.01), 2))
        frames.append(
            pd.DataFrame(
                {
                    "product_name": f"P{s}",
                    "observed_on": pd.date_range("2021-01-01", periods=n, freq="MS"),
                    "selling_price": values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestHorizonReport:
    @staticmethod
    def _report():
        report = HorizonReport()
        for model, horizon, mae in [
            ("naive_last", 1, 0.05),
            ("naive_last", 12, 0.30),
            ("ma3", 1, 0.04),
            ("ma3", 12, 0.50),
        ]:
            report.results.append(HorizonResult(model, "P0", horizon, 0, mae, 3.0, 20, horizon))
        return report

    def test_curve_aggregates_by_model_and_horizon(self):
        curve = self._report().curve()
        assert len(curve) == 4
        assert set(curve.columns) >= {"model", "horizon", "mae", "n"}

    def test_pivot_puts_horizons_in_columns(self):
        table = self._report().pivot()
        assert list(table.columns) == [1, 12]
        # sorted by the shortest horizon, so ma3 (0.04) comes first
        assert list(table.index) == ["ma3", "naive_last"]

    def test_empty_report(self):
        assert HorizonReport().curve().empty
        assert "no results" in HorizonReport().summary()

    def test_skill_vs_naive_is_paired(self):
        skill = self._report().skill_vs_naive()
        h1 = skill[skill["horizon"] == 1].iloc[0]
        h12 = skill[skill["horizon"] == 12].iloc[0]
        assert h1["median_improvement_pct"] > 0  # ma3 better at h=1
        assert h12["median_improvement_pct"] < 0  # worse at h=12

    def test_skill_returns_empty_without_the_baseline(self):
        report = HorizonReport()
        report.results.append(HorizonResult("only_model", "P0", 1, 0, 0.1, 3.0, 20, 1))
        assert report.skill_vs_naive().empty

    def test_summary_shows_win_rate_next_to_improvement(self):
        """A near-zero median with a ~50% win rate means indistinguishable, not
        broken -- the summary has to make that readable."""
        text = self._report().summary()
        assert "win rate" in text
        assert "indistinguishable" in text


class TestEvaluateHorizons:
    def test_produces_results_for_each_horizon(self):
        grid = resample_panel(monthly_panel(), min_observed=12)
        report = evaluate_horizons(grid, [NaiveLast], horizons=(1, 3), n_splits=2)
        assert set(report.to_frame()["horizon"]) == {1, 3}

    def test_error_grows_with_horizon(self):
        """The core property: forecasting further ahead is harder."""
        grid = resample_panel(monthly_panel(n=60), min_observed=12)
        report = evaluate_horizons(grid, [NaiveLast], horizons=(1, 12), n_splits=2)
        curve = report.curve().set_index("horizon")["mae"]
        assert curve[12] > curve[1]

    def test_empty_grid_is_recorded_not_crashed(self):
        report = evaluate_horizons(pd.DataFrame(), [NaiveLast])
        assert report.results == []
        assert "empty_grid" in report.skipped

    def test_series_too_short_for_a_horizon_are_skipped(self):
        grid = resample_panel(monthly_panel(n=20), min_observed=12)
        report = evaluate_horizons(grid, [NaiveLast], horizons=(1, 12), n_splits=3)
        assert any("too_short" in k for k in report.skipped)

    def test_fresh_model_per_fold(self):
        grid = resample_panel(monthly_panel(n_series=2, n=48), min_observed=12)
        report = evaluate_horizons(grid, [NaiveLast], horizons=(1,), n_splits=3)
        assert report.to_frame()["mae"].nunique() > 1

    def test_multiple_models_scored_on_identical_folds(self):
        """Fair comparison requires every model to see the same partitions."""
        grid = resample_panel(monthly_panel(), min_observed=12)
        report = evaluate_horizons(
            grid, [NaiveLast, lambda: MovingAverage(3)], horizons=(1,), n_splits=2
        )
        frame = report.to_frame()
        keys = frame.groupby("model").apply(
            lambda g: set(zip(g["series"], g["fold"], strict=False)), include_groups=False
        )
        assert keys.iloc[0] == keys.iloc[1]

    def test_a_failing_model_does_not_kill_the_run(self):
        class Broken(NaiveLast):
            name = "broken"

            def _fit(self, y):
                raise RuntimeError("boom")

        grid = resample_panel(monthly_panel(), min_observed=12)
        report = evaluate_horizons(grid, [NaiveLast, Broken], horizons=(1,), n_splits=2)
        assert set(report.to_frame()["model"]) == {"naive_last"}
        assert any(k.startswith("model_error") for k in report.skipped)

    def test_ungrouped_frame_is_handled(self):
        grid = resample_panel(monthly_panel(n_series=1, n=48), min_observed=12)
        report = evaluate_horizons(
            grid.drop(columns=["product_name"]), [NaiveLast], horizons=(1,), n_splits=2
        )
        assert report.results

    def test_mean_is_worse_than_naive_at_short_horizon(self):
        """Sanity check on the ranking: predicting the global mean should lose
        badly one step ahead on a level-shifting series."""
        grid = resample_panel(monthly_panel(n=60), min_observed=12)
        report = evaluate_horizons(grid, [NaiveLast, Mean], horizons=(1,), n_splits=2)
        curve = report.curve().set_index("model")["mae"]
        assert curve["mean"] > curve["naive_last"]
