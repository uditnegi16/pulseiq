"""Tests for ARIMA, Prophet, the evaluation harness and the training CLI.

Prophet fits are slow (~0.3s each), so tests that touch it are marked `slow`
and kept to a handful of series. Run the fast suite with:

    pytest tests/unit -m "not slow"
"""

import numpy as np
import pandas as pd
import pytest

from pulseiq.evaluation.harness import (
    HarnessReport,
    SeriesResult,
    evaluate_models,
    paired_comparison,
)
from pulseiq.features.resample import resample_panel
from pulseiq.features.splits import split_per_product
from pulseiq.training.forecasting.arima_model import ARIMAForecaster
from pulseiq.training.forecasting.baseline import Mean, NaiveLast, NotFittedError
from pulseiq.training.forecasting.train_forecast import build_factories


def trending(n=36, slope=0.4, noise=0.5, seed=1):
    rng = np.random.default_rng(seed)
    return 20 + np.arange(n) * slope + rng.normal(0, noise, n)


def panel(n_series=3, n=30, seed=2):
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_series):
        frames.append(
            pd.DataFrame(
                {
                    "product_name": f"P{s}",
                    "observed_on": pd.date_range("2024-01-01", periods=n, freq="MS"),
                    "selling_price": 10 + s + np.cumsum(rng.normal(0, 0.2, n)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestARIMA:
    def test_fits_and_predicts(self):
        model = ARIMAForecaster(auto=True)
        preds = model.fit_predict(trending(), 6)
        assert len(preds) == 6
        assert np.all(np.isfinite(preds))
        assert not model.used_fallback

    def test_selects_an_order(self):
        model = ARIMAForecaster(auto=True).fit(trending())
        assert model.selected_order in ARIMAForecaster().grid

    def test_explicit_order_is_respected(self):
        model = ARIMAForecaster(order=(1, 1, 1), auto=False).fit(trending())
        assert model.selected_order == (1, 1, 1)

    def test_beats_naive_on_a_clear_trend(self):
        """Not a guarantee in general -- on a strong linear trend it should."""
        y = trending(n=40, slope=0.5, noise=0.2)
        train, test = y[:34], y[34:]
        arima_mae = np.mean(np.abs(test - ARIMAForecaster().fit_predict(train, len(test))))
        naive_mae = np.mean(np.abs(test - NaiveLast().fit_predict(train, len(test))))
        assert arima_mae < naive_mae

    def test_short_series_falls_back_visibly(self):
        """A silent fallback would credit ARIMA for a naive forecast."""
        model = ARIMAForecaster(auto=True)
        preds = model.fit_predict([1.0, 2.0, 3.0], 2)
        assert model.used_fallback is True
        assert preds.tolist() == [3.0, 3.0]

    def test_constant_series_does_not_raise(self):
        preds = ARIMAForecaster(auto=True).fit_predict([5.0] * 20, 3)
        assert np.all(np.isfinite(preds))

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            ARIMAForecaster().predict(1)

    def test_horizon_length_respected(self):
        for horizon in (1, 5, 12):
            assert len(ARIMAForecaster().fit_predict(trending(), horizon)) == horizon


@pytest.mark.slow
# Prophet does bare-int timedelta arithmetic internally, which numpy >= 2.5
# deprecates. The project promotes DeprecationWarning to an error, so without
# this scoped ignore Prophet's fit raises and silently falls back to naive.
# Scoped to this class ONLY: the global rule stays strict for our own code,
# which is how it caught a real bug in splits.py.
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestProphet:
    def test_fits_and_predicts(self):
        from pulseiq.training.forecasting.prophet_model import ProphetForecaster

        model = ProphetForecaster()
        preds = model.fit_predict(trending(), 6)
        assert len(preds) == 6
        assert np.all(np.isfinite(preds))
        assert not model.used_fallback

    def test_short_series_falls_back_visibly(self):
        from pulseiq.training.forecasting.prophet_model import ProphetForecaster

        model = ProphetForecaster()
        preds = model.fit_predict([1.0, 2.0, 3.0], 2)
        assert model.used_fallback is True
        assert preds.tolist() == [3.0, 3.0]

    def test_yearly_seasonality_off_on_short_series(self):
        """Prophet will fit an annual cycle to 14 months. That is fitted noise."""
        from pulseiq.training.forecasting.prophet_model import ProphetForecaster

        short = ProphetForecaster().fit(trending(n=14))
        long = ProphetForecaster().fit(trending(n=30))
        assert short.yearly_seasonality_used is False
        assert long.yearly_seasonality_used is True


class TestHarnessReport:
    @staticmethod
    def _report():
        report = HarnessReport()
        for i, (model, mae) in enumerate([("a", 1.0), ("a", 2.0), ("b", 3.0), ("b", 4.0)]):
            report.results.append(
                SeriesResult(
                    model=model,
                    series=f"S{i % 2}",
                    mae=mae,
                    rmse=mae,
                    mape=10.0,
                    smape=10.0,
                    mase=1.0,
                    n_train=10,
                    n_test=3,
                )
            )
        return report

    def test_leaderboard_sorted_by_mae(self):
        board = self._report().leaderboard()
        assert list(board["model"]) == ["a", "b"]
        assert board.iloc[0]["mae"] == 1.5

    def test_empty_report_returns_empty_frame_with_columns(self):
        board = HarnessReport().leaderboard()
        assert board.empty

    def test_skips_are_counted_by_reason(self):
        report = HarnessReport()
        report.skip("train_too_short")
        report.skip("train_too_short")
        report.skip("empty_test")
        assert report.skipped == {"train_too_short": 2, "empty_test": 1}

    def test_fallback_rate_is_reported(self):
        """A good MAE with a 60% fallback rate is the baseline in disguise."""
        report = HarnessReport()
        for i, fb in enumerate([True, True, False]):
            report.results.append(
                SeriesResult("m", f"S{i}", 1.0, 1.0, None, 1.0, 1.0, 10, 3, fallback=fb)
            )
        assert report.leaderboard().iloc[0]["fallback_rate"] == pytest.approx(2 / 3)

    def test_summary_is_readable(self):
        text = self._report().summary()
        assert "model" in text and "MAE" in text


class TestPairedComparison:
    def test_win_rate_on_shared_series(self):
        report = HarnessReport()
        for series, a_mae, b_mae in [("S1", 1.0, 2.0), ("S2", 3.0, 2.0), ("S3", 1.0, 5.0)]:
            report.results.append(SeriesResult("a", series, a_mae, a_mae, None, 1, 1, 10, 3))
            report.results.append(SeriesResult("b", series, b_mae, b_mae, None, 1, 1, 10, 3))
        result = paired_comparison(report, "a", "b")
        assert result["n"] == 3
        assert result["win_rate"] == pytest.approx(2 / 3)

    def test_no_shared_series_returns_nan_not_a_crash(self):
        report = HarnessReport()
        report.results.append(SeriesResult("a", "S1", 1.0, 1.0, None, 1, 1, 10, 3))
        report.results.append(SeriesResult("b", "S2", 1.0, 1.0, None, 1, 1, 10, 3))
        assert paired_comparison(report, "a", "b")["n"] == 0

    def test_detects_a_median_win_that_is_not_a_real_win(self):
        """The check that matters: model a has the better median but loses on
        most series -- the aggregate gain comes from one outlier."""
        report = HarnessReport()
        pairs = [("S1", 5.0, 1.0), ("S2", 5.0, 1.0), ("S3", 0.1, 40.0)]
        for series, a_mae, b_mae in pairs:
            report.results.append(SeriesResult("a", series, a_mae, a_mae, None, 1, 1, 10, 3))
            report.results.append(SeriesResult("b", series, b_mae, b_mae, None, 1, 1, 10, 3))
        assert paired_comparison(report, "a", "b")["win_rate"] < 0.5


class TestEvaluateModels:
    def test_scores_every_model_on_every_series(self):
        grid = resample_panel(panel(n_series=3, n=30), min_observed=8)
        split = split_per_product(grid, 0.2, min_observations=10)
        report = evaluate_models(split, [NaiveLast, Mean])
        assert len(report.results) == 6
        assert set(report.to_frame()["model"]) == {"naive_last", "mean"}

    def test_fresh_model_per_series(self):
        """Shared state would leak one product's history into the next."""
        grid = resample_panel(panel(n_series=3, n=30), min_observed=8)
        split = split_per_product(grid, 0.2, min_observations=10)
        report = evaluate_models(split, [NaiveLast])
        frame = report.to_frame()
        assert frame["mae"].nunique() > 1

    def test_series_with_too_little_training_data_are_skipped(self):
        grid = resample_panel(panel(n_series=2, n=30), min_observed=8)
        split = split_per_product(grid, 0.2, min_observations=10)
        report = evaluate_models(split, [NaiveLast], min_train=1000)
        assert report.results == []
        assert report.skipped.get("train_too_short", 0) > 0

    def test_asserts_no_leakage_before_scoring(self):
        from pulseiq.features.splits import Split

        frame = panel(n_series=1, n=30)
        shuffled = frame.sample(frac=1.0, random_state=0)
        bad = Split(shuffled.iloc[:24], shuffled.iloc[24:], pd.Timestamp("2024-01-01"))
        with pytest.raises(AssertionError, match="LEAKAGE"):
            evaluate_models(bad, [NaiveLast])

    def test_a_failing_model_does_not_kill_the_run(self):
        class Broken(NaiveLast):
            name = "broken"

            def _fit(self, y):
                raise RuntimeError("boom")

        grid = resample_panel(panel(n_series=2, n=30), min_observed=8)
        split = split_per_product(grid, 0.2, min_observations=10)
        report = evaluate_models(split, [NaiveLast, Broken])
        assert set(report.to_frame()["model"]) == {"naive_last"}
        assert any(k.startswith("model_error") for k in report.skipped)

    def test_imputed_test_rows_are_excluded_from_scoring(self):
        grid = resample_panel(panel(n_series=2, n=30), min_observed=8)
        split = split_per_product(grid, 0.2, min_observations=10)
        scored = evaluate_models(split, [NaiveLast], score_observed_only=True)
        unscored = evaluate_models(split, [NaiveLast], score_observed_only=False)
        assert scored.results[0].n_test <= unscored.results[0].n_test


class TestBuildFactories:
    def test_baselines_only(self):
        factories = build_factories(use_arima=False, use_prophet=False)
        assert len(factories) == 6

    def test_with_arima(self):
        assert len(build_factories(use_arima=True, use_prophet=False)) == 7

    def test_factories_return_unfitted_instances(self):
        for factory in build_factories(use_arima=True, use_prophet=False):
            with pytest.raises(NotFittedError):
                factory().predict(1)

    def test_model_names_are_unique(self):
        names = [f().name for f in build_factories(use_arima=True, use_prophet=False)]
        assert len(names) == len(set(names))

    def test_a_failed_fit_is_recorded_not_hidden(self, monkeypatch):
        """The property this project depends on.

        When Prophet cannot fit -- a dependency incompatibility, a bad series,
        anything -- it must fall back visibly. A silent fallback would credit
        Prophet with the naive baseline's score and corrupt the whole model
        comparison.
        """
        from pulseiq.training.forecasting import prophet_model

        model = prophet_model.ProphetForecaster()

        def explode(*args, **kwargs):
            raise RuntimeError("simulated dependency failure")

        monkeypatch.setattr(model, "_frame", explode)
        preds = model.fit_predict(trending(n=30), 3)

        assert model.used_fallback is True
        assert preds.tolist() == [pytest.approx(trending(n=30)[-1])] * 3
