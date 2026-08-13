"""Tests for the CI metric regression gate.

The gate's job is to fail a build when a model gets worse. These tests confirm
it fires when it should, stays quiet when it should, and -- the part that is
easy to get wrong -- also fires when a result is implausibly *good*.
"""

import json

import pandas as pd
import pytest
import yaml

from pulseiq.evaluation.regression_gate import (
    DEFAULT_THRESHOLDS,
    Status,
    check_maximum,
    check_minimum,
    load_report,
    load_thresholds,
    run_gate,
)

HEALTHY_SENTIMENT = {
    "accuracy": 0.9413,
    "macro_f1": 0.9413,
    "recall": 0.9333,
    "precision": 0.9485,
}


def healthy_curve() -> pd.DataFrame:
    rows = [
        {"model": "naive_last", "horizon": h, "mae": m}
        for h, m in [(1, 0.0), (3, 0.010), (6, 0.033), (12, 0.042)]
    ]
    rows += [
        {"model": "arima_auto", "horizon": h, "mae": m}
        for h, m in [(1, 0.009), (3, 0.021), (6, 0.043), (12, 0.062)]
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def reports(tmp_path):
    directory = tmp_path / "reports"
    directory.mkdir()
    return directory


def write_sentiment(directory, payload):
    (directory / "sentiment_finetuned.json").write_text(json.dumps(payload), encoding="utf-8")


def write_curve(directory, frame):
    frame.to_csv(directory / "horizon_curve.csv", index=False)


class TestShippedThresholds:
    def test_thresholds_file_is_valid(self):
        """Guards against a YAML typo shipping unnoticed -- this file is edited
        by hand whenever a model improves."""
        config = load_thresholds()
        assert "sentiment" in config
        assert "forecasting" in config

    def test_minimums_sit_below_recorded_values(self):
        """A threshold at or above the recorded value fails on the very run that
        set it, the first time a seed shifts."""
        config = load_thresholds()
        recorded = config["sentiment"]["recorded"]
        for metric, minimum in config["sentiment"]["minimum"].items():
            assert minimum < recorded[metric], f"{metric} threshold leaves no tolerance"

    def test_forecast_maximums_sit_above_recorded_values(self):
        config = load_thresholds()
        recorded = config["forecasting"]["recorded"]
        for metric, maximum in config["forecasting"]["maximum"].items():
            assert maximum >= recorded[metric], f"{metric} threshold is unachievable"

    def test_thresholds_yaml_is_parseable_directly(self):
        yaml.safe_load(DEFAULT_THRESHOLDS.read_text(encoding="utf-8"))


class TestCheckHelpers:
    def test_minimum_passes_at_the_boundary(self):
        assert check_minimum("m", 0.91, 0.91).status is Status.PASS

    def test_minimum_fails_below(self):
        check = check_minimum("m", 0.90, 0.91)
        assert check.status is Status.FAIL
        assert "regressed by" in check.detail

    def test_maximum_passes_at_the_boundary(self):
        assert check_maximum("m", 0.05, 0.05).status is Status.PASS

    def test_maximum_fails_above(self):
        assert check_maximum("m", 0.06, 0.05).status is Status.FAIL

    def test_absent_metric_skips_rather_than_fails(self):
        assert check_minimum("m", None, 0.9).status is Status.SKIP


class TestGateOutcomes:
    def test_missing_reports_skip_without_blocking(self, reports):
        """A PR touching only the scraper must not be blocked because nobody
        re-ran the fine-tune."""
        report, code = run_gate(reports_dir=reports)
        assert code == 0
        assert len(report.skipped) == 2

    def test_healthy_metrics_pass(self, reports):
        write_sentiment(reports, HEALTHY_SENTIMENT)
        write_curve(reports, healthy_curve())
        report, code = run_gate(reports_dir=reports)
        assert code == 0
        assert report.failed == []

    def test_regressed_accuracy_fails(self, reports):
        write_sentiment(reports, {**HEALTHY_SENTIMENT, "accuracy": 0.87, "macro_f1": 0.87})
        report, code = run_gate(reports_dir=reports)
        assert code == 1
        assert any(c.name == "sentiment.accuracy" for c in report.failed)

    def test_regressed_recall_fails_even_when_accuracy_holds(self, reports):
        """The fine-tune's entire value was recall. A change trading recall for
        precision keeps accuracy stable while undoing the result."""
        write_sentiment(reports, {**HEALTHY_SENTIMENT, "recall": 0.85, "precision": 0.99})
        report, code = run_gate(reports_dir=reports)
        assert code == 1
        assert any(c.name == "sentiment.recall" for c in report.failed)

    def test_fine_tune_no_longer_beating_baseline_fails(self, reports):
        write_sentiment(reports, {**HEALTHY_SENTIMENT, "accuracy": 0.8800, "macro_f1": 0.92})
        report, code = run_gate(reports_dir=reports)
        assert code == 1
        assert any("vs_baseline" in c.name for c in report.failed)

    def test_forecast_error_growth_fails(self, reports):
        curve = healthy_curve()
        curve.loc[curve["model"] == "naive_last", "mae"] = [0.05, 0.09, 0.15, 0.30]
        write_curve(reports, curve)
        report, code = run_gate(reports_dir=reports)
        assert code == 1
        assert any(c.name.startswith("forecast.mae") for c in report.failed)


class TestLeakageTripwire:
    """Every leakage bug in this project's history made metrics look BETTER.
    A gate that only checks the downside would have caught none of them."""

    def test_implausible_improvement_fails(self, reports):
        write_curve(
            reports,
            pd.DataFrame(
                [
                    {"model": "naive_last", "horizon": 3, "mae": 0.010},
                    {"model": "arima_auto", "horizon": 3, "mae": 0.002},  # 80% better
                ]
            ),
        )
        report, code = run_gate(reports_dir=reports)
        assert code == 1
        failure = next(c for c in report.failed if "leakage" in c.name)
        assert "verify the split" in failure.detail

    def test_modest_improvement_passes(self, reports):
        """A real gain must not trip the wire -- 10% is plausible."""
        write_curve(
            reports,
            pd.DataFrame(
                [
                    {"model": "naive_last", "horizon": 3, "mae": 0.010},
                    {"model": "arima_auto", "horizon": 3, "mae": 0.009},
                ]
            ),
        )
        report, _ = run_gate(reports_dir=reports)
        assert not any("leakage" in c.name for c in report.failed)

    def test_zero_baseline_error_does_not_divide_by_zero(self):
        """naive_last has median MAE of exactly 0.0000 at h=1 on real data."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pd.DataFrame(
                [
                    {"model": "naive_last", "horizon": 1, "mae": 0.0},
                    {"model": "arima_auto", "horizon": 1, "mae": 0.009},
                ]
            ).to_csv(directory / "horizon_curve.csv", index=False)
            report, _ = run_gate(reports_dir=directory)
            assert not any("leakage" in c.name for c in report.failed)


class TestRobustness:
    def test_malformed_json_skips_rather_than_crashing(self, reports):
        """A truncated report from an interrupted run must not block every other
        check from running."""
        (reports / "sentiment_finetuned.json").write_text("{not valid json", encoding="utf-8")
        write_curve(reports, healthy_curve())
        report, code = run_gate(reports_dir=reports)
        assert code == 0
        assert any(c.status is Status.SKIP for c in report.checks)

    def test_load_report_returns_none_when_absent(self, tmp_path):
        assert load_report(tmp_path / "nope.json") is None

    def test_curve_without_naive_baseline_skips(self, reports):
        write_curve(reports, pd.DataFrame([{"model": "arima_auto", "horizon": 1, "mae": 0.01}]))
        report, code = run_gate(reports_dir=reports)
        assert code == 0
        assert any(c.status is Status.SKIP for c in report.checks)

    def test_strict_mode_turns_skips_into_failures(self, reports):
        report, code = run_gate(reports_dir=reports, strict=True)
        assert code == 1
        assert report.skipped

    def test_missing_thresholds_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_gate(reports_dir=tmp_path, thresholds_path=tmp_path / "nope.yaml")

    def test_summary_is_readable(self, reports):
        write_sentiment(reports, HEALTHY_SENTIMENT)
        report, _ = run_gate(reports_dir=reports)
        text = report.summary()
        assert "status" in text and "checks:" in text
