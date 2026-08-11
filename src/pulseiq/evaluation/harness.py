"""Evaluation harness: score every model on every series, identically.

The whole point is that each model sees exactly the same train/test partition
of exactly the same series, and is scored by exactly the same code. Any
per-model special-casing here would silently advantage one model over another,
which is how forecasting comparisons usually go wrong.

Two levels of result:
  * per-series  -- one row per (model, series), for distributions and outliers
  * aggregate   -- median across series per model, for the headline table

Median rather than mean across series: prices here span 1.50 to 30.00, so one
expensive product's absolute error would dominate a mean and the leaderboard
would rank models by which handled a single series best.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pulseiq.evaluation.metrics import EmptyEvaluationError, evaluate_forecast, mase
from pulseiq.features.resample import observed_only
from pulseiq.features.splits import Split
from pulseiq.training.forecasting.baseline import Forecaster

logger = logging.getLogger(__name__)

DATE_COL = "observed_on"
GROUP_COL = "product_name"
PRICE_COL = "selling_price"

ModelFactory = Callable[[], Forecaster]


@dataclass
class SeriesResult:
    """One model's score on one series."""

    model: str
    series: str
    mae: float
    rmse: float
    mape: float | None
    smape: float
    mase: float | None
    n_train: int
    n_test: int
    fallback: bool = False


@dataclass
class HarnessReport:
    """Everything a training run produced, plus what it skipped and why."""

    results: list[SeriesResult] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def to_frame(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(
                columns=[
                    "model",
                    "series",
                    "mae",
                    "rmse",
                    "mape",
                    "smape",
                    "mase",
                    "n_train",
                    "n_test",
                    "fallback",
                ]
            )
        return pd.DataFrame([r.__dict__ for r in self.results])

    def leaderboard(self) -> pd.DataFrame:
        """Median metrics per model, best MAE first.

        Also reports `fallback_rate` -- the share of series where a model could
        not fit and silently degraded to a naive forecast. A model with a good
        MAE and a 60% fallback rate is not a good model; it is mostly the
        baseline wearing a different name.
        """
        frame = self.to_frame()
        if frame.empty:
            return frame

        agg = (
            frame.groupby("model")
            .agg(
                series=("series", "count"),
                mae=("mae", "median"),
                rmse=("rmse", "median"),
                smape=("smape", "median"),
                mase=("mase", "median"),
                fallback_rate=("fallback", "mean"),
            )
            .sort_values("mae")
            .reset_index()
        )
        return agg

    def summary(self) -> str:
        board = self.leaderboard()
        if board.empty:
            return f"no results. skipped: {self.skipped}"
        lines = [
            f"{len(self.results)} evaluations in {self.duration_seconds:.1f}s",
            f"skipped: {self.skipped or 'none'}",
            "",
            f"{'model':<20}{'series':>8}{'MAE':>10}{'RMSE':>10}{'sMAPE':>9}{'MASE':>8}{'fallbk':>8}",
        ]
        for row in board.itertuples(index=False):
            mase_txt = f"{row.mase:.3f}" if pd.notna(row.mase) else "n/a"
            lines.append(
                f"{row.model:<20}{row.series:>8}{row.mae:>10.4f}{row.rmse:>10.4f}"
                f"{row.smape:>8.2f}%{mase_txt:>8}{row.fallback_rate:>7.0%}"
            )
        return "\n".join(lines)


def evaluate_models(
    split: Split,
    factories: Sequence[ModelFactory],
    *,
    min_train: int = 8,
    score_observed_only: bool = True,
) -> HarnessReport:
    """Fit and score every model on every series in `split`.

    Args:
        min_train: series with fewer training points are skipped entirely, for
            every model, so the comparison stays on a common set of series.
        score_observed_only: drop forward-filled test rows before scoring.
            Leave this True. Imputed rows are copies of their predecessor, so
            scoring on them rewards whichever model changes its prediction
            least -- usually the naive baseline.
    """
    started = time.perf_counter()
    report = HarnessReport()

    split.assert_no_leakage()

    test_frame = observed_only(split.test) if score_observed_only else split.test
    if test_frame.empty:
        report.skip("no_observed_test_rows")
        report.duration_seconds = time.perf_counter() - started
        return report

    grouped = (
        test_frame.groupby(GROUP_COL) if GROUP_COL in test_frame.columns else [("all", test_frame)]
    )

    for series_name, test_group in grouped:
        train_group = (
            split.train[split.train[GROUP_COL] == series_name]
            if GROUP_COL in split.train.columns
            else split.train
        )

        if len(train_group) < min_train:
            report.skip("train_too_short")
            continue
        if test_group.empty:
            report.skip("empty_test")
            continue

        y_train = train_group[PRICE_COL].to_numpy(dtype=float)
        y_test = test_group[PRICE_COL].to_numpy(dtype=float)
        horizon = len(y_test)

        for factory in factories:
            # A fresh instance per series: Forecaster holds fitted state, so
            # reuse would leak one product's history into the next.
            model = factory()
            try:
                preds = model.fit_predict(y_train, horizon)
                metrics = evaluate_forecast(y_test, preds)
            except EmptyEvaluationError:
                report.skip("empty_evaluation")
                continue
            except Exception as exc:  # noqa: BLE001 - one bad series must not kill the run
                logger.warning("%s failed on %s: %s", model.name, series_name, exc)
                report.skip(f"model_error:{model.name}")
                continue

            report.results.append(
                SeriesResult(
                    model=model.name,
                    series=str(series_name),
                    mae=metrics.mae,
                    rmse=metrics.rmse,
                    mape=metrics.mape,
                    smape=metrics.smape,
                    mase=mase(y_test, preds, y_train),
                    n_train=len(y_train),
                    n_test=horizon,
                    fallback=bool(getattr(model, "used_fallback", False)),
                )
            )

    report.duration_seconds = time.perf_counter() - started
    logger.info(
        "harness complete: %d results in %.1fs", len(report.results), report.duration_seconds
    )
    return report


def paired_comparison(report: HarnessReport, model_a: str, model_b: str) -> dict[str, float]:
    """Compare two models on the series both were scored on.

    An unpaired comparison of medians can mislead when models were skipped on
    different series. `win_rate` is the share of shared series where model_a had
    the lower MAE -- a more robust signal than a median difference when the
    error distribution is skewed, which it is here.
    """
    frame = report.to_frame()
    a = frame[frame["model"] == model_a].set_index("series")["mae"]
    b = frame[frame["model"] == model_b].set_index("series")["mae"]
    shared = a.index.intersection(b.index)

    if len(shared) == 0:
        return {"n": 0, "win_rate": float("nan"), "median_delta": float("nan")}

    a_shared, b_shared = a.loc[shared], b.loc[shared]
    return {
        "n": int(len(shared)),
        "win_rate": float((a_shared < b_shared).mean()),
        "median_delta": float(np.median(a_shared - b_shared)),
        "median_pct_improvement": float(
            np.median((b_shared - a_shared) / b_shared.replace(0, np.nan)) * 100
        ),
    }
