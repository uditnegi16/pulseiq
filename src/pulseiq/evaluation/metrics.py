"""Forecasting and classification metrics.

Implemented here rather than called inline so that every model is scored by
identical code, and so the edge cases are handled once and tested once.

MAPE is included because stakeholders ask for it, but it is a bad default and
this module says so: it is undefined at zero, unbounded when actuals are small,
and asymmetric (it punishes over-forecasting more than under-forecasting).
sMAPE and MAE are reported alongside so a single flattering number cannot be
cherry-picked.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

logger = logging.getLogger(__name__)


class EmptyEvaluationError(ValueError):
    """Raised when there is nothing to score.

    Deliberately loud. A silent 0.0 for an empty test set is how a broken
    evaluation gets reported as a perfect one.
    """


@dataclass(frozen=True)
class ForecastMetrics:
    """Point-forecast metrics for one model on one test set."""

    mae: float
    rmse: float
    mape: float | None
    smape: float
    n: int

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)

    def __str__(self) -> str:
        mape = f"{self.mape:.2f}%" if self.mape is not None else "n/a"
        return (
            f"MAE={self.mae:.4f} RMSE={self.rmse:.4f} "
            f"MAPE={mape} sMAPE={self.smape:.2f}% n={self.n}"
        )


def _clean_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Align, coerce to float, and drop pairs containing NaN."""
    true_arr = np.asarray(y_true, dtype=float).ravel()
    pred_arr = np.asarray(y_pred, dtype=float).ravel()

    if true_arr.shape != pred_arr.shape:
        raise ValueError(f"shape mismatch: y_true {true_arr.shape} vs y_pred {pred_arr.shape}")
    if true_arr.size == 0:
        raise EmptyEvaluationError("no observations to score")

    mask = np.isfinite(true_arr) & np.isfinite(pred_arr)
    dropped = int((~mask).sum())
    if dropped:
        logger.warning("dropped %d non-finite pair(s) before scoring", dropped)
    if not mask.any():
        raise EmptyEvaluationError("all pairs were non-finite")

    return true_arr[mask], pred_arr[mask]


def mae(y_true, y_pred) -> float:
    """Mean absolute error. In the units of the target, which is why it is the
    primary metric here -- 'off by 0.42 euros' is interpretable."""
    true_arr, pred_arr = _clean_pair(y_true, y_pred)
    return float(np.mean(np.abs(true_arr - pred_arr)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error. Punishes large misses harder than MAE, so a big
    gap between the two indicates a few severe errors rather than uniform drift."""
    true_arr, pred_arr = _clean_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))


def mape(y_true, y_pred, *, epsilon: float = 1e-9) -> float | None:
    """Mean absolute percentage error, or None if it cannot be computed.

    Returns None rather than a large number when actuals are at or near zero.
    A MAPE of 4.7 billion percent is not a metric, it is a division artefact,
    and reporting it as if it meant something is worse than reporting nothing.
    """
    true_arr, pred_arr = _clean_pair(y_true, y_pred)
    usable = np.abs(true_arr) > epsilon
    if not usable.any():
        return None
    if usable.sum() < len(true_arr):
        logger.warning(
            "MAPE computed on %d/%d points; %d actuals were ~zero",
            int(usable.sum()),
            len(true_arr),
            int((~usable).sum()),
        )
    return float(np.mean(np.abs((true_arr[usable] - pred_arr[usable]) / true_arr[usable])) * 100.0)


def smape(y_true, y_pred, *, epsilon: float = 1e-9) -> float:
    """Symmetric MAPE, bounded at 200%.

    Preferred over MAPE for reporting: defined at zero, bounded, and it does not
    systematically favour models that under-forecast.
    """
    true_arr, pred_arr = _clean_pair(y_true, y_pred)
    denominator = (np.abs(true_arr) + np.abs(pred_arr)) / 2.0
    usable = denominator > epsilon
    if not usable.any():
        return 0.0
    return float(np.mean(np.abs(true_arr[usable] - pred_arr[usable]) / denominator[usable]) * 100.0)


def evaluate_forecast(y_true, y_pred) -> ForecastMetrics:
    """All point-forecast metrics at once. The single entry point for scoring."""
    true_arr, pred_arr = _clean_pair(y_true, y_pred)
    return ForecastMetrics(
        mae=mae(true_arr, pred_arr),
        rmse=rmse(true_arr, pred_arr),
        mape=mape(true_arr, pred_arr),
        smape=smape(true_arr, pred_arr),
        n=len(true_arr),
    )


def mase(y_true, y_pred, y_train, *, seasonality: int = 1) -> float | None:
    """Mean absolute scaled error.

    MAE divided by the MAE of a naive forecast on the training data. Scale-free,
    and it answers the only question that matters about a forecast: is this
    better than doing nothing?

      < 1.0  beats the naive baseline
      = 1.0  matches it
      > 1.0  worse than doing nothing

    Returns None when the training series has no variation to scale by.
    """
    true_arr, pred_arr = _clean_pair(y_true, y_pred)
    train_arr = np.asarray(y_train, dtype=float).ravel()
    train_arr = train_arr[np.isfinite(train_arr)]

    if len(train_arr) <= seasonality:
        return None

    naive_errors = np.abs(train_arr[seasonality:] - train_arr[:-seasonality])
    scale = float(np.mean(naive_errors))
    if scale <= 0:
        return None

    return float(np.mean(np.abs(true_arr - pred_arr)) / scale)
