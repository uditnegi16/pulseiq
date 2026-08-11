"""Baseline forecasters.

These exist to be beaten. A published MAE means nothing on its own -- MAE 0.42
is excellent if the naive baseline scores 1.80 and embarrassing if it scores
0.31. Reporting ARIMA without a baseline is how forecasting projects claim
results they have not earned.

Every model here follows the same tiny interface (`fit`, `predict`) so the
evaluation harness treats baselines and real models identically. Nothing here
imports statsmodels or prophet; these run in milliseconds and are fully tested.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class NotFittedError(RuntimeError):
    """Raised when predict() is called before fit()."""


class Forecaster(ABC):
    """Minimal forecaster interface shared by baselines, ARIMA and Prophet."""

    name: str = "forecaster"

    def __init__(self) -> None:
        self._fitted = False
        self._history: np.ndarray | None = None

    @abstractmethod
    def _fit(self, y: np.ndarray) -> None: ...

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray: ...

    def fit(self, y) -> Forecaster:
        arr = np.asarray(y, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError(f"{self.name}: cannot fit on an empty series")
        self._history = arr
        self._fit(arr)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        if not self._fitted:
            raise NotFittedError(f"{self.name}: call fit() before predict()")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        out = np.asarray(self._predict(horizon), dtype=float).ravel()
        if out.shape != (horizon,):
            raise RuntimeError(f"{self.name}: predicted shape {out.shape}, expected ({horizon},)")
        return out

    def fit_predict(self, y, horizon: int) -> np.ndarray:
        return self.fit(y).predict(horizon)


class NaiveLast(Forecaster):
    """Predict the last observed value, forever.

    The one every other model must beat. On price data this is a genuinely
    strong competitor -- prices are step functions, so "tomorrow costs what
    today costs" is right most of the time.
    """

    name = "naive_last"

    def _fit(self, y: np.ndarray) -> None:
        self._value = float(y[-1])

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._value)


class MovingAverage(Forecaster):
    """Predict the mean of the last `window` observations.

    Smoother than NaiveLast, which helps on noisy series and hurts on series
    with genuine level shifts -- exactly the trade-off worth measuring.
    """

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self.name = f"moving_average_{window}"

    def _fit(self, y: np.ndarray) -> None:
        self._value = float(np.mean(y[-self.window :]))

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._value)


class SeasonalNaive(Forecaster):
    """Predict the value from `period` steps ago.

    On a monthly grid, period=12 means "this month last year". Captures annual
    retail seasonality without estimating anything. Falls back to the last value
    when history is shorter than one period, rather than failing -- most series
    here are under three years.
    """

    def __init__(self, period: int = 12) -> None:
        super().__init__()
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}")
        self.period = period
        self.name = f"seasonal_naive_{period}"

    def _fit(self, y: np.ndarray) -> None:
        self._season = y[-self.period :] if len(y) >= self.period else None
        self._fallback = float(y[-1])

    def _predict(self, horizon: int) -> np.ndarray:
        if self._season is None:
            return np.full(horizon, self._fallback)
        reps = int(np.ceil(horizon / self.period))
        return np.tile(self._season, reps)[:horizon]


class Drift(Forecaster):
    """Extrapolate the straight line from the first to the last observation.

    Captures trend with zero parameters. On a series with a genuine upward drift
    it beats NaiveLast; on a flat noisy series it is worse. Included so trend is
    represented among the baselines.
    """

    name = "drift"

    def _fit(self, y: np.ndarray) -> None:
        self._last = float(y[-1])
        self._slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0.0

    def _predict(self, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1, dtype=float)
        return self._last + self._slope * steps


class Mean(Forecaster):
    """Predict the mean of the whole history.

    Deliberately weak. If a sophisticated model cannot beat this, the series has
    no exploitable structure and that is the finding worth reporting.
    """

    name = "mean"

    def _fit(self, y: np.ndarray) -> None:
        self._value = float(np.mean(y))

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._value)


def default_baselines() -> list[Forecaster]:
    """The standard baseline set, constructed fresh each call.

    Fresh instances matter: a Forecaster holds fitted state, so reusing one
    across series would leak the previous product's history into the next.
    """
    return [NaiveLast(), MovingAverage(3), MovingAverage(6), Drift(), Mean()]
