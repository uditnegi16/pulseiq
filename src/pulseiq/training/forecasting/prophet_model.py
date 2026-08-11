"""Prophet forecaster, behind the shared Forecaster interface.

Prophet expects a DataFrame with `ds` (timestamp) and `y` (value), so this
adapter reconstructs a monthly date index from the array it is handed. That is
sound because every series reaching a model has already been put on the monthly
grid by `features/resample.py` -- positions map one-to-one onto months.

CONFIGURATION CHOICES
---------------------
`yearly_seasonality` is only enabled when the series covers at least two years.
Prophet will happily fit an annual cycle to 14 months of data; the resulting
seasonality is fitted noise, and it makes forecasts worse while looking
sophisticated.

Weekly and daily seasonality are always off: the data is monthly, so those
components have nothing to fit and only add parameters.

Prophet is chatty on stdout (cmdstan output) and would otherwise flood a
training run over 200 series, so its logger is quietened at import.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from pulseiq.training.forecasting.baseline import Forecaster

logger = logging.getLogger(__name__)

# cmdstanpy writes progress to stdout for every fit; over 200 series that is
# thousands of lines of noise around the numbers that matter.
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)
os.environ.setdefault("CMDSTANPY_LOG_LEVEL", "CRITICAL")


def _silence_prophet_logging() -> None:
    """Silence Prophet/cmdstanpy output.

    Setting the level is not enough: cmdstanpy attaches its *own* StreamHandler
    when imported, so the first fit escapes a level set beforehand. The handlers
    have to be removed as well. Over hundreds of series this is the difference
    between a readable leaderboard and thousands of lines of chain output.
    """
    for name in ("prophet", "cmdstanpy", "prophet.forecaster", "prophet.models"):
        noisy = logging.getLogger(name)
        noisy.handlers.clear()
        noisy.addHandler(logging.NullHandler())
        noisy.setLevel(logging.CRITICAL)
        noisy.propagate = False


_silence_prophet_logging()

MIN_OBSERVATIONS = 8
MONTHS_FOR_YEARLY_SEASONALITY = 24


class ProphetForecaster(Forecaster):
    """Facebook Prophet on a monthly grid.

    Falls back to the last observed value when the series is too short or the
    fit fails. The fallback is recorded in `used_fallback` and logged, so a run
    that quietly degraded to a naive forecast is visible in the comparison
    rather than being credited to Prophet.
    """

    name = "prophet"

    def __init__(
        self,
        *,
        freq: str = "MS",
        changepoint_prior_scale: float = 0.05,
        seasonality_mode: str = "additive",
        anchor: pd.Timestamp | None = None,
    ) -> None:
        super().__init__()
        self.freq = freq
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.anchor = anchor
        self.used_fallback = False
        self.yearly_seasonality_used = False

    def _frame(self, y: np.ndarray) -> pd.DataFrame:
        end = self.anchor or pd.Timestamp("2000-01-01")
        dates = pd.date_range(end=end, periods=len(y), freq=self.freq)
        return pd.DataFrame({"ds": dates, "y": y})

    def _fit(self, y: np.ndarray) -> None:
        self.used_fallback = False
        self._model = None
        self._last = float(y[-1])

        if len(y) < MIN_OBSERVATIONS:
            logger.debug("prophet: %d points below minimum %d", len(y), MIN_OBSERVATIONS)
            self.used_fallback = True
            return

        # Only fit an annual cycle when there is at least one full cycle plus
        # enough to distinguish it from trend.
        self.yearly_seasonality_used = len(y) >= MONTHS_FOR_YEARLY_SEASONALITY

        try:
            from prophet import Prophet

            # Importing prophet pulls in cmdstanpy, which re-attaches its own
            # handler. Silence again now that those handlers exist.
            _silence_prophet_logging()

            model = Prophet(
                yearly_seasonality=self.yearly_seasonality_used,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_mode=self.seasonality_mode,
                uncertainty_samples=0,  # we score point forecasts; sampling is wasted time
            )
            self._train_frame = self._frame(y)
            model.fit(self._train_frame)
            self._model = model
        except Exception as exc:  # noqa: BLE001
            logger.warning("prophet: fit failed (%s), falling back to last value", exc)
            self.used_fallback = True

    def _predict(self, horizon: int) -> np.ndarray:
        if self.used_fallback or self._model is None:
            return np.full(horizon, self._last)
        try:
            future = self._model.make_future_dataframe(
                periods=horizon, freq=self.freq, include_history=False
            )
            forecast = self._model.predict(future)
            return np.asarray(forecast["yhat"].to_numpy()[:horizon], dtype=float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("prophet: predict failed (%s), using last value", exc)
            return np.full(horizon, self._last)
