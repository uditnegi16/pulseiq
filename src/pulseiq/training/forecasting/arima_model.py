"""ARIMA forecaster (statsmodels), behind the shared Forecaster interface.

WHY THE GRID IS NON-NEGOTIABLE HERE
-----------------------------------
ARIMA models the relationship between step t and step t-1 and assumes those
steps are evenly spaced. The Open Prices observations are not: median gap 17
days, p90 63. Fitting ARIMA to raw irregular timestamps produces a model that
silently believes every gap is identical, and its metrics are meaningless.

That is why `features/resample.py` exists and why every model in this project is
evaluated on the same monthly grid.

ORDER SELECTION
---------------
`auto=True` runs a small grid search over (p, d, q) scored by AIC on the
TRAINING data only. Selecting an order by test-set performance would be
leakage of exactly the kind this project is built to avoid -- the test set must
not influence any decision, including hyperparameters.

The grid is deliberately small. Series here have a median of ~32 monthly points;
searching a large space on 32 observations overfits the order-selection itself.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from pulseiq.training.forecasting.baseline import Forecaster

logger = logging.getLogger(__name__)

# Small grid: p,q in 0..2 and d in 0..1. On ~32 points anything larger is
# fitting noise in the model-selection step.
DEFAULT_GRID: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
    (2, 1, 2),
)

MIN_OBSERVATIONS = 8


class ARIMAForecaster(Forecaster):
    """ARIMA with optional AIC-based order selection.

    Falls back to the last observed value if every candidate order fails to
    converge. That fallback is logged, never silent: a "successful" run that was
    secretly a naive forecast would corrupt the model comparison.
    """

    def __init__(
        self,
        order: tuple[int, int, int] | None = None,
        *,
        auto: bool = True,
        grid: tuple[tuple[int, int, int], ...] = DEFAULT_GRID,
    ) -> None:
        super().__init__()
        self.order = order
        self.auto = auto and order is None
        self.grid = grid
        self.selected_order: tuple[int, int, int] | None = None
        self.used_fallback = False
        self.name = "arima_auto" if self.auto else f"arima_{order or (1, 1, 1)}"

    def _fit_one(self, y: np.ndarray, order: tuple[int, int, int]):
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            # statsmodels is noisy about convergence on short series; we handle
            # failure explicitly below rather than reading warnings.
            warnings.simplefilter("ignore")
            model = ARIMA(y, order=order, enforce_stationarity=False, enforce_invertibility=False)
            return model.fit()

    def _fit(self, y: np.ndarray) -> None:
        self.used_fallback = False
        self._result = None

        if len(y) < MIN_OBSERVATIONS:
            logger.debug(
                "%s: %d points is below the %d minimum, falling back to last value",
                self.name,
                len(y),
                MIN_OBSERVATIONS,
            )
            self.used_fallback = True
            self._last = float(y[-1])
            return

        candidates = self.grid if self.auto else ((self.order or (1, 1, 1)),)
        best_aic = np.inf
        best = None
        best_order = None

        for order in candidates:
            # An ARIMA of order (p,d,q) needs more observations than parameters.
            if sum(order) >= len(y):
                continue
            try:
                fitted = self._fit_one(y, order)
                aic = float(fitted.aic)
            except Exception as exc:  # noqa: BLE001 - any failure just disqualifies the order
                logger.debug("%s: order %s failed (%s)", self.name, order, type(exc).__name__)
                continue
            if np.isfinite(aic) and aic < best_aic:
                best_aic, best, best_order = aic, fitted, order

        if best is None:
            logger.warning(
                "%s: no ARIMA order converged on %d points, falling back to last value",
                self.name,
                len(y),
            )
            self.used_fallback = True
            self._last = float(y[-1])
            return

        self._result = best
        self.selected_order = best_order
        logger.debug("%s: selected order %s (AIC %.2f)", self.name, best_order, best_aic)

    def _predict(self, horizon: int) -> np.ndarray:
        if self.used_fallback or self._result is None:
            return np.full(horizon, self._last)
        try:
            return np.asarray(self._result.forecast(steps=horizon), dtype=float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: forecast failed (%s), using last value", self.name, exc)
            return np.full(horizon, float(self._history[-1]))
