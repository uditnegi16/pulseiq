"""Forecasting endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from pulseiq.api.deps import cache_ttl, get_cache_dep, get_db
from pulseiq.api.models import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    ProductSummary,
)
from pulseiq.llm.cache import make_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecast", tags=["forecast"])

# The finding this API has to carry honestly. Measured across 56 series at
# h=1/3/6/12: no model beats the naive forecast, and median h=1 error is exactly
# zero. Returning an ARIMA number without this note would imply a sophistication
# the evaluation does not support.
BASELINE_NOTE = (
    "Measured on this dataset, no model beats a naive 'price stays the same' "
    "forecast at any horizon (median h=1 MAE 0.0000). Treat any prediction "
    "beyond the last observed price with caution. See docs/metrics.md."
)

AVAILABLE_MODELS = {
    "naive_last": "Repeat the last observed price. Best measured performer.",
    "moving_average_3": "Mean of the last 3 observations.",
    "moving_average_6": "Mean of the last 6 observations.",
    "drift": "Linear extrapolation from first to last observation.",
    "mean": "Mean of the entire history. Weak, included as a floor.",
    "arima_auto": "ARIMA with AIC-based order selection.",
}


def _build_model(name: str):
    from pulseiq.training.forecasting.baseline import Drift, Mean, MovingAverage, NaiveLast

    if name == "naive_last":
        return NaiveLast()
    if name == "moving_average_3":
        return MovingAverage(3)
    if name == "moving_average_6":
        return MovingAverage(6)
    if name == "drift":
        return Drift()
    if name == "mean":
        return Mean()
    if name == "arima_auto":
        from pulseiq.training.forecasting.arima_model import ARIMAForecaster

        return ARIMAForecaster(auto=True)
    raise HTTPException(
        status_code=400,
        detail=f"Unknown model '{name}'. Available: {', '.join(sorted(AVAILABLE_MODELS))}",
    )


@router.get("/models")
def list_models() -> dict[str, str]:
    """Available forecasting models and what each does."""
    return AVAILABLE_MODELS


@router.get("/products", response_model=list[ProductSummary])
def list_products(
    limit: int = Query(default=50, ge=1, le=500),
    min_observations: int = Query(default=8, ge=1),
    session=Depends(get_db),
) -> list[ProductSummary]:
    """Products with enough history to forecast."""
    from pulseiq.storage.repository import load_price_history

    frame = load_price_history(session, min_observations=min_observations)
    if frame.empty:
        return []

    summaries = []
    for name, group in frame.groupby("product_name"):
        ordered = group.sort_values("observed_on")
        summaries.append(
            ProductSummary(
                product_name=str(name),
                n_observations=len(ordered),
                first_observed=ordered["observed_on"].iloc[0],
                last_observed=ordered["observed_on"].iloc[-1],
                last_price=float(ordered["selling_price"].iloc[-1]),
            )
        )

    summaries.sort(key=lambda s: s.n_observations, reverse=True)
    return summaries[:limit]


@router.get("/history")
def price_history(
    product_name: str,
    limit: int = Query(default=12, ge=1, le=120),
    session=Depends(get_db),
) -> dict:
    """Recent observed prices for a product, oldest first.

    Exists so the dashboard can plot a forecast against the series it extends.
    A prediction shown without its history cannot be judged plausible or not.
    """
    import pandas as pd

    from pulseiq.features.resample import resample_panel
    from pulseiq.storage.repository import load_price_history

    frame = load_price_history(session, product_name=product_name)
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"No price history for '{product_name}'.")

    grid = resample_panel(frame, freq="MS", min_observed=1, max_fill_periods=3)
    recent = grid.sort_values("observed_on").tail(limit)

    # to_dict("records") rather than itertuples(): itertuples types every field
    # as a wide Scalar union, so `.date()` and `float()` cannot be verified
    # statically. Explicit pd.Timestamp() makes the conversion checkable.
    return {
        "product_name": product_name,
        "history": [
            {
                "date": pd.Timestamp(row["observed_on"]).date().isoformat(),
                "price": float(row["selling_price"]),
            }
            for row in recent.to_dict("records")
        ],
    }


@router.post("", response_model=ForecastResponse)
def forecast(
    request: ForecastRequest,
    session=Depends(get_db),
    cache=Depends(get_cache_dep),
) -> ForecastResponse:
    """Forecast future prices for one product."""
    from pulseiq.features.resample import resample_panel
    from pulseiq.storage.repository import load_price_history

    key = make_key("forecast", request.product_name, request.model, request.horizon)
    if (cached := cache.get(key)) is not None:
        return ForecastResponse(**{**cached, "cached": True})

    frame = load_price_history(session, product_name=request.product_name)
    if frame.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No price history for '{request.product_name}'. "
            f"See /forecast/products for available products.",
        )

    grid = resample_panel(frame, freq="MS", min_observed=4, max_fill_periods=3)
    if grid.empty or len(grid) < 4:
        raise HTTPException(
            status_code=422,
            detail=f"'{request.product_name}' has {len(frame)} observations, "
            f"too few to forecast. At least 4 monthly points are needed.",
        )

    ordered = grid.sort_values("observed_on")
    series = ordered["selling_price"].to_numpy(dtype=float)

    model = _build_model(request.model)
    try:
        predictions = model.fit_predict(series, request.horizon)
    except Exception as exc:  # noqa: BLE001
        logger.exception("forecast failed for %s", request.product_name)
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc

    response = ForecastResponse(
        product_name=request.product_name,
        model=model.name,
        n_observations=len(series),
        last_observed_price=float(series[-1]),
        last_observed_date=ordered["observed_on"].iloc[-1].date(),
        forecast=[
            ForecastPoint(period=i + 1, predicted_price=round(float(p), 2))
            for i, p in enumerate(predictions)
        ],
        baseline_note=BASELINE_NOTE,
    )

    cache.set(key, response.model_dump(mode="json"), ttl=cache_ttl())
    return response
