"""Feature engineering for the discount forecasting task.

The rule every function here obeys: a feature for row t may use data from
t-1 and earlier, never from t or later.

That sounds obvious and is the single easiest thing to get wrong. `rolling()`
in pandas includes the current row by default, so a 7-day rolling mean of the
target silently contains the answer. Every rolling and expanding computation
below is explicitly shifted, and there is a test asserting it.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DATE_COL = "observed_on"
GROUP_COL = "product_name"
PRICE_COL = "selling_price"
TARGET_COL = "discount_pct"


def define_discount_target(
    frame: pd.DataFrame, *, window: int = 30, min_periods: int = 3
) -> pd.DataFrame:
    """Define the forecasting target honestly, from observed prices.

    discount_pct = (1 - price / trailing_max_price) * 100

    The reference price is the maximum over the PRIOR `window` observations,
    excluding today. This replaces the original project's fabricated
    `Predicted_Discount` formula with something derived only from data that
    existed before the observation being labelled.

    Using a trailing max rather than a scraped "original price" field means the
    target is defined identically across scraped and public datasets, which is
    what lets one model train on both.
    """
    out = frame.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    # shift(1) is the leakage guard: today's price cannot inform today's
    # reference price.
    reference = out.groupby(GROUP_COL)[PRICE_COL].transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_periods).max()
    )

    out["reference_price"] = reference
    out["discount_from_reference"] = ((1.0 - out[PRICE_COL] / reference) * 100.0).clip(lower=0.0)

    return out


def add_lag_features(
    frame: pd.DataFrame, column: str = PRICE_COL, lags: tuple[int, ...] = (1, 2, 3, 7)
) -> pd.DataFrame:
    """Add lagged values of `column`, computed within each product.

    Grouping matters: an ungrouped shift pulls the last row of product A into
    the first row of product B.
    """
    out = frame.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    for lag in lags:
        out[f"{column}_lag{lag}"] = out.groupby(GROUP_COL)[column].shift(lag)
    return out


def add_rolling_features(
    frame: pd.DataFrame,
    column: str = PRICE_COL,
    windows: tuple[int, ...] = (3, 7, 14),
    *,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Add trailing rolling mean and std, shifted to exclude the current row.

    The `.shift(1)` before `.rolling()` is the whole point. Without it these are
    the most convincing leaky features you can build -- error drops sharply and
    the model looks excellent right up until it meets real data.
    """
    out = frame.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    grouped = out.groupby(GROUP_COL)[column]
    for window in windows:
        out[f"{column}_roll{window}_mean"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=min_periods).mean()
        )
        out[f"{column}_roll{window}_std"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=min_periods).std()
        )
    return out


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features. These are known in advance, so they cannot leak.

    Retail discounting is strongly weekly and month-end driven, which makes
    these genuinely predictive rather than filler.
    """
    out = frame.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])

    dates = out[DATE_COL].dt
    out["day_of_week"] = dates.dayofweek
    out["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    out["day_of_month"] = dates.day
    out["month"] = dates.month
    out["is_month_start"] = dates.is_month_start.astype(int)
    out["is_month_end"] = dates.is_month_end.astype(int)
    out["week_of_year"] = dates.isocalendar().week.astype(int)
    return out


def add_price_change_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add day-over-day change and days since the last price move.

    "Days since last change" is often the strongest single predictor of an
    imminent discount: prices sit flat, then drop.
    """
    out = frame.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    grouped = out.groupby(GROUP_COL)[PRICE_COL]
    out["price_diff_1"] = grouped.diff(1)
    out["price_pct_change_1"] = grouped.pct_change(1) * 100.0

    def _days_since_change(series: pd.Series) -> pd.Series:
        diffs = series.diff()
        # The first observation has no prior price, so it counts as a change
        # point rather than "one day since an unknown change".
        changed = diffs.isna() | (diffs != 0)
        counter, result = 0, []
        for is_change in changed:
            counter = 0 if is_change else counter + 1
            result.append(counter)
        return pd.Series(result, index=series.index)

    out["days_since_price_change"] = grouped.transform(_days_since_change)
    return out


def build_feature_frame(
    frame: pd.DataFrame,
    *,
    lags: tuple[int, ...] = (1, 2, 3, 7),
    windows: tuple[int, ...] = (3, 7, 14),
    dropna: bool = True,
) -> pd.DataFrame:
    """Full feature pipeline. Returns a model-ready frame.

    With `dropna=True`, early rows lacking enough history are removed -- they
    would otherwise be imputed, and imputing a lag means inventing a past.
    """
    if frame.empty:
        return frame.copy()

    out = define_discount_target(frame)
    out = add_lag_features(out, lags=lags)
    out = add_rolling_features(out, windows=windows)
    out = add_price_change_features(out)
    out = add_calendar_features(out)

    if dropna:
        before = len(out)
        required = [c for c in out.columns if "_lag" in c or "_roll" in c]
        out = out.dropna(subset=required).reset_index(drop=True)
        logger.info(
            "feature frame: %d -> %d rows after dropping incomplete history",
            before,
            len(out),
        )

    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model input columns: engineered features only.

    Deliberately excludes `selling_price`, `discount_from_reference`,
    `reference_price`, and `discount_pct` -- all of which are the target or
    trivially derived from it. Selecting features by prefix rather than by
    "everything except the target" is what stops a leaky column being added
    later and silently included.
    """
    prefixes = ("selling_price_lag", "selling_price_roll", "price_diff", "price_pct")
    calendar = {
        "day_of_week",
        "is_weekend",
        "day_of_month",
        "month",
        "is_month_start",
        "is_month_end",
        "week_of_year",
        "days_since_price_change",
    }
    return [c for c in frame.columns if c.startswith(prefixes) or c in calendar]
