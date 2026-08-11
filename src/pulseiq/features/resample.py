"""Resample irregular price observations onto a regular time grid.

WHY THIS EXISTS
---------------
The Open Prices data is crowdsourced, so observations arrive whenever somebody
photographs a shelf tag. Measured on the real export: median gap 17 days, p75
31, p90 63. That is not a regular series.

ARIMA assumes evenly spaced observations. Handing it irregular timestamps
produces numbers that look fine and mean nothing, because the model believes
step t-1 is always the same distance from step t. Prophet handles irregular
timestamps natively, but running the models on different grids would make their
metrics incomparable -- so every model is evaluated on the same monthly grid.

FORWARD FILL IS A MODELLING CHOICE, NOT A CONVENIENCE
-----------------------------------------------------
A price is a step function: it holds until somebody changes it. "The price in
March was whatever it was in February, because nobody observed a change" is a
defensible statement about prices in a way it would not be about, say,
temperature.

THE TRAP: NEVER SCORE AGAINST AN IMPUTED VALUE
----------------------------------------------
A forward-filled point is a copy of the previous point. A naive "predict the
last value" model scores *perfectly* on it. If imputed rows stay in the test
set, every model's error collapses toward zero and the best model appears to be
the dumbest one.

So every row is tagged `is_imputed`, and evaluation filters to observed rows
only. Imputed rows remain available for *training* -- they carry the correct
"price did not change" signal -- but never for scoring.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DATE_COL = "observed_on"
GROUP_COL = "product_name"
PRICE_COL = "selling_price"


def resample_series(
    frame: pd.DataFrame,
    *,
    freq: str = "MS",
    aggregate: str = "median",
    max_fill_periods: int | None = 3,
) -> pd.DataFrame:
    """Put one product's observations on a regular grid.

    Args:
        freq: pandas offset alias. "MS" = month start, the natural bucket for a
            17-day median gap. Weekly would be mostly holes.
        aggregate: how to combine multiple observations inside one bucket.
            Median, not mean -- crowdsourced prices contain typos, and one
            mistyped 999.00 would drag a mean badly.
        max_fill_periods: how far a value may be carried forward. None means
            unlimited. The default of 3 means a price is not assumed to hold for
            more than a quarter without evidence; beyond that the gap stays NaN
            and the series is treated as discontinuous.

    Returns a frame with an added boolean `is_imputed` column.
    """
    if frame.empty:
        out = frame.copy()
        out["is_imputed"] = pd.Series(dtype=bool)
        return out

    work = frame.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])
    work = work.sort_values(DATE_COL)

    grouped = work.set_index(DATE_COL)[PRICE_COL].resample(freq)
    observed = grouped.agg(aggregate)

    # Count of real observations per bucket: 0 means the bucket is a gap.
    counts = grouped.count()

    filled = observed.ffill(limit=max_fill_periods)

    out = pd.DataFrame(
        {
            DATE_COL: filled.index,
            PRICE_COL: filled.to_numpy(),
            "is_imputed": (counts.to_numpy() == 0),
        }
    )

    # Leading buckets before the first observation, and gaps beyond the fill
    # limit, are genuinely unknown -- drop rather than invent.
    out = out[out[PRICE_COL].notna()].reset_index(drop=True)

    if GROUP_COL in work.columns:
        out[GROUP_COL] = work[GROUP_COL].iloc[0]
    if "source" in work.columns:
        out["source"] = work["source"].iloc[0]

    return out


def resample_panel(
    frame: pd.DataFrame,
    *,
    freq: str = "MS",
    aggregate: str = "median",
    max_fill_periods: int | None = 3,
    min_observed: int = 8,
) -> pd.DataFrame:
    """Resample every product independently and recombine.

    `min_observed` counts *real* observations, not grid points. A series can
    have 30 monthly rows of which 25 are forward-filled copies; that series has
    5 facts in it, not 30, and cannot support a forecast.
    """
    if frame.empty:
        return frame.copy()

    work = frame.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])

    if GROUP_COL not in work.columns:
        return resample_series(
            work, freq=freq, aggregate=aggregate, max_fill_periods=max_fill_periods
        )

    pieces: list[pd.DataFrame] = []
    dropped = 0

    for _product, group in work.groupby(GROUP_COL, sort=True):
        resampled = resample_series(
            group, freq=freq, aggregate=aggregate, max_fill_periods=max_fill_periods
        )
        observed_count = int((~resampled["is_imputed"]).sum())
        if observed_count < min_observed:
            dropped += 1
            continue
        pieces.append(resampled)

    if dropped:
        logger.info(
            "dropped %d series with <%d real observations after resampling",
            dropped,
            min_observed,
        )

    if not pieces:
        logger.warning("no series survived resampling at freq=%s", freq)
        return pd.DataFrame(columns=[GROUP_COL, DATE_COL, PRICE_COL, "is_imputed"])

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    imputed_share = out["is_imputed"].mean()
    logger.info(
        "resampled to %s: %d rows across %d series, %.1f%% imputed",
        freq,
        len(out),
        out[GROUP_COL].nunique(),
        imputed_share * 100,
    )
    return out


def observed_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop imputed rows. Apply to any frame before scoring.

    Forward-filled rows are copies of their predecessor, so a naive last-value
    model predicts them exactly. Scoring on them rewards the model that learns
    least.
    """
    if "is_imputed" not in frame.columns:
        return frame
    return frame[~frame["is_imputed"]].reset_index(drop=True)


def grid_report(frame: pd.DataFrame) -> str:
    """One-line summary of a resampled panel, for logs and the phase log."""
    if frame.empty:
        return "empty grid"
    observed = int((~frame["is_imputed"]).sum())
    per_series = frame.groupby(GROUP_COL).size() if GROUP_COL in frame.columns else frame.size
    return (
        f"{len(frame)} grid points | "
        f"{frame[GROUP_COL].nunique() if GROUP_COL in frame.columns else 1} series | "
        f"{observed} observed ({observed / len(frame):.1%}) | "
        f"median points/series {pd.Series(per_series).median():.0f} | "
        f"{frame[DATE_COL].min().date()}..{frame[DATE_COL].max().date()}"
    )
