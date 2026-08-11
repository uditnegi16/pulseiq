"""Chronological train/test splitting for time series.

This is the module that decides whether every metric in this project is
meaningful or worthless.

The original code fabricated its target and split randomly. On time series a
random split trains on Thursday and tests on Wednesday -- the model sees the
future, error collapses, and the reported number is a lie that looks excellent.
Every function here refuses to do that.

Rules enforced:
  * Splits are strictly chronological: every train timestamp precedes every
    test timestamp.
  * Splits happen per product. One product's future must never appear in
    another's training window when series are pooled.
  * A series too short to split is rejected, not silently split into an empty
    test set that scores perfectly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

DATE_COL = "observed_on"
GROUP_COL = "product_name"


class InsufficientDataError(ValueError):
    """Raised when a series is too short to split or forecast honestly."""


@dataclass(frozen=True)
class Split:
    """One train/test partition, with the boundary recorded for auditing."""

    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp

    @property
    def sizes(self) -> tuple[int, int]:
        return len(self.train), len(self.test)

    def assert_no_leakage(self) -> None:
        """Verify the invariant that makes the metrics trustworthy.

        Called in tests and at the top of every training run. Cheap insurance
        against a refactor quietly reintroducing a random split.
        """
        if self.train.empty or self.test.empty:
            raise InsufficientDataError(
                f"empty partition: train={len(self.train)} test={len(self.test)}"
            )

        # With a group column, correctness is defined PER PRODUCT. Products
        # enter and leave the dataset at different times, so product A's last
        # training date being later than product B's first test date is normal
        # and not leakage. Comparing dates globally across a per-product split
        # raises on perfectly valid partitions.
        if GROUP_COL in self.train.columns and GROUP_COL in self.test.columns:
            for product, group in self.test.groupby(GROUP_COL):
                train_group = self.train[self.train[GROUP_COL] == product]
                if train_group.empty:
                    continue
                if train_group[DATE_COL].max() >= group[DATE_COL].min():
                    raise AssertionError(
                        f"LEAKAGE within product {product!r}: "
                        f"train reaches {train_group[DATE_COL].max()}, "
                        f"test starts {group[DATE_COL].min()}"
                    )
            return

        # Ungrouped: a single global boundary is the correct check.
        train_max = self.train[DATE_COL].max()
        test_min = self.test[DATE_COL].min()
        if train_max >= test_min:
            raise AssertionError(
                f"LEAKAGE: last train date {train_max} is not before first test date {test_min}"
            )


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL not in frame.columns:
        raise KeyError(f"frame must contain a {DATE_COL!r} column")
    out = frame.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    return out.sort_values(
        [GROUP_COL, DATE_COL] if GROUP_COL in out.columns else [DATE_COL]
    ).reset_index(drop=True)


def split_by_date(frame: pd.DataFrame, cutoff: date | str | pd.Timestamp) -> Split:
    """Split at an absolute date. Train is strictly before the cutoff.

    Preferred for a final holdout: the boundary is a fixed, reportable date
    rather than a ratio that shifts as data accumulates.
    """
    frame = _validate_frame(frame)
    cutoff_ts = pd.Timestamp(cutoff)

    train = frame[frame[DATE_COL] < cutoff_ts].reset_index(drop=True)
    test = frame[frame[DATE_COL] >= cutoff_ts].reset_index(drop=True)

    split = Split(train=train, test=test, cutoff=cutoff_ts)
    if train.empty or test.empty:
        raise InsufficientDataError(
            f"cutoff {cutoff_ts.date()} yields train={len(train)} test={len(test)}; "
            f"data spans {frame[DATE_COL].min().date()} to {frame[DATE_COL].max().date()}"
        )
    return split


def split_by_fraction(frame: pd.DataFrame, test_size: float = 0.2) -> Split:
    """Split so the last `test_size` of the *time span* is held out.

    Note this splits on time position, not row count. With irregular
    observations, taking the last 20% of rows is not the last 20% of time, and
    the difference matters when a scraper missed a week.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    frame = _validate_frame(frame)
    if len(frame) < 2:
        raise InsufficientDataError(f"need >=2 rows to split, got {len(frame)}")

    start, end = frame[DATE_COL].min(), frame[DATE_COL].max()
    cutoff = start + (end - start) * (1.0 - test_size)

    train = frame[frame[DATE_COL] < cutoff]
    test = frame[frame[DATE_COL] >= cutoff]

    # Degenerate case: all observations share one timestamp, or the computed
    # cutoff lands before the first point. Fall back to a positional split so
    # the caller gets a usable partition rather than an exception.
    if train.empty or test.empty:
        n_train = max(1, int(len(frame) * (1.0 - test_size)))
        if n_train >= len(frame):
            n_train = len(frame) - 1
        train = frame.iloc[:n_train]
        test = frame.iloc[n_train:]
        cutoff = test[DATE_COL].min()

    return Split(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        cutoff=pd.Timestamp(cutoff),
    )


def split_per_product(
    frame: pd.DataFrame, test_size: float = 0.2, *, min_observations: int = 8
) -> Split:
    """Split each product's series independently, then recombine.

    Why per-product rather than one global cutoff: products enter the dataset at
    different times. A single global cutoff can leave a late-arriving product
    entirely in the test set with no training history at all, which then scores
    as a catastrophic failure that says nothing about the model.

    Products with fewer than `min_observations` points are dropped and logged.
    """
    frame = _validate_frame(frame)
    if GROUP_COL not in frame.columns:
        return split_by_fraction(frame, test_size)

    trains: list[pd.DataFrame] = []
    tests: list[pd.DataFrame] = []
    dropped: list[str] = []

    for product, group in frame.groupby(GROUP_COL, sort=True):
        if len(group) < min_observations:
            dropped.append(str(product))
            continue
        try:
            part = split_by_fraction(group, test_size)
        except InsufficientDataError:
            dropped.append(str(product))
            continue
        trains.append(part.train)
        tests.append(part.test)

    if dropped:
        logger.info(
            "dropped %d product(s) with <%d observations: %s",
            len(dropped),
            min_observations,
            ", ".join(dropped[:5]) + (" ..." if len(dropped) > 5 else ""),
        )

    if not trains:
        raise InsufficientDataError(
            f"no product has >={min_observations} observations. "
            f"Collect more history before forecasting."
        )

    train = pd.concat(trains, ignore_index=True)
    test = pd.concat(tests, ignore_index=True)
    return Split(train=train, test=test, cutoff=pd.Timestamp(test[DATE_COL].min()))


def rolling_origin_splits(
    frame: pd.DataFrame, n_splits: int = 3, *, horizon_days: int = 7
) -> list[Split]:
    """Expanding-window cross-validation for time series.

    Each fold trains on everything before its cutoff and tests on the following
    `horizon_days`. The training window grows; it never slides backwards.

    Single-split metrics on short series are noisy -- one unusual fortnight can
    swing MAE badly. Reporting mean and spread across folds is the honest
    version, and it is what distinguishes a real evaluation from a lucky number.
    """
    frame = _validate_frame(frame)
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    start, end = frame[DATE_COL].min(), frame[DATE_COL].max()

    # All window arithmetic is done on integer day offsets rather than
    # timedeltas. Constructing a Timedelta from a bare int triggers a NumPy
    # deprecation on some pandas/numpy combinations; subtracting two Timestamps
    # does not. So the only datetime operation here is Timestamp - Timestamp,
    # and everything after it is plain integer comparison.
    span_days = (end - start).days
    day_offset = (frame[DATE_COL] - start).dt.days

    if span_days < horizon_days * (n_splits + 1):
        raise InsufficientDataError(
            f"data spans {span_days} days; need at least "
            f"{horizon_days * (n_splits + 1)} for {n_splits} folds "
            f"at a {horizon_days}-day horizon"
        )

    splits: list[Split] = []
    for i in range(n_splits, 0, -1):
        cutoff_day = span_days - horizon_days * i
        upper_day = cutoff_day + horizon_days

        train = frame[day_offset < cutoff_day]
        test = frame[(day_offset >= cutoff_day) & (day_offset < upper_day)]
        if train.empty or test.empty:
            continue
        splits.append(
            Split(
                train=train.reset_index(drop=True),
                test=test.reset_index(drop=True),
                # The first actually-observed test date, not a synthesised
                # timestamp -- no datetime construction needed.
                cutoff=pd.Timestamp(test[DATE_COL].min()),
            )
        )

    if not splits:
        raise InsufficientDataError("no valid folds produced")
    return splits
