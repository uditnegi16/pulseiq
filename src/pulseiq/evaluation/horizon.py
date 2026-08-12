"""Horizon-curve evaluation: how does model advantage change with forecast distance?

WHY THIS EXISTS
---------------
The first Phase 2 run used a single 20% holdout, which on series spanning
2010-2026 meant forecasting roughly three years ahead on a monthly grid -- a
~36-step horizon. No method beats the naive forecast that far out, so the run
could only ever conclude "naive wins".

That answers "what will this cost in three years?". The useful question is
"what will this cost next month?".

Forecast skill is horizon-dependent, and reporting a single number hides that.
This module evaluates at h = 1, 3, 6, 12 using expanding-window origins, which
is how forecasting results are actually reported in the literature.

METHOD
------
For each horizon h, `rolling_origin_splits` produces several folds. Each fold
trains on everything before its cutoff and scores the next h steps. Averaging
across folds matters: on short series a single origin is noisy, and one unusual
quarter can swing MAE enough to reverse a ranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pulseiq.evaluation.harness import ModelFactory
from pulseiq.evaluation.metrics import EmptyEvaluationError, evaluate_forecast
from pulseiq.features.resample import observed_only
from pulseiq.features.splits import InsufficientDataError, rolling_origin_splits

logger = logging.getLogger(__name__)

DATE_COL = "observed_on"
GROUP_COL = "product_name"
PRICE_COL = "selling_price"

DEFAULT_HORIZONS = (1, 3, 6, 12)


@dataclass
class HorizonResult:
    """One model's score at one horizon on one series-fold."""

    model: str
    series: str
    horizon: int
    fold: int
    mae: float
    smape: float
    n_train: int
    n_test: int
    fallback: bool = False


@dataclass
class HorizonReport:
    results: list[HorizonResult] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def to_frame(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame(
                columns=[
                    "model",
                    "series",
                    "horizon",
                    "fold",
                    "mae",
                    "smape",
                    "n_train",
                    "n_test",
                    "fallback",
                ]
            )
        return pd.DataFrame([r.__dict__ for r in self.results])

    def curve(self) -> pd.DataFrame:
        """Median MAE per (model, horizon). The headline table."""
        frame = self.to_frame()
        if frame.empty:
            return frame
        return (
            frame.groupby(["model", "horizon"])
            .agg(
                mae=("mae", "median"),
                smape=("smape", "median"),
                n=("mae", "count"),
                fallback_rate=("fallback", "mean"),
            )
            .reset_index()
        )

    def pivot(self) -> pd.DataFrame:
        """Models as rows, horizons as columns, sorted by h=1 performance."""
        curve = self.curve()
        if curve.empty:
            return curve
        table = curve.pivot(index="model", columns="horizon", values="mae")
        first = table.columns.min()
        return table.sort_values(first)

    def skill_vs_naive(self, baseline: str = "naive_last") -> pd.DataFrame:
        """Percentage improvement over the baseline, per model per horizon.

        Positive means better than naive. Computed on the *paired* series-folds
        both models were scored on, so a model skipped on hard series cannot
        look good by comparison.
        """
        frame = self.to_frame()
        if frame.empty or baseline not in set(frame["model"]):
            return pd.DataFrame()

        key = ["series", "horizon", "fold"]
        base = frame[frame["model"] == baseline].set_index(key)["mae"]

        rows = []
        for model in sorted(set(frame["model"]) - {baseline}):
            other = frame[frame["model"] == model].set_index(key)["mae"]
            shared = base.index.intersection(other.index)
            if len(shared) == 0:
                continue
            paired = pd.DataFrame(
                {"base": base.loc[shared], "model": other.loc[shared]}
            ).reset_index()
            for horizon, group in paired.groupby("horizon"):
                improvement = (group["base"] - group["model"]) / group["base"].replace(0, np.nan)
                rows.append(
                    {
                        "model": model,
                        "horizon": int(horizon),
                        "median_improvement_pct": float(np.nanmedian(improvement) * 100),
                        "win_rate": float((group["model"] < group["base"]).mean()),
                        "n": int(len(group)),
                    }
                )
        return pd.DataFrame(rows)

    def summary(self, baseline: str = "naive_last") -> str:
        table = self.pivot()
        if table.empty:
            return f"no results. skipped: {self.skipped}"

        horizons = list(table.columns)
        lines = [
            f"{len(self.results)} evaluations | skipped: {self.skipped or 'none'}",
            "",
            "median MAE by forecast horizon (months ahead)",
            "",
            f"{'model':<20}" + "".join(f"{f'h={h}':>10}" for h in horizons),
        ]
        for model, row in table.iterrows():
            cells = "".join(
                f"{row[h]:>10.4f}" if pd.notna(row[h]) else f"{'-':>10}" for h in horizons
            )
            lines.append(f"{str(model):<20}{cells}")

        skill = self.skill_vs_naive(baseline)
        if not skill.empty:
            lines += [
                "",
                f"vs {baseline}: median paired improvement % (win rate)",
                "",
            ]
            lines.append(f"{'model':<20}" + "".join(f"{f'h={h}':>16}" for h in horizons))
            for model, group in skill.groupby("model"):
                by_h = group.set_index("horizon")
                cells = ""
                for h in horizons:
                    if h in by_h.index:
                        pct = by_h.loc[h, "median_improvement_pct"]
                        win = by_h.loc[h, "win_rate"]
                        cells += f"{pct:>+8.1f}% ({win:>3.0%})"
                    else:
                        cells += f"{'-':>16}"
                lines.append(f"{str(model):<20}{cells}")
            lines += [
                "",
                "Median improvement near 0% with a win rate near 50% means the two",
                "models are indistinguishable on this data -- not that the comparison",
                "failed. A win rate above 50% with a small median gain is a real but",
                "modest edge.",
            ]

        return "\n".join(lines)


def evaluate_horizons(
    grid: pd.DataFrame,
    factories: list[ModelFactory],
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    n_splits: int = 3,
    min_train: int = 12,
    score_observed_only: bool = True,
) -> HorizonReport:
    """Score every model at every horizon using expanding-window origins.

    Horizons are in grid periods -- months, given the monthly resample. A
    horizon of 1 is "next month"; 12 is "a year out".
    """
    report = HorizonReport()

    if grid.empty:
        report.skip("empty_grid")
        return report

    work = grid.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])

    groups = work.groupby(GROUP_COL) if GROUP_COL in work.columns else [("all", work)]

    for series_name, series_frame in groups:
        series_frame = series_frame.sort_values(DATE_COL).reset_index(drop=True)

        for horizon in horizons:
            # Horizons are monthly steps; rolling_origin_splits works in days.
            horizon_days = horizon * 31
            try:
                folds = rolling_origin_splits(
                    series_frame, n_splits=n_splits, horizon_days=horizon_days
                )
            except InsufficientDataError:
                report.skip(f"too_short_for_h{horizon}")
                continue

            for fold_index, fold in enumerate(folds):
                if len(fold.train) < min_train:
                    report.skip("train_too_short")
                    continue

                test_frame = observed_only(fold.test) if score_observed_only else fold.test
                if test_frame.empty:
                    report.skip("no_observed_test_rows")
                    continue

                y_train = fold.train[PRICE_COL].to_numpy(dtype=float)
                y_test = test_frame[PRICE_COL].to_numpy(dtype=float)

                for factory in factories:
                    model = factory()
                    try:
                        preds = model.fit_predict(y_train, len(y_test))
                        metrics = evaluate_forecast(y_test, preds)
                    except EmptyEvaluationError:
                        report.skip("empty_evaluation")
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "%s failed on %s h=%d: %s", model.name, series_name, horizon, exc
                        )
                        report.skip(f"model_error:{model.name}")
                        continue

                    report.results.append(
                        HorizonResult(
                            model=model.name,
                            series=str(series_name),
                            horizon=horizon,
                            fold=fold_index,
                            mae=metrics.mae,
                            smape=metrics.smape,
                            n_train=len(y_train),
                            n_test=len(y_test),
                            fallback=bool(getattr(model, "used_fallback", False)),
                        )
                    )

    logger.info("horizon evaluation: %d results", len(report.results))
    return report
