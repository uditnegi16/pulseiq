"""Forecasting training run: data -> grid -> split -> models -> metrics -> MLflow.

Orchestration only. Every step it calls is implemented and tested elsewhere;
this file just wires them in the right order and records what happened.

Usage (project root, venv active):

    python -m pulseiq.training.forecasting.train_forecast --source db
    python -m pulseiq.training.forecasting.train_forecast --source open-prices
    python -m pulseiq.training.forecasting.train_forecast --source open-prices \\
        --max-series 50 --no-prophet          # fast iteration
    python -m pulseiq.training.forecasting.train_forecast --source db --no-mlflow

WHY MLFLOW
----------
Twelve models across hundreds of series produces numbers that are impossible to
remember and easy to misreport later. MLflow stores parameters, metrics and the
leaderboard artifact per run, so "which configuration produced MAE 0.31" has an
answer six weeks from now. Defaults to `file:./mlruns`; set MLFLOW_TRACKING_URI
to point elsewhere (an S3-backed store in Phase 7) with no code change.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from config.settings import settings
from pulseiq.evaluation.harness import evaluate_models, paired_comparison
from pulseiq.features.resample import grid_report, resample_panel
from pulseiq.features.splits import InsufficientDataError, split_per_product
from pulseiq.training.forecasting.arima_model import ARIMAForecaster
from pulseiq.training.forecasting.baseline import (
    Drift,
    Mean,
    MovingAverage,
    NaiveLast,
    SeasonalNaive,
)

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,  # stdout, not a file: CloudWatch captures stdout
    )


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------


def load_from_db(min_observations: int = 8) -> pd.DataFrame:
    """Read price history from the relational store."""
    from pulseiq.storage.relational import get_engine, init_db, session_scope
    from pulseiq.storage.repository import load_price_history

    engine = get_engine()
    init_db(engine)
    with session_scope(engine) as session:
        frame = load_price_history(session, min_observations=min_observations)
    logger.info("loaded %d rows from %s", len(frame), settings.database_url)
    return frame


def load_from_open_prices(min_observations: int = 8, max_series: int | None = None) -> pd.DataFrame:
    """Read straight from the Open Prices export, bypassing the database.

    Useful for a first run before anything has been ingested.
    """
    from pulseiq.ingestion.seed_open_prices import read_open_prices, transform_open_prices
    from pulseiq.ingestion.validation import validate_price_snapshots

    raw = read_open_prices()
    rows = transform_open_prices(raw, min_observations=min_observations, max_series=max_series)
    records, report = validate_price_snapshots(rows)
    logger.info("open prices: %s", report.summary().splitlines()[0])
    return pd.DataFrame(
        [
            {
                "product_name": r.product_name,
                "observed_on": r.observed_on,
                "selling_price": r.selling_price,
                "source": r.source,
            }
            for r in records
        ]
    )


# --------------------------------------------------------------------------
# model set
# --------------------------------------------------------------------------


def build_factories(*, use_arima: bool = True, use_prophet: bool = True) -> list:
    """Model factories, not instances.

    Factories because each series needs a fresh model -- a fitted Forecaster
    holds the previous product's history.
    """
    factories = [
        NaiveLast,
        lambda: MovingAverage(3),
        lambda: MovingAverage(6),
        Drift,
        Mean,
        lambda: SeasonalNaive(12),
    ]
    if use_arima:
        factories.append(lambda: ARIMAForecaster(auto=True))
    if use_prophet:
        from pulseiq.training.forecasting.prophet_model import ProphetForecaster

        factories.append(ProphetForecaster)
    return factories


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def run(
    *,
    source: str = "db",
    freq: str = "MS",
    test_size: float = 0.2,
    min_observations: int = 8,
    max_series: int | None = None,
    max_fill_periods: int = 3,
    use_arima: bool = True,
    use_prophet: bool = True,
    use_mlflow: bool = True,
    output_dir: Path = Path("reports"),
) -> int:
    """Execute one full forecasting experiment. Returns a process exit code."""
    frame = (
        load_from_db(min_observations)
        if source == "db"
        else load_from_open_prices(min_observations, max_series)
    )

    if frame.empty:
        logger.error(
            "no data. Run ingestion first, or use --source open-prices to read the export directly."
        )
        return 1

    grid = resample_panel(
        frame,
        freq=freq,
        min_observed=min_observations,
        max_fill_periods=max_fill_periods,
    )
    if grid.empty:
        logger.error("no series survived resampling at freq=%s", freq)
        return 1
    print(f"\ngrid: {grid_report(grid)}")

    try:
        split = split_per_product(grid, test_size=test_size, min_observations=min_observations + 2)
    except InsufficientDataError as exc:
        logger.error("cannot split: %s", exc)
        return 1

    split.assert_no_leakage()
    print(f"split: train={len(split.train)} test={len(split.test)} cutoff={split.cutoff.date()}")

    factories = build_factories(use_arima=use_arima, use_prophet=use_prophet)
    print(f"evaluating {len(factories)} models...\n")

    report = evaluate_models(split, factories)
    if not report.results:
        logger.error("no results produced. skipped: %s", report.skipped)
        return 1

    print(report.summary())

    board = report.leaderboard()
    best = board.iloc[0]
    baseline_mae = float(board.loc[board["model"] == "naive_last", "mae"].iloc[0])
    improvement = (baseline_mae - float(best["mae"])) / baseline_mae * 100 if baseline_mae else 0.0

    print(f"\nbest: {best['model']} (MAE {best['mae']:.4f})")
    print(f"naive_last baseline: MAE {baseline_mae:.4f}")
    print(f"improvement over baseline: {improvement:+.1f}%")

    if best["model"] != "naive_last":
        paired = paired_comparison(report, str(best["model"]), "naive_last")
        print(
            f"paired vs naive on {paired['n']} shared series: "
            f"win rate {paired['win_rate']:.1%}, median delta {paired['median_delta']:+.4f}"
        )
        # A median-MAE win can come from a handful of series while the model
        # loses on most of them. The paired win rate is the honest read, and
        # saying so here stops a flattering headline being quoted alone.
        if paired["win_rate"] < 0.5:
            print(
                f"  VERDICT: despite the better median, {best['model']} beats naive_last on "
                f"only {paired['win_rate']:.0%} of series. The aggregate gain is driven by a "
                f"minority of series, not a general improvement."
            )
        else:
            print(
                f"  VERDICT: {best['model']} beats naive_last on {paired['win_rate']:.0%} "
                f"of series -- a consistent improvement, not an aggregation artefact."
            )
    else:
        print("  VERDICT: no model beat the naive baseline. Report this, do not hide it.")

    # --- artifacts --------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    board_path = output_dir / "forecast_leaderboard.csv"
    detail_path = output_dir / "forecast_per_series.csv"
    board.to_csv(board_path, index=False)
    report.to_frame().to_csv(detail_path, index=False)

    metrics_path = output_dir / "forecast_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_model": str(best["model"]),
                "best_mae": float(best["mae"]),
                "baseline_mae": baseline_mae,
                "improvement_pct": improvement,
                "n_series": int(best["series"]),
                "n_evaluations": len(report.results),
                "freq": freq,
                "test_size": test_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {board_path}, {detail_path}, {metrics_path}")

    # --- mlflow -----------------------------------------------------------
    if use_mlflow:
        try:
            _log_to_mlflow(
                board=board,
                report=report,
                params={
                    "source": source,
                    "freq": freq,
                    "test_size": test_size,
                    "min_observations": min_observations,
                    "max_fill_periods": max_fill_periods,
                    "n_series": int(best["series"]),
                    "use_arima": use_arima,
                    "use_prophet": use_prophet,
                },
                artifacts=[board_path, detail_path, metrics_path],
            )
        except Exception:  # noqa: BLE001 - tracking must never lose a completed run
            logger.exception("MLflow logging failed; results are still on disk")

    return 0


def _log_to_mlflow(*, board, report, params: dict, artifacts: list[Path]) -> None:
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="forecast_comparison"):
        mlflow.log_params(params)

        # One metric set per model, so the MLflow UI can rank them directly.
        for row in board.itertuples(index=False):
            safe = str(row.model).replace(".", "_")
            mlflow.log_metric(f"{safe}__mae", float(row.mae))
            mlflow.log_metric(f"{safe}__rmse", float(row.rmse))
            mlflow.log_metric(f"{safe}__smape", float(row.smape))
            mlflow.log_metric(f"{safe}__fallback_rate", float(row.fallback_rate))
            if pd.notna(row.mase):
                mlflow.log_metric(f"{safe}__mase", float(row.mase))

        best = board.iloc[0]
        mlflow.log_metric("best_mae", float(best["mae"]))
        mlflow.set_tag("best_model", str(best["model"]))
        mlflow.set_tag("n_evaluations", len(report.results))

        for path in artifacts:
            mlflow.log_artifact(str(path))

    logger.info("logged run to MLflow at %s", settings.mlflow_tracking_uri)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pulseiq.training.forecasting.train_forecast",
        description="Compare forecasting models on price history and log to MLflow.",
    )
    parser.add_argument(
        "--source",
        choices=["db", "open-prices"],
        default="db",
        help="read from the database, or straight from the Open Prices export",
    )
    parser.add_argument("--freq", default="MS", help="resample frequency (default: month start)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-observations", type=int, default=8)
    parser.add_argument("--max-series", type=int, default=None)
    parser.add_argument("--max-fill-periods", type=int, default=3)
    parser.add_argument("--no-arima", action="store_true")
    parser.add_argument("--no-prophet", action="store_true", help="much faster iteration")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--log-level", default=settings.log_level)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    return run(
        source=args.source,
        freq=args.freq,
        test_size=args.test_size,
        min_observations=args.min_observations,
        max_series=args.max_series,
        max_fill_periods=args.max_fill_periods,
        use_arima=not args.no_arima,
        use_prophet=not args.no_prophet,
        use_mlflow=not args.no_mlflow,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
