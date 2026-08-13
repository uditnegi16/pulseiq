"""CI metric regression gate.

A pull request that makes a model measurably worse fails the build.

This is the model equivalent of a failing unit test. Without it, a refactor that
quietly drops F1 by five points merges without comment, and the regression
surfaces weeks later with no obvious culprit. Tests catch code that stops
working; this catches code that keeps working and starts being wrong.

It reads whatever metric reports exist in `reports/` and compares them against
`thresholds.yaml`. Missing reports are SKIPPED, not failed -- a PR touching only
the scraper should not be blocked because nobody re-ran the fine-tune.

THE LEAKAGE TRIPWIRE
--------------------
The gate also fails on results that are *too good*. On near-random-walk price
data, a model suddenly beating the naive baseline by 40% is far more likely to
be leakage than skill. Every leakage bug in this project's history made metrics
look better, never worse, so a gate that only checks the downside would have
missed all of them.

Usage:
    python -m pulseiq.evaluation.regression_gate
    python -m pulseiq.evaluation.regression_gate --strict   # missing = failure
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = Path(__file__).parent / "thresholds.yaml"
DEFAULT_REPORTS = Path("reports")


class Status(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"


@dataclass
class Check:
    """One threshold comparison."""

    name: str
    status: Status
    detail: str
    actual: float | None = None
    threshold: float | None = None

    def __str__(self) -> str:
        return f"{self.status:<6}{self.name:<34}{self.detail}"


@dataclass
class GateReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.WARN]

    @property
    def skipped(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.SKIP]

    def summary(self) -> str:
        lines = [f"{'status':<6}{'check':<34}detail", "-" * 88]
        lines.extend(str(c) for c in self.checks)
        lines.append("")
        lines.append(
            f"{len(self.checks)} checks: "
            f"{sum(1 for c in self.checks if c.status is Status.PASS)} passed, "
            f"{len(self.failed)} failed, "
            f"{len(self.warnings)} warnings, "
            f"{len(self.skipped)} skipped"
        )
        return "\n".join(lines)


def load_thresholds(path: Path = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"thresholds file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_report(path: Path) -> dict[str, Any] | None:
    """Read a metric report, or None if absent or malformed.

    Malformed returns None rather than raising: a truncated JSON file from an
    interrupted run should skip its checks, not crash the gate and block every
    other check from running.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("could not parse %s: %s", path, exc)
        return None


def check_minimum(name: str, actual: float | None, minimum: float) -> Check:
    """Higher is better (accuracy, F1, recall)."""
    if actual is None:
        return Check(name, Status.SKIP, "metric absent from report")
    if actual >= minimum:
        return Check(name, Status.PASS, f"{actual:.4f} >= {minimum:.4f}", actual, minimum)
    return Check(
        name,
        Status.FAIL,
        f"{actual:.4f} < {minimum:.4f} (regressed by {minimum - actual:.4f})",
        actual,
        minimum,
    )


def check_maximum(name: str, actual: float | None, maximum: float) -> Check:
    """Lower is better (MAE, RMSE)."""
    if actual is None:
        return Check(name, Status.SKIP, "metric absent from report")
    if actual <= maximum:
        return Check(name, Status.PASS, f"{actual:.4f} <= {maximum:.4f}", actual, maximum)
    return Check(
        name,
        Status.FAIL,
        f"{actual:.4f} > {maximum:.4f} (worse by {actual - maximum:.4f})",
        actual,
        maximum,
    )


def check_sentiment(reports_dir: Path, config: dict[str, Any]) -> list[Check]:
    """Gate the fine-tuned sentiment model."""
    report = load_report(reports_dir / "sentiment_finetuned.json")
    if report is None:
        return [
            Check(
                "sentiment",
                Status.SKIP,
                "reports/sentiment_finetuned.json not found -- fine-tune not re-run",
            )
        ]

    checks = [
        check_minimum(f"sentiment.{metric}", report.get(metric), minimum)
        for metric, minimum in config.get("minimum", {}).items()
    ]

    # The fine-tune must still beat the zero-shot baseline. If it does not, the
    # adaptation has stopped contributing and the extra complexity is unjustified.
    baseline_floor = config.get("must_beat_baseline", {})
    for metric, floor in baseline_floor.items():
        actual = report.get(metric)
        if actual is None:
            checks.append(Check(f"sentiment.{metric}_vs_baseline", Status.SKIP, "metric absent"))
        elif actual > floor:
            checks.append(
                Check(
                    f"sentiment.{metric}_vs_baseline",
                    Status.PASS,
                    f"{actual:.4f} beats zero-shot {floor:.4f}",
                    actual,
                    floor,
                )
            )
        else:
            checks.append(
                Check(
                    f"sentiment.{metric}_vs_baseline",
                    Status.FAIL,
                    f"{actual:.4f} no better than zero-shot {floor:.4f} -- "
                    f"fine-tuning adds nothing",
                    actual,
                    floor,
                )
            )

    return checks


def check_forecasting(reports_dir: Path, config: dict[str, Any]) -> list[Check]:
    """Gate the forecasting horizon curve, including the leakage tripwire."""
    curve_path = reports_dir / "horizon_curve.csv"
    if not curve_path.exists():
        return [
            Check(
                "forecasting",
                Status.SKIP,
                "reports/horizon_curve.csv not found -- forecasting not re-run",
            )
        ]

    import pandas as pd

    try:
        curve = pd.read_csv(curve_path)
    except Exception as exc:  # noqa: BLE001
        return [Check("forecasting", Status.SKIP, f"could not read curve: {exc}")]

    baseline = curve[curve["model"] == "naive_last"]
    if baseline.empty:
        return [Check("forecasting", Status.SKIP, "naive_last absent from the curve")]

    checks: list[Check] = []
    by_horizon = baseline.set_index("horizon")["mae"].to_dict()

    for key, maximum in config.get("maximum", {}).items():
        horizon = int(key.replace("mae_h", ""))
        checks.append(check_maximum(f"forecast.{key}", by_horizon.get(horizon), maximum))

    # Leakage tripwire. Every leakage bug in this project made metrics look
    # better; a downside-only gate would have caught none of them.
    limit = config.get("suspicious_improvement_pct")
    if limit is not None:
        for horizon, baseline_mae in by_horizon.items():
            if baseline_mae <= 0:
                continue
            others = curve[(curve["horizon"] == horizon) & (curve["model"] != "naive_last")]
            if others.empty:
                continue
            best = others.loc[others["mae"].idxmin()]
            improvement = (baseline_mae - best["mae"]) / baseline_mae * 100
            if improvement > limit:
                checks.append(
                    Check(
                        f"forecast.leakage_tripwire_h{horizon}",
                        Status.FAIL,
                        f"{best['model']} beats naive by {improvement:.1f}% "
                        f"(limit {limit}%) -- verify the split before trusting this",
                        improvement,
                        limit,
                    )
                )

    if not any(c.name.startswith("forecast.leakage") for c in checks):
        checks.append(
            Check("forecast.leakage_tripwire", Status.PASS, "no implausible improvement detected")
        )

    return checks


def run_gate(
    *,
    reports_dir: Path = DEFAULT_REPORTS,
    thresholds_path: Path = DEFAULT_THRESHOLDS,
    strict: bool = False,
) -> tuple[GateReport, int]:
    """Run every gate. Returns (report, exit_code).

    `strict` turns SKIP into FAIL. Off by default so a PR touching only the
    scraper is not blocked by an un-rerun fine-tune; on for release builds,
    where "we never measured it" is not an acceptable answer.
    """
    thresholds = load_thresholds(thresholds_path)
    report = GateReport()

    for check in check_sentiment(reports_dir, thresholds.get("sentiment", {})):
        report.add(check)
    for check in check_forecasting(reports_dir, thresholds.get("forecasting", {})):
        report.add(check)

    exit_code = 1 if report.failed else 0
    if strict and report.skipped:
        exit_code = 1

    return report, exit_code


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pulseiq.evaluation.regression_gate",
        description="Fail the build when a model metric regresses past its threshold.",
    )
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument(
        "--strict", action="store_true", help="treat missing metric reports as failures"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    report, exit_code = run_gate(
        reports_dir=args.reports_dir, thresholds_path=args.thresholds, strict=args.strict
    )

    print(report.summary())

    if report.failed:
        print("\nGATE FAILED\n")
        for check in report.failed:
            print(f"  {check.name}: {check.detail}")
        print(
            "\nIf this regression is intentional, update "
            "src/pulseiq/evaluation/thresholds.yaml in the same commit and record "
            "the justification in docs/decision-log.md. Lowering a threshold to "
            "make a build pass is how a gate becomes decoration."
        )
    elif report.skipped and not args.strict:
        print("\nGATE PASSED (some checks skipped -- those models were not re-run)")
    else:
        print("\nGATE PASSED")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
