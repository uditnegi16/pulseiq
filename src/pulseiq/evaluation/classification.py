"""Classification metrics for the sentiment task.

Separate from `metrics.py` (forecasting) because the failure modes are
different. The one that matters here:

ACCURACY IS ALMOST ALWAYS THE WRONG HEADLINE
--------------------------------------------
On an imbalanced set, a model that always predicts the majority class scores
well and has learned nothing. The sentiment set is deliberately balanced, so
accuracy is meaningful *there* -- but the discount-detection task in this same
project has a 1.9% positive rate, where 98% accuracy is the do-nothing result.

Every function here therefore reports macro-F1 and per-class recall alongside
accuracy, and `majority_baseline` exists so the do-nothing score is always
visible next to the model's.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

logger = logging.getLogger(__name__)


class EmptyEvaluationError(ValueError):
    """Raised when there is nothing to score."""


@dataclass(frozen=True)
class ClassificationMetrics:
    """Binary classification results, with the confusion matrix kept intact.

    The four cells are stored rather than only the summary statistics, because
    "which mistakes" is a different and often more useful question than "how
    many". A model that misses negatives and one that over-predicts them can
    share an F1 score and need opposite fixes.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @property
    def confusion_matrix(self) -> list[list[int]]:
        """[[TN, FP], [FN, TP]] -- the sklearn convention."""
        return [
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives],
        ]

    @property
    def negative_recall(self) -> float:
        """Share of true negatives correctly identified.

        Reported explicitly because it is the first thing to collapse when a
        model drifts toward predicting the majority class, and an overall
        accuracy figure hides that completely.
        """
        denominator = self.true_negatives + self.false_positives
        return self.true_negatives / denominator if denominator else 0.0

    def __str__(self) -> str:
        return (
            f"acc={self.accuracy:.4f} f1={self.f1:.4f} macro_f1={self.macro_f1:.4f} "
            f"prec={self.precision:.4f} rec={self.recall:.4f} n={self.n}"
        )

    def confusion_table(self) -> str:
        """Human-readable confusion matrix for logs and reports."""
        return (
            f"{'':>12}{'pred neg':>10}{'pred pos':>10}\n"
            f"{'true neg':>12}{self.true_negatives:>10}{self.false_positives:>10}\n"
            f"{'true pos':>12}{self.false_negatives:>10}{self.true_positives:>10}"
        )


def _clean_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true_arr = np.asarray(y_true).ravel()
    pred_arr = np.asarray(y_pred).ravel()

    if true_arr.shape != pred_arr.shape:
        raise ValueError(f"shape mismatch: {true_arr.shape} vs {pred_arr.shape}")
    if true_arr.size == 0:
        raise EmptyEvaluationError("no predictions to score")

    return true_arr.astype(int), pred_arr.astype(int)


def evaluate_classification(y_true, y_pred) -> dict[str, float | int]:
    """All binary classification metrics. The single scoring entry point.

    `zero_division=0` throughout: when a model never predicts the positive
    class, precision is undefined. Reporting 0 is correct and honest; letting
    sklearn warn and return 0 silently is the same number without the record.
    """
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    true_arr, pred_arr = _clean_pair(y_true, y_pred)

    matrix = confusion_matrix(true_arr, pred_arr, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics = ClassificationMetrics(
        accuracy=float((true_arr == pred_arr).mean()),
        precision=float(precision_score(true_arr, pred_arr, zero_division=0)),
        recall=float(recall_score(true_arr, pred_arr, zero_division=0)),
        f1=float(f1_score(true_arr, pred_arr, zero_division=0)),
        macro_f1=float(f1_score(true_arr, pred_arr, average="macro", zero_division=0)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        n=int(len(true_arr)),
    )
    return metrics.as_dict()


def to_metrics(payload: dict) -> ClassificationMetrics:
    """Rebuild the dataclass from a dict, for reporting helpers."""
    fields = {
        k: payload[k]
        for k in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "macro_f1",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
            "n",
        )
    }
    return ClassificationMetrics(**fields)


def majority_baseline(y_true) -> dict[str, float | int]:
    """Score a model that always predicts the majority class.

    Report this next to every classification result. It is the floor: any model
    not clearing it has learned nothing, and on an imbalanced set that floor can
    be surprisingly high.
    """
    true_arr = np.asarray(y_true).ravel().astype(int)
    if true_arr.size == 0:
        raise EmptyEvaluationError("no labels")

    majority = int(np.bincount(true_arr, minlength=2).argmax())
    return evaluate_classification(true_arr, np.full_like(true_arr, majority))


def compare(before: dict, after: dict) -> dict[str, float]:
    """Absolute and relative change between two metric sets.

    `error_reduction_pct` is the honest way to report a gain at high accuracy.
    Going from 91% to 94% is "+3 points", which sounds modest, but it removes a
    third of the remaining errors -- and at 98%+ the point difference becomes
    actively misleading.
    """
    keys = ("accuracy", "f1", "macro_f1", "precision", "recall")
    delta = {f"{k}_delta": float(after[k] - before[k]) for k in keys if k in before and k in after}

    if "accuracy" in before and "accuracy" in after:
        remaining_error = 1.0 - before["accuracy"]
        delta["error_reduction_pct"] = (
            float((after["accuracy"] - before["accuracy"]) / remaining_error * 100)
            if remaining_error > 1e-9
            else 0.0
        )
    return delta
