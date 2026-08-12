"""Zero-shot sentiment baseline — the "before" number.

This is what the original project did: call a pretrained sentiment pipeline and
report its output. It is a genuinely strong baseline, because
`distilbert-base-uncased-finetuned-sst-2-english` was already fine-tuned on
movie-review sentiment and transfers reasonably to product reviews.

That strength is the point. Fine-tuning has to beat a real competitor for the
before/after comparison to mean anything. A weak baseline chosen to flatter the
fine-tune would make the improvement number worthless.

WHAT THE COMPARISON ACTUALLY MEASURES
-------------------------------------
Both models are DistilBERT. The zero-shot one was tuned on SST-2 (movie
reviews); the fine-tuned one is adapted to Amazon product reviews with LoRA. So
the delta isolates *domain adaptation*, not model capacity -- a cleaner
experiment than comparing two different architectures, where any difference
could be attributed to either cause.

Runs on CPU in a couple of minutes for a few thousand rows. No GPU needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASELINE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# The SST-2 model emits LABEL/POSITIVE strings; map to our integer labels.
LABEL_MAP = {
    "NEGATIVE": 0,
    "POSITIVE": 1,
    "LABEL_0": 0,
    "LABEL_1": 1,
}


def map_pipeline_label(raw: str) -> int | None:
    """Normalise a pipeline label string to 0/1. None if unrecognised."""
    if raw is None:
        return None
    return LABEL_MAP.get(str(raw).strip().upper())


def predict_zeroshot(
    texts: list[str],
    *,
    model_name: str = BASELINE_MODEL,
    batch_size: int = 32,
    max_length: int = 512,
    device: int = -1,
) -> np.ndarray:
    """Run the zero-shot pipeline over `texts`. Returns an array of 0/1.

    `device=-1` is CPU, `0` is the first GPU. Defaults to CPU because this
    baseline is cheap and there is no reason to require a GPU for it.

    Truncation is explicit rather than left to the pipeline default: a review
    longer than 512 tokens would otherwise raise, and silently dropping long
    reviews would bias the baseline toward short, blunt ones.
    """
    from transformers import pipeline

    classifier = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        truncation=True,
        max_length=max_length,
    )

    predictions: list[int] = []
    unmapped = 0

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        for result in classifier(chunk):
            label = map_pipeline_label(result.get("label"))
            if label is None:
                unmapped += 1
                # Default to the majority class rather than crashing. Counted
                # and logged so it cannot pass unnoticed.
                label = 1
            predictions.append(label)

    if unmapped:
        logger.warning("%d prediction(s) had unrecognised labels; defaulted to positive", unmapped)

    return np.asarray(predictions, dtype=int)


def evaluate_baseline(
    test_frame: pd.DataFrame,
    *,
    model_name: str = BASELINE_MODEL,
    device: int = -1,
) -> dict[str, float]:
    """Score the zero-shot model on the held-out test set."""
    from pulseiq.evaluation.classification import evaluate_classification

    texts = test_frame["text"].tolist()
    y_true = test_frame["label"].to_numpy()

    logger.info("running zero-shot baseline on %d reviews", len(texts))
    y_pred = predict_zeroshot(texts, model_name=model_name, device=device)

    metrics = evaluate_classification(y_true, y_pred)
    logger.info("baseline: %s", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from pulseiq.training.sentiment.dataset import load_splits

    parser = argparse.ArgumentParser(
        prog="python -m pulseiq.training.sentiment.baseline_zeroshot",
        description="Score the zero-shot sentiment baseline on the held-out test set.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model", default=BASELINE_MODEL)
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0 for GPU")
    parser.add_argument("--limit", type=int, default=None, help="score only N rows (smoke test)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    _, _, test = load_splits(args.data_dir)
    if args.limit:
        test = test.head(args.limit)

    metrics = evaluate_baseline(test, model_name=args.model, device=args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "sentiment_baseline.json"
    path.write_text(
        json.dumps({"model": args.model, "n_test": len(test), **metrics}, indent=2),
        encoding="utf-8",
    )

    print("\nzero-shot baseline")
    print(f"  model    : {args.model}")
    print(f"  test rows: {len(test)}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<9}: {value:.4f}")
    print(f"\nwrote {path}")

    if not args.no_mlflow:
        try:
            import mlflow

            from config.settings import settings

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            with mlflow.start_run(run_name="sentiment_baseline_zeroshot"):
                mlflow.log_params(
                    {"model": args.model, "n_test": len(test), "approach": "zero-shot"}
                )
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                )
                mlflow.log_artifact(str(path))
            logger.info("logged baseline to MLflow")
        except Exception:  # noqa: BLE001 - tracking must not lose a completed run
            logger.exception("MLflow logging failed; results are still on disk")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
