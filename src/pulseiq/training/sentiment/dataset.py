"""Review dataset construction for sentiment fine-tuning.

DATA — Amazon Reviews'23 (McAuley Lab), Electronics subset.
See docs/decision-log.md D-007 for the licensing position: non-commercial
research use with citation.

    Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. (2024).
    Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952

LABELS ARE A PROXY, AND THIS MATTERS MORE THAN THE MODEL
--------------------------------------------------------
Sentiment is derived from the star rating: 1-2 negative, 4-5 positive, 3
dropped. This is standard practice and it is also *wrong* a measurable fraction
of the time -- people leave 5 stars with complaints in the text, and 1 star
because delivery was late on a product they liked.

The consequence: the achievable accuracy ceiling is set by label noise, not by
model capacity. A fine-tune reaching 92% where the labels are ~95% faithful has
essentially saturated the task, and pushing further would mean fitting the
noise. Any reported number has to be read against that ceiling.

Neutral (3-star) reviews are dropped rather than kept as a third class. A
3-star review is where the star-to-sentiment mapping is least trustworthy --
it is genuinely mixed text, and forcing it into a class teaches the model that
mixed text has a definite label.

STREAMING
---------
The full corpus is 571M reviews. The loader streams and stops once it has
enough, so nothing large is ever downloaded or held in memory.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

HF_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
DEFAULT_CATEGORY = "raw_review_Electronics"

LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1
LABEL_NAMES = ["negative", "positive"]

MIN_WORDS = 5
MAX_CHARS = 2000


@dataclass
class DatasetStats:
    """What the loader actually produced, and what it discarded."""

    total_seen: int = 0
    kept: int = 0
    dropped: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dropped is None:
            self.dropped = Counter()

    def drop(self, reason: str) -> None:
        self.dropped[reason] += 1

    def summary(self) -> str:
        lines = [
            f"seen={self.total_seen} kept={self.kept} "
            f"dropped={sum(self.dropped.values())} "
            f"keep_rate={self.kept / self.total_seen:.1%}"
            if self.total_seen
            else "no rows seen"
        ]
        for reason, count in self.dropped.most_common():
            lines.append(f"  - {reason}: {count}")
        return "\n".join(lines)


def rating_to_label(rating: float) -> int | None:
    """Map a star rating to a binary sentiment label.

    Returns None for 3 stars, which are excluded rather than treated as a
    third class -- see the module docstring.
    """
    if rating is None:
        return None
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return None

    if value <= 2.0:
        return LABEL_NEGATIVE
    if value >= 4.0:
        return LABEL_POSITIVE
    return None


def clean_text(title: str | None, text: str | None) -> str | None:
    """Combine a review's title and body into one training string.

    Titles carry real sentiment signal ("Waste of money") and are often the
    clearest part of a review, so they are prepended rather than discarded.
    Truncated at MAX_CHARS because DistilBERT sees 512 tokens regardless, and
    carrying more just slows tokenisation.
    """
    parts = [p.strip() for p in (title, text) if p and p.strip()]
    if not parts:
        return None
    combined = " ".join(parts)
    combined = " ".join(combined.split())
    return combined[:MAX_CHARS] if combined else None


def prepare_rows(
    records: Iterator[dict[str, Any]],
    *,
    max_rows: int = 20_000,
    balance: bool = True,
    min_words: int = MIN_WORDS,
) -> tuple[list[dict[str, Any]], DatasetStats]:
    """Turn raw review records into labelled training rows. PURE (no network).

    `balance=True` caps each class at max_rows/2. Amazon reviews skew heavily
    positive -- roughly 80% are 4-5 stars -- and an unbalanced set produces a
    model that scores well by predicting "positive" almost always. Balancing
    makes accuracy an honest metric instead of a restatement of the base rate.
    """
    stats = DatasetStats()
    per_class_cap = max_rows // 2 if balance else max_rows
    counts = {LABEL_NEGATIVE: 0, LABEL_POSITIVE: 0}
    seen_text: set[str] = set()
    rows: list[dict[str, Any]] = []

    for record in records:
        stats.total_seen += 1

        label = rating_to_label(record.get("rating"))
        if label is None:
            stats.drop("neutral_or_unparseable_rating")
            continue

        if balance and counts[label] >= per_class_cap:
            stats.drop("class_quota_full")
            if all(c >= per_class_cap for c in counts.values()):
                break
            continue

        text = clean_text(record.get("title"), record.get("text"))
        if text is None:
            stats.drop("empty_text")
            continue
        if len(text.split()) < min_words:
            stats.drop("too_short")
            continue

        key = text.lower()
        if key in seen_text:
            stats.drop("duplicate_text")
            continue
        seen_text.add(key)

        rows.append(
            {
                "text": text,
                "label": label,
                "rating": float(record["rating"]),
                "verified_purchase": record.get("verified_purchase"),
                "timestamp": record.get("timestamp"),
            }
        )
        counts[label] += 1
        stats.kept += 1

        if not balance and stats.kept >= max_rows:
            break

    logger.info("dataset preparation:\n%s", stats.summary())
    logger.info("class balance: negative=%d positive=%d", counts[0], counts[1])
    return rows, stats


def load_reviews(
    *,
    category: str = DEFAULT_CATEGORY,
    max_rows: int = 20_000,
    balance: bool = True,
    scan_limit: int | None = None,
) -> pd.DataFrame:
    """Stream reviews from HuggingFace and return a labelled DataFrame.

    `scan_limit` caps how many raw records are examined. Balancing needs to see
    many records to find enough negatives (they are ~20% of the corpus), so the
    default allows 15x the target before giving up.
    """
    from datasets import load_dataset

    scan_limit = scan_limit or max_rows * 15
    logger.info("streaming %s / %s (scan limit %d)", HF_DATASET, category, scan_limit)

    stream = load_dataset(
        HF_DATASET, category, split="full", streaming=True, trust_remote_code=True
    )

    def limited() -> Iterator[dict[str, Any]]:
        for i, record in enumerate(stream):
            if i >= scan_limit:
                logger.warning("hit scan limit of %d records", scan_limit)
                break
            yield record

    rows, _ = prepare_rows(limited(), max_rows=max_rows, balance=balance)
    return pd.DataFrame(rows)


def stratified_split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test, preserving class balance in each.

    A random split is correct here, unlike the forecasting task: reviews are
    independent observations, not a time series, so there is no temporal
    ordering to respect and no leakage from shuffling.

    Three-way because the validation set selects the checkpoint and the test set
    must stay untouched until the final number is reported. Tuning on the test
    set is the classification equivalent of the leakage this project fixed
    elsewhere.
    """
    from sklearn.model_selection import train_test_split

    if frame.empty:
        empty = frame.copy()
        return empty, empty.copy(), empty.copy()

    train_val, test = train_test_split(
        frame, test_size=test_size, stratify=frame["label"], random_state=seed
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["label"], random_state=seed
    )

    logger.info(
        "split: train=%d val=%d test=%d (positive share: %.1f%% / %.1f%% / %.1f%%)",
        len(train),
        len(val),
        len(test),
        train["label"].mean() * 100,
        val["label"].mean() * 100,
        test["label"].mean() * 100,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_splits(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, output_dir: Path
) -> dict[str, Path]:
    """Persist splits as parquet.

    Written to disk so the baseline and the fine-tune are scored on byte-identical
    test data. Regenerating the split between runs -- even with the same seed but
    a different library version -- would make the before/after comparison invalid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        path = output_dir / f"reviews_{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = path
        logger.info("wrote %d rows to %s", len(frame), path)
    return paths


def load_splits(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read splits back. Raises if they were never generated."""
    frames = []
    for name in ("train", "val", "test"):
        path = output_dir / f"reviews_{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run dataset preparation first "
                f"(python -m pulseiq.training.sentiment.dataset --max-rows 20000)."
            )
        frames.append(pd.read_parquet(path))
    return frames[0], frames[1], frames[2]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pulseiq.training.sentiment.dataset",
        description="Build the sentiment dataset from Amazon Reviews'23.",
    )
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--max-rows", type=int, default=20_000)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    frame = load_reviews(
        category=args.category, max_rows=args.max_rows, balance=not args.no_balance
    )
    if frame.empty:
        logger.error("no rows produced")
        return 1

    train, val, test = stratified_split(frame, seed=args.seed)
    save_splits(train, val, test, args.output_dir)

    print(f"\n{len(frame)} labelled reviews")
    print(f"  positive: {int(frame['label'].sum())}")
    print(f"  negative: {int((1 - frame['label']).sum())}")
    print(f"  median length: {frame['text'].str.split().str.len().median():.0f} words")
    print(f"\nsplits written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
