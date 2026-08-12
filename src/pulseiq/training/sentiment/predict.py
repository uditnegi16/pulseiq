"""Local inference with the fine-tuned LoRA adapter.

The adapter trains on a GPU and runs on a CPU. That asymmetry is the point of
LoRA here: a few MB of weights, committed to the repo, loaded locally, no GPU
required at inference time.

The model is cached in a module-level singleton because loading DistilBERT takes
several seconds — acceptable once at API startup, unacceptable per request.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ADAPTER_DIR = Path("models/sentiment_lora")
LABEL_NAMES = ["negative", "positive"]


class AdapterNotFoundError(FileNotFoundError):
    """Raised when the adapter has not been downloaded from the training run."""


@lru_cache(maxsize=2)
def load_model(adapter_dir: str = str(DEFAULT_ADAPTER_DIR)):
    """Load the base model with the LoRA adapter applied. Cached per path.

    Raises with an actionable message rather than a bare FileNotFoundError,
    because the most likely cause is "the notebook ran but the adapter was never
    downloaded", and the fix is not obvious from a missing-path error.
    """
    path = Path(adapter_dir)
    if not path.exists() or not any(path.iterdir()):
        raise AdapterNotFoundError(
            f"No adapter at {path}. Run notebooks/finetune_sentiment.ipynb on a "
            f"Colab GPU, download sentiment_lora.zip, and unzip it into {path}."
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from pulseiq.training.sentiment.finetune_lora import BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(str(path))
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    model = PeftModel.from_pretrained(base, str(path))
    model.eval()

    logger.info("loaded adapter from %s (device=cpu)", path)
    return model, tokenizer, torch


def predict(
    texts: list[str],
    *,
    adapter_dir: str = str(DEFAULT_ADAPTER_DIR),
    max_length: int = 256,
    batch_size: int = 32,
) -> list[dict[str, float | str]]:
    """Classify texts. Returns label, confidence, and both class probabilities.

    Confidence is included because a downstream consumer needs to know when the
    model is unsure. A 0.51 positive and a 0.99 positive are the same label and
    very different facts.
    """
    if not texts:
        return []

    model, tokenizer, torch = load_model(adapter_dir)
    results: list[dict[str, float | str]] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
            )
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).numpy()

            for row in probabilities:
                index = int(np.argmax(row))
                results.append(
                    {
                        "label": LABEL_NAMES[index],
                        "confidence": float(row[index]),
                        "negative": float(row[0]),
                        "positive": float(row[1]),
                    }
                )

    return results


def predict_one(text: str, **kwargs) -> dict[str, float | str]:
    """Single-text convenience wrapper."""
    return predict([text], **kwargs)[0]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m pulseiq.training.sentiment.predict",
        description="Classify review sentiment with the fine-tuned adapter.",
    )
    parser.add_argument("--text", action="append", help="text to classify (repeatable)")
    parser.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER_DIR))
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(message)s")

    texts = args.text or [
        "Battery died within a week, complete waste of money.",
        "Sound quality is incredible and the noise cancellation is superb.",
        "It works fine I suppose, nothing special about it.",
    ]

    try:
        results = predict(texts, adapter_dir=args.adapter_dir)
    except AdapterNotFoundError as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps([{"text": t, **r} for t, r in zip(texts, results, strict=True)], indent=2))
    else:
        for text, result in zip(texts, results, strict=True):
            preview = text if len(text) <= 60 else text[:57] + "..."
            print(f"{result['label']:<9} {result['confidence']:.3f}  {preview}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
