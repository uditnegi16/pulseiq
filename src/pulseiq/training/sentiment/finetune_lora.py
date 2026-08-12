"""DistilBERT + LoRA fine-tuning for product-review sentiment.

WHY LoRA RATHER THAN FULL FINE-TUNING
-------------------------------------
Full fine-tuning updates all ~67M DistilBERT parameters and produces a ~250MB
checkpoint per experiment. LoRA freezes the base model and trains small
low-rank matrices injected into the attention layers -- roughly 0.5-1% of the
parameters, and an adapter of a few MB.

Three consequences that matter for this project:

1. The adapter is small enough to commit to git. The result is reproducible by
   anyone cloning the repo, which a 250MB checkpoint would not be.
2. Training fits comfortably on a free-tier T4, and is feasible (if slow) on CPU.
3. The base model is untouched, so the before/after comparison isolates the
   adaptation rather than confounding it with catastrophic forgetting.

The trade-off is a small accuracy ceiling relative to full fine-tuning. On a
binary task with proxy labels, that ceiling is set by label noise long before
it is set by LoRA's capacity, so the trade is close to free here.

DESIGN NOTES
------------
* Checkpoint selection uses the VALIDATION set. The test set is scored exactly
  once, at the end. Selecting on test is the classification equivalent of the
  temporal leakage this project fixed in forecasting.
* Metrics come from `evaluation/classification.py` -- the same code that scores
  the baseline, so the comparison is apples to apples.
* Runs on GPU when available and CPU otherwise, with no code change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_MODEL = "distilbert-base-uncased"
DEFAULT_ADAPTER_DIR = Path("models/sentiment_lora")


@dataclass
class LoRAConfig:
    """Hyperparameters, in one place so a run is fully described by its config.

    Defaults are the standard starting point for a small encoder on a binary
    task: r=16 with alpha=32 (the common 2x ratio), targeting the attention
    projections. Not tuned exhaustively -- with ~20k balanced examples the model
    saturates quickly, and effort spent on hyperparameter search buys less than
    effort spent on label quality.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list[str] = field(default_factory=lambda: ["q_lin", "v_lin"])

    learning_rate: float = 2e-4  # higher than full fine-tuning: fewer, smaller params
    batch_size: int = 32
    epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 256  # median review is well under this; 512 doubles time for little gain
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": ",".join(self.target_modules),
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_length": self.max_length,
            "seed": self.seed,
        }


def detect_device() -> str:
    """Return 'cuda' or 'cpu'. Logged loudly, because a silent CPU fallback
    turns a 10-minute run into a 4-hour one."""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("CUDA available: %s", name)
        return "cuda"
    logger.warning("no GPU detected -- training on CPU will be roughly 20x slower")
    return "cpu"


def build_datasets(
    train: pd.DataFrame,
    val: pd.DataFrame,
    tokenizer,
    max_length: int = 256,
):
    """Tokenise the splits into HuggingFace Datasets."""
    from datasets import Dataset

    def encode(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=max_length)

    columns = ["text", "label"]
    train_ds = Dataset.from_pandas(train[columns], preserve_index=False)
    val_ds = Dataset.from_pandas(val[columns], preserve_index=False)

    train_ds = train_ds.map(encode, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(encode, batched=True, remove_columns=["text"])
    return train_ds, val_ds


def build_model(config: LoRAConfig, base_model: str = BASE_MODEL):
    """Load DistilBERT and wrap it with LoRA adapters."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )

    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "trainable parameters: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        trainable / total * 100,
    )
    return model, {"trainable_params": trainable, "total_params": total}


def compute_metrics_fn(eval_pred):
    """Metrics during training, from the shared classification module."""
    from pulseiq.evaluation.classification import evaluate_classification

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = evaluate_classification(labels, predictions)
    return {
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "macro_f1": metrics["macro_f1"],
    }


def train(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    *,
    config: LoRAConfig | None = None,
    base_model: str = BASE_MODEL,
    output_dir: Path = DEFAULT_ADAPTER_DIR,
    checkpoint_dir: Path = Path("models/_checkpoints"),
) -> tuple[Any, Any, dict[str, Any]]:
    """Fine-tune and return (model, tokenizer, run_info).

    Only the LoRA adapter is written to `output_dir` -- a few MB rather than the
    full ~250MB checkpoint.
    """
    from transformers import (
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    config = config or LoRAConfig()
    set_seed(config.seed)
    device = detect_device()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_ds, val_ds = build_datasets(train_frame, val_frame, tokenizer, config.max_length)
    model, param_info = build_model(config, base_model)

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # Selected on VALIDATION macro-F1, never on the test set.
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        seed=config.seed,
        fp16=(device == "cuda"),  # ~2x faster on a T4; unsupported on CPU
        report_to="none",  # MLflow logging is handled explicitly below
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics_fn,
    )

    logger.info(
        "training %d epochs on %d examples (device=%s)",
        config.epochs,
        len(train_frame),
        device,
    )
    result = trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("adapter saved to %s", output_dir)

    adapter_bytes = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())

    run_info = {
        "device": device,
        "train_runtime_seconds": float(result.metrics.get("train_runtime", 0.0)),
        "train_loss": float(result.metrics.get("train_loss", 0.0)),
        "adapter_size_mb": round(adapter_bytes / 1e6, 2),
        **param_info,
        **config.as_dict(),
    }
    return model, tokenizer, run_info


def evaluate_on_test(
    model, tokenizer, test_frame: pd.DataFrame, *, max_length: int = 256, batch_size: int = 64
) -> dict[str, float | int]:
    """Score the fine-tuned model on the held-out test set. Runs once."""
    import torch

    from pulseiq.evaluation.classification import evaluate_classification

    device = next(model.parameters()).device
    model.eval()

    predictions: list[int] = []
    texts = test_frame["text"].tolist()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded).logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    metrics = evaluate_classification(test_frame["label"].to_numpy(), np.asarray(predictions))
    logger.info("test metrics: %s", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    import argparse

    from pulseiq.training.sentiment.dataset import load_splits

    parser = argparse.ArgumentParser(
        prog="python -m pulseiq.training.sentiment.finetune_lora",
        description="Fine-tune DistilBERT with LoRA on product-review sentiment.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="cap training rows (smoke test)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    train_frame, val_frame, test_frame = load_splits(args.data_dir)
    if args.limit:
        train_frame = train_frame.head(args.limit)
        val_frame = val_frame.head(max(args.limit // 5, 20))

    config = LoRAConfig(
        r=args.lora_r,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
    )

    model, tokenizer, run_info = train(
        train_frame, val_frame, config=config, output_dir=args.output_dir
    )
    metrics = evaluate_on_test(model, tokenizer, test_frame, max_length=config.max_length)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    path = args.reports_dir / "sentiment_finetuned.json"
    path.write_text(json.dumps({**run_info, **metrics}, indent=2), encoding="utf-8")

    from pulseiq.evaluation.classification import to_metrics

    print("\nfine-tuned (LoRA)")
    print(
        f"  trainable: {run_info['trainable_params']:,} / {run_info['total_params']:,} "
        f"({run_info['trainable_params'] / run_info['total_params'] * 100:.2f}%)"
    )
    print(f"  adapter  : {run_info['adapter_size_mb']} MB")
    print(f"  runtime  : {run_info['train_runtime_seconds']:.0f}s on {run_info['device']}")
    print(f"\n{to_metrics(metrics)}")
    print(f"\n{to_metrics(metrics).confusion_table()}")

    baseline_path = args.reports_dir / "sentiment_baseline.json"
    if baseline_path.exists():
        from pulseiq.evaluation.classification import compare

        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        delta = compare(baseline, metrics)
        print("\nvs zero-shot baseline")
        for key, value in delta.items():
            print(f"  {key:<22}: {value:+.4f}")

    print(f"\nwrote {path}")

    if not args.no_mlflow:
        try:
            import mlflow

            from config.settings import settings

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.set_experiment(settings.mlflow_experiment_name)
            with mlflow.start_run(run_name="sentiment_lora_finetune"):
                mlflow.log_params(
                    {
                        **config.as_dict(),
                        "base_model": BASE_MODEL,
                        "n_train": len(train_frame),
                        "n_test": len(test_frame),
                    }
                )
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                )
                mlflow.log_metrics(
                    {
                        k: v
                        for k, v in run_info.items()
                        if isinstance(v, (int, float)) and k not in config.as_dict()
                    }
                )
                mlflow.log_artifact(str(path))
            logger.info("logged fine-tune to MLflow")
        except Exception:  # noqa: BLE001
            logger.exception("MLflow logging failed; results are still on disk")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
