"""Fine-tune a pretrained transformer on PubMed RCT 20k."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import load_config, parse_args
from src.data import LABEL_NAMES, create_transformer_tokenizer, load_pubmed_rct20k
from src.error_analysis import save_error_analysis
from src.evaluation import save_metrics_and_confusion
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import set_global_seed


def tokenize_dataset(dataset, tokenizer, text_column: str, max_length: int):
    """Tokenize dataset splits for transformer training."""

    def _tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    tokenized = dataset.map(_tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def compute_hf_metrics(eval_pred):
    """Compute evaluation metrics compatible with Hugging Face Trainer."""
    metric = evaluate.load("f1")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_macro = metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = (preds == labels).mean().item()
    return {"accuracy": accuracy, "macro_f1": f1_macro}


def main() -> None:
    """Run transformer fine-tuning and test evaluation."""
    args = parse_args()
    cfg: Dict = load_config(args.config)
    cfg["quick_mode"] = cfg["quick_mode"] or args.quick_mode

    output_dir = Path(args.output_dir) / "transformer"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(output_dir / "train.log")

    set_global_seed(cfg["seed"])
    dataset = load_pubmed_rct20k(cfg["dataset_name"], cfg["quick_mode"], cfg)

    model_name = cfg["transformer"]["model_name"]
    tokenizer = create_transformer_tokenizer(model_name)
    tokenized = tokenize_dataset(dataset, tokenizer, cfg["text_column"], cfg["max_length"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_NAMES),
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )

    train_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=cfg["transformer"]["learning_rate"],
        num_train_epochs=cfg["transformer"]["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        weight_decay=cfg["transformer"]["weight_decay"],
        warmup_ratio=cfg["transformer"]["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=cfg["seed"],
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_hf_metrics,
    )

    trainer.train()
    predictions = trainer.predict(tokenized["test"])
    y_pred = np.argmax(predictions.predictions, axis=1).tolist()
    y_true = predictions.label_ids.tolist()

    metrics = save_metrics_and_confusion(
        y_true,
        y_pred,
        LABEL_NAMES,
        output_dir,
        prefix="transformer_test",
    )

    test_texts = [row[cfg["text_column"]] for row in dataset["test"]]
    save_error_analysis(
        texts=test_texts,
        labels=y_true,
        predictions=y_pred,
        label_names=LABEL_NAMES,
        output_dir=output_dir,
        prefix="transformer_test",
    )

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Transformer model complete\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")

    with (output_dir / "trainer_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(predictions.metrics, f, indent=2)

    logger.info("Transformer training and evaluation complete")


if __name__ == "__main__":
    main()
