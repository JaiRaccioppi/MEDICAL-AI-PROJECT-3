"""Train and evaluate the LSTM baseline on PubMed RCT 20k."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from src.config import load_config, parse_args
from src.data import LABEL_NAMES, create_rnn_dataloaders, load_pubmed_rct20k
from src.error_analysis import save_error_analysis
from src.evaluation import save_metrics_and_confusion
from src.models.lstm_baseline import LSTMClassifier
from src.utils.logging_utils import get_logger
from src.utils.reproducibility import set_global_seed


def run_epoch(
    model: LSTMClassifier,
    data_loader,
    criterion: nn.Module,
    optimizer: Adam | None,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Run one epoch for training or evaluation.

    Args:
        model: LSTM classifier instance.
        data_loader: PyTorch dataloader.
        criterion: Loss function.
        optimizer: Optimizer or None for eval mode.
        device: CUDA or CPU device.

    Returns:
        Tuple with loss, accuracy, labels, and predictions.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    y_true, y_pred = [], []

    for batch in tqdm(data_loader, disable=False):
        input_ids = batch.input_ids.to(device)
        lengths = batch.lengths.to(device)
        labels = batch.labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    return total_loss / total_examples, total_correct / total_examples, y_true, y_pred


def compute_class_weights(dataset, label_column: str, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights to mitigate imbalance.

    Args:
        dataset: Training split.
        label_column: Label field name.
        device: Device to store tensor.

    Returns:
        Tensor of normalized class weights.
    """
    labels = [int(row[label_column]) for row in dataset]
    counts = torch.bincount(torch.tensor(labels), minlength=len(LABEL_NAMES)).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    return weights.to(device)


def main() -> None:
    """Run full LSTM training/validation/test workflow."""
    args = parse_args()
    cfg: Dict = load_config(args.config)
    cfg["quick_mode"] = cfg["quick_mode"] or args.quick_mode

    output_dir = Path(args.output_dir) / "lstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(output_dir / "train.log")

    set_global_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    dataset = load_pubmed_rct20k(cfg["dataset_name"], cfg["quick_mode"], cfg)
    train_loader, val_loader, test_loader, vocab = create_rnn_dataloaders(dataset, cfg)
    logger.info("Vocab size: %d", len(vocab))

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=cfg["lstm"]["embedding_dim"],
        hidden_dim=cfg["lstm"]["hidden_dim"],
        num_layers=cfg["lstm"]["num_layers"],
        num_classes=len(LABEL_NAMES),
        dropout=cfg["lstm"]["dropout"],
        bidirectional=cfg["lstm"]["bidirectional"],
    ).to(device)

    class_weights = compute_class_weights(dataset["train"], cfg["label_column"], device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=cfg["lstm"]["learning_rate"])

    best_val_acc = -1.0
    best_path = output_dir / "best_model.pt"
    history = []

    for epoch in range(cfg["lstm"]["epochs"]):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, None, device)

        logger.info(
            "Epoch %d | train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device))
    _, test_acc, y_true, y_pred = run_epoch(model, test_loader, criterion, None, device)
    logger.info("Test accuracy: %.4f", test_acc)

    metrics = save_metrics_and_confusion(y_true, y_pred, LABEL_NAMES, output_dir, prefix="lstm_test")

    test_texts = [row[cfg["text_column"]] for row in dataset["test"]]
    save_error_analysis(test_texts, y_true, y_pred, LABEL_NAMES, output_dir, prefix="lstm_test")

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("LSTM baseline complete\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")


if __name__ == "__main__":
    main()
