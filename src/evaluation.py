"""Evaluation helpers used by both baseline and transformer models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_metrics(y_true: List[int], y_pred: List[int], label_names: List[str]) -> Dict:
    """Compute classification metrics.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels.
        label_names: Ordered class names.

    Returns:
        Dictionary with scalar and per-class metrics.
    """
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "classification_report": report,
    }
    return result


def save_metrics_and_confusion(
    y_true: List[int],
    y_pred: List[int],
    label_names: List[str],
    output_dir: Path,
    prefix: str,
) -> Dict:
    """Save metrics JSON and confusion matrix CSV.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels.
        label_names: Ordered class names.
        output_dir: Destination directory.
        prefix: File name prefix.

    Returns:
        Metrics dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, label_names)

    with (output_dir / f"{prefix}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")

    # Save a normalized confusion matrix for easier class imbalance interpretation.
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm_df = pd.DataFrame(cm_norm, index=label_names, columns=label_names)
    cm_norm_df.to_csv(output_dir / f"{prefix}_confusion_matrix_normalized.csv")

    return metrics
