"""Error analysis utilities for qualitative model behavior review."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import pandas as pd


def save_error_analysis(
    texts: List[str],
    labels: List[int],
    predictions: List[int],
    label_names: List[str],
    output_dir: Path,
    prefix: str,
) -> None:
    """Persist error examples and confusion pairs for manual inspection.

    Args:
        texts: Original sentences.
        labels: Ground truth label ids.
        predictions: Predicted label ids.
        label_names: Ordered label names.
        output_dir: Destination directory.
        prefix: File name prefix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pair_counter = Counter()
    examples_by_pair = defaultdict(list)

    for text, true_id, pred_id in zip(texts, labels, predictions):
        if true_id != pred_id:
            true_name = label_names[true_id]
            pred_name = label_names[pred_id]
            pair_counter[(true_name, pred_name)] += 1
            if len(examples_by_pair[(true_name, pred_name)]) < 5:
                examples_by_pair[(true_name, pred_name)].append(text)

    for (true_name, pred_name), count in pair_counter.most_common(20):
        rows.append(
            {
                "true_label": true_name,
                "predicted_label": pred_name,
                "count": count,
                "sample_errors": " ||| ".join(examples_by_pair[(true_name, pred_name)]),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_error_analysis.csv", index=False)
