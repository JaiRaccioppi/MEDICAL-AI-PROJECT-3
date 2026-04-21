# PubMed RCT 20k Clinical Sentence Classification

Sentence-level biomedical NLP pipeline that classifies clinical trial abstract sentences into five rhetorical roles: Background, Objective, Methods, Results, and Conclusions.

## Clinical Context

This project is designed for biomedical NLP researchers, clinical informatics students, and evidence synthesis teams who need structured trial abstracts to support literature review automation and downstream clinical decision-support workflows.

## Project Scope

- Task: Multi-class sentence classification for PubMed trial abstracts.
- Dataset: Hugging Face [armanc/pubmed-rct20k](https://huggingface.co/datasets/armanc/pubmed-rct20k).
- Required models implemented:
	- Recurrent baseline: BiLSTM classifier.
	- Pretrained transformer: DistilBERT fine-tuning (configurable to Bio/Clinical models).
- Shared data splits: Official train/validation/test splits from dataset.

## Quick Start

### 1. Environment Setup

- Python version: 3.10+ (tested with Python 3.11).
- Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Pipeline

- Quick debug mode (smaller subsets):

```bash
bash scripts/run_quick_pipeline.sh
```

- Full experiment mode:

```bash
bash scripts/run_full_pipeline.sh
```

- Or run each model independently:

```bash
python -m src.train_lstm --config configs/default_config.json --output-dir outputs
python -m src.train_transformer --config configs/default_config.json --output-dir outputs
```

### 3. Runtime and Compute Requirements

- Quick mode: ~10 to 30 minutes on CPU; faster on GPU.
- Full mode: several hours depending on hardware.
- Recommended:
	- CPU: 8+ cores, 16+ GB RAM minimum.
	- GPU: NVIDIA GPU with 8+ GB VRAM for transformer fine-tuning.

## Usage Guide

1. Edit experiment settings in `configs/default_config.json` (batch size, epochs, model name, max sequence length).
2. Run LSTM baseline training (`src/train_lstm.py`).
3. Run transformer fine-tuning (`src/train_transformer.py`).
4. Review outputs in `outputs/lstm/` and `outputs/transformer/`.
5. Compare metrics from `*_metrics.json` and confusion matrices from `*_confusion_matrix.csv`.
6. Review error analysis in `*_error_analysis.csv` for common failure modes.

Expected outputs include:

- `history.json` for per-epoch learning curves (LSTM).
- `*_metrics.json` with Accuracy, Precision, Recall, F1.
- `*_confusion_matrix.csv` and normalized confusion matrix.
- `*_error_analysis.csv` with top confusion pairs and sample errors.

## Data Description

- Source: Hugging Face dataset `armanc/pubmed-rct20k`.
- Task labels:
	- BACKGROUND
	- OBJECTIVE
	- METHODS
	- RESULTS
	- CONCLUSIONS
- Data format:
	- `text`: sentence string.
	- `label`: integer class id.
- Splits:
	- Train: ~180k sentences.
	- Validation: ~30k sentences.
	- Test: ~30k sentences.
- License and citation:
	- Follow the dataset card license and citation instructions on Hugging Face.
- Data acquisition:
	- No manual download needed; scripts load data automatically via `datasets.load_dataset`.

## Results Summary

This repository is scaffolded for reproducible experiments. After running training, summarize in this section:

- LSTM baseline metrics (accuracy, macro F1, class-wise precision/recall).
- Transformer metrics on same test split.
- Comparison of performance, runtime, and error patterns.
- Clinical interpretation of typical confusions (e.g., Objective vs Background).

## Interpretability and Error Analysis Plan

- Identify representative correct and incorrect predictions from error analysis CSVs.
- Categorize failure modes:
	- Ambiguous wording between rhetorical roles.
	- Long or complex biomedical sentences.
	- Rare terminology and class overlap.
- Discuss overfitting signals using train/validation curves and validation metrics.
- Use confusion matrices to assess class imbalance effects and per-class weakness.

## Project Structure

```text
.
├── configs/
│   └── default_config.json
├── reports/
│   └── REPORT_TEMPLATE.md
├── scripts/
│   ├── run_full_pipeline.sh
│   └── run_quick_pipeline.sh
├── src/
│   ├── config.py
│   ├── data.py
│   ├── error_analysis.py
│   ├── evaluation.py
│   ├── train_lstm.py
│   ├── train_transformer.py
│   ├── models/
│   │   └── lstm_baseline.py
│   └── utils/
│       ├── logging_utils.py
│       └── reproducibility.py
├── requirements.txt
└── README.md
```

## Authors and Contributions

Document final team members and roles in both places:

- `reports/REPORT_TEMPLATE.md` title page table.
- This section before submission.

Template:

| Name | Role | Contributions |
|---|---|---|
| Member 1 | Project lead | Coordination, experiment design |
| Member 2 | Baseline engineer | LSTM implementation and tuning |
| Member 3 | Transformer engineer | Transformer fine-tuning and analysis |
| Member 4 | Evaluation lead | Metrics, error analysis, report/presentation |

## Dependencies

All pinned versions are listed in `requirements.txt`:

- datasets==2.20.0
- evaluate==0.4.2
- huggingface-hub==0.24.6
- matplotlib==3.9.2
- numpy==1.26.4
- pandas==2.2.2
- scikit-learn==1.5.1
- seaborn==0.13.2
- torch==2.4.1
- transformers==4.44.2
- tqdm==4.66.5

## Code Quality and Reproducibility Notes

- Random seeds set in `src/utils/reproducibility.py`.
- Logging and output artifacts saved for experiment traceability.
- Config-driven paths and hyperparameters; no hard-coded dataset file paths.
- Function-level docstrings included across source modules.
- PEP 8-compatible naming conventions and modular pipeline design.