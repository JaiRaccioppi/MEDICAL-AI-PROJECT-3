#!/usr/bin/env bash
set -euo pipefail

# Debug run on smaller subsets to validate the pipeline quickly.
python -m src.train_lstm --config configs/default_config.json --output-dir outputs --quick-mode
python -m src.train_transformer --config configs/default_config.json --output-dir outputs --quick-mode
