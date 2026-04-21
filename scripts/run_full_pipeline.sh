#!/usr/bin/env bash
set -euo pipefail

# Runs both required models with full training data.
python -m src.train_lstm --config configs/default_config.json --output-dir outputs
python -m src.train_transformer --config configs/default_config.json --output-dir outputs
