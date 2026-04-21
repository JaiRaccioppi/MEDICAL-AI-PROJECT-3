"""Configuration utilities for experiment setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments shared by all entrypoint scripts."""
    parser = argparse.ArgumentParser(description="PubMed RCT 20k experiment runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default_config.json"),
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save models and metrics.",
    )
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Run on smaller data subsets for debugging.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from JSON.

    Args:
        config_path: File path to JSON configuration.

    Returns:
        Dictionary with configuration values.
    """
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)
