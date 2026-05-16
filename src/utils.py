"""Shared utilities: seeding, logging setup, config loading, directory creation."""

import logging
import random
from pathlib import Path

import numpy as np
import yaml


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² that returns 0 when variance of y_true is 0 (degenerate case)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)
