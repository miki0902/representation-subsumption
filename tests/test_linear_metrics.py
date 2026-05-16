import numpy as np
import pytest
from src.metrics_linear import compute_linear_metrics


def test_identical_features_r2_near_one():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((200, 64)).astype(np.float32)
    m = compute_linear_metrics(F, F)
    assert m.r2_l_to_s > 0.99
    assert m.r2_s_to_l > 0.99


def test_directional_gap_sign():
    # Large has more info than Small: L→S should be easier than S→L
    rng = np.random.default_rng(1)
    F_L = rng.standard_normal((300, 128)).astype(np.float32)
    # Small is a noisy low-rank projection of Large
    W = rng.standard_normal((128, 16)).astype(np.float32)
    F_S = F_L @ W + 0.01 * rng.standard_normal((300, 16)).astype(np.float32)
    m = compute_linear_metrics(F_L, F_S)
    assert m.r2_l_to_s > m.r2_s_to_l


def test_ridge_runs_without_error():
    rng = np.random.default_rng(2)
    F_L = rng.standard_normal((100, 32)).astype(np.float32)
    F_S = rng.standard_normal((100, 16)).astype(np.float32)
    m = compute_linear_metrics(F_L, F_S, use_ridge=True, ridge_alpha=10.0)
    assert isinstance(m.r2_l_to_s, float)
