import numpy as np
import pytest
from src.metrics_geometry import compute_cka, compute_rsa, compute_mutual_knn, compute_geometric_metrics


def test_cka_identical():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((100, 32)).astype(np.float32)
    assert compute_cka(F, F) > 0.99


def test_cka_random_low():
    rng = np.random.default_rng(42)
    F_L = rng.standard_normal((200, 64)).astype(np.float32)
    F_S = rng.standard_normal((200, 64)).astype(np.float32)
    # Random independent features should have low CKA
    assert compute_cka(F_L, F_S) < 0.3


def test_mutual_knn_identical():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((100, 32)).astype(np.float32)
    assert compute_mutual_knn(F, F, k=10) > 0.99


def test_mutual_knn_random_low():
    rng = np.random.default_rng(7)
    F_L = rng.standard_normal((200, 64)).astype(np.float32)
    F_S = rng.standard_normal((200, 64)).astype(np.float32)
    assert compute_mutual_knn(F_L, F_S, k=10) < 0.5


def test_rsa_identical():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((50, 16)).astype(np.float32)
    assert compute_rsa(F, F) > 0.99


def test_knn_k_too_large():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((10, 4)).astype(np.float32)
    with pytest.raises(ValueError):
        compute_mutual_knn(F, F, k=10)
