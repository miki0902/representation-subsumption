import numpy as np
import pytest
from src.metrics_subspace import compute_subspace_metrics, _subspace_containment, _top_components, _center


def test_containment_identical_subspace():
    rng = np.random.default_rng(0)
    F = rng.standard_normal((200, 64)).astype(np.float32)
    F_c = _center(F)
    V = _top_components(F_c, 16)
    # V is contained in itself exactly
    c = _subspace_containment(V, V)
    assert c > 0.99


def test_containment_orthogonal_subspaces():
    # Two orthogonal subspaces should have zero containment
    d = 64
    V_L = np.eye(d)[:, :16].astype(np.float32)
    V_S = np.eye(d)[:, 32:48].astype(np.float32)
    c = _subspace_containment(V_S, V_L)
    assert c < 1e-5


def test_subspace_metrics_large_subsumes_small():
    rng = np.random.default_rng(3)
    # Small is a low-rank projection of Large
    F_L = rng.standard_normal((300, 128)).astype(np.float32)
    W = rng.standard_normal((128, 32)).astype(np.float32)
    F_S = F_L @ W
    m = compute_subspace_metrics(F_L, F_S, n_components_list=[16, 32])
    # Small's space should be more contained in Large than vice versa
    assert m.containment_s_in_l[16] > m.containment_l_in_s[16]
