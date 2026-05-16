"""
Geometric alignment metrics: CKA, RSA, mutual kNN.

Why: Linear regression measures coordinate-level recovery, but two representations
can be geometrically equivalent under a rotation (CKA) or preserve rank-order
distances (RSA) without being linearly predictable. Mutual kNN directly measures
whether each sample's neighborhood is consistent across models.
"""

from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import euclidean_distances


@dataclass
class GeometricMetrics:
    cka: float
    rsa_spearman: float
    mutual_knn: dict[int, float]  # k -> overlap rate


def compute_geometric_metrics(
    F_L: np.ndarray,
    F_S: np.ndarray,
    knn_ks: list[int] = (5, 10, 20),
) -> GeometricMetrics:
    return GeometricMetrics(
        cka=compute_cka(F_L, F_S),
        rsa_spearman=compute_rsa(F_L, F_S),
        mutual_knn={k: compute_mutual_knn(F_L, F_S, k) for k in knn_ks},
    )


def compute_cka(F_L: np.ndarray, F_S: np.ndarray) -> float:
    """Linear CKA between F_L and F_S.

    Why: CKA is invariant to orthogonal transforms and isotropic scaling,
    making it a principled measure of representational similarity that doesn't
    penalize rotations. (Kornblith et al., 2019)
    """
    K = _gram(F_L)
    L = _gram(F_S)
    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)
    if hsic_kk == 0 or hsic_ll == 0:
        return 0.0
    return float(hsic_kl / np.sqrt(hsic_kk * hsic_ll))


def _gram(F: np.ndarray) -> np.ndarray:
    return F @ F.T


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator via centered Gram matrices.

    Why: Centering removes the mean effect, isolating covariance structure.
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return float(np.trace(Kc @ Lc) / (n - 1) ** 2)


def compute_rsa(F_L: np.ndarray, F_S: np.ndarray) -> float:
    """Representational Similarity Analysis: Spearman correlation of RDMs.

    Why: RSA compares rank-order structure of pairwise distances, which is
    insensitive to monotone transformations and captures relational geometry.
    """
    rdm_L = _upper_tri(euclidean_distances(F_L))
    rdm_S = _upper_tri(euclidean_distances(F_S))
    rho, _ = spearmanr(rdm_L, rdm_S)
    return float(rho)


def _upper_tri(D: np.ndarray) -> np.ndarray:
    """Extract upper triangle of distance matrix (excluding diagonal)."""
    idx = np.triu_indices(D.shape[0], k=1)
    return D[idx]


def compute_mutual_knn(F_L: np.ndarray, F_S: np.ndarray, k: int) -> float:
    """Mutual k-NN overlap: fraction of k nearest neighbors shared across spaces.

    Why: kNN overlap tests whether local neighborhoods are consistent between
    models. High overlap means the models agree on which samples are "close,"
    even if their coordinates differ. (Huh et al., 2024)
    """
    n = F_L.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be less than n_samples={n}")

    nn_L = _knn_indices(F_L, k)
    nn_S = _knn_indices(F_S, k)

    overlaps = np.array([
        len(np.intersect1d(nn_L[i], nn_S[i])) / k
        for i in range(n)
    ])
    return float(overlaps.mean())


def _knn_indices(F: np.ndarray, k: int) -> np.ndarray:
    """Return (n, k) array of k nearest neighbor indices (excluding self)."""
    D = euclidean_distances(F)
    np.fill_diagonal(D, np.inf)  # exclude self
    return np.argsort(D, axis=1)[:, :k]
