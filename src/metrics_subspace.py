"""
Subspace containment: measures how much of Small's principal subspace lies within Large's.

Why: Linear R² tests coordinate recovery; subspace containment tests whether the
*directions* that matter most to Small are representable in Large's span, regardless
of specific coordinates. This is a stronger geometric claim than RSA/CKA.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SubspaceMetrics:
    # dict keys are r values
    containment_s_in_l: dict[int, float]
    containment_l_in_s: dict[int, float]
    containment_gap: dict[int, float]


def compute_subspace_metrics(
    F_L: np.ndarray,
    F_S: np.ndarray,
    n_components_list: list[int] = (16, 32, 64, 128),
) -> SubspaceMetrics:
    F_L_c = _center(F_L)
    F_S_c = _center(F_S)

    V_L_full = _top_components(F_L_c, max(n_components_list))
    V_S_full = _top_components(F_S_c, max(n_components_list))

    containment_s_in_l = {}
    containment_l_in_s = {}
    containment_gap = {}

    for r in n_components_list:
        r_eff = min(r, V_L_full.shape[1], V_S_full.shape[1])
        V_L = V_L_full[:, :r_eff]
        V_S = V_S_full[:, :r_eff]

        c_s_in_l = _subspace_containment(V_S, V_L)
        c_l_in_s = _subspace_containment(V_L, V_S)

        containment_s_in_l[r] = c_s_in_l
        containment_l_in_s[r] = c_l_in_s
        containment_gap[r] = c_s_in_l - c_l_in_s

    return SubspaceMetrics(
        containment_s_in_l=containment_s_in_l,
        containment_l_in_s=containment_l_in_s,
        containment_gap=containment_gap,
    )


def _center(F: np.ndarray) -> np.ndarray:
    return F - F.mean(axis=0)


def _top_components(F_centered: np.ndarray, r: int) -> np.ndarray:
    """Extract top-r right singular vectors via SVD.

    Why: SVD on the centered feature matrix gives principal components without
    explicitly computing the covariance matrix, which is more numerically stable
    for high-dimensional features.
    Returns V of shape (d, r).
    """
    r_eff = min(r, min(F_centered.shape))
    _, _, Vt = np.linalg.svd(F_centered, full_matrices=False)
    return Vt[:r_eff].T  # shape: (d, r_eff)


def _subspace_containment(V_query: np.ndarray, V_base: np.ndarray) -> float:
    """Measure how much of V_query's column space lies in V_base's column space.

    Formula: ||P_{V_base} V_query||_F^2 / ||V_query||_F^2

    Why: The Frobenius norm of the projection quantifies the total "energy" of
    V_query that is explainable by V_base directions. Value of 1 means complete
    containment; 0 means orthogonal subspaces.

    Assumes columns of V_base are orthonormal (from SVD).
    """
    # P_{V_base} V_query = V_base (V_base^T V_query)
    proj = V_base @ (V_base.T @ V_query)
    numerator = float(np.linalg.norm(proj, "fro") ** 2)
    denominator = float(np.linalg.norm(V_query, "fro") ** 2)
    if denominator == 0:
        return 0.0
    return numerator / denominator
