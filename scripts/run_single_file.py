"""
Self-contained single-file analysis script for Representation Subsumption.

All logic from feature_io, metrics_linear, metrics_geometry, metrics_subspace,
report, and utils is inlined here. No package installation needed beyond the
standard dependencies listed in requirements.txt.

Usage:
  python scripts/run_single_file.py --large features/large.npy --small features/small.npy
  python scripts/run_single_file.py --large features/large.pt --small features/small.pt \\
      --large-model resnet50 --small-model resnet18 \\
      --knn-k 5 10 20 --n-components 16 32 64 --output-dir results
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# feature_io
# ---------------------------------------------------------------------------


@dataclass
class FeatureSet:
    features: np.ndarray  # shape: (n_samples, d)
    sample_ids: Optional[np.ndarray]
    source_path: str
    model_name: str


def load_features(path: str, model_name: str = "") -> FeatureSet:
    """Load features from .pt, .npy, or .npz file.

    Why: Research environments use varied serialization formats; unified loading
    avoids per-script format handling.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    suffix = p.suffix.lower()
    sample_ids = None

    if suffix == ".pt":
        import torch
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            features = data["features"]
            sample_ids = data.get("sample_ids", None)
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            if sample_ids is not None and isinstance(sample_ids, torch.Tensor):
                sample_ids = sample_ids.numpy()
        elif isinstance(data, torch.Tensor):
            features = data.numpy()
        else:
            raise ValueError(f"Unsupported .pt content type: {type(data)}")

    elif suffix == ".npy":
        features = np.load(path)

    elif suffix == ".npz":
        data = np.load(path)
        if "features" in data:
            features = data["features"]
            sample_ids = data.get("sample_ids", None)
        else:
            key = list(data.keys())[0]
            features = data[key]

    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .pt, .npy, or .npz")

    features = np.asarray(features, dtype=np.float32)
    _validate_features(features, path)

    logger.info(f"Loaded {model_name} features: shape={features.shape}, path={path}")
    return FeatureSet(features=features, sample_ids=sample_ids, source_path=path, model_name=model_name)


def _validate_features(features: np.ndarray, source: str) -> None:
    """Check for NaN/Inf which would silently corrupt all downstream metrics."""
    if np.any(np.isnan(features)):
        raise ValueError(f"NaN detected in features from {source}")
    if np.any(np.isinf(features)):
        raise ValueError(f"Inf detected in features from {source}")
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {features.shape} from {source}")


def align_features(large: FeatureSet, small: FeatureSet) -> tuple[np.ndarray, np.ndarray]:
    """Align Large and Small features on common sample_ids.

    Why: When sample_ids differ (e.g., subsets from different extraction runs),
    we must find the intersection to ensure row-wise correspondence.
    Returns (F_L, F_S) aligned arrays.
    """
    if large.sample_ids is None or small.sample_ids is None:
        n_L, n_S = len(large.features), len(small.features)
        if n_L != n_S:
            raise ValueError(
                f"Feature count mismatch without sample_ids: large={n_L}, small={n_S}. "
                "Provide sample_ids or ensure equal sample counts."
            )
        return large.features, small.features

    common_ids = np.intersect1d(large.sample_ids, small.sample_ids)
    if len(common_ids) == 0:
        raise ValueError("No common sample_ids between large and small feature sets.")

    dropped = (len(large.sample_ids) - len(common_ids)) + (len(small.sample_ids) - len(common_ids))
    if dropped > 0:
        logger.warning(
            f"Dropped {dropped} samples due to sample_id mismatch; using {len(common_ids)} common samples."
        )

    large_idx = np.where(np.isin(large.sample_ids, common_ids))[0]
    small_idx = np.where(np.isin(small.sample_ids, common_ids))[0]

    large_order = np.argsort(large.sample_ids[large_idx])
    small_order = np.argsort(small.sample_ids[small_idx])

    F_L = large.features[large_idx[large_order]]
    F_S = small.features[small_idx[small_order]]

    logger.info(f"Aligned features: n_samples={len(common_ids)}, d_L={F_L.shape[1]}, d_S={F_S.shape[1]}")
    return F_L, F_S


# ---------------------------------------------------------------------------
# metrics_linear
# ---------------------------------------------------------------------------


@dataclass
class LinearMetrics:
    r2_l_to_s: float
    r2_s_to_l: float
    mse_l_to_s: float
    mse_s_to_l: float
    directional_gap: float  # r2_l_to_s - r2_s_to_l


def compute_linear_metrics(
    F_L: np.ndarray,
    F_S: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    use_ridge: bool = False,
    ridge_alpha: float = 1.0,
) -> LinearMetrics:
    """Compute bidirectional linear regression R² between F_L and F_S.

    Train/test split prevents overfitting artifacts in high-dimensional settings.
    Ridge regression is available to handle near-collinear features.

    Returns LinearMetrics with R², MSE, and directional_gap.
    """
    F_L_train, F_L_test, F_S_train, F_S_test = train_test_split(
        F_L, F_S, test_size=test_size, random_state=random_state
    )

    r2_l_to_s, mse_l_to_s = _fit_and_eval(F_L_train, F_S_train, F_L_test, F_S_test, use_ridge, ridge_alpha)
    r2_s_to_l, mse_s_to_l = _fit_and_eval(F_S_train, F_L_train, F_S_test, F_L_test, use_ridge, ridge_alpha)

    return LinearMetrics(
        r2_l_to_s=r2_l_to_s,
        r2_s_to_l=r2_s_to_l,
        mse_l_to_s=mse_l_to_s,
        mse_s_to_l=mse_s_to_l,
        directional_gap=r2_l_to_s - r2_s_to_l,
    )


def _fit_and_eval(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    use_ridge: bool,
    alpha: float,
) -> tuple[float, float]:
    """Fit regression X→Y and evaluate on held-out test set.

    Why: Evaluating on test set ensures the R² reflects generalization,
    not memorization of training features.
    """
    reg = Ridge(alpha=alpha) if use_ridge else LinearRegression()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    r2 = float(r2_score(Y_test, Y_pred, multioutput="uniform_average"))
    mse = float(mean_squared_error(Y_test, Y_pred))
    return r2, mse


# ---------------------------------------------------------------------------
# metrics_geometry
# ---------------------------------------------------------------------------


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
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


# ---------------------------------------------------------------------------
# metrics_subspace
# ---------------------------------------------------------------------------


@dataclass
class SubspaceMetrics:
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
    proj = V_base @ (V_base.T @ V_query)
    numerator = float(np.linalg.norm(proj, "fro") ** 2)
    denominator = float(np.linalg.norm(V_query, "fro") ** 2)
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def generate_report(
    cfg: dict,
    linear: LinearMetrics,
    geometry: GeometricMetrics,
    subspace: SubspaceMetrics,
    n_samples: int,
    d_L: int,
    d_S: int,
    output_path: str,
    layer_results: list[dict] | None = None,
) -> str:
    """Build and write Markdown summary report.

    Returns the report string for further use.
    """
    lines = []
    exp = cfg.get("experiment", {})

    lines += [
        "# Representation Subsumption Analysis",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Experiment**: {exp.get('name', 'N/A')}",
        f"**Large model**: `{exp.get('large_model', 'N/A')}`",
        f"**Small model**: `{exp.get('small_model', 'N/A')}`",
        "",
        "## Dataset",
        "",
        "| Key | Value |",
        "|-----|-------|",
        f"| Samples | {n_samples} |",
        f"| d_L | {d_L} |",
        f"| d_S | {d_S} |",
        "",
    ]

    lines += [
        "## Linear Containment",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| R²_L→S | {linear.r2_l_to_s:.4f} |",
        f"| R²_S→L | {linear.r2_s_to_l:.4f} |",
        f"| MSE_L→S | {linear.mse_l_to_s:.6f} |",
        f"| MSE_S→L | {linear.mse_s_to_l:.6f} |",
        f"| Directional Gap (R²_L→S − R²_S→L) | {linear.directional_gap:.4f} |",
        "",
    ]

    lines += [
        "## Geometric Alignment",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| CKA | {geometry.cka:.4f} |",
        f"| RSA (Spearman ρ) | {geometry.rsa_spearman:.4f} |",
    ]
    for k, v in sorted(geometry.mutual_knn.items()):
        lines.append(f"| mutual_kNN@{k} | {v:.4f} |")
    lines.append("")

    lines += [
        "## Subspace Containment",
        "",
        "| r | S_in_L | L_in_S | Gap (S_in_L − L_in_S) |",
        "|---|--------|--------|------------------------|",
    ]
    for r in sorted(subspace.containment_s_in_l):
        s_in_l = subspace.containment_s_in_l[r]
        l_in_s = subspace.containment_l_in_s[r]
        gap = subspace.containment_gap[r]
        lines.append(f"| {r} | {s_in_l:.4f} | {l_in_s:.4f} | {gap:.4f} |")
    lines.append("")

    lines += _interpretation_block(linear, geometry, subspace)

    if layer_results:
        lines += _layer_results_section(layer_results)

    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    return report


def _interpretation_block(
    linear: LinearMetrics,
    geometry: GeometricMetrics,
    subspace: SubspaceMetrics,
) -> list[str]:
    lines = ["## Key Observations", ""]

    if linear.r2_l_to_s > 0.9 and linear.r2_s_to_l < 0.5:
        obs = "Large linearly subsumes Small (high R²_L→S, low R²_S→L)."
    elif linear.r2_l_to_s > 0.8 and linear.r2_s_to_l > 0.8:
        obs = "Both directions high: representations are nearly isomorphic."
    elif linear.r2_l_to_s > 0.5 and linear.r2_s_to_l > 0.5:
        obs = "Moderate bidirectional R²: shared components exist but no full containment."
    else:
        obs = "Low R²_L→S: Small has components not linearly explained by Large."
    lines += [f"- **Linear**: {obs}"]

    if geometry.cka > 0.9:
        lines += ["- **CKA**: Very high — representations are geometrically similar."]
    elif geometry.cka > 0.6:
        lines += ["- **CKA**: Moderate — partial geometric alignment."]
    else:
        lines += ["- **CKA**: Low — representations differ in geometry."]

    max_r = max(subspace.containment_s_in_l)
    c_s_in_l = subspace.containment_s_in_l[max_r]
    if c_s_in_l > 0.9:
        lines += [f"- **Subspace (r={max_r})**: Small's principal directions are well-contained in Large."]
    elif c_s_in_l > 0.6:
        lines += [f"- **Subspace (r={max_r})**: Partial containment of Small in Large."]
    else:
        lines += [f"- **Subspace (r={max_r})**: Small has principal directions outside Large's subspace."]

    lines.append("")
    return lines


def _layer_results_section(layer_results: list[dict]) -> list[str]:
    lines = ["## Layer-Pair Analysis", ""]
    lines += ["| Large Layer | Small Layer | R²_L→S | R²_S→L | CKA | RSA | kNN@10 |"]
    lines += ["|-------------|-------------|--------|--------|-----|-----|--------|"]
    for r in layer_results:
        lines.append(
            f"| {r['large_layer']} | {r['small_layer']} | "
            f"{r.get('r2_l_to_s', 0.0):.4f} | {r.get('r2_s_to_l', 0.0):.4f} | "
            f"{r.get('cka', 0.0):.4f} | {r.get('rsa', 0.0):.4f} | "
            f"{r.get('knn_10', 0.0):.4f} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(linear: LinearMetrics, geometry: GeometricMetrics, subspace: SubspaceMetrics, figs_dir: str) -> None:
    """Generate and save analysis figures. Requires matplotlib."""
    import matplotlib.pyplot as plt
    out = Path(figs_dir)

    # Directional gap bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["R²_L→S", "R²_S→L", "Gap"]
    values = [linear.r2_l_to_s, linear.r2_s_to_l, linear.directional_gap]
    colors = ["steelblue", "salmon", "mediumseagreen"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Linear Containment: Directional R²")
    ax.set_ylabel("R²")
    fig.tight_layout()
    fig.savefig(out / "linear_directional_gap.png", dpi=150)
    plt.close(fig)

    # Subspace containment line chart
    rs = sorted(subspace.containment_s_in_l)
    s_in_l = [subspace.containment_s_in_l[r] for r in rs]
    l_in_s = [subspace.containment_l_in_s[r] for r in rs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rs, s_in_l, marker="o", label="Small in Large")
    ax.plot(rs, l_in_s, marker="s", label="Large in Small")
    ax.set_xlabel("r (# components)")
    ax.set_ylabel("Containment")
    ax.set_title("Subspace Containment vs. r")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "subspace_containment.png", dpi=150)
    plt.close(fig)

    # mutual kNN line chart
    ks = sorted(geometry.mutual_knn)
    knn_vals = [geometry.mutual_knn[k] for k in ks]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ks, knn_vals, marker="o", color="darkorchid")
    ax.set_xlabel("k")
    ax.set_ylabel("mutual kNN overlap")
    ax.set_title("Mutual kNN Overlap vs. k")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "mutual_knn.png", dpi=150)
    plt.close(fig)

    logger.info(f"Figures saved to {out}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Representation Subsumption Analysis (single-file mode)"
    )
    p.add_argument("--large", required=True, help="Path to large model features (.pt/.npy/.npz)")
    p.add_argument("--small", required=True, help="Path to small model features (.pt/.npy/.npz)")
    p.add_argument("--large-model", default="large", help="Name label for large model")
    p.add_argument("--small-model", default="small", help="Name label for small model")
    p.add_argument("--output-dir", default="results", help="Root output directory")
    p.add_argument("--knn-k", nargs="+", type=int, default=[5, 10, 20], help="k values for mutual kNN")
    p.add_argument("--n-components", nargs="+", type=int, default=[16, 32, 64, 128],
                   help="Number of PCA components for subspace analysis")
    p.add_argument("--test-size", type=float, default=0.2, help="Train/test split ratio for linear metrics")
    p.add_argument("--ridge", action="store_true", help="Use Ridge regression instead of OLS")
    p.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization strength")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--figures", action="store_true", help="Generate figures (requires matplotlib)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    set_seed(args.seed)

    figures_dir = str(Path(args.output_dir) / "figures")
    tables_dir = str(Path(args.output_dir) / "tables")
    reports_dir = str(Path(args.output_dir) / "reports")
    ensure_dirs(figures_dir, tables_dir, reports_dir)

    large_feat = load_features(args.large, model_name=args.large_model)
    small_feat = load_features(args.small, model_name=args.small_model)

    F_L, F_S = align_features(large_feat, small_feat)
    n_samples, d_L = F_L.shape
    d_S = F_S.shape[1]
    logger.info(f"Analysis: n={n_samples}, d_L={d_L}, d_S={d_S}")

    linear = compute_linear_metrics(
        F_L, F_S,
        test_size=args.test_size,
        random_state=args.seed,
        use_ridge=args.ridge,
        ridge_alpha=args.ridge_alpha,
    )
    logger.info(f"Linear: R²_L→S={linear.r2_l_to_s:.4f}, R²_S→L={linear.r2_s_to_l:.4f}, gap={linear.directional_gap:.4f}")

    geometry = compute_geometric_metrics(F_L, F_S, knn_ks=args.knn_k)
    logger.info(f"Geometry: CKA={geometry.cka:.4f}, RSA={geometry.rsa_spearman:.4f}")

    subspace = compute_subspace_metrics(F_L, F_S, n_components_list=args.n_components)

    cfg = {
        "experiment": {
            "name": "run_single_file",
            "large_model": args.large_model,
            "small_model": args.small_model,
        }
    }
    report_path = str(Path(reports_dir) / "summary.md")
    report = generate_report(
        cfg=cfg,
        linear=linear,
        geometry=geometry,
        subspace=subspace,
        n_samples=n_samples,
        d_L=d_L,
        d_S=d_S,
        output_path=report_path,
    )

    print(report)
    logger.info(f"Report written to {report_path}")

    if args.figures:
        try:
            make_figures(linear, geometry, subspace, figures_dir)
        except ImportError:
            logger.warning("matplotlib not available; skipping figures.")


if __name__ == "__main__":
    main()
