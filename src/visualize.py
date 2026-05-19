"""
幾何可視化: Large / Small / pseudo-Large / merge 空間の2D比較図。

3パネル構成:
  (A) CCA common space: Z_L vs Z_S
      d_L≠d_S のため可視化専用に CCA で共通空間へ射影。写像前の生の表現を比較。
  (B) post-inclusion (Large feature space): F_L vs pseudo-Large
      Small を S→L 線形写像で Large 空間に持ち上げた pseudo-Large と本物の Large を比較。
  (C) merge space: Z_L vs Z_S
      CCA 正準空間上での Large と Small の幾何的配置を確認。

同じ座標系保証:
  各パネルで比較する全群を結合してから reducer を1回 fit し、
  全群を同じ reducer で transform することで座標系を統一する。
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def _project_cca(
    F: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray | None,
    W: np.ndarray,
) -> np.ndarray:
    """Center, optionally scale, and project F using CCA weights W.

    Args:
        F: (n, d) feature matrix
        mean: (d,) centering mean
        std: (d,) scaling std, or None if not standardized
        W: (d, k) CCA projection weights

    Returns:
        (n, k) projected coordinates
    """
    F_c = F - mean
    if std is not None:
        F_c = F_c / std
    return F_c @ W


def _build_reducer(
    X_joint: np.ndarray,
    seed: int,
    pca_pre_dim: int = 50,
) -> tuple[np.ndarray, str]:
    """Fit a 2D reducer on X_joint and return coords + method name.

    Args:
        X_joint: (N_total, d) joint data for all groups
        seed: random seed
        pca_pre_dim: if d > pca_pre_dim, first reduce with PCA

    Returns:
        (coords_2d, method_name) where coords_2d has shape (N_total, 2)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_joint)

    # Optional PCA pre-reduction
    n_samples, n_features = X_joint.shape
    effective_pre_dim = min(pca_pre_dim, n_samples - 1, n_features - 1)
    if n_features > effective_pre_dim and effective_pre_dim >= 2:
        pca_pre = PCA(n_components=effective_pre_dim, random_state=seed)
        X_scaled = pca_pre.fit_transform(X_scaled)

    # Try UMAP, fall back to PCA
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        coords_2d = reducer.fit_transform(X_scaled)
        method_name = "UMAP"
    except ImportError:
        logger.warning("umap-learn not installed, falling back to PCA for visualization.")
        reducer = PCA(n_components=2, random_state=seed)
        coords_2d = reducer.fit_transform(X_scaled)
        method_name = "PCA"

    return coords_2d, method_name


def compute_geometry_aux_metrics(
    F_L: np.ndarray,
    F_PL: np.ndarray,
    Z_L: np.ndarray,
    Z_S: np.ndarray,
) -> dict:
    """Compute auxiliary geometry metrics for the visualization.

    Args:
        F_L: (n, d_L) Large features (subsampled)
        F_PL: (n, d_L) pseudo-Large features (S→L linear map)
        Z_L: (n, k) Large CCA projections
        Z_S: (n, k) Small CCA projections

    Returns:
        dict with centroid_dist_L_PL, centroid_dist_merge_ZL_ZS,
             mean_nn_dist_L_PL, silhouette_merge
    """
    from scipy.spatial.distance import cdist

    # Centroid distance in feature space
    centroid_L = F_L.mean(axis=0)
    centroid_PL = F_PL.mean(axis=0)
    centroid_dist_L_PL = float(np.linalg.norm(centroid_L - centroid_PL))

    # Centroid distance in CCA space
    centroid_ZL = Z_L.mean(axis=0)
    centroid_ZS = Z_S.mean(axis=0)
    centroid_dist_merge_ZL_ZS = float(np.linalg.norm(centroid_ZL - centroid_ZS))

    # Mean nearest-neighbor distance from F_L to F_PL
    dists = cdist(F_L, F_PL)
    mean_nn_dist_L_PL = float(dists.min(axis=1).mean())

    # Silhouette score
    try:
        from sklearn.metrics import silhouette_score
        Z_both = np.concatenate([Z_L, Z_S], axis=0)
        labels = np.array([0] * len(Z_L) + [1] * len(Z_S))
        silhouette_merge = float(silhouette_score(Z_both, labels))
    except Exception:
        silhouette_merge = float("nan")

    return {
        "centroid_dist_L_PL": centroid_dist_L_PL,
        "centroid_dist_merge_ZL_ZS": centroid_dist_merge_ZL_ZS,
        "mean_nn_dist_L_PL": mean_nn_dist_L_PL,
        "silhouette_merge": silhouette_merge,
    }


def _draw_scatter(
    ax,
    coords_list: list[np.ndarray],
    labels_list: list[str],
    colors: list[str],
    title: str,
    method_name: str,
    draw_connections: bool = False,
    point_labels: np.ndarray | None = None,
) -> None:
    """Draw scatter plot with multiple groups.

    Args:
        ax: matplotlib axis
        coords_list: list of (n_i, 2) coordinate arrays
        labels_list: list of group names
        colors: list of colors per group
        title: panel title
        method_name: "UMAP" or "PCA"
        draw_connections: if True and groups 0,1 have same n, draw thin lines
        point_labels: (n,) integer class labels for marker shapes
    """
    markers = ['o', 's', '^', 'D', 'v']

    # Draw connections between first two groups if requested
    if draw_connections and len(coords_list) >= 2:
        c0, c1 = coords_list[0], coords_list[1]
        if len(c0) == len(c1):
            for i in range(len(c0)):
                ax.plot(
                    [c0[i, 0], c1[i, 0]],
                    [c0[i, 1], c1[i, 1]],
                    color="gray", alpha=0.2, linewidth=0.5, zorder=0,
                )

    for coords, label, color in zip(coords_list, labels_list, colors):
        if point_labels is not None and len(point_labels) == len(coords):
            unique_cls = sorted(set(point_labels.tolist()))
            for cls_i, cls in enumerate(unique_cls[:5]):
                mask = (point_labels == cls)
                marker = markers[cls_i % len(markers)]
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color, marker=marker, alpha=0.6, s=20,
                    label=f"{label} (cls {cls})" if cls_i == 0 else None,
                )
        else:
            ax.scatter(
                coords[:, 0], coords[:, 1],
                c=color, alpha=0.6, s=20, label=label,
            )

    ax.set_title(title)
    ax.set_xlabel(f"{method_name} dim 1")
    ax.set_ylabel(f"{method_name} dim 2")
    ax.legend(fontsize=8)


def _compute_panel_coords(
    F_L_sub: np.ndarray,
    F_S_sub: np.ndarray,
    cca,
    reg_s_to_l,
    seed: int,
    pca_pre_dim: int,
) -> tuple:
    """Compute 2D coordinates for all three panels.

    Args:
        F_L_sub: (n, d_L) subsampled Large features
        F_S_sub: (n, d_S) subsampled Small features
        cca: CCAMetrics with projection matrices
        reg_s_to_l: fitted sklearn regressor S→L
        seed: random seed for reducers
        pca_pre_dim: PCA pre-reduction dim threshold

    Returns:
        (coords_A_L, coords_A_S, coords_B_L, coords_B_PL,
         coords_C_L, coords_C_S, F_PL, method_name)
    """
    mean_L = cca.mean_L.astype(np.float64)
    mean_S = cca.mean_S.astype(np.float64)
    std_L = cca.std_L.astype(np.float64) if cca.std_L is not None else None
    std_S = cca.std_S.astype(np.float64) if cca.std_S is not None else None
    W_L = cca.W_L.astype(np.float64)
    W_S = cca.W_S.astype(np.float64)

    Z_L = _project_cca(F_L_sub, mean_L, std_L, W_L)
    Z_S = _project_cca(F_S_sub, mean_S, std_S, W_S)
    F_PL = reg_s_to_l.predict(F_S_sub).astype(np.float64)

    n_L = len(Z_L)

    # Panel A — CCA space Z_L vs Z_S
    joint_A = np.concatenate([Z_L, Z_S], axis=0)
    coords_A, method_name = _build_reducer(joint_A, seed=seed, pca_pre_dim=pca_pre_dim)
    coords_A_L = coords_A[:n_L]
    coords_A_S = coords_A[n_L:]

    # Panel B — Feature space F_L vs F_PL
    joint_B = np.concatenate([F_L_sub, F_PL], axis=0)
    coords_B, _ = _build_reducer(joint_B, seed=seed, pca_pre_dim=pca_pre_dim)
    coords_B_L = coords_B[:n_L]
    coords_B_PL = coords_B[n_L:]

    # Panel C — Merge space: Z_L vs Z_S (independent reducer)
    joint_C = np.concatenate([Z_L, Z_S], axis=0)
    coords_C, _ = _build_reducer(joint_C, seed=seed, pca_pre_dim=pca_pre_dim)
    coords_C_L = coords_C[:n_L]
    coords_C_S = coords_C[n_L:]

    return (coords_A_L, coords_A_S, coords_B_L, coords_B_PL,
            coords_C_L, coords_C_S, F_PL, method_name)


def visualize_geometry(
    F_L: np.ndarray,
    F_S: np.ndarray,
    cca,
    reg_s_to_l,
    output_dir: str,
    max_points: int = 500,
    seed: int = 42,
    pca_pre_dim: int = 50,
    draw_connections: bool = False,
    labels: np.ndarray | None = None,
    layer_name: str = "",
    shuffle_baseline: bool = True,
    # CCA recompute params (needed for shuffle)
    n_components: int = 16,
    use_regularized: bool | None = None,
    lambda_L: float = 1e-3,
    lambda_S: float = 1e-3,
    standardize: bool = True,
    # Linear recompute params
    use_ridge: bool = False,
    ridge_alpha: float = 1.0,
    random_state: int = 42,
) -> dict:
    """Generate geometry visualization figure (1×3 or 2×3 with shuffle baseline).

    Args:
        F_L: (n, d_L) full Large features
        F_S: (n, d_S) full Small features
        cca: CCAMetrics with W_L, W_S, mean_L, mean_S, std_L, std_S
        reg_s_to_l: fitted sklearn regressor S→L
        output_dir: base output directory
        max_points: max samples to subsample
        seed: random seed
        pca_pre_dim: PCA pre-reduction dim threshold
        draw_connections: connect same-sample points with thin lines
        labels: (n,) integer class labels
        layer_name: suffix for output filename
        shuffle_baseline: if True, add a 2nd row with shuffled F_S baseline
        n_components: CCA components for shuffle recompute
        use_regularized: regularized CCA flag for shuffle recompute
        lambda_L: CCA lambda_L for shuffle recompute
        lambda_S: CCA lambda_S for shuffle recompute
        standardize: CCA standardize flag for shuffle recompute
        use_ridge: use Ridge regression for shuffle linear projector
        ridge_alpha: Ridge alpha for shuffle linear projector
        random_state: random state for shuffle CCA recompute

    Returns:
        dict of auxiliary metrics
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cca.W_L is None:
        logger.warning("cca.W_L is None — skipping geometry visualization.")
        return {}

    # Step 1: Subsample
    n = len(F_L)
    rng = np.random.default_rng(seed)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n)

    F_L_sub = F_L[idx].astype(np.float64)
    F_S_sub = F_S[idx].astype(np.float64)
    labels_sub = labels[idx] if labels is not None else None

    # Step 2: Compute real panel coords
    (coords_A_L, coords_A_S, coords_B_L, coords_B_PL,
     coords_C_L, coords_C_S, F_PL, method_name) = _compute_panel_coords(
        F_L_sub, F_S_sub, cca, reg_s_to_l, seed, pca_pre_dim
    )

    # Step 3: Optionally compute shuffle baseline
    shuf_coords = None
    if shuffle_baseline:
        try:
            from .metrics_cca import compute_cca_metrics
            from .metrics_linear import fit_linear_projector

            rng_shuf = np.random.default_rng(seed + 999)
            shuf_order = rng_shuf.permutation(len(F_S_sub))
            F_S_shuf = F_S_sub[shuf_order]

            cca_shuf = compute_cca_metrics(
                F_L_sub, F_S_shuf,
                n_components=n_components,
                r_values=[1],
                knn_ks=[1],
                test_size=0.2,
                random_state=random_state,
                use_regularized=use_regularized,
                lambda_L=lambda_L,
                lambda_S=lambda_S,
                standardize=standardize,
            )

            if cca_shuf.W_L is None:
                logger.warning("cca_shuf.W_L is None — skipping shuffle baseline row.")
            else:
                reg_shuf = fit_linear_projector(
                    F_S_shuf, F_L_sub,
                    use_ridge=use_ridge,
                    ridge_alpha=ridge_alpha,
                )
                shuf_coords = _compute_panel_coords(
                    F_L_sub, F_S_shuf, cca_shuf, reg_shuf, seed, pca_pre_dim
                )
        except Exception as e:
            logger.warning(f"シャッフルベースライン計算中にエラー: {e} — スキップします。")

    # Step 4: Create figure
    n_rows = 2 if (shuffle_baseline and shuf_coords is not None) else 1
    fig_height = 6 * n_rows
    fig, axes_all = plt.subplots(n_rows, 3, figsize=(18, fig_height))

    # Normalize axes to 2D array
    if n_rows == 1:
        axes_row0 = axes_all
    else:
        axes_row0 = axes_all[0]

    _draw_scatter(
        axes_row0[0],
        [coords_A_L, coords_A_S],
        ["Large", "Small"],
        ["steelblue", "salmon"],
        "(A) CCA Common Space: Large vs Small",
        method_name,
        draw_connections=draw_connections,
        point_labels=labels_sub,
    )

    _draw_scatter(
        axes_row0[1],
        [coords_B_L, coords_B_PL],
        ["Large", "pseudo-Large (S→L)"],
        ["steelblue", "mediumseagreen"],
        "(B) Large Feature Space: Original vs Reconstructed",
        method_name,
        draw_connections=draw_connections,
        point_labels=labels_sub,
    )

    _draw_scatter(
        axes_row0[2],
        [coords_C_L, coords_C_S],
        ["Large", "Small"],
        ["steelblue", "salmon"],
        "(C) Merge Space (CCA canonical)",
        method_name,
        draw_connections=draw_connections,
        point_labels=labels_sub,
    )

    # Row 1: shuffle baseline
    if n_rows == 2 and shuf_coords is not None:
        (sc_A_L, sc_A_S, sc_B_L, sc_B_PL,
         sc_C_L, sc_C_S, _F_PL_shuf, shuf_method) = shuf_coords
        axes_row1 = axes_all[1]

        _draw_scatter(
            axes_row1[0],
            [sc_A_L, sc_A_S],
            ["Large", "Small"],
            ["steelblue", "salmon"],
            "(A) CCA Common Space: Large vs Small [Shuffled baseline]",
            shuf_method,
            draw_connections=draw_connections,
            point_labels=labels_sub,
        )

        _draw_scatter(
            axes_row1[1],
            [sc_B_L, sc_B_PL],
            ["Large", "pseudo-Large (S→L)"],
            ["steelblue", "mediumseagreen"],
            "(B) Large Feature Space: Original vs Reconstructed [Shuffled baseline]",
            shuf_method,
            draw_connections=draw_connections,
            point_labels=labels_sub,
        )

        _draw_scatter(
            axes_row1[2],
            [sc_C_L, sc_C_S],
            ["Large", "Small"],
            ["steelblue", "salmon"],
            "(C) Merge Space (CCA canonical) [Shuffled baseline]",
            shuf_method,
            draw_connections=draw_connections,
            point_labels=labels_sub,
        )

    fig.tight_layout()

    # Save figure
    geo_dir = os.path.join(output_dir, "geometry")
    os.makedirs(geo_dir, exist_ok=True)

    if layer_name:
        fig_path = os.path.join(geo_dir, f"geometry_umap_{layer_name}.png")
    else:
        fig_path = os.path.join(geo_dir, "geometry_umap.png")

    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"幾何可視化図を保存しました: {fig_path}")

    # Step 5: Compute aux metrics (unchanged)
    # Recompute Z_L / Z_S for aux metrics using real cca projections
    _mean_L = cca.mean_L.astype(np.float64)
    _mean_S = cca.mean_S.astype(np.float64)
    _std_L = cca.std_L.astype(np.float64) if cca.std_L is not None else None
    _std_S = cca.std_S.astype(np.float64) if cca.std_S is not None else None
    _W_L = cca.W_L.astype(np.float64)
    _W_S = cca.W_S.astype(np.float64)
    _Z_L = _project_cca(F_L_sub, _mean_L, _std_L, _W_L)
    _Z_S = _project_cca(F_S_sub, _mean_S, _std_S, _W_S)
    aux = compute_geometry_aux_metrics(F_L_sub, F_PL, _Z_L, _Z_S)

    # Save aux metrics to text file
    metrics_path = os.path.join(geo_dir, "geometry_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in aux.items():
            f.write(f"{k}={v}\n")

    return aux
