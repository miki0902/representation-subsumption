"""
幾何可視化: Large / Small の表現類似性を3パネルで比較。

3パネル構成:
  (A) CCA common space: Z_L vs Z_S
      CCA で共通空間へ射影し UMAP/PCA で2D可視化。
      バイアス注記: CCA は相関最大化を目的とするため重なりやすい傾向あり。
      Shuffle行との比較でバイアスを確認できる。

  (B) Joint PCA→UMAP (bias-free): F_L vs F_S + ペア接続線
      CCAを介さずに F_L・F_S を独立に PCA 圧縮してから
      concat して joint UMAP。同一サンプル i を細線で結ぶ。
      線が短い ≈ 同一画像の表現が近い ≈ 表現包摂の証拠。
      CCA と異なり「重なるように」学習しない。

  (C) ペア距離ヒストグラム: 実データ vs Shuffle
      Joint UMAP 空間での ||coords_L[i] - coords_S[i]|| の分布。
      実データ << Shuffle → 本物の類似性あり。
      実データ ≈ Shuffle → 偶然の一致。

行構成 (shuffle_baseline=True のとき 2×3):
  Row 1: 実データ  (A, B, C)
  Row 2: Shuffle  (A', B', メトリクスサマリ)
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _project_cca(
    F: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray | None,
    W: np.ndarray,
) -> np.ndarray:
    """Center, optionally scale, and project F using CCA weights W."""
    F_c = F - mean
    if std is not None:
        F_c = F_c / std
    return F_c @ W


def _build_reducer(
    X_joint: np.ndarray,
    seed: int,
    pca_pre_dim: int = 50,
) -> tuple[np.ndarray, str]:
    """Fit a 2D UMAP/PCA reducer on X_joint.

    Applies StandardScaler then optional PCA pre-reduction before UMAP.
    Falls back to PCA if umap-learn is not installed.

    Returns:
        (coords_2d (N, 2), method_name)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_joint)

    n_samples, n_features = X_scaled.shape
    effective_pre_dim = min(pca_pre_dim, n_samples - 1, n_features - 1)
    if n_features > effective_pre_dim and effective_pre_dim >= 2:
        X_scaled = PCA(n_components=effective_pre_dim, random_state=seed).fit_transform(X_scaled)

    try:
        import umap  # noqa: PLC0415
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        coords_2d = reducer.fit_transform(X_scaled)
        method_name = "UMAP"
    except ImportError:
        logger.warning("umap-learn not installed, falling back to PCA for visualization.")
        coords_2d = PCA(n_components=2, random_state=seed).fit_transform(X_scaled)
        method_name = "PCA"

    return coords_2d, method_name


def _compute_cca_coords(
    F_L: np.ndarray,
    F_S: np.ndarray,
    cca,
    seed: int,
    pca_pre_dim: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Project F_L, F_S via CCA weights then UMAP.

    Returns:
        (coords_L (n,2), coords_S (n,2), method_name)
    """
    mean_L = cca.mean_L.astype(np.float64)
    mean_S = cca.mean_S.astype(np.float64)
    std_L = cca.std_L.astype(np.float64) if cca.std_L is not None else None
    std_S = cca.std_S.astype(np.float64) if cca.std_S is not None else None
    W_L = cca.W_L.astype(np.float64)
    W_S = cca.W_S.astype(np.float64)

    Z_L = _project_cca(F_L, mean_L, std_L, W_L)
    Z_S = _project_cca(F_S, mean_S, std_S, W_S)

    n = len(Z_L)
    coords, method_name = _build_reducer(
        np.concatenate([Z_L, Z_S], axis=0), seed=seed, pca_pre_dim=pca_pre_dim
    )
    return coords[:n], coords[n:], method_name


def _compute_joint_umap_coords(
    F_L: np.ndarray,
    F_S: np.ndarray,
    seed: int,
    pca_pre_dim: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Joint PCA→UMAP without any alignment objective (bias-free).

    Steps:
      1. Independently standardize F_L and F_S.
      2. Independently PCA-reduce each to pca_pre_dim dimensions.
      3. Concatenate and run a single UMAP fit (joint coordinate system).
      4. Compute per-pair euclidean distances in 2D space.

    The key property: UMAP sees a flat feature vector with no knowledge of
    which group each row belongs to — no correlation maximization is performed.

    Returns:
        coords_L:   (n, 2) Large coordinates
        coords_S:   (n, 2) Small coordinates
        pair_dists: (n,)   ||coords_L[i] - coords_S[i]|| for each sample i
        method_name: "UMAP" or "PCA"
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n = len(F_L)
    pca_dim = min(pca_pre_dim, n - 1, F_L.shape[1] - 1, F_S.shape[1] - 1)
    pca_dim = max(pca_dim, 2)

    # Independent standardization
    F_L_s = StandardScaler().fit_transform(F_L)
    F_S_s = StandardScaler().fit_transform(F_S)

    # Independent PCA (each model's own structure preserved)
    F_L_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(F_L_s)
    F_S_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(F_S_s)

    # Joint UMAP — no alignment, no correlation maximization
    X_joint = np.concatenate([F_L_pca, F_S_pca], axis=0)
    try:
        import umap  # noqa: PLC0415
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        coords_2d = reducer.fit_transform(X_joint)
        method_name = "UMAP"
    except ImportError:
        coords_2d = PCA(n_components=2, random_state=seed).fit_transform(X_joint)
        method_name = "PCA"

    coords_L = coords_2d[:n]
    coords_S = coords_2d[n:]
    pair_dists = np.linalg.norm(coords_L - coords_S, axis=1)

    return coords_L, coords_S, pair_dists, method_name


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
        draw_connections: if True, draw thin gray lines between groups[0] and groups[1].
    """
    markers = ['o', 's', '^', 'D', 'v']

    if draw_connections and len(coords_list) >= 2:
        c0, c1 = coords_list[0], coords_list[1]
        if len(c0) == len(c1):
            for i in range(len(c0)):
                ax.plot(
                    [c0[i, 0], c1[i, 0]], [c0[i, 1], c1[i, 1]],
                    color="gray", alpha=0.25, linewidth=0.5, zorder=0,
                )

    for coords, label, color in zip(coords_list, labels_list, colors):
        if point_labels is not None and len(point_labels) == len(coords):
            for cls_i, cls in enumerate(sorted(set(point_labels.tolist()))[:5]):
                mask = point_labels == cls
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color, marker=markers[cls_i % len(markers)],
                    alpha=0.6, s=20,
                    label=f"{label} (cls {cls})" if cls_i == 0 else None,
                )
        else:
            ax.scatter(coords[:, 0], coords[:, 1], c=color, alpha=0.6, s=20, label=label)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"{method_name} dim 1")
    ax.set_ylabel(f"{method_name} dim 2")
    ax.legend(fontsize=8)


def _draw_pair_distance_hist(
    ax,
    real_dists: np.ndarray,
    shuf_dists: np.ndarray | None = None,
    title: str = "Pair Distance Distribution (Joint UMAP space)",
) -> None:
    """Histogram of per-sample pair distances ||coords_L[i] - coords_S[i]||.

    Shows real data (blue) vs shuffle baseline (salmon) in the same panel.
    A left-shifted blue distribution indicates genuine representational similarity.
    """
    bins = min(30, max(10, len(real_dists) // 5))

    ax.hist(real_dists, bins=bins, alpha=0.7, color="steelblue",
            label=f"Real   mean={real_dists.mean():.3f}")
    ax.axvline(real_dists.mean(), color="steelblue", linestyle="--", linewidth=1.5)

    if shuf_dists is not None:
        ax.hist(shuf_dists, bins=bins, alpha=0.7, color="salmon",
                label=f"Shuffle  mean={shuf_dists.mean():.3f}")
        ax.axvline(shuf_dists.mean(), color="salmon", linestyle="--", linewidth=1.5)

        ratio = real_dists.mean() / (shuf_dists.mean() + 1e-12)
        ax.set_title(f"{title}\nreal/shuffle mean ratio = {ratio:.3f}", fontsize=9)
    else:
        ax.set_title(title, fontsize=10)

    ax.set_xlabel("Euclidean distance in Joint UMAP space")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)


def _draw_metrics_text(ax, metrics_real: dict, metrics_shuf: dict | None = None) -> None:
    """Display geometry metrics as text in a panel."""
    ax.axis("off")
    lines = ["── Geometry Metrics ──\n"]

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "N/A"

    for k, v in metrics_real.items():
        line = f"[Real]    {k} = {_fmt(v)}"
        if metrics_shuf and k in metrics_shuf:
            line += f"\n[Shuffle] {k} = {_fmt(metrics_shuf[k])}"
        lines.append(line)

    ax.text(
        0.05, 0.95, "\n\n".join(lines),
        transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )


# ---------------------------------------------------------------------------
# Aux metrics (updated: pseudo-Large removed)
# ---------------------------------------------------------------------------

def compute_geometry_aux_metrics(
    Z_L: np.ndarray,
    Z_S: np.ndarray,
    pair_dists_joint: np.ndarray | None = None,
) -> dict:
    """Compute auxiliary geometry metrics.

    Args:
        Z_L:               (n, k) Large CCA projections
        Z_S:               (n, k) Small CCA projections
        pair_dists_joint:  (n,) per-sample distances in Joint UMAP space (optional)

    Returns:
        dict with metric name → float value
    """
    centroid_ZL = Z_L.mean(axis=0)
    centroid_ZS = Z_S.mean(axis=0)
    centroid_dist_merge_ZL_ZS = float(np.linalg.norm(centroid_ZL - centroid_ZS))

    try:
        from sklearn.metrics import silhouette_score
        Z_both = np.concatenate([Z_L, Z_S], axis=0)
        lbl = np.array([0] * len(Z_L) + [1] * len(Z_S))
        silhouette_merge = float(silhouette_score(Z_both, lbl))
    except Exception:
        silhouette_merge = float("nan")

    result = {
        "centroid_dist_merge_ZL_ZS": centroid_dist_merge_ZL_ZS,
        "silhouette_merge": silhouette_merge,
    }

    if pair_dists_joint is not None:
        result["mean_pair_dist_joint_umap"] = float(pair_dists_joint.mean())

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def visualize_geometry(
    F_L: np.ndarray,
    F_S: np.ndarray,
    cca,
    reg_s_to_l,          # kept for API compatibility, no longer used
    output_dir: str,
    max_points: int = 500,
    seed: int = 42,
    pca_pre_dim: int = 50,
    draw_connections: bool = True,
    labels: np.ndarray | None = None,
    layer_name: str = "",
    shuffle_baseline: bool = True,
    # CCA recompute params (for shuffle row)
    n_components: int = 16,
    use_regularized: bool | None = None,
    lambda_L: float = 1e-3,
    lambda_S: float = 1e-3,
    standardize: bool = True,
    use_ridge: bool = False,
    ridge_alpha: float = 1.0,
    random_state: int = 42,
) -> dict:
    """Generate geometry visualization figure.

    Panel layout:
      (A) CCA common space  — Z_L vs Z_S (UMAP after CCA projection)
      (B) Joint PCA→UMAP   — F_L vs F_S, bias-free, pair connections drawn
      (C) Pair distance histogram — real vs shuffle in one plot

    When shuffle_baseline=True, a 2nd row is added:
      (A') CCA space [Shuffle]
      (B') Joint UMAP [Shuffle]   ← longer pair lines expected
      [Metrics summary text]

    Args:
        F_L:           (n, d_L) Large features
        F_S:           (n, d_S) Small features
        cca:           CCAMetrics with W_L, W_S, mean_L, mean_S, std_L, std_S
        reg_s_to_l:    (unused) kept for backward compatibility
        output_dir:    base output directory
        max_points:    max samples to subsample for visualization
        seed:          random seed
        pca_pre_dim:   PCA pre-reduction dim (for CCA UMAP panel)
        draw_connections: draw pair lines in Panel B/B'
        labels:        (n,) integer class labels for marker shapes
        layer_name:    suffix for output filename
        shuffle_baseline: add 2nd row with shuffled F_S
        n_components:  CCA components for shuffle recompute
        use_regularized, lambda_L, lambda_S, standardize: CCA params for shuffle
        use_ridge, ridge_alpha: linear projector params for shuffle
        random_state:  random state for shuffle CCA

    Returns:
        dict of auxiliary metrics
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cca.W_L is None:
        logger.warning("cca.W_L is None — skipping geometry visualization.")
        return {}

    # ------------------------------------------------------------------
    # Step 1: Subsample
    # ------------------------------------------------------------------
    n = len(F_L)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(max_points, n), replace=False)
    idx = np.sort(idx)

    F_L_sub = F_L[idx].astype(np.float64)
    F_S_sub = F_S[idx].astype(np.float64)
    labels_sub = labels[idx] if labels is not None else None

    # ------------------------------------------------------------------
    # Step 2: Real data — Panel A (CCA), Panel B (Joint UMAP)
    # ------------------------------------------------------------------
    coords_A_L, coords_A_S, method_A = _compute_cca_coords(
        F_L_sub, F_S_sub, cca, seed=seed, pca_pre_dim=pca_pre_dim
    )
    coords_B_L, coords_B_S, pair_dists_real, method_B = _compute_joint_umap_coords(
        F_L_sub, F_S_sub, seed=seed, pca_pre_dim=pca_pre_dim
    )

    # CCA projections for aux metrics
    _Z_L = _project_cca(
        F_L_sub,
        cca.mean_L.astype(np.float64),
        cca.std_L.astype(np.float64) if cca.std_L is not None else None,
        cca.W_L.astype(np.float64),
    )
    _Z_S = _project_cca(
        F_S_sub,
        cca.mean_S.astype(np.float64),
        cca.std_S.astype(np.float64) if cca.std_S is not None else None,
        cca.W_S.astype(np.float64),
    )

    # ------------------------------------------------------------------
    # Step 3: Shuffle baseline
    # ------------------------------------------------------------------
    shuf_data = None
    if shuffle_baseline:
        try:
            from .metrics_cca import compute_cca_metrics  # noqa: PLC0415

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
                coords_shA_L, coords_shA_S, method_shA = _compute_cca_coords(
                    F_L_sub, F_S_shuf, cca_shuf, seed=seed, pca_pre_dim=pca_pre_dim
                )
                coords_shB_L, coords_shB_S, pair_dists_shuf, method_shB = _compute_joint_umap_coords(
                    F_L_sub, F_S_shuf, seed=seed, pca_pre_dim=pca_pre_dim
                )
                _Z_L_sh = _project_cca(
                    F_L_sub,
                    cca_shuf.mean_L.astype(np.float64),
                    cca_shuf.std_L.astype(np.float64) if cca_shuf.std_L is not None else None,
                    cca_shuf.W_L.astype(np.float64),
                )
                _Z_S_sh = _project_cca(
                    F_S_shuf,
                    cca_shuf.mean_S.astype(np.float64),
                    cca_shuf.std_S.astype(np.float64) if cca_shuf.std_S is not None else None,
                    cca_shuf.W_S.astype(np.float64),
                )
                shuf_data = dict(
                    coords_A_L=coords_shA_L, coords_A_S=coords_shA_S, method_A=method_shA,
                    coords_B_L=coords_shB_L, coords_B_S=coords_shB_S, method_B=method_shB,
                    pair_dists=pair_dists_shuf, Z_L=_Z_L_sh, Z_S=_Z_S_sh,
                )

        except Exception as e:
            logger.warning(f"Shuffle baseline failed: {e} — skipping.")

    # ------------------------------------------------------------------
    # Step 4: Aux metrics
    # ------------------------------------------------------------------
    metrics_real = compute_geometry_aux_metrics(_Z_L, _Z_S, pair_dists_real)
    metrics_shuf = (
        compute_geometry_aux_metrics(
            shuf_data["Z_L"], shuf_data["Z_S"], shuf_data["pair_dists"]
        )
        if shuf_data is not None else None
    )

    # ------------------------------------------------------------------
    # Step 5: Build figure
    # ------------------------------------------------------------------
    n_rows = 2 if shuf_data is not None else 1
    fig_height = 6 * n_rows
    fig, axes_all = plt.subplots(n_rows, 3, figsize=(18, fig_height))
    if n_rows == 1:
        axes_row0 = axes_all
    else:
        axes_row0 = axes_all[0]

    # Row 0 — real data
    _draw_scatter(
        axes_row0[0],
        [coords_A_L, coords_A_S],
        ["Large", "Small"],
        ["steelblue", "salmon"],
        f"(A) CCA Common Space\nLarge vs Small",
        method_A,
        draw_connections=False,
        point_labels=labels_sub,
    )

    _draw_scatter(
        axes_row0[1],
        [coords_B_L, coords_B_S],
        ["Large", "Small"],
        ["steelblue", "salmon"],
        f"(B) Joint PCA→{method_B} (bias-free)\nPair lines: short = similar",
        method_B,
        draw_connections=draw_connections,
        point_labels=labels_sub,
    )

    _draw_pair_distance_hist(
        axes_row0[2],
        real_dists=pair_dists_real,
        shuf_dists=shuf_data["pair_dists"] if shuf_data is not None else None,
        title="(C) Pair Distance: Real vs Shuffle",
    )

    # Row 1 — shuffle baseline (if available)
    if n_rows == 2 and shuf_data is not None:
        axes_row1 = axes_all[1]

        _draw_scatter(
            axes_row1[0],
            [shuf_data["coords_A_L"], shuf_data["coords_A_S"]],
            ["Large", "Small"],
            ["steelblue", "salmon"],
            f"(A’) CCA Common Space [Shuffle]\nCorrespondence destroyed",
            shuf_data["method_A"],
            draw_connections=False,
            point_labels=labels_sub,
        )

        _draw_scatter(
            axes_row1[1],
            [shuf_data["coords_B_L"], shuf_data["coords_B_S"]],
            ["Large", "Small"],
            ["steelblue", "salmon"],
            f"(B’) Joint PCA→{shuf_data['method_B']} [Shuffle]\nLines should be longer than (B)",
            shuf_data["method_B"],
            draw_connections=draw_connections,
            point_labels=labels_sub,
        )

        _draw_metrics_text(axes_row1[2], metrics_real, metrics_shuf)

    # Overall title
    layer_str = f" [{layer_name}]" if layer_name else ""
    fig.suptitle(
        f"Representation Geometry{layer_str}\n"
        f"(B)(C): bias-free Joint PCA→{method_B}  |  "
        f"(A): CCA projection (structural overlap bias present)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ------------------------------------------------------------------
    # Step 6: Save
    # ------------------------------------------------------------------
    geo_dir = os.path.join(output_dir, "geometry")
    os.makedirs(geo_dir, exist_ok=True)

    suffix = f"_{layer_name}" if layer_name else ""
    fig_path = os.path.join(geo_dir, f"geometry_umap{suffix}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Geometry figure saved: {fig_path}")

    # Save metrics
    metrics_path = os.path.join(geo_dir, "geometry_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics_real.items():
            f.write(f"{k}={v}\n")
        if metrics_shuf:
            for k, v in metrics_shuf.items():
                f.write(f"shuffle_{k}={v}\n")
    logger.info(f"Geometry metrics saved: {metrics_path}")

    return metrics_real
