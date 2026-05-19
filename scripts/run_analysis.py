"""
マルチファイル分析スクリプト: 設定ファイルを読み込み、全指標を計算してレポートと図を出力する。

使い方:
  python scripts/run_analysis.py --config configs/experiment.yaml
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from src import utils
from src.feature_io import load_features, align_features
from src.metrics_linear import compute_linear_metrics, fit_linear_projector
from src.metrics_geometry import compute_geometric_metrics
from src.metrics_cca import compute_cca_metrics
from src.report import generate_report

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Representation Subsumption 分析")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--debug", action="store_true")
    # dry-run: パイプライン全体が動くかを少数サンプルで確認するモード
    p.add_argument("--dry-run", action="store_true",
                   help="dry-run モード: 少数サンプルで全パイプラインの動作確認を行う")
    p.add_argument("--dry-run-samples", type=int, default=5,
                   help="dry-run 時に使用するサンプル数（デフォルト: 5）")
    return p.parse_args()


def _apply_dry_run(
    F_L: np.ndarray,
    F_S: np.ndarray,
    cfg: dict,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """dry-run 用にサンプルを切り出し、設定を少数サンプル向けに上書きする。

    なぜ cfg を上書きするか:
      mutual kNN (k < n_test)・CCA (n_components < n_train) の制約を
      サンプル数から逆算して満たすため。元の cfg は変更せず、コピーを返す。
    """
    import copy
    cfg = copy.deepcopy(cfg)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F_L), size=min(n_samples, len(F_L)), replace=False)
    F_L = F_L[idx]
    F_S = F_S[idx]
    n = len(F_L)

    # test_size を調整して n_test >= 2 を保証する
    test_size = max(cfg.get("linear", {}).get("test_size", 0.2), 2 / n)
    n_test = max(2, int(np.ceil(n * test_size)))
    n_train = n - n_test
    k_max = min(n_test, n) - 1

    # linear / geometry の設定を上書きする
    cfg.setdefault("linear", {})["test_size"] = test_size
    cfg.setdefault("geometry", {})["knn_k"] = [k for k in cfg["geometry"].get("knn_k", [5, 10, 20]) if k <= k_max] or [max(1, k_max)]

    # CCA の設定を上書きする
    cca_cfg = cfg.setdefault("cca", {})
    cca_cfg["test_size"] = test_size
    n_comp = min(cca_cfg.get("n_components", 16), max(1, n_train - 1))
    cca_cfg["n_components"] = n_comp
    cca_cfg["r_values"] = [r for r in cca_cfg.get("r_values", [4, 8, 16]) if r <= n_comp] or [1]
    cca_cfg["knn_k"] = [k for k in cca_cfg.get("knn_k", [5, 10, 20]) if k <= k_max] or [max(1, k_max)]
    cca_cfg["use_regularized"] = True  # 少数サンプルでは共分散行列が必ず特異になる

    logger.warning(
        f"[dry-run] n={n}, n_train={n_train}, n_test={n_test}, "
        f"knn_k={cfg['geometry']['knn_k']}, n_components={n_comp}, "
        f"r_values={cca_cfg['r_values']}, Regularized CCA を強制使用"
    )
    return F_L, F_S, cfg


def _log_config(cfg: dict, dry_run: bool, dry_run_samples: int) -> None:
    """実行設定をログに出力する。"""
    exp = cfg.get("experiment", {})
    feat = cfg.get("features", {})
    lin = cfg.get("linear", {})
    geo = cfg.get("geometry", {})
    cca = cfg.get("cca", {})
    out = cfg.get("output", {})
    logger.info("=" * 50)
    logger.info("[設定] 実験")
    logger.info(f"  name         : {exp.get('name', 'N/A')}")
    logger.info(f"  large_model  : {exp.get('large_model', 'N/A')}")
    logger.info(f"  small_model  : {exp.get('small_model', 'N/A')}")
    logger.info("[設定] 特徴量パス")
    logger.info(f"  large_path   : {feat.get('large_path', 'N/A')}")
    logger.info(f"  small_path   : {feat.get('small_path', 'N/A')}")
    logger.info("[設定] 線形包含")
    logger.info(f"  test_size    : {lin.get('test_size', 0.2)}")
    logger.info(f"  random_state : {lin.get('random_state', 42)}")
    logger.info(f"  use_ridge    : {lin.get('use_ridge', False)}")
    logger.info(f"  ridge_alpha  : {lin.get('ridge_alpha', 1.0)}")
    logger.info("[設定] 幾何的整合")
    logger.info(f"  knn_k        : {geo.get('knn_k', [5, 10, 20])}")
    logger.info("[設定] CCA")
    logger.info(f"  n_components : {cca.get('n_components', 16)}")
    logger.info(f"  r_values     : {cca.get('r_values', [4, 8, 16])}")
    logger.info(f"  lambda_L     : {cca.get('lambda_L', 1e-3)}")
    logger.info(f"  lambda_S     : {cca.get('lambda_S', 1e-3)}")
    logger.info(f"  use_regularized: {cca.get('use_regularized', None)}")
    logger.info(f"  standardize  : {cca.get('standardize', True)}")
    logger.info("[設定] 出力先")
    logger.info(f"  figures_dir  : {out.get('figures_dir', 'results/figures')}")
    logger.info(f"  reports_dir  : {out.get('reports_dir', 'results/reports')}")
    if dry_run:
        logger.info(f"[設定] dry-run  : True (samples={dry_run_samples})")
    logger.info("=" * 50)


def run(cfg: dict, dry_run: bool = False, dry_run_samples: int = 5) -> None:
    """設定辞書を受け取り、全指標の計算・レポート生成・図保存を行う。"""
    _log_config(cfg, dry_run, dry_run_samples)
    out_cfg = cfg.get("output", {})
    utils.ensure_dirs(
        out_cfg.get("figures_dir", "results/figures"),
        out_cfg.get("tables_dir", "results/tables"),
        out_cfg.get("reports_dir", "results/reports"),
    )

    # 特徴量の読み込みとアライメント
    feat_cfg = cfg.get("features", {})
    large_feat = load_features(feat_cfg["large_path"], model_name=cfg["experiment"].get("large_model", "large"))
    small_feat = load_features(feat_cfg["small_path"], model_name=cfg["experiment"].get("small_model", "small"))

    F_L, F_S = align_features(large_feat, small_feat)

    # dry-run: サンプルを切り出して設定を調整する
    if dry_run:
        seed = cfg.get("linear", {}).get("random_state", 42)
        F_L, F_S, cfg = _apply_dry_run(F_L, F_S, cfg, dry_run_samples, seed)

    n_samples, d_L = F_L.shape
    d_S = F_S.shape[1]
    logger.info(f"分析対象: n={n_samples}, d_L={d_L}, d_S={d_S}")

    # 線形包含指標の計算
    lin_cfg = cfg.get("linear", {})
    linear = compute_linear_metrics(
        F_L, F_S,
        test_size=lin_cfg.get("test_size", 0.2),
        random_state=lin_cfg.get("random_state", 42),
        use_ridge=lin_cfg.get("use_ridge", False),
        ridge_alpha=lin_cfg.get("ridge_alpha", 1.0),
    )
    logger.info(f"線形: R²_L→S={linear.r2_l_to_s:.4f}, R²_S→L={linear.r2_s_to_l:.4f}, gap={linear.directional_gap:.4f}")

    # 幾何的整合指標の計算
    geo_cfg = cfg.get("geometry", {})
    geometry = compute_geometric_metrics(F_L, F_S, knn_ks=geo_cfg.get("knn_k", [5, 10, 20]))
    logger.info(f"幾何: CKA={geometry.cka:.4f}, RSA={geometry.rsa_spearman:.4f}")

    # CCA / Regularized CCA 指標の計算
    cca_cfg = cfg.get("cca", {})
    cca = compute_cca_metrics(
        F_L, F_S,
        n_components=cca_cfg.get("n_components", 16),
        r_values=cca_cfg.get("r_values", [4, 8, 16]),
        knn_ks=cca_cfg.get("knn_k", [5, 10, 20]),
        test_size=cca_cfg.get("test_size", 0.2),
        random_state=cca_cfg.get("random_state", 42),
        use_regularized=cca_cfg.get("use_regularized", None),
        lambda_L=cca_cfg.get("lambda_L", 1e-3),
        lambda_S=cca_cfg.get("lambda_S", 1e-3),
        standardize=cca_cfg.get("standardize", True),
    )
    method = "Regularized CCA" if cca.used_regularized else "標準 CCA"
    logger.info(f"CCA ({method}): n_components={cca.n_components}, meanCCA@{min(cca.mean_cca.keys())}={list(cca.mean_cca.values())[0]:.4f}")

    # レポートの生成
    report_path = Path(out_cfg.get("reports_dir", "results/reports")) / "summary.md"
    report = generate_report(
        cfg=cfg,
        linear=linear,
        geometry=geometry,
        cca=cca,
        n_samples=n_samples,
        d_L=d_L,
        d_S=d_S,
        output_path=str(report_path),
    )

    logger.info(f"レポートを出力しました: {report_path}")
    print(report)

    # 図の生成（matplotlib がない場合はスキップする）
    try:
        _make_figures(linear, geometry, cca, out_cfg)
    except ImportError:
        logger.warning("matplotlib が見つかりません。図の生成をスキップします。")

    # 幾何可視化（matplotlib が必要、エラー時はスキップ）
    try:
        from src.visualize import visualize_geometry
        figures_dir = out_cfg.get("figures_dir", "results/figures")
        reg_s_to_l = fit_linear_projector(
            F_S, F_L,
            use_ridge=lin_cfg.get("use_ridge", False),
            ridge_alpha=lin_cfg.get("ridge_alpha", 1.0),
        )
        aux = visualize_geometry(F_L, F_S, cca, reg_s_to_l, figures_dir)
        logger.info(
            f"幾何可視化: centroid_dist_L_PL={aux.get('centroid_dist_L_PL', float('nan')):.4f}, "
            f"silhouette_merge={aux.get('silhouette_merge', float('nan')):.4f}"
        )
    except Exception as e:
        logger.warning(f"幾何可視化をスキップしました: {e}")


def _make_figures(linear, geometry, cca, out_cfg: dict) -> None:
    """分析図を生成して figures_dir に保存する。matplotlib が必要。"""
    import matplotlib.pyplot as plt
    figs_dir = Path(out_cfg.get("figures_dir", "results/figures"))

    # Directional R² bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["R2_L->S", "R2_S->L", "Gap"]
    values = [linear.r2_l_to_s, linear.r2_s_to_l, linear.directional_gap]
    colors = ["steelblue", "salmon", "mediumseagreen"]
    ax.bar(labels, values, color=colors)
    y_margin = max(1.0, max(abs(v) for v in values) * 1.1)
    ax.set_ylim(-y_margin, y_margin)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Linear Containment: Directional R2")
    ax.set_ylabel("R2")
    fig.tight_layout()
    fig.savefig(figs_dir / "linear_directional_gap.png", dpi=150)
    plt.close(fig)

    # Canonical correlation spectrum bar chart (train vs test)
    n_show = min(cca.n_components, 16)
    components = list(range(1, n_show + 1))
    rho_train = cca.canonical_correlations_train[:n_show]
    rho_test = cca.canonical_correlations[:n_show]
    x = np.arange(n_show)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, n_show * 0.5 + 1), 4))
    ax.bar(x - width / 2, rho_train, width, label="Train", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, rho_test, width, label="Test", color="salmon", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in components])
    ax.set_xlabel("Canonical Component")
    ax.set_ylabel("Canonical Correlation rho")
    ax.set_title("Canonical Correlation Spectrum (Train vs Test)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "canonical_correlations.png", dpi=150)
    plt.close(fig)

    # Cumulative shared score curve
    r_vals = sorted(cca.shared_score.keys())
    ss_vals = [cca.shared_score[r] for r in r_vals]
    nss_vals = [cca.normalized_shared_score[r] for r in r_vals]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r_vals, ss_vals, marker="o", label="sharedScore@r", color="steelblue")
    ax.plot(r_vals, nss_vals, marker="s", label="normalizedSharedScore@r", color="darkorchid")
    ax.set_xlabel("r (top components)")
    ax.set_ylabel("Score")
    ax.set_title("Cumulative Shared Score vs. r")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "shared_score.png", dpi=150)
    plt.close(fig)

    # Mutual kNN overlap (original space vs merge space)
    ks = sorted(geometry.mutual_knn)
    knn_vals = [geometry.mutual_knn[k] for k in ks]
    merge_knn_vals = [cca.merge_mutual_knn.get(k, float("nan")) for k in ks]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ks, knn_vals, marker="o", color="darkorchid", label="Original Space")
    ax.plot(ks, merge_knn_vals, marker="s", color="teal", label="Merge Space")
    ax.set_xlabel("k")
    ax.set_ylabel("Mutual kNN Overlap")
    ax.set_title("Mutual kNN Overlap vs. k")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "mutual_knn.png", dpi=150)
    plt.close(fig)

    # Train vs Test canonical correlation scatter (overfitting check)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(cca.canonical_correlations_train, cca.canonical_correlations,
               alpha=0.7, color="steelblue", edgecolors="black", linewidths=0.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("Train Canonical Correlation rho")
    ax.set_ylabel("Test Canonical Correlation rho")
    ax.set_title("Train vs Test Canonical Correlation")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "train_vs_test_cca.png", dpi=150)
    plt.close(fig)

    logger.info(f"図を保存しました: {figs_dir}")


def main() -> None:
    args = parse_args()
    utils.setup_logging(logging.DEBUG if args.debug else logging.INFO)
    cfg = utils.load_yaml(args.config)
    run(cfg, dry_run=args.dry_run, dry_run_samples=args.dry_run_samples)


if __name__ == "__main__":
    main()
