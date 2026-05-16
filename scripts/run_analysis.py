"""
Multi-file analysis script: loads config, runs all metrics, writes report and figures.

Usage:
  python scripts/run_analysis.py --config configs/experiment.yaml
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from src import utils
from src.feature_io import load_features, align_features
from src.metrics_linear import compute_linear_metrics
from src.metrics_geometry import compute_geometric_metrics
from src.metrics_subspace import compute_subspace_metrics
from src.report import generate_report

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Representation Subsumption Analysis")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def run(cfg: dict) -> None:
    out_cfg = cfg.get("output", {})
    utils.ensure_dirs(
        out_cfg.get("figures_dir", "results/figures"),
        out_cfg.get("tables_dir", "results/tables"),
        out_cfg.get("reports_dir", "results/reports"),
    )

    feat_cfg = cfg.get("features", {})
    large_feat = load_features(feat_cfg["large_path"], model_name=cfg["experiment"].get("large_model", "large"))
    small_feat = load_features(feat_cfg["small_path"], model_name=cfg["experiment"].get("small_model", "small"))

    F_L, F_S = align_features(large_feat, small_feat)
    n_samples, d_L = F_L.shape
    d_S = F_S.shape[1]
    logger.info(f"Analysis: n={n_samples}, d_L={d_L}, d_S={d_S}")

    lin_cfg = cfg.get("linear", {})
    linear = compute_linear_metrics(
        F_L, F_S,
        test_size=lin_cfg.get("test_size", 0.2),
        random_state=lin_cfg.get("random_state", 42),
        use_ridge=lin_cfg.get("use_ridge", False),
        ridge_alpha=lin_cfg.get("ridge_alpha", 1.0),
    )

    geo_cfg = cfg.get("geometry", {})
    geometry = compute_geometric_metrics(F_L, F_S, knn_ks=geo_cfg.get("knn_k", [5, 10, 20]))

    sub_cfg = cfg.get("subspace", {})
    subspace = compute_subspace_metrics(F_L, F_S, n_components_list=sub_cfg.get("n_components", [16, 32, 64, 128]))

    report_path = Path(out_cfg.get("reports_dir", "results/reports")) / "summary.md"
    report = generate_report(
        cfg=cfg,
        linear=linear,
        geometry=geometry,
        subspace=subspace,
        n_samples=n_samples,
        d_L=d_L,
        d_S=d_S,
        output_path=str(report_path),
    )

    logger.info(f"Report written to {report_path}")
    print(report)

    try:
        _make_figures(linear, geometry, subspace, out_cfg)
    except ImportError:
        logger.warning("matplotlib not available; skipping figures.")


def _make_figures(linear, geometry, subspace, out_cfg: dict) -> None:
    import matplotlib.pyplot as plt
    figs_dir = Path(out_cfg.get("figures_dir", "results/figures"))

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
    fig.savefig(figs_dir / "linear_directional_gap.png", dpi=150)
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
    fig.savefig(figs_dir / "subspace_containment.png", dpi=150)
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
    fig.savefig(figs_dir / "mutual_knn.png", dpi=150)
    plt.close(fig)

    logger.info(f"Figures saved to {figs_dir}")


def main() -> None:
    args = parse_args()
    utils.setup_logging(logging.DEBUG if args.debug else logging.INFO)
    cfg = utils.load_yaml(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
