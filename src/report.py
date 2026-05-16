"""
Report generation: structured Markdown suitable for GitHub Issues.

Why: Experiment results should be reproducible and traceable. Outputting
a structured Markdown file makes it trivial to copy results into GitHub Issues,
preserving tables and headers for readability.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .metrics_linear import LinearMetrics
from .metrics_geometry import GeometricMetrics
from .metrics_subspace import SubspaceMetrics


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
        f"# Representation Subsumption Analysis",
        f"",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Experiment**: {exp.get('name', 'N/A')}",
        f"**Large model**: `{exp.get('large_model', 'N/A')}`",
        f"**Small model**: `{exp.get('small_model', 'N/A')}`",
        f"",
        f"## Dataset",
        f"",
        f"| Key | Value |",
        f"|-----|-------|",
        f"| Samples | {n_samples} |",
        f"| d_L | {d_L} |",
        f"| d_S | {d_S} |",
        f"",
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

    # Linear interpretation
    gap = linear.directional_gap
    if linear.r2_l_to_s > 0.9 and linear.r2_s_to_l < 0.5:
        obs = "Large linearly subsumes Small (high R²_L→S, low R²_S→L)."
    elif linear.r2_l_to_s > 0.8 and linear.r2_s_to_l > 0.8:
        obs = "Both directions high: representations are nearly isomorphic."
    elif linear.r2_l_to_s > 0.5 and linear.r2_s_to_l > 0.5:
        obs = "Moderate bidirectional R²: shared components exist but no full containment."
    else:
        obs = "Low R²_L→S: Small has components not linearly explained by Large."
    lines += [f"- **Linear**: {obs}"]

    # Geometric interpretation
    if geometry.cka > 0.9:
        lines += ["- **CKA**: Very high — representations are geometrically similar."]
    elif geometry.cka > 0.6:
        lines += ["- **CKA**: Moderate — partial geometric alignment."]
    else:
        lines += ["- **CKA**: Low — representations differ in geometry."]

    # Subspace interpretation
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
            f"{r.get('r2_l_to_s', 'N/A'):.4f} | {r.get('r2_s_to_l', 'N/A'):.4f} | "
            f"{r.get('cka', 'N/A'):.4f} | {r.get('rsa', 'N/A'):.4f} | "
            f"{r.get('knn_10', 'N/A'):.4f} |"
        )
    lines.append("")
    return lines
