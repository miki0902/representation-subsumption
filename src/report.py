"""
Markdown レポート生成: GitHub Issue への貼り付けに適した構造化出力。

なぜ Markdown ファイルとして出力するか:
  実験結果は再現可能かつ追跡可能である必要がある。
  構造化 Markdown を出力することで GitHub Issue への貼り付けが容易になり、
  テーブルや見出しが読みやすい形式で保存・共有できる。
"""

from datetime import datetime
from pathlib import Path

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
    """Markdown サマリーレポートを構築してファイルに書き出す。

    戻り値: レポート文字列（後続の処理でも利用できるよう返す）。
    """
    lines = []
    exp = cfg.get("experiment", {})

    lines += [
        "# Representation Subsumption 分析レポート",
        "",
        f"**日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**実験名**: {exp.get('name', 'N/A')}",
        f"**Large モデル**: `{exp.get('large_model', 'N/A')}`",
        f"**Small モデル**: `{exp.get('small_model', 'N/A')}`",
        "",
        "## データセット",
        "",
        "| 項目 | 値 |",
        "|------|---|",
        f"| サンプル数 | {n_samples} |",
        f"| d_L | {d_L} |",
        f"| d_S | {d_S} |",
        "",
    ]

    lines += [
        "## 線形包含",
        "",
        "| 指標 | 値 |",
        "|------|---|",
        f"| R²_L→S | {linear.r2_l_to_s:.4f} |",
        f"| R²_S→L | {linear.r2_s_to_l:.4f} |",
        f"| MSE_L→S | {linear.mse_l_to_s:.6f} |",
        f"| MSE_S→L | {linear.mse_s_to_l:.6f} |",
        f"| Directional Gap (R²_L→S − R²_S→L) | {linear.directional_gap:.4f} |",
        "",
    ]

    lines += [
        "## 幾何的整合",
        "",
        "| 指標 | 値 |",
        "|------|---|",
        f"| CKA | {geometry.cka:.4f} |",
        f"| RSA (Spearman ρ) | {geometry.rsa_spearman:.4f} |",
    ]
    for k, v in sorted(geometry.mutual_knn.items()):
        lines.append(f"| mutual_kNN@{k} | {v:.4f} |")
    lines.append("")

    lines += [
        "## 部分空間包含",
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
    lines = ["## 主要な観察", ""]

    # 線形包含の解釈
    if linear.r2_l_to_s > 0.9 and linear.r2_s_to_l < 0.5:
        obs = "Large が Small を線形的に包含しています（R²_L→S が高く R²_S→L が低い）。"
    elif linear.r2_l_to_s > 0.8 and linear.r2_s_to_l > 0.8:
        obs = "両方向とも高い: 表現はほぼ同型です。"
    elif linear.r2_l_to_s > 0.5 and linear.r2_s_to_l > 0.5:
        obs = "双方向 R² が中程度: 共有成分はあるが完全包含ではありません。"
    else:
        obs = "R²_L→S が低い: Small には Large から線形説明できない成分があります。"
    lines += [f"- **線形**: {obs}"]

    # 幾何的整合の解釈
    if geometry.cka > 0.9:
        lines += ["- **CKA**: 非常に高い — 表現は幾何的に類似しています。"]
    elif geometry.cka > 0.6:
        lines += ["- **CKA**: 中程度 — 部分的な幾何的整合があります。"]
    else:
        lines += ["- **CKA**: 低い — 表現の幾何構造が異なります。"]

    # 部分空間包含の解釈
    max_r = max(subspace.containment_s_in_l)
    c_s_in_l = subspace.containment_s_in_l[max_r]
    if c_s_in_l > 0.9:
        lines += [f"- **部分空間 (r={max_r})**: Small の主方向は Large の空間によく含まれています。"]
    elif c_s_in_l > 0.6:
        lines += [f"- **部分空間 (r={max_r})**: Small の Large への部分的な包含があります。"]
    else:
        lines += [f"- **部分空間 (r={max_r})**: Small には Large の部分空間外の主方向が存在します。"]

    lines.append("")
    return lines


def _layer_results_section(layer_results: list[dict]) -> list[str]:
    lines = ["## 層ペア分析", ""]
    lines += ["| Large 層 | Small 層 | R²_L→S | R²_S→L | CKA | RSA | kNN@10 |"]
    lines += ["|----------|----------|--------|--------|-----|-----|--------|"]
    for r in layer_results:
        lines.append(
            f"| {r['large_layer']} | {r['small_layer']} | "
            f"{r.get('r2_l_to_s', 'N/A'):.4f} | {r.get('r2_s_to_l', 'N/A'):.4f} | "
            f"{r.get('cka', 'N/A'):.4f} | {r.get('rsa', 'N/A'):.4f} | "
            f"{r.get('knn_10', 'N/A'):.4f} |"
        )
    lines.append("")
    return lines
