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
from .metrics_cca import CCAMetrics


def generate_report(
    cfg: dict,
    linear: LinearMetrics,
    geometry: GeometricMetrics,
    cca: CCAMetrics,
    n_samples: int,
    d_L: int,
    d_S: int,
    output_path: str,
    layer_results: list[dict] | None = None,
) -> str:
    """Markdown サマリーレポートを構築してファイルに書き出す。

    レポートの構成:
      1. ヘッダー（実験名・モデル名・日時）
      2. データセット（サンプル数・d_L・d_S）
      3. 線形包含（R² テーブル）
      4. 幾何的整合（CKA・RSA・kNN テーブル）
      5. CCA / Regularized CCA（正準相関テーブル・meanCCA@r・sharedScore@r・normalizedSharedScore@r）
      6. Merge 空間幾何（CKA・RSA・kNN on Z_L, Z_S）
      7. Train/Test Gap（過学習チェック）
      8. 主要な観察（自動解釈）
      9. 層ペア分析（オプション）

    戻り値: レポート文字列（後続の処理でも利用できるよう返す）。
    """
    lines = []
    exp = cfg.get("experiment", {})

    # 1. ヘッダー
    lines += [
        "# Representation Subsumption 分析レポート",
        "",
        f"**日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**実験名**: {exp.get('name', 'N/A')}",
        f"**Large モデル**: `{exp.get('large_model', 'N/A')}`",
        f"**Small モデル**: `{exp.get('small_model', 'N/A')}`",
        "",
    ]

    # 2. データセット
    lines += [
        "## データセット",
        "",
        "| 項目 | 値 |",
        "|------|---|",
        f"| サンプル数 | {n_samples} |",
        f"| d_L | {d_L} |",
        f"| d_S | {d_S} |",
        "",
    ]

    # 3. 線形包含
    lines += [
        "## 線形包含",
        "",
        "| 指標 | 値 |",
        "|------|---|",
        f"| R²_L→S | {linear.r2_l_to_s:.4f} |",
        f"| R²_S→L | {linear.r2_s_to_l:.4f} |",
        f"| Directional Gap (R²_L→S − R²_S→L) | {linear.directional_gap:.4f} |",
        "",
    ]

    # 4. 幾何的整合
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

    # 5. CCA / Regularized CCA
    cca_method = "Regularized CCA" if cca.used_regularized else "標準 CCA"
    lines += [
        "## CCA / Regularized CCA",
        "",
        f"**手法**: {cca_method}  ",
        f"**正準成分数**: {cca.n_components}  ",
    ]
    if cca.used_regularized:
        lines += [f"**λ_L**: {cca.lambda_L}  ", f"**λ_S**: {cca.lambda_S}  "]
    lines.append("")

    # 正準相関テーブル（上位 min(n_components, 8) 成分を表示する）
    n_show = min(cca.n_components, 8)
    lines += [
        "### 正準相関（テストセット）",
        "",
        "| 成分 | 正準相関 ρ |",
        "|------|-----------|",
    ]
    for i in range(n_show):
        lines.append(f"| {i + 1} | {cca.canonical_correlations[i]:.4f} |")
    if cca.n_components > n_show:
        lines.append(f"| ... | （{cca.n_components - n_show} 成分省略）|")
    lines.append("")

    # 集約指標テーブル（実際の r_values に合わせて動的生成）
    _r_keys = sorted(cca.mean_cca.keys())
    _r_fmt = lambda d, r: f"{d[r]:.4f}"
    _col_header = " | ".join(f"r={r}" for r in _r_keys)
    _col_sep = "|------|" + "------|" * len(_r_keys)
    lines += [
        "### 集約指標",
        "",
        f"| 指標 | {_col_header} |",
        _col_sep,
        "| meanCCA@r | " + " | ".join(_r_fmt(cca.mean_cca, r) for r in _r_keys) + " |",
        "| sharedScore@r | " + " | ".join(_r_fmt(cca.shared_score, r) for r in _r_keys) + " |",
        "| normalizedSharedScore@r | " + " | ".join(_r_fmt(cca.normalized_shared_score, r) for r in _r_keys) + " |",
        "",
    ]

    # 6. Merge 空間幾何
    _fmt_float = lambda v: f"{v:.4f}" if not (v != v) else "N/A (n_test<3)"
    lines += [
        "## Merge 空間幾何（Z_L vs Z_S）",
        "",
        "| 指標 | 値 |",
        "|------|---|",
        f"| CKA (merge) | {cca.merge_cka:.4f} |",
        f"| RSA Spearman ρ (merge) | {_fmt_float(cca.merge_rsa_spearman)} |",
    ]
    for k, v in sorted(cca.merge_mutual_knn.items()):
        lines.append(f"| mutual_kNN@{k} (merge) | {v:.4f} |")
    lines.append("")

    # 7. Train/Test Gap（過学習チェック）
    lines += [
        "## Train/Test Gap（過学習チェック）",
        "",
        "| 成分 | 訓練 ρ | テスト ρ | Gap |",
        "|------|--------|---------|-----|",
    ]
    for i in range(n_show):
        rho_train = cca.canonical_correlations_train[i]
        rho_test = cca.canonical_correlations[i]
        gap = cca.train_test_gap[i]
        lines.append(f"| {i + 1} | {rho_train:.4f} | {rho_test:.4f} | {gap:.4f} |")
    lines.append("")

    # 8. 主要な観察
    lines += _interpretation_block(linear, geometry, cca)

    # 9. 層ペア分析（オプション）
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
    cca: CCAMetrics,
) -> list[str]:
    """各指標の値を自動解釈して観察コメントを生成する。"""
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

    # CCA の解釈（利用可能な最大の r 値を使用する）
    r_values_available = sorted(cca.mean_cca.keys())
    if r_values_available:
        max_r = r_values_available[-1]
        mean_corr = cca.mean_cca[max_r]
        if mean_corr > 0.8:
            lines += [f"- **CCA (r={max_r})**: 非常に高い平均正準相関 — 強い共有潜在構造があります。"]
        elif mean_corr > 0.5:
            lines += [f"- **CCA (r={max_r})**: 中程度の平均正準相関 — 部分的な共有構造があります。"]
        else:
            lines += [f"- **CCA (r={max_r})**: 低い平均正準相関 — 共有潜在構造が少ない可能性があります。"]

    # Merge 空間の解釈
    if cca.merge_cka > 0.8:
        lines += ["- **Merge 空間 CKA**: 高い — CCA 写像後の表現は非常に類似しています。"]
    elif cca.merge_cka > 0.5:
        lines += ["- **Merge 空間 CKA**: 中程度 — CCA 写像後に一定の類似構造があります。"]
    else:
        lines += ["- **Merge 空間 CKA**: 低い — CCA 写像後も表現構造が異なります。"]

    # Train/Test Gap の解釈（過学習の警告）
    mean_gap = float(cca.train_test_gap.mean())
    if mean_gap > 0.2:
        lines += [f"- **⚠ Train/Test Gap が大きい** (平均 {mean_gap:.3f}): 過学習の可能性があります。Regularized CCA の λ を大きくすることを検討してください。"]
    else:
        lines += [f"- **Train/Test Gap**: 小さい (平均 {mean_gap:.3f}) — 正準相関の推定が安定しています。"]

    lines.append("")
    return lines


def _layer_results_section(layer_results: list[dict]) -> list[str]:
    """層ペア分析結果のテーブルセクションを生成する。"""
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
