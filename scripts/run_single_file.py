"""
Representation Subsumption 分析の自己完結型単一ファイルスクリプト。

feature_io, metrics_linear, metrics_geometry, metrics_subspace, report, utils
のすべてのロジックをここにインライン化しています。
requirements.txt に記載の標準依存パッケージ以外のインストール不要です。

使い方:
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
    """y_true の分散が 0 の縮退ケースで 0 を返す R²。"""
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
    """特徴量を .pt / .npy / .npz ファイルから読み込む。

    なぜ統一ロード関数が必要か:
      研究環境では保存形式がスクリプトごとに異なることが多い。
      形式ごとの分岐を各スクリプトに書かずに済むよう、ここで吸収する。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"特徴量ファイルが見つかりません: {path}")

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
            raise ValueError(f"非対応の .pt コンテンツ型: {type(data)}")

    elif suffix == ".npy":
        features = np.load(path)

    elif suffix == ".npz":
        data = np.load(path)
        if "features" in data:
            features = data["features"]
            sample_ids = data.get("sample_ids", None)
        else:
            # "features" キーがない場合は最初の配列を特徴量として使う
            key = list(data.keys())[0]
            features = data[key]

    else:
        raise ValueError(f"非対応の拡張子: {suffix}。.pt / .npy / .npz を使用してください")

    features = np.asarray(features, dtype=np.float32)
    _validate_features(features, path)

    logger.info(f"{model_name} の特徴量を読み込みました: shape={features.shape}, path={path}")
    return FeatureSet(features=features, sample_ids=sample_ids, source_path=path, model_name=model_name)


def _validate_features(features: np.ndarray, source: str) -> None:
    """NaN / Inf を検出して明示的にエラーを出す。

    なぜ明示的に検出するか:
      NaN / Inf が混入していると後続の指標計算がすべて静かに壊れる。
      早期に検出して問題箇所を特定しやすくするため。
    """
    if np.any(np.isnan(features)):
        raise ValueError(f"{source} の特徴量に NaN が含まれています")
    if np.any(np.isinf(features)):
        raise ValueError(f"{source} の特徴量に Inf が含まれています")
    if features.ndim != 2:
        raise ValueError(f"2次元の特徴量配列を期待しましたが、{source} の shape は {features.shape} です")


def align_features(large: FeatureSet, small: FeatureSet) -> tuple[np.ndarray, np.ndarray]:
    """Large と Small の特徴量を共通 sample_id で行揃えする。

    なぜアライメントが必要か:
      異なる抽出ランからのサブセットなど、sample_id が食い違う場合がある。
      共通部分の積集合を取ることで行ごとの対応を保証する。
    戻り値: アライメント済みの (F_L, F_S) タプル。
    """
    if large.sample_ids is None or small.sample_ids is None:
        n_L, n_S = len(large.features), len(small.features)
        if n_L != n_S:
            raise ValueError(
                f"sample_ids なしでサンプル数が不一致: large={n_L}, small={n_S}。"
                " sample_ids を付与するか、サンプル数を揃えてください。"
            )
        return large.features, small.features

    common_ids = np.intersect1d(large.sample_ids, small.sample_ids)
    if len(common_ids) == 0:
        raise ValueError("large と small に共通する sample_id がありません。")

    dropped = (len(large.sample_ids) - len(common_ids)) + (len(small.sample_ids) - len(common_ids))
    if dropped > 0:
        logger.warning(
            f"sample_id 不一致により {dropped} サンプルを除外しました。共通サンプル数: {len(common_ids)}"
        )

    large_idx = np.where(np.isin(large.sample_ids, common_ids))[0]
    small_idx = np.where(np.isin(small.sample_ids, common_ids))[0]

    # 両側を common_ids でソートして行の対応を確定させる
    large_order = np.argsort(large.sample_ids[large_idx])
    small_order = np.argsort(small.sample_ids[small_idx])

    F_L = large.features[large_idx[large_order]]
    F_S = small.features[small_idx[small_order]]

    logger.info(f"アライメント完了: n_samples={len(common_ids)}, d_L={F_L.shape[1]}, d_S={F_S.shape[1]}")
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
    """F_L と F_S の双方向線形回帰 R² を計算する。

    なぜ train/test split を入れるか:
      高次元特徴量では訓練データへの過適合が起きやすく、
      R² が見かけ上高くなって包含の強さを過大評価してしまう。
      テストセットで評価することで汎化を確認する。

    Ridge 回帰は近似共線的な特徴量を扱う場合のオプション。
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
    """X→Y の回帰をフィットし、ホールドアウトテストセットで評価する。

    なぜテストセットで評価するか:
      訓練データで R² を測ると過適合の影響で実際の汎化能力より高く見える。
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
    mutual_knn: dict[int, float]  # k -> 重なり率


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
    """F_L と F_S の線形 CKA を計算する。

    なぜ CKA を使うか:
      直交変換・等方スケーリングに対して不変であり、
      座標系の違いで罰せられない原理的な表現類似度の尺度となる。（Kornblith et al., 2019）
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
    """中心化 Gram 行列による不偏 HSIC 推定量。

    なぜ中心化するか:
      中心化によって平均の影響を取り除き、共分散構造のみを取り出す。
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return float(np.trace(Kc @ Lc) / (n - 1) ** 2)


def compute_rsa(F_L: np.ndarray, F_S: np.ndarray) -> float:
    """表現類似性分析（RSA）: RDM の Spearman 相関。

    なぜ RSA を使うか:
      ペアワイズ距離のランク順構造を比較し、単調変換に頑健な幾何構造を捉える。
    """
    rdm_L = _upper_tri(euclidean_distances(F_L))
    rdm_S = _upper_tri(euclidean_distances(F_S))
    rho, _ = spearmanr(rdm_L, rdm_S)
    return float(rho)


def _upper_tri(D: np.ndarray) -> np.ndarray:
    """距離行列の上三角部分（対角除く）を抽出する。"""
    idx = np.triu_indices(D.shape[0], k=1)
    return D[idx]


def compute_mutual_knn(F_L: np.ndarray, F_S: np.ndarray, k: int) -> float:
    """Mutual k-NN 重なり率: 両空間で共有される k 近傍の割合。

    なぜ mutual kNN を使うか:
      座標が異なっていても「どのサンプルが近いか」という局所的な
      近傍構造がモデル間で一致するかを直接測定する。（Huh et al., 2024）
    """
    n = F_L.shape[0]
    if k >= n:
        raise ValueError(f"k={k} は n_samples={n} より小さくなければなりません")

    nn_L = _knn_indices(F_L, k)
    nn_S = _knn_indices(F_S, k)

    overlaps = np.array([
        len(np.intersect1d(nn_L[i], nn_S[i])) / k
        for i in range(n)
    ])
    return float(overlaps.mean())


def _knn_indices(F: np.ndarray, k: int) -> np.ndarray:
    """各サンプルの k 近傍インデックスを返す（自分自身を除く）。shape: (n, k)"""
    D = euclidean_distances(F)
    np.fill_diagonal(D, np.inf)  # 自分自身を近傍候補から除外する
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

    same_dim = F_L.shape[1] == F_S.shape[1]

    V_L_full = _top_components(F_L_c, max(n_components_list))
    V_S_full = _top_components(F_S_c, max(n_components_list))

    containment_s_in_l = {}
    containment_l_in_s = {}
    containment_gap = {}

    for r in n_components_list:
        if same_dim:
            # 共通環境空間での直接射影
            r_eff = min(r, V_L_full.shape[1], V_S_full.shape[1])
            V_L = V_L_full[:, :r_eff]
            V_S = V_S_full[:, :r_eff]
            c_s_in_l = _subspace_containment(V_S, V_L)
            c_l_in_s = _subspace_containment(V_L, V_S)
        else:
            # 次元が異なる場合: データを介した R² で包含度を代替測定する
            r_eff = min(r, V_L_full.shape[1], V_S_full.shape[1])
            V_L = V_L_full[:, :r_eff]
            V_S = V_S_full[:, :r_eff]
            scores_S = F_S_c @ V_S
            scores_L = F_L_c @ V_L
            c_s_in_l = _regression_r2(F_L_c, scores_S)
            c_l_in_s = _regression_r2(F_S_c, scores_L)

        containment_s_in_l[r] = c_s_in_l
        containment_l_in_s[r] = c_l_in_s
        containment_gap[r] = c_s_in_l - c_l_in_s

    return SubspaceMetrics(
        containment_s_in_l=containment_s_in_l,
        containment_l_in_s=containment_l_in_s,
        containment_gap=containment_gap,
    )


def _regression_r2(X: np.ndarray, Y: np.ndarray) -> float:
    """X の線形関数で Y の分散をどれだけ説明できるかを in-sample R² で測る。

    なぜここでは in-sample R² を使うか:
      サブスペース包含は汎化特性ではなく幾何的特性を問うている。
      PCA スコアのターゲットは構成上単位分散を持つため、
      in-sample R² が包含度の tight な代理指標となる。
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, Y)
    Y_pred = reg.predict(X)
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(max(0.0, 1.0 - ss_res / ss_tot))


def _center(F: np.ndarray) -> np.ndarray:
    return F - F.mean(axis=0)


def _top_components(F_centered: np.ndarray, r: int) -> np.ndarray:
    """SVD で上位 r 右特異ベクトルを抽出する。

    なぜ共分散行列ではなく SVD を使うか:
      高次元特徴量で共分散行列を明示的に計算すると数値的に不安定になる。
    戻り値の shape: (d, r)
    """
    r_eff = min(r, min(F_centered.shape))
    _, _, Vt = np.linalg.svd(F_centered, full_matrices=False)
    return Vt[:r_eff].T  # shape: (d, r_eff)


def _subspace_containment(V_query: np.ndarray, V_base: np.ndarray) -> float:
    """V_query の列空間が V_base の列空間にどれだけ含まれるかを測る。

    計算式: ||P_{V_base} V_query||_F^2 / ||V_query||_F^2

    なぜ Frobenius ノルムで測るか:
      射影の Frobenius ノルムは、V_query の「エネルギー」のうち
      V_base の方向で説明できる割合を定量化する。1 = 完全包含、0 = 直交。

    前提: V_base の列は SVD により正規直交である。
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

    if linear.r2_l_to_s > 0.9 and linear.r2_s_to_l < 0.5:
        obs = "Large が Small を線形的に包含しています（R²_L→S が高く R²_S→L が低い）。"
    elif linear.r2_l_to_s > 0.8 and linear.r2_s_to_l > 0.8:
        obs = "両方向とも高い: 表現はほぼ同型です。"
    elif linear.r2_l_to_s > 0.5 and linear.r2_s_to_l > 0.5:
        obs = "双方向 R² が中程度: 共有成分はあるが完全包含ではありません。"
    else:
        obs = "R²_L→S が低い: Small には Large から線形説明できない成分があります。"
    lines += [f"- **線形**: {obs}"]

    if geometry.cka > 0.9:
        lines += ["- **CKA**: 非常に高い — 表現は幾何的に類似しています。"]
    elif geometry.cka > 0.6:
        lines += ["- **CKA**: 中程度 — 部分的な幾何的整合があります。"]
    else:
        lines += ["- **CKA**: 低い — 表現の幾何構造が異なります。"]

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
            f"{r.get('r2_l_to_s', 0.0):.4f} | {r.get('r2_s_to_l', 0.0):.4f} | "
            f"{r.get('cka', 0.0):.4f} | {r.get('rsa', 0.0):.4f} | "
            f"{r.get('knn_10', 0.0):.4f} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 図の生成
# ---------------------------------------------------------------------------


def make_figures(linear: LinearMetrics, geometry: GeometricMetrics, subspace: SubspaceMetrics, figs_dir: str) -> None:
    """分析図を生成して保存する。matplotlib が必要。"""
    import matplotlib.pyplot as plt
    out = Path(figs_dir)

    # 方向性ギャップの棒グラフ
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["R²_L→S", "R²_S→L", "Gap"]
    values = [linear.r2_l_to_s, linear.r2_s_to_l, linear.directional_gap]
    colors = ["steelblue", "salmon", "mediumseagreen"]
    ax.bar(labels, values, color=colors)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("線形包含: 方向性 R²")
    ax.set_ylabel("R²")
    fig.tight_layout()
    fig.savefig(out / "linear_directional_gap.png", dpi=150)
    plt.close(fig)

    # 部分空間包含の折れ線グラフ
    rs = sorted(subspace.containment_s_in_l)
    s_in_l = [subspace.containment_s_in_l[r] for r in rs]
    l_in_s = [subspace.containment_l_in_s[r] for r in rs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rs, s_in_l, marker="o", label="Small in Large")
    ax.plot(rs, l_in_s, marker="s", label="Large in Small")
    ax.set_xlabel("r（主成分数）")
    ax.set_ylabel("包含度")
    ax.set_title("部分空間包含度 vs. r")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "subspace_containment.png", dpi=150)
    plt.close(fig)

    # mutual kNN の折れ線グラフ
    ks = sorted(geometry.mutual_knn)
    knn_vals = [geometry.mutual_knn[k] for k in ks]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ks, knn_vals, marker="o", color="darkorchid")
    ax.set_xlabel("k")
    ax.set_ylabel("mutual kNN 重なり率")
    ax.set_title("Mutual kNN 重なり率 vs. k")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out / "mutual_knn.png", dpi=150)
    plt.close(fig)

    logger.info(f"図を保存しました: {out}")


# ---------------------------------------------------------------------------
# CLI エントリーポイント
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Representation Subsumption 分析（単一ファイルモード）"
    )
    p.add_argument("--large", required=True, help="Large モデルの特徴量ファイルパス（.pt/.npy/.npz）")
    p.add_argument("--small", required=True, help="Small モデルの特徴量ファイルパス（.pt/.npy/.npz）")
    p.add_argument("--large-model", default="large", help="Large モデルの名前ラベル")
    p.add_argument("--small-model", default="small", help="Small モデルの名前ラベル")
    p.add_argument("--output-dir", default="results", help="結果出力のルートディレクトリ")
    p.add_argument("--knn-k", nargs="+", type=int, default=[5, 10, 20], help="mutual kNN の k 値リスト")
    p.add_argument("--n-components", nargs="+", type=int, default=[16, 32, 64, 128],
                   help="部分空間分析の PCA 主成分数リスト")
    p.add_argument("--test-size", type=float, default=0.2, help="線形指標の train/test 分割比率")
    p.add_argument("--ridge", action="store_true", help="OLS の代わりに Ridge 回帰を使用する")
    p.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge の正則化強度")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--figures", action="store_true", help="図を生成する（matplotlib が必要）")
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
    logger.info(f"分析対象: n={n_samples}, d_L={d_L}, d_S={d_S}")

    linear = compute_linear_metrics(
        F_L, F_S,
        test_size=args.test_size,
        random_state=args.seed,
        use_ridge=args.ridge,
        ridge_alpha=args.ridge_alpha,
    )
    logger.info(f"線形: R²_L→S={linear.r2_l_to_s:.4f}, R²_S→L={linear.r2_s_to_l:.4f}, gap={linear.directional_gap:.4f}")

    geometry = compute_geometric_metrics(F_L, F_S, knn_ks=args.knn_k)
    logger.info(f"幾何: CKA={geometry.cka:.4f}, RSA={geometry.rsa_spearman:.4f}")

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
    logger.info(f"レポートを出力しました: {report_path}")

    if args.figures:
        try:
            make_figures(linear, geometry, subspace, figures_dir)
        except ImportError:
            logger.warning("matplotlib が見つかりません。図の生成をスキップします。")


if __name__ == "__main__":
    main()
