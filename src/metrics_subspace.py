"""
部分空間包含度: Small の主部分空間が Large の主部分空間にどれだけ含まれるかを測る。

なぜ線形 R² とは別にこの指標が必要か:
  線形 R² は座標レベルの復元を測るが、部分空間包含は
  「Small にとって最重要な方向」が Large の張る空間で表現可能かを測る。
  特定の座標ではなく、方向そのものの包含関係を問う、より強い幾何的主張である。

d_L == d_S の場合:
  共通の環境空間で直接 Frobenius ノルムによる射影を計算できる（高速・厳密）。

d_L != d_S の場合（幅の異なるモデルを比較する典型ケース）:
  環境空間が異なるため直接射影できない。
  代わりに「データを介した測定」として、一方のモデルの特徴量から
  他方の上位 r 主成分スコアをどれだけ線形予測できるかを R² で測る。
  これは次元が一致する場合の幾何的測定と整合する。
"""

from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class SubspaceMetrics:
    # dict のキーは r の値
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

    if same_dim:
        # 共通環境空間での直接射影: 高速かつ厳密
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
    else:
        # 次元が異なる場合: データ射影 R² で包含度を代替測定する
        V_L_full = _top_components(F_L_c, max(n_components_list))
        V_S_full = _top_components(F_S_c, max(n_components_list))

        containment_s_in_l = {}
        containment_l_in_s = {}
        containment_gap = {}

        for r in n_components_list:
            r_eff_L = min(r, V_L_full.shape[1])
            r_eff_S = min(r, V_S_full.shape[1])
            r_eff = min(r_eff_L, r_eff_S)

            V_L = V_L_full[:, :r_eff]
            V_S = V_S_full[:, :r_eff]

            # 各モデルの PC 空間でのスコア (n_samples, r_eff)
            scores_L = F_L_c @ V_L
            scores_S = F_S_c @ V_S

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
      in-sample R² が包含度の tight な代理指標となり、
      train/test split によるノイズを避けられる。
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
      中心化後の特徴量行列への直接 SVD はより安定した主成分を与える。
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
      V_base の方向で説明できる割合を定量化する。
      1 = 完全包含、0 = 直交する部分空間。

    前提: V_query と V_base は同一環境空間（d_L == d_S）に存在する。
    前提: V_base の列は SVD により正規直交である。
    """
    # P_{V_base} V_query = V_base (V_base^T V_query)
    proj = V_base @ (V_base.T @ V_query)
    numerator = float(np.linalg.norm(proj, "fro") ** 2)
    denominator = float(np.linalg.norm(V_query, "fro") ** 2)
    if denominator == 0:
        return 0.0
    return numerator / denominator
