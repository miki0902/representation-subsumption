"""
幾何的整合指標: CKA, RSA, mutual kNN。

なぜ線形回帰に加えてこれらが必要か:
  線形回帰は座標レベルの復元を測るが、表現が回転的に等価な場合（CKA）や
  距離のランク順を保つ場合（RSA）は線形予測できなくても幾何的には整合している。
  Mutual kNN は各サンプルの近傍が両モデルで一致するかを直接測定する。
"""

from dataclasses import dataclass
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import euclidean_distances


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
      CKA は直交変換・等方スケーリングに対して不変であり、
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
      RSA はペアワイズ距離のランク順構造を比較する。
      単調変換に対して頑健であり、関係的な幾何構造を捉える。
    """
    if F_L.shape[0] < 3:
        return float("nan")
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
      kNN の重なりは、座標が異なっていても「どのサンプルが近いか」という
      局所的な近傍構造がモデル間で一致するかを直接測定する。（Huh et al., 2024）
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
