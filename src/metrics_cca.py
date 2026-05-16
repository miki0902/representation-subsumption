"""
CCA / Regularized CCA による共通 merge 空間分析。

なぜ CCA を使うか:
  d_L != d_S の場合、Small の主成分方向 V_S と Large の主成分方向 V_L は
  異なる環境空間に存在するため、直接的な部分空間包含は定義できない。
  CCA は Large と Small をそれぞれ別の線形写像で共通空間へ写像し、
  両モデルにとってフェアな共有潜在構造を評価できる。

なぜ Regularized CCA も実装するか:
  高次元特徴（d >> n）では共分散行列が特異になり標準 CCA が数値的に不安定になる。
  共分散行列に ridge 正則化を加えることで安定化する。
  n < d_L または n < d_S の場合はデフォルトで Regularized CCA を推奨する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .metrics_geometry import compute_cka, compute_rsa, compute_mutual_knn


@dataclass
class CCAMetrics:
    """CCA 分析の結果を格納するデータクラス。"""

    # テストセットでの正準相関。shape: (n_components,)
    canonical_correlations: np.ndarray
    # 訓練セットでの正準相関（train/test ギャップ計算用）。shape: (n_components,)
    canonical_correlations_train: np.ndarray
    # r -> meanCCA@r = 上位 r 次元の平均相関
    mean_cca: dict[int, float]
    # r -> 上位 r 次元の二乗相関和
    shared_score: dict[int, float]
    # r -> 上位 r 次元の平均二乗相関
    normalized_shared_score: dict[int, float]
    # 訓練相関 - テスト相関。shape: (n_components,)
    train_test_gap: np.ndarray
    # merge 空間での CKA(Z_L_test, Z_S_test)
    merge_cka: float
    # merge 空間での RSA(Z_L_test, Z_S_test)
    merge_rsa_spearman: float
    # k -> merge 空間での mutual kNN 重なり率
    merge_mutual_knn: dict[int, float]
    # Regularized CCA を使用したかどうか
    used_regularized: bool
    # Large 側の ridge 正則化係数
    lambda_L: float
    # Small 側の ridge 正則化係数
    lambda_S: float
    # 実際に使用した正準成分数
    n_components: int


def compute_cca_metrics(
    F_L: np.ndarray,
    F_S: np.ndarray,
    n_components: int = 16,
    r_values: list[int] = (4, 8, 16),
    knn_ks: list[int] = (5, 10, 20),
    test_size: float = 0.2,
    random_state: int = 42,
    use_regularized: bool | None = None,   # None = 自動検出
    lambda_L: float = 1e-3,
    lambda_S: float = 1e-3,
    standardize: bool = True,
) -> CCAMetrics:
    """CCA / Regularized CCA を用いて共通 merge 空間の分析指標を計算する。

    引数:
        F_L: Large モデルの特徴量行列。shape: (n_samples, d_L)
        F_S: Small モデルの特徴量行列。shape: (n_samples, d_S)
        n_components: 正準成分数。min(d_L, d_S, n_train - 1) でクリップされる。
        r_values: meanCCA・sharedScore 計算に使う r 値のリスト。
        knn_ks: merge 空間 mutual kNN の k 値リスト。
        test_size: テストセットの割合。
        random_state: train/test split の乱数シード。
        use_regularized: None のとき n_train < d_L または n_train < d_S であれば自動で Regularized CCA を使用する。
        lambda_L: Regularized CCA の Large 側 ridge 係数。
        lambda_S: Regularized CCA の Small 側 ridge 係数。
        standardize: True のとき中心化と標準化（各次元の標準偏差で割る）を行う。

    戻り値:
        CCAMetrics インスタンス。
    """
    # Step 1: train/test 分割
    idx = np.arange(F_L.shape[0])
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state)

    F_L_train_raw, F_L_test_raw = F_L[idx_train], F_L[idx_test]
    F_S_train_raw, F_S_test_raw = F_S[idx_train], F_S[idx_test]

    n_train = F_L_train_raw.shape[0]
    d_L = F_L.shape[1]
    d_S = F_S.shape[1]

    # 自動検出: n_train < d_L または n_train < d_S のとき Regularized CCA を使用する
    if use_regularized is None:
        use_regularized = (n_train < d_L) or (n_train < d_S)

    # n_components の上限クリップ
    n_components_eff = min(n_components, d_L, d_S, n_train - 1)
    if n_components_eff < 1:
        n_components_eff = 1

    # 中心化（と標準化）: 訓練セットで統計量を計算し、テストセットにも同じ変換を適用する
    F_L_train, F_L_test = _standardize(F_L_train_raw, F_L_test_raw, standardize)
    F_S_train, F_S_test = _standardize(F_S_train_raw, F_S_test_raw, standardize)

    # Step 2: 訓練セットで共分散行列を計算する
    denom = n_train - 1
    Sigma_LL = (F_L_train.T @ F_L_train) / denom
    Sigma_SS = (F_S_train.T @ F_S_train) / denom
    Sigma_LS = (F_L_train.T @ F_S_train) / denom

    # Step 3: Regularized CCA の場合は ridge を加算する
    if use_regularized:
        Sigma_LL_reg = Sigma_LL + lambda_L * np.eye(d_L)
        Sigma_SS_reg = Sigma_SS + lambda_S * np.eye(d_S)
    else:
        # 正則化なし（= lambda = 0 相当）
        Sigma_LL_reg = Sigma_LL
        Sigma_SS_reg = Sigma_SS

    # Step 4: SVD による CCA の解法（数値的に安定）
    # M = Sigma_LL_reg^{-1/2} @ Sigma_LS @ Sigma_SS_reg^{-1/2}
    SLL_inv_sqrt = _matrix_sqrt_inv(Sigma_LL_reg)
    SSS_inv_sqrt = _matrix_sqrt_inv(Sigma_SS_reg)

    M = SLL_inv_sqrt @ Sigma_LS @ SSS_inv_sqrt  # shape: (d_L, d_S)

    # SVD: 特異値が正準相関（訓練セット）に相当する
    U, s_train, Vt = np.linalg.svd(M, full_matrices=False)

    # 上位 n_components_eff 成分のみ使用する
    U = U[:, :n_components_eff]
    s_train = s_train[:n_components_eff]
    Vt = Vt[:n_components_eff, :]

    # 正準相関は [-1, 1] に収まるはずだが浮動小数点誤差でわずかに外れる場合があるのでクリップする
    s_train = np.clip(s_train, -1.0, 1.0)

    # 正準方向（射影行列）
    # W_L: F_L を正準空間に写像する行列。shape: (d_L, n_components_eff)
    W_L = SLL_inv_sqrt @ U
    # W_S: F_S を正準空間に写像する行列。shape: (d_S, n_components_eff)
    W_S = SSS_inv_sqrt @ Vt.T

    # Step 5: テストセットへの射影
    Z_L_test = F_L_test @ W_L  # shape: (n_test, n_components_eff)
    Z_S_test = F_S_test @ W_S  # shape: (n_test, n_components_eff)

    # Step 6: テストセットでの正準相関を各次元ごとに Pearson r で計算する
    s_test = np.array([
        _pearson_r(Z_L_test[:, i], Z_S_test[:, i])
        for i in range(n_components_eff)
    ])
    # 浮動小数点誤差をクリップする
    s_test = np.clip(s_test, -1.0, 1.0)

    # Step 7: 派生指標を計算する
    mean_cca = _compute_mean_cca(s_test, r_values)
    shared_score = _compute_shared_score(s_test, r_values)
    normalized_shared_score = _compute_normalized_shared_score(s_test, r_values)
    train_test_gap = s_train - s_test

    # merge 空間での幾何指標を計算する
    merge_cka = compute_cka(Z_L_test, Z_S_test)
    merge_rsa_spearman = compute_rsa(Z_L_test, Z_S_test)

    # kNN は n_test > k を要求するため k をクリップして安全に計算する
    n_test = Z_L_test.shape[0]
    safe_ks = [k for k in knn_ks if k < n_test]
    merge_mutual_knn = {k: compute_mutual_knn(Z_L_test, Z_S_test, k) for k in safe_ks}

    return CCAMetrics(
        canonical_correlations=s_test,
        canonical_correlations_train=s_train,
        mean_cca=mean_cca,
        shared_score=shared_score,
        normalized_shared_score=normalized_shared_score,
        train_test_gap=train_test_gap,
        merge_cka=merge_cka,
        merge_rsa_spearman=merge_rsa_spearman,
        merge_mutual_knn=merge_mutual_knn,
        used_regularized=use_regularized,
        lambda_L=lambda_L if use_regularized else 0.0,
        lambda_S=lambda_S if use_regularized else 0.0,
        n_components=n_components_eff,
    )


# ---------------------------------------------------------------------------
# 行列演算ユーティリティ
# ---------------------------------------------------------------------------


def _matrix_sqrt_inv(A: np.ndarray) -> np.ndarray:
    """対称正定値行列 A の逆平方根 A^{-1/2} を SVD で計算する。

    なぜ SVD を使うか:
      固有値分解より SVD の方が数値的に安定で近似的に対称な行列にも対応できる。
      固有値（特異値）を 1e-10 でクリップして特異に近い行列でも安定動作させる。

    戻り値: shape (d, d) の行列 A^{-1/2}。
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # 特異値が小さすぎる場合は数値的安定性のためクリップする
    s_clipped = np.maximum(s, 1e-10)
    s_inv_sqrt = 1.0 / np.sqrt(s_clipped)
    return U @ np.diag(s_inv_sqrt) @ Vt


def _standardize(
    F_train: np.ndarray,
    F_test: np.ndarray,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """訓練セットで統計量を計算し、訓練・テスト双方を変換する。

    standardize=True のとき各次元を平均 0・標準偏差 1 に正規化する。
    standardize=False のとき中心化のみ行う（標準偏差 1 に正規化しない）。
    """
    mean = F_train.mean(axis=0)
    F_train_c = F_train - mean
    F_test_c = F_test - mean

    if standardize:
        std = F_train_c.std(axis=0)
        # 標準偏差がほぼ 0 の次元では割り算が数値的に不安定になるためクリップする
        std = np.maximum(std, 1e-8)
        F_train_c = F_train_c / std
        F_test_c = F_test_c / std

    return F_train_c, F_test_c


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """1次元ベクトル x と y の Pearson 相関係数を計算する。

    標準偏差がほぼ 0 の縮退ケースでは 0 を返す。
    """
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = np.sqrt((x_c ** 2).sum() * (y_c ** 2).sum())
    if denom < 1e-10:
        return 0.0
    return float(np.dot(x_c, y_c) / denom)


# ---------------------------------------------------------------------------
# 派生指標の計算
# ---------------------------------------------------------------------------


def _compute_mean_cca(correlations: np.ndarray, r_values: list[int]) -> dict[int, float]:
    """上位 r 次元の平均正準相関 meanCCA@r を計算する。

    なぜ平均を使うか:
      先頭次元が最も高い相関を持つため、上位 r 次元の平均は
      モデル間で共有される構造の「密度」を示す代表値になる。
    """
    result = {}
    for r in r_values:
        r_eff = min(r, len(correlations))
        result[r] = float(correlations[:r_eff].mean()) if r_eff > 0 else 0.0
    return result


def _compute_shared_score(correlations: np.ndarray, r_values: list[int]) -> dict[int, float]:
    """上位 r 次元の二乗相関和 sharedScore@r を計算する。

    解釈: 上位 r 正準成分でどれだけ共有される分散があるかの総量を示す。
    """
    result = {}
    for r in r_values:
        r_eff = min(r, len(correlations))
        result[r] = float((correlations[:r_eff] ** 2).sum()) if r_eff > 0 else 0.0
    return result


def _compute_normalized_shared_score(correlations: np.ndarray, r_values: list[int]) -> dict[int, float]:
    """上位 r 次元の平均二乗相関 normalizedSharedScore@r を計算する。

    解釈: [0, 1] に正規化されており、1 = 上位 r 成分すべてで完全相関。
    """
    result = {}
    for r in r_values:
        r_eff = min(r, len(correlations))
        result[r] = float((correlations[:r_eff] ** 2).mean()) if r_eff > 0 else 0.0
    return result
