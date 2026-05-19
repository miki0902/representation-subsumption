"""
Representation Subsumption 分析の自己完結型単一ファイルスクリプト。

feature_io, metrics_linear, metrics_geometry, metrics_cca, report, utils
のすべてのロジックをここにインライン化しています。
requirements.txt に記載の標準依存パッケージ以外のインストール不要です。

使い方:
  python scripts/run_single_file.py --large features/large.npy --small features/small.npy
  python scripts/run_single_file.py --large features/large.pt --small features/small.pt \\
      --large-model resnet50 --small-model resnet18 \\
      --knn-k 5 10 20 --n-components 16 --r-values 4 8 16 --output-dir results
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from dataclasses import dataclass, field
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
    """ログフォーマットとレベルを設定する。"""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=level,
    )


def set_seed(seed: int) -> None:
    """Python・NumPy の乱数シードを固定して再現性を確保する。"""
    random.seed(seed)
    np.random.seed(seed)


def load_yaml(path: str) -> dict:
    """YAML ファイルを読み込んで辞書として返す。"""
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dirs(*paths: str) -> None:
    """指定されたディレクトリを存在しない場合は再帰的に作成する。"""
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
    """特徴量ファイルの読み込み結果を格納するデータクラス。"""

    features: np.ndarray       # shape: (n_samples, d)
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
    """線形回帰による包含指標を格納するデータクラス。"""

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


def fit_linear_projector(
    F_X: np.ndarray,
    F_Y: np.ndarray,
    use_ridge: bool = False,
    ridge_alpha: float = 1.0,
) -> object:
    """F_X → F_Y の線形射影器を全データで学習して返す（可視化用）。
    戻り値: fitted sklearn regressor (has .predict(X) method)
    """
    reg = Ridge(alpha=ridge_alpha) if use_ridge else LinearRegression()
    reg.fit(F_X, F_Y)
    return reg


# ---------------------------------------------------------------------------
# metrics_geometry
# ---------------------------------------------------------------------------


@dataclass
class GeometricMetrics:
    """CKA・RSA・mutual kNN などの幾何的整合指標を格納するデータクラス。"""

    cka: float
    rsa_spearman: float
    mutual_knn: dict[int, float]  # k -> 重なり率


def compute_geometric_metrics(
    F_L: np.ndarray,
    F_S: np.ndarray,
    knn_ks: list[int] = (5, 10, 20),
) -> GeometricMetrics:
    """CKA・RSA・mutual kNN をまとめて計算する。"""
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
    """Gram 行列 F @ F^T を計算する。"""
    return F @ F.T


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """中心化 Gram 行列による不偏 HSIC 推定量を計算する。

    なぜ中心化するか:
      中心化によって平均の影響を取り除き、共分散構造のみを取り出す。
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return float(np.trace(Kc @ Lc) / (n - 1) ** 2)


def compute_rsa(F_L: np.ndarray, F_S: np.ndarray) -> float:
    """表現類似性分析（RSA）: RDM の Spearman 相関を計算する。

    なぜ RSA を使うか:
      ペアワイズ距離のランク順構造を比較し、単調変換に頑健な幾何構造を捉える。
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
    """Mutual k-NN 重なり率: 両空間で共有される k 近傍の割合を計算する。

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
# metrics_cca
# ---------------------------------------------------------------------------


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
    # Projection matrices (stored for visualization)
    W_L: np.ndarray | None = field(default=None, repr=False)  # (d_L, n_components)
    W_S: np.ndarray | None = field(default=None, repr=False)  # (d_S, n_components)
    mean_L: np.ndarray | None = field(default=None, repr=False)  # (d_L,) centering mean
    mean_S: np.ndarray | None = field(default=None, repr=False)  # (d_S,) centering mean
    std_L: np.ndarray | None = field(default=None, repr=False)   # (d_L,) scaling std
    std_S: np.ndarray | None = field(default=None, repr=False)   # (d_S,) scaling std


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

    アルゴリズム概要:
      1. train/test 分割と中心化（標準化）
      2. 訓練セットで共分散行列を計算する
      3. Regularized CCA の場合は ridge 正則化を加える
      4. SVD により正準方向と訓練相関を求める
      5. テストセットに射影して正準相関を評価する
      6. meanCCA・sharedScore・normalizedSharedScore を計算する
      7. merge 空間の幾何指標（CKA・RSA・kNN）を計算する

    自動検出ルール: n_train < d_L または n_train < d_S のとき Regularized CCA を使用する。
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

    # 訓練セットで統計量を計算してテストセットにも同じ変換を適用する
    F_L_train, F_L_test, mu_L, sigma_L = _cca_standardize(F_L_train_raw, F_L_test_raw, standardize)
    F_S_train, F_S_test, mu_S, sigma_S = _cca_standardize(F_S_train_raw, F_S_test_raw, standardize)

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
        Sigma_LL_reg = Sigma_LL
        Sigma_SS_reg = Sigma_SS

    # Step 4: SVD による CCA の解法（数値的に安定）
    SLL_inv_sqrt = _matrix_sqrt_inv(Sigma_LL_reg)
    SSS_inv_sqrt = _matrix_sqrt_inv(Sigma_SS_reg)
    M = SLL_inv_sqrt @ Sigma_LS @ SSS_inv_sqrt  # shape: (d_L, d_S)

    U, s_train, Vt = np.linalg.svd(M, full_matrices=False)
    U = U[:, :n_components_eff]
    s_train = s_train[:n_components_eff]
    Vt = Vt[:n_components_eff, :]
    s_train = np.clip(s_train, -1.0, 1.0)

    # 正準方向（投影行列）を計算する
    W_L_raw = SLL_inv_sqrt @ U          # shape: (d_L, n_components_eff)
    W_S_raw = SSS_inv_sqrt @ Vt.T       # shape: (d_S, n_components_eff)

    # Step 5: テストセットへの射影
    Z_L_test = F_L_test @ W_L_raw       # shape: (n_test, n_components_eff)
    Z_S_test = F_S_test @ W_S_raw       # shape: (n_test, n_components_eff)

    # Step 6: テストセットでの正準相関を Pearson r で計算する
    s_test = np.array([
        _pearson_r_1d(Z_L_test[:, i], Z_S_test[:, i])
        for i in range(n_components_eff)
    ])
    s_test = np.clip(s_test, -1.0, 1.0)

    # Step 7: 派生指標を計算する
    mean_cca_dict = _cca_mean(s_test, r_values)
    shared_score_dict = _cca_shared(s_test, r_values)
    norm_shared_dict = _cca_norm_shared(s_test, r_values)
    train_test_gap = s_train - s_test

    # merge 空間での幾何指標を計算する
    merge_cka = compute_cka(Z_L_test, Z_S_test)
    merge_rsa = compute_rsa(Z_L_test, Z_S_test)

    n_test = Z_L_test.shape[0]
    safe_ks = [k for k in knn_ks if k < n_test]
    merge_mutual_knn_dict = {k: compute_mutual_knn(Z_L_test, Z_S_test, k) for k in safe_ks}

    return CCAMetrics(
        canonical_correlations=s_test,
        canonical_correlations_train=s_train,
        mean_cca=mean_cca_dict,
        shared_score=shared_score_dict,
        normalized_shared_score=norm_shared_dict,
        train_test_gap=train_test_gap,
        merge_cka=merge_cka,
        merge_rsa_spearman=merge_rsa,
        merge_mutual_knn=merge_mutual_knn_dict,
        used_regularized=use_regularized,
        lambda_L=lambda_L if use_regularized else 0.0,
        lambda_S=lambda_S if use_regularized else 0.0,
        n_components=n_components_eff,
        W_L=W_L_raw[:, :n_components_eff],
        W_S=W_S_raw[:, :n_components_eff],
        mean_L=mu_L,
        mean_S=mu_S,
        std_L=sigma_L if standardize else None,
        std_S=sigma_S if standardize else None,
    )


def _matrix_sqrt_inv(A: np.ndarray) -> np.ndarray:
    """対称正定値行列 A の逆平方根 A^{-1/2} を SVD で計算する。

    特異値を 1e-10 でクリップして特異に近い行列でも安定動作させる。
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_clipped = np.maximum(s, 1e-10)
    s_inv_sqrt = 1.0 / np.sqrt(s_clipped)
    return U @ np.diag(s_inv_sqrt) @ Vt


def _cca_standardize(
    F_train: np.ndarray,
    F_test: np.ndarray,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """訓練セットで統計量を計算し、訓練・テスト双方を変換する。

    standardize=True のとき各次元を平均 0・標準偏差 1 に正規化する。
    standardize=False のとき中心化のみ行う。

    戻り値: (F_train_transformed, F_test_transformed, mean, std_or_None)
    """
    mean = F_train.mean(axis=0)
    F_train_c = F_train - mean
    F_test_c = F_test - mean

    if standardize:
        std = F_train_c.std(axis=0)
        std = np.maximum(std, 1e-8)
        F_train_c = F_train_c / std
        F_test_c = F_test_c / std
        return F_train_c, F_test_c, mean, std

    return F_train_c, F_test_c, mean, None


def _pearson_r_1d(x: np.ndarray, y: np.ndarray) -> float:
    """1次元ベクトル x と y の Pearson 相関係数を計算する。

    標準偏差がほぼ 0 の縮退ケースでは 0 を返す。
    """
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = np.sqrt((x_c ** 2).sum() * (y_c ** 2).sum())
    if denom < 1e-10:
        return 0.0
    return float(np.dot(x_c, y_c) / denom)


def _cca_mean(corr: np.ndarray, r_values: list[int]) -> dict[int, float]:
    """上位 r 次元の平均正準相関 meanCCA@r を計算する。"""
    result = {}
    for r in r_values:
        r_eff = min(r, len(corr))
        result[r] = float(corr[:r_eff].mean()) if r_eff > 0 else 0.0
    return result


def _cca_shared(corr: np.ndarray, r_values: list[int]) -> dict[int, float]:
    """上位 r 次元の二乗相関和 sharedScore@r を計算する。"""
    result = {}
    for r in r_values:
        r_eff = min(r, len(corr))
        result[r] = float((corr[:r_eff] ** 2).sum()) if r_eff > 0 else 0.0
    return result


def _cca_norm_shared(corr: np.ndarray, r_values: list[int]) -> dict[int, float]:
    """上位 r 次元の平均二乗相関 normalizedSharedScore@r を計算する。"""
    result = {}
    for r in r_values:
        r_eff = min(r, len(corr))
        result[r] = float((corr[:r_eff] ** 2).mean()) if r_eff > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


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
    _fmt_float = lambda v: f"{v:.4f}" if not (v != v) else "N/A (n_test<3)"  # nan check
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

    # 7. Train/Test Gap
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
    lines += _interpretation_block_inline(linear, geometry, cca)

    # 9. 層ペア分析（オプション）
    if layer_results:
        lines += _layer_results_section(layer_results)

    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    return report


def _interpretation_block_inline(
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

    # CCA の解釈
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

    # Train/Test Gap の解釈
    mean_gap = float(cca.train_test_gap.mean())
    if mean_gap > 0.2:
        lines += [f"- **Train/Test Gap が大きい** (平均 {mean_gap:.3f}): 過学習の可能性があります。λ を大きくすることを検討してください。"]
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
            f"{r.get('r2_l_to_s', 0.0):.4f} | {r.get('r2_s_to_l', 0.0):.4f} | "
            f"{r.get('cka', 0.0):.4f} | {r.get('rsa', 0.0):.4f} | "
            f"{r.get('knn_10', 0.0):.4f} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# 図の生成
# ---------------------------------------------------------------------------


def make_figures(
    linear: LinearMetrics,
    geometry: GeometricMetrics,
    cca: CCAMetrics,
    figs_dir: str,
) -> None:
    """分析図を生成して保存する。matplotlib が必要。"""
    import matplotlib.pyplot as plt
    out = Path(figs_dir)

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
    fig.savefig(out / "linear_directional_gap.png", dpi=150)
    plt.close(fig)

    # Canonical correlation spectrum bar chart (train vs test)
    n_show = min(cca.n_components, 16)
    x = np.arange(n_show)
    width = 0.35
    rho_train = cca.canonical_correlations_train[:n_show]
    rho_test = cca.canonical_correlations[:n_show]
    fig, ax = plt.subplots(figsize=(max(6, n_show * 0.5 + 1), 4))
    ax.bar(x - width / 2, rho_train, width, label="Train", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, rho_test, width, label="Test", color="salmon", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(n_show)])
    ax.set_xlabel("Canonical Component")
    ax.set_ylabel("Canonical Correlation rho")
    ax.set_title("Canonical Correlation Spectrum (Train vs Test)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "canonical_correlations.png", dpi=150)
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
    fig.savefig(out / "shared_score.png", dpi=150)
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
    fig.savefig(out / "mutual_knn.png", dpi=150)
    plt.close(fig)

    # Train vs Test canonical correlation scatter (overfitting check)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        cca.canonical_correlations_train,
        cca.canonical_correlations,
        alpha=0.7, color="steelblue", edgecolors="black", linewidths=0.5,
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel("Train Canonical Correlation rho")
    ax.set_ylabel("Test Canonical Correlation rho")
    ax.set_title("Train vs Test Canonical Correlation")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "train_vs_test_cca.png", dpi=150)
    plt.close(fig)

    logger.info(f"図を保存しました: {out}")


# ---------------------------------------------------------------------------
# 幾何可視化 (visualize.py inline)
# ---------------------------------------------------------------------------


def _project_cca_inline(
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


def _build_reducer_inline(
    X_joint: np.ndarray,
    seed: int,
    pca_pre_dim: int = 50,
) -> tuple[np.ndarray, str]:
    """Fit a 2D reducer on X_joint and return coords + method name."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_joint)

    n_samples_j, n_features_j = X_joint.shape
    effective_pre_dim = min(pca_pre_dim, n_samples_j - 1, n_features_j - 1)
    if n_features_j > effective_pre_dim and effective_pre_dim >= 2:
        pca_pre = PCA(n_components=effective_pre_dim, random_state=seed)
        X_scaled = pca_pre.fit_transform(X_scaled)

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        coords_2d = reducer.fit_transform(X_scaled)
        method_name = "UMAP"
    except ImportError:
        logger.warning("umap-learn not installed, falling back to PCA for visualization.")
        reducer = PCA(n_components=2, random_state=seed)
        coords_2d = reducer.fit_transform(X_scaled)
        method_name = "PCA"

    return coords_2d, method_name


def _compute_geometry_aux_metrics_inline(
    F_L: np.ndarray,
    F_PL: np.ndarray,
    Z_L: np.ndarray,
    Z_S: np.ndarray,
) -> dict:
    """Compute auxiliary geometry metrics."""
    from scipy.spatial.distance import cdist

    centroid_L = F_L.mean(axis=0)
    centroid_PL = F_PL.mean(axis=0)
    centroid_dist_L_PL = float(np.linalg.norm(centroid_L - centroid_PL))

    centroid_ZL = Z_L.mean(axis=0)
    centroid_ZS = Z_S.mean(axis=0)
    centroid_dist_merge_ZL_ZS = float(np.linalg.norm(centroid_ZL - centroid_ZS))

    dists = cdist(F_L, F_PL)
    mean_nn_dist_L_PL = float(dists.min(axis=1).mean())

    try:
        from sklearn.metrics import silhouette_score
        Z_both = np.concatenate([Z_L, Z_S], axis=0)
        sil_labels = np.array([0] * len(Z_L) + [1] * len(Z_S))
        silhouette_merge = float(silhouette_score(Z_both, sil_labels))
    except Exception:
        silhouette_merge = float("nan")

    return {
        "centroid_dist_L_PL": centroid_dist_L_PL,
        "centroid_dist_merge_ZL_ZS": centroid_dist_merge_ZL_ZS,
        "mean_nn_dist_L_PL": mean_nn_dist_L_PL,
        "silhouette_merge": silhouette_merge,
    }


def _draw_scatter_inline(
    ax,
    coords_list: list,
    labels_list: list,
    colors: list,
    title: str,
    method_name: str,
    draw_connections: bool = False,
    point_labels=None,
) -> None:
    """Draw scatter plot with multiple groups."""
    markers = ['o', 's', '^', 'D', 'v']

    if draw_connections and len(coords_list) >= 2:
        c0, c1 = coords_list[0], coords_list[1]
        if len(c0) == len(c1):
            for i in range(len(c0)):
                ax.plot(
                    [c0[i, 0], c1[i, 0]],
                    [c0[i, 1], c1[i, 1]],
                    color="gray", alpha=0.2, linewidth=0.5, zorder=0,
                )

    for coords, label, color in zip(coords_list, labels_list, colors):
        if point_labels is not None and len(point_labels) == len(coords):
            unique_cls = sorted(set(point_labels.tolist()))
            for cls_i, cls in enumerate(unique_cls[:5]):
                mask = (point_labels == cls)
                marker = markers[cls_i % len(markers)]
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color, marker=marker, alpha=0.6, s=20,
                    label=f"{label} (cls {cls})" if cls_i == 0 else None,
                )
        else:
            ax.scatter(
                coords[:, 0], coords[:, 1],
                c=color, alpha=0.6, s=20, label=label,
            )

    ax.set_title(title)
    ax.set_xlabel(f"{method_name} dim 1")
    ax.set_ylabel(f"{method_name} dim 2")
    ax.legend(fontsize=8)


def visualize_geometry_inline(
    F_L: np.ndarray,
    F_S: np.ndarray,
    cca: CCAMetrics,
    reg_s_to_l,
    output_dir: str,
    max_points: int = 500,
    seed: int = 42,
    pca_pre_dim: int = 50,
    draw_connections: bool = False,
    labels: np.ndarray | None = None,
    layer_name: str = "",
) -> dict:
    """Generate 3-panel geometry visualization figure (inline version)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cca.W_L is None:
        logger.warning("cca.W_L is None — skipping geometry visualization.")
        return {}

    n = len(F_L)
    rng = np.random.default_rng(seed)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n)

    F_L_sub = F_L[idx].astype(np.float64)
    F_S_sub = F_S[idx].astype(np.float64)
    labels_sub = labels[idx] if labels is not None else None

    mean_L = cca.mean_L.astype(np.float64)
    mean_S = cca.mean_S.astype(np.float64)
    std_L = cca.std_L.astype(np.float64) if cca.std_L is not None else None
    std_S = cca.std_S.astype(np.float64) if cca.std_S is not None else None
    W_L = cca.W_L.astype(np.float64)
    W_S = cca.W_S.astype(np.float64)

    Z_L = _project_cca_inline(F_L_sub, mean_L, std_L, W_L)
    Z_S = _project_cca_inline(F_S_sub, mean_S, std_S, W_S)
    F_PL = reg_s_to_l.predict(F_S_sub).astype(np.float64)
    Z_PL = _project_cca_inline(F_PL, mean_L, std_L, W_L)

    n_L = len(Z_L)

    joint_A = np.concatenate([Z_L, Z_S], axis=0)
    coords_A, method_name = _build_reducer_inline(joint_A, seed=seed, pca_pre_dim=pca_pre_dim)
    coords_A_L = coords_A[:n_L]
    coords_A_S = coords_A[n_L:]

    joint_B = np.concatenate([F_L_sub, F_PL], axis=0)
    coords_B, _ = _build_reducer_inline(joint_B, seed=seed, pca_pre_dim=pca_pre_dim)
    coords_B_L = coords_B[:n_L]
    coords_B_PL = coords_B[n_L:]

    joint_C = np.concatenate([Z_L, Z_S, Z_PL], axis=0)
    coords_C, _ = _build_reducer_inline(joint_C, seed=seed, pca_pre_dim=pca_pre_dim)
    coords_C_L = coords_C[:n_L]
    coords_C_S = coords_C[n_L: 2 * n_L]
    coords_C_PL = coords_C[2 * n_L:]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    _draw_scatter_inline(
        axes[0],
        [coords_A_L, coords_A_S],
        ["Large", "Small"],
        ["steelblue", "salmon"],
        "(A) Pre-inclusion: CCA space",
        method_name,
        draw_connections=draw_connections,
        point_labels=labels_sub,
    )

    _draw_scatter_inline(
        axes[1],
        [coords_B_L, coords_B_PL],
        ["Large", "pseudo-Large"],
        ["steelblue", "mediumseagreen"],
        "(B) Post-inclusion: Feature space",
        method_name,
        draw_connections=draw_connections,
        point_labels=labels_sub,
    )

    _draw_scatter_inline(
        axes[2],
        [coords_C_L, coords_C_S, coords_C_PL],
        ["Large", "Small", "pseudo-Large"],
        ["steelblue", "salmon", "mediumseagreen"],
        "(C) Merge space: CCA + pseudo-Large",
        method_name,
        draw_connections=False,
        point_labels=labels_sub,
    )

    fig.tight_layout()

    import os
    geo_dir = os.path.join(output_dir, "geometry")
    os.makedirs(geo_dir, exist_ok=True)

    if layer_name:
        fig_path = os.path.join(geo_dir, f"geometry_umap_{layer_name}.png")
    else:
        fig_path = os.path.join(geo_dir, "geometry_umap.png")

    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"幾何可視化図を保存しました: {fig_path}")

    aux = _compute_geometry_aux_metrics_inline(F_L_sub, F_PL, Z_L, Z_S)

    metrics_path = os.path.join(geo_dir, "geometry_metrics.txt")
    with open(metrics_path, "w") as mf:
        for k, v in aux.items():
            mf.write(f"{k}={v}\n")

    return aux


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
    p.add_argument("--n-components", type=int, default=16, help="CCA の正準成分数")
    p.add_argument("--r-values", nargs="+", type=int, default=[4, 8, 16],
                   help="meanCCA・sharedScore を計算する r 値リスト")
    p.add_argument("--test-size", type=float, default=0.2, help="線形・CCA 指標の train/test 分割比率")
    p.add_argument("--ridge", action="store_true", help="OLS の代わりに Ridge 回帰を使用する")
    p.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge の正則化強度")
    p.add_argument("--lambda-l", type=float, default=1e-3, help="Regularized CCA の Large 側 λ")
    p.add_argument("--lambda-s", type=float, default=1e-3, help="Regularized CCA の Small 側 λ")
    p.add_argument("--regularized", action="store_true", help="Regularized CCA を強制使用する")
    p.add_argument("--no-standardize", action="store_true", help="CCA の標準化をスキップして中心化のみ行う")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--figures", action="store_true", help="図を生成する（matplotlib が必要）")
    # dry-run: パイプライン全体が動くかを少数サンプルで確認するモード
    p.add_argument("--dry-run", action="store_true",
                   help="dry-run モード: 少数サンプルで全パイプラインの動作確認を行う")
    p.add_argument("--dry-run-samples", type=int, default=5,
                   help="dry-run 時に使用するサンプル数（デフォルト: 5）")
    return p.parse_args()


def _apply_dry_run(
    F_L: np.ndarray,
    F_S: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """dry-run 用にサンプルを切り出し、パラメータを少数サンプル向けに調整する。

    なぜパラメータ調整が必要か:
      mutual kNN は k < n_test を要求し、CCA は n_components < n_train を要求する。
      5 サンプル程度では元のデフォルト値がこれらの制約を破るため、
      サンプル数から逆算して安全な値に上書きする。
    """
    n_total = args.dry_run_samples
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(F_L), size=min(n_total, len(F_L)), replace=False)
    F_L = F_L[idx]
    F_S = F_S[idx]
    n = len(F_L)

    # test_size を調整して n_test >= 2 を保証する
    # （merge 空間 kNN は k < n_test を必要とするため最低 2 サンプル必要）
    args.test_size = max(args.test_size, 2 / n)
    n_test = max(2, int(np.ceil(n * args.test_size)))
    n_train = n - n_test

    # knn_k: k < min(n_test, n) を満たす値のみ残す
    k_max = min(n_test, n) - 1
    args.knn_k = [k for k in args.knn_k if k <= k_max] or [max(1, k_max)]

    # n_components: n_train - 1 を超えられない（CCA の制約）
    args.n_components = min(args.n_components, max(1, n_train - 1))

    # r_values: n_components 以下の値のみ残す
    args.r_values = [r for r in args.r_values if r <= args.n_components] or [1]

    # 少数サンプルでは共分散行列が必ず特異になるため正則化を強制する
    args.regularized = True

    # n << d では OLS が underdetermined になり R² が発散するため Ridge を強制する
    # alpha は n_train 以上の値に引き上げて数値的安定性を確保する
    args.ridge = True
    args.ridge_alpha = max(args.ridge_alpha, float(n_train))

    logger.warning(
        f"[dry-run] n={n}, n_train={n_train}, n_test={n_test}, "
        f"knn_k={args.knn_k}, n_components={args.n_components}, "
        f"r_values={args.r_values}, Regularized CCA を強制使用"
    )
    return F_L, F_S


def _log_config(args: argparse.Namespace) -> None:
    """実行設定をログに出力する。"""
    logger.info("=" * 50)
    logger.info("[設定] モデル・データ")
    logger.info(f"  large_model  : {args.large_model}")
    logger.info(f"  small_model  : {args.small_model}")
    logger.info(f"  large_path   : {args.large}")
    logger.info(f"  small_path   : {args.small}")
    logger.info(f"  output_dir   : {args.output_dir}")
    logger.info("[設定] 線形包含")
    logger.info(f"  test_size    : {args.test_size}")
    logger.info(f"  seed         : {args.seed}")
    logger.info(f"  use_ridge    : {args.ridge}")
    logger.info(f"  ridge_alpha  : {args.ridge_alpha}")
    logger.info("[設定] 幾何的整合")
    logger.info(f"  knn_k        : {args.knn_k}")
    logger.info("[設定] CCA")
    logger.info(f"  n_components : {args.n_components}")
    logger.info(f"  r_values     : {args.r_values}")
    logger.info(f"  lambda_l     : {args.lambda_l}")
    logger.info(f"  lambda_s     : {args.lambda_s}")
    logger.info(f"  regularized  : {args.regularized}")
    logger.info(f"  standardize  : {not args.no_standardize}")
    if args.dry_run:
        logger.info(f"[設定] dry-run  : True (samples={args.dry_run_samples})")
    logger.info("=" * 50)


def main() -> None:
    args = parse_args()
    setup_logging(logging.DEBUG if args.debug else logging.INFO)
    set_seed(args.seed)

    _log_config(args)

    figures_dir = str(Path(args.output_dir) / "figures")
    tables_dir = str(Path(args.output_dir) / "tables")
    reports_dir = str(Path(args.output_dir) / "reports")
    ensure_dirs(figures_dir, tables_dir, reports_dir)

    # 特徴量の読み込みとアライメント
    large_feat = load_features(args.large, model_name=args.large_model)
    small_feat = load_features(args.small, model_name=args.small_model)

    F_L, F_S = align_features(large_feat, small_feat)

    # dry-run: サンプルを切り出してパラメータを調整する
    if args.dry_run:
        F_L, F_S = _apply_dry_run(F_L, F_S, args)

    n_samples, d_L = F_L.shape
    d_S = F_S.shape[1]
    logger.info(f"分析対象: n={n_samples}, d_L={d_L}, d_S={d_S}")

    # 線形包含指標の計算
    linear = compute_linear_metrics(
        F_L, F_S,
        test_size=args.test_size,
        random_state=args.seed,
        use_ridge=args.ridge,
        ridge_alpha=args.ridge_alpha,
    )
    logger.info(f"線形: R²_L→S={linear.r2_l_to_s:.4f}, R²_S→L={linear.r2_s_to_l:.4f}, gap={linear.directional_gap:.4f}")

    # 幾何的整合指標の計算
    geometry = compute_geometric_metrics(F_L, F_S, knn_ks=args.knn_k)
    logger.info(f"幾何: CKA={geometry.cka:.4f}, RSA={geometry.rsa_spearman:.4f}")

    # CCA / Regularized CCA 指標の計算
    use_reg = True if args.regularized else None  # None = 自動検出
    cca = compute_cca_metrics(
        F_L, F_S,
        n_components=args.n_components,
        r_values=args.r_values,
        knn_ks=args.knn_k,
        test_size=args.test_size,
        random_state=args.seed,
        use_regularized=use_reg,
        lambda_L=args.lambda_l,
        lambda_S=args.lambda_s,
        standardize=not args.no_standardize,
    )
    method = "Regularized CCA" if cca.used_regularized else "標準 CCA"
    logger.info(f"CCA ({method}): n_components={cca.n_components}")

    # レポートの生成
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
        cca=cca,
        n_samples=n_samples,
        d_L=d_L,
        d_S=d_S,
        output_path=report_path,
    )

    print(report)
    logger.info(f"レポートを出力しました: {report_path}")

    # 図の生成（matplotlib がない場合はスキップする）
    if args.figures:
        try:
            make_figures(linear, geometry, cca, figures_dir)
        except ImportError:
            logger.warning("matplotlib が見つかりません。図の生成をスキップします。")

        # 幾何可視化
        try:
            reg_s_to_l = fit_linear_projector(
                F_S, F_L,
                use_ridge=args.ridge,
                ridge_alpha=args.ridge_alpha,
            )
            aux = visualize_geometry_inline(
                F_L, F_S, cca, reg_s_to_l,
                output_dir=str(Path(args.output_dir) / "figures"),
                seed=args.seed,
            )
            logger.info(
                f"幾何可視化: centroid_dist_L_PL={aux.get('centroid_dist_L_PL', float('nan')):.4f}, "
                f"silhouette_merge={aux.get('silhouette_merge', float('nan')):.4f}"
            )
        except Exception as e:
            logger.warning(f"幾何可視化をスキップしました: {e}")


if __name__ == "__main__":
    main()
