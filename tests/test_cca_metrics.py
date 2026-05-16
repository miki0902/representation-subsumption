"""
CCA 指標のテスト。

境界条件:
  - 強く相関する合成データでは正準相関が高くなること
  - 独立な合成データでは正準相関が低くなること
  - Regularized CCA で特異な共分散行列でも落ちないこと
  - n_components が min(d_L, d_S) を超えても落ちないこと
  - train/test 相関差（train_test_gap）が計算されること
"""

import numpy as np
import pytest

from src.metrics_cca import (
    compute_cca_metrics,
    _compute_mean_cca,
    _compute_shared_score,
    _compute_normalized_shared_score,
    _matrix_sqrt_inv,
)


def test_correlated_data_high_cca():
    """強く相関する合成データでは上位正準相関が高くなること。

    F_S = F_L[:, :d_S] + 小さなノイズ という構造を持つデータを使用する。
    """
    rng = np.random.default_rng(0)
    n, d_L, d_S = 200, 32, 16
    F_L = rng.standard_normal((n, d_L)).astype(np.float32)
    noise = rng.standard_normal((n, d_S)).astype(np.float32) * 0.05
    F_S = F_L[:, :d_S] + noise

    result = compute_cca_metrics(
        F_L, F_S,
        n_components=8,
        r_values=[4, 8],
        knn_ks=[5],
        test_size=0.2,
        random_state=42,
        use_regularized=False,
    )

    # 上位成分の平均正準相関は高いはずである
    assert result.mean_cca[4] > 0.8, f"相関が低すぎます: mean_cca@4={result.mean_cca[4]:.4f}"


def test_independent_data_low_cca():
    """独立な合成データでは正準相関が低くなること。"""
    rng = np.random.default_rng(1)
    n, d_L, d_S = 300, 32, 16
    F_L = rng.standard_normal((n, d_L)).astype(np.float32)
    F_S = rng.standard_normal((n, d_S)).astype(np.float32)  # F_L と独立

    result = compute_cca_metrics(
        F_L, F_S,
        n_components=8,
        r_values=[4, 8],
        knn_ks=[5],
        test_size=0.2,
        random_state=42,
        use_regularized=False,
    )

    # 独立なデータでは平均相関は低くなるはずである
    assert result.mean_cca[4] < 0.5, f"相関が高すぎます: mean_cca@4={result.mean_cca[4]:.4f}"


def test_regularized_cca_singular_covariance():
    """n < d（特異な共分散行列）でも Regularized CCA がエラーなく完了すること。"""
    rng = np.random.default_rng(2)
    # n=30 < d_L=64, d_S=64 — 共分散行列が特異になるケース
    n, d_L, d_S = 30, 64, 64
    F_L = rng.standard_normal((n, d_L)).astype(np.float32)
    F_S = rng.standard_normal((n, d_S)).astype(np.float32)

    # use_regularized=True を明示指定して特異行列でも動作することを確認する
    result = compute_cca_metrics(
        F_L, F_S,
        n_components=16,
        r_values=[4, 8],
        knn_ks=[5],
        test_size=0.3,
        random_state=42,
        use_regularized=True,
        lambda_L=1e-2,
        lambda_S=1e-2,
    )

    # すべての相関値が有限値であることを確認する
    assert np.all(np.isfinite(result.canonical_correlations)), "正準相関に非有限値が含まれています"
    assert result.used_regularized is True


def test_n_components_clipped():
    """n_components が min(d_L, d_S) を超えたとき自動クリップされてエラーにならないこと。"""
    rng = np.random.default_rng(3)
    n, d_L, d_S = 200, 32, 16
    F_L = rng.standard_normal((n, d_L)).astype(np.float32)
    F_S = rng.standard_normal((n, d_S)).astype(np.float32)

    # 256 を要求するが min(d_L, d_S) = 16 でクリップされるはずである
    result = compute_cca_metrics(
        F_L, F_S,
        n_components=256,
        r_values=[4, 8, 16],
        knn_ks=[5],
        test_size=0.2,
        random_state=42,
    )

    # 実際の成分数は min(d_L, d_S) 以下に収まるはずである
    assert len(result.canonical_correlations) <= d_S, (
        f"正準相関の長さが d_S={d_S} を超えています: {len(result.canonical_correlations)}"
    )
    assert result.n_components <= d_S


def test_train_test_gap_shape():
    """train_test_gap が canonical_correlations と同じ shape を持つこと。"""
    rng = np.random.default_rng(4)
    n, d_L, d_S = 200, 32, 16
    F_L = rng.standard_normal((n, d_L)).astype(np.float32)
    F_S = rng.standard_normal((n, d_S)).astype(np.float32)

    result = compute_cca_metrics(
        F_L, F_S,
        n_components=8,
        r_values=[4, 8],
        knn_ks=[5],
        test_size=0.2,
        random_state=42,
    )

    assert result.train_test_gap.shape == result.canonical_correlations.shape, (
        f"train_test_gap の shape {result.train_test_gap.shape} が "
        f"canonical_correlations の shape {result.canonical_correlations.shape} と一致しません"
    )


def test_merge_metrics_correlated():
    """強く相関するデータでは merge_cka が高くなること。"""
    rng = np.random.default_rng(5)
    n, d_L, d_S = 200, 32, 16
    F_L = rng.standard_normal((n, d_L)).astype(np.float32)
    noise = rng.standard_normal((n, d_S)).astype(np.float32) * 0.05
    F_S = F_L[:, :d_S] + noise

    result = compute_cca_metrics(
        F_L, F_S,
        n_components=8,
        r_values=[4, 8],
        knn_ks=[5],
        test_size=0.2,
        random_state=42,
    )

    # merge 空間（Z_L_test, Z_S_test）の CKA は高いはずである
    assert result.merge_cka > 0.5, f"merge_cka が低すぎます: {result.merge_cka:.4f}"


def test_auto_detect_regularized():
    """n < d のとき use_regularized=None で自動的に Regularized CCA が選択されること。"""
    rng = np.random.default_rng(6)
    n, d_L, d_S = 20, 64, 64
    F_L = rng.standard_normal((n, d_L)).astype(np.float32)
    F_S = rng.standard_normal((n, d_S)).astype(np.float32)

    result = compute_cca_metrics(
        F_L, F_S,
        n_components=8,
        r_values=[4],
        knn_ks=[5],
        test_size=0.3,
        random_state=42,
        use_regularized=None,  # 自動検出
    )

    # n_train < d_L のため自動で Regularized CCA が選択されるはずである
    assert result.used_regularized is True


def test_matrix_sqrt_inv_identity():
    """単位行列の逆平方根は単位行列になること。"""
    A = np.eye(8)
    A_inv_sqrt = _matrix_sqrt_inv(A)
    assert np.allclose(A_inv_sqrt, np.eye(8), atol=1e-6), "単位行列の逆平方根が単位行列になっていません"


def test_derived_metrics_consistency():
    """派生指標の計算の整合性を確認する。

    - sharedScore@r = 上位 r の二乗相関の和
    - normalizedSharedScore@r = sharedScore@r / r
    - meanCCA@r = 上位 r の相関の平均
    """
    corr = np.array([0.9, 0.7, 0.5, 0.3, 0.1])

    mean_cca = _compute_mean_cca(corr, [2, 4])
    shared = _compute_shared_score(corr, [2, 4])
    norm_shared = _compute_normalized_shared_score(corr, [2, 4])

    # meanCCA@2 = (0.9 + 0.7) / 2 = 0.8
    assert abs(mean_cca[2] - 0.8) < 1e-6, f"meanCCA@2 の値が不正: {mean_cca[2]}"

    # sharedScore@2 = 0.9^2 + 0.7^2 = 0.81 + 0.49 = 1.30
    assert abs(shared[2] - (0.81 + 0.49)) < 1e-6, f"sharedScore@2 の値が不正: {shared[2]}"

    # normalizedSharedScore@2 = sharedScore@2 / 2 = 0.65
    assert abs(norm_shared[2] - 0.65) < 1e-6, f"normalizedSharedScore@2 の値が不正: {norm_shared[2]}"
