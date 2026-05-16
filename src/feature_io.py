"""
特徴量の読み込み・サンプルアライメント・バリデーション。

なぜこの処理が必要か:
  異なるモデルの特徴量を比較するには、同一サンプルに対応した行が揃っている必要がある。
  サンプルがずれていると、どの指標も無意味な値を返してしまう。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


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
        # sample_id 情報がない場合は既にアライメント済みと仮定する
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
