"""
Feature loading with sample alignment and validation.

Why: Features from different models must be aligned on the same samples to compare
representations. Misaligned samples would produce meaningless metrics.
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
    """Load features from .pt, .npy, or .npz file.

    Why: Research environments use varied serialization formats; unified loading
    avoids per-script format handling.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

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
            raise ValueError(f"Unsupported .pt content type: {type(data)}")

    elif suffix == ".npy":
        features = np.load(path)

    elif suffix == ".npz":
        data = np.load(path)
        if "features" in data:
            features = data["features"]
            sample_ids = data.get("sample_ids", None)
        else:
            # Use first array as features
            key = list(data.keys())[0]
            features = data[key]

    else:
        raise ValueError(f"Unsupported file extension: {suffix}. Use .pt, .npy, or .npz")

    features = np.asarray(features, dtype=np.float32)
    _validate_features(features, path)

    logger.info(f"Loaded {model_name} features: shape={features.shape}, path={path}")
    return FeatureSet(features=features, sample_ids=sample_ids, source_path=path, model_name=model_name)


def _validate_features(features: np.ndarray, source: str) -> None:
    """Check for NaN/Inf which would silently corrupt all downstream metrics."""
    if np.any(np.isnan(features)):
        raise ValueError(f"NaN detected in features from {source}")
    if np.any(np.isinf(features)):
        raise ValueError(f"Inf detected in features from {source}")
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {features.shape} from {source}")


def align_features(large: FeatureSet, small: FeatureSet) -> tuple[np.ndarray, np.ndarray]:
    """Align Large and Small features on common sample_ids.

    Why: When sample_ids differ (e.g., subsets from different extraction runs),
    we must find the intersection to ensure row-wise correspondence.
    Returns (F_L, F_S) aligned arrays.
    """
    if large.sample_ids is None or small.sample_ids is None:
        # No sample_id info: assume already aligned
        n_L, n_S = len(large.features), len(small.features)
        if n_L != n_S:
            raise ValueError(
                f"Feature count mismatch without sample_ids: large={n_L}, small={n_S}. "
                "Provide sample_ids or ensure equal sample counts."
            )
        return large.features, small.features

    common_ids = np.intersect1d(large.sample_ids, small.sample_ids)
    if len(common_ids) == 0:
        raise ValueError("No common sample_ids between large and small feature sets.")

    dropped = (len(large.sample_ids) - len(common_ids)) + (len(small.sample_ids) - len(common_ids))
    if dropped > 0:
        logger.warning(
            f"Dropped {dropped} samples due to sample_id mismatch; using {len(common_ids)} common samples."
        )

    large_idx = np.where(np.isin(large.sample_ids, common_ids))[0]
    small_idx = np.where(np.isin(small.sample_ids, common_ids))[0]

    # Sort both by common_ids to ensure correspondence
    large_order = np.argsort(large.sample_ids[large_idx])
    small_order = np.argsort(small.sample_ids[small_idx])

    F_L = large.features[large_idx[large_order]]
    F_S = small.features[small_idx[small_order]]

    logger.info(f"Aligned features: n_samples={len(common_ids)}, d_L={F_L.shape[1]}, d_S={F_S.shape[1]}")
    return F_L, F_S
