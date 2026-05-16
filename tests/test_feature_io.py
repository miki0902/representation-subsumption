import numpy as np
import pytest
import tempfile
import os
from src.feature_io import load_features, align_features, FeatureSet


def test_load_npy(tmp_path):
    F = np.random.randn(50, 32).astype(np.float32)
    p = tmp_path / "feat.npy"
    np.save(p, F)
    fs = load_features(str(p), "test")
    assert fs.features.shape == (50, 32)


def test_load_npz_with_ids(tmp_path):
    F = np.random.randn(50, 32).astype(np.float32)
    ids = np.arange(50)
    p = tmp_path / "feat.npz"
    np.savez(p, features=F, sample_ids=ids)
    fs = load_features(str(p), "test")
    assert fs.features.shape == (50, 32)
    assert fs.sample_ids is not None


def test_nan_raises():
    import tempfile, os
    F = np.random.randn(10, 4).astype(np.float32)
    F[3, 2] = float("nan")
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, F)
        fname = f.name
    try:
        with pytest.raises(ValueError, match="NaN"):
            load_features(fname, "test")
    finally:
        os.unlink(fname)


def test_inf_raises():
    import tempfile, os
    F = np.random.randn(10, 4).astype(np.float32)
    F[1, 1] = float("inf")
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, F)
        fname = f.name
    try:
        with pytest.raises(ValueError, match="Inf"):
            load_features(fname, "test")
    finally:
        os.unlink(fname)


def test_sample_id_alignment():
    F_L = np.random.randn(5, 8).astype(np.float32)
    F_S = np.random.randn(5, 4).astype(np.float32)
    # ids: large has [0,1,2,3,4], small has [1,2,3,4,5] -> common: [1,2,3,4]
    ids_L = np.array([0, 1, 2, 3, 4])
    ids_S = np.array([1, 2, 3, 4, 5])
    fs_L = FeatureSet(features=F_L, sample_ids=ids_L, source_path="", model_name="large")
    fs_S = FeatureSet(features=F_S, sample_ids=ids_S, source_path="", model_name="small")
    AL, AS = align_features(fs_L, fs_S)
    assert AL.shape[0] == 4
    assert AS.shape[0] == 4
