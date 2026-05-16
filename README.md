# Representation Subsumption / Feature Hierarchy Analysis

A research toolkit for quantifying whether and how the representations of a **large** neural network model subsume (contain) those of a **small** model. Implements three complementary families of metrics: linear containment, geometric alignment, and subspace containment.

---

## Purpose

When scaling neural networks, a natural question arises: does a larger model simply "know everything" the smaller one does, plus more? This is *representation subsumption*. Concretely:

- **Linear subsumption**: Can small-model features be linearly predicted from large-model features (but not vice versa)?
- **Geometric alignment**: Are the relational structures (distances, neighborhoods) consistent across models?
- **Subspace containment**: Do the principal directions of the small model's representation lie within the span of the large model's representation?

This codebase provides all three analyses with a unified pipeline, reproducible configs, and structured Markdown reports.

---

## Installation

```bash
cd representation_subsumption
pip install -r requirements.txt
```

Python 3.10+ is required (uses `X | Y` union type hints and `match/case` patterns).

---

## Usage

### Multi-file mode (recommended)

```bash
# Edit configs/experiment.yaml to point to your feature files, then:
python scripts/run_analysis.py --config configs/experiment.yaml

# With debug logging:
python scripts/run_analysis.py --config configs/experiment.yaml --debug
```

### Single-file mode (submission / no-package environments)

All logic is inlined in `scripts/run_single_file.py`. No package installation beyond `requirements.txt` is needed.

```bash
python scripts/run_single_file.py \
    --large features/large_features.npy \
    --small features/small_features.npy \
    --large-model resnet50 \
    --small-model resnet18 \
    --knn-k 5 10 20 \
    --n-components 16 32 64 128 \
    --output-dir results \
    --figures

# With Ridge regression:
python scripts/run_single_file.py \
    --large features/large.pt \
    --small features/small.pt \
    --ridge --ridge-alpha 10.0
```

---

## Directory Structure

```
representation_subsumption/
  README.md               # This file
  requirements.txt        # Python dependencies
  pyproject.toml          # pytest configuration
  configs/
    experiment.yaml       # Experiment configuration
  src/
    __init__.py
    feature_io.py         # Load/align features (.pt, .npy, .npz)
    metrics_linear.py     # Bidirectional R² / MSE
    metrics_geometry.py   # CKA, RSA, mutual kNN
    metrics_subspace.py   # Subspace containment via SVD
    report.py             # Markdown report generation
    utils.py              # Logging, seeding, YAML loading
  scripts/
    run_analysis.py       # Config-driven multi-file pipeline
    run_single_file.py    # Self-contained single-file version
  tests/
    test_feature_io.py
    test_linear_metrics.py
    test_geometry_metrics.py
    test_subspace_metrics.py
  results/
    figures/              # PNG plots
    tables/               # CSV tables (future use)
    reports/              # Markdown summary reports
```

---

## Config Reference (`configs/experiment.yaml`)

| Key | Description | Default |
|-----|-------------|---------|
| `experiment.name` | Experiment label in report | `"representation_subsumption_analysis"` |
| `experiment.large_model` | Name label for large model | `"large_model"` |
| `experiment.small_model` | Name label for small model | `"small_model"` |
| `features.large_path` | Path to large model features | required |
| `features.small_path` | Path to small model features | required |
| `features.large_layers` | List of layer-wise feature paths (large) | `[]` |
| `features.small_layers` | List of layer-wise feature paths (small) | `[]` |
| `linear.test_size` | Fraction of data held out for test | `0.2` |
| `linear.random_state` | RNG seed for train/test split | `42` |
| `linear.use_ridge` | Use Ridge instead of OLS | `false` |
| `linear.ridge_alpha` | Ridge regularization alpha | `1.0` |
| `geometry.knn_k` | List of k values for mutual kNN | `[5, 10, 20]` |
| `subspace.n_components` | List of r values for subspace analysis | `[16, 32, 64, 128]` |
| `output.results_dir` | Root results directory | `"results"` |
| `output.figures_dir` | Figures output path | `"results/figures"` |
| `output.tables_dir` | Tables output path | `"results/tables"` |
| `output.reports_dir` | Reports output path | `"results/reports"` |

---

## Feature File Format

Features can be stored as:

- **`.npy`**: Plain `(n_samples, d)` float32 array.
- **`.npz`**: Compressed archive with key `"features"` (shape `(n_samples, d)`) and optionally `"sample_ids"` (shape `(n_samples,)`).
- **`.pt`**: PyTorch file. Either a `Tensor` of shape `(n_samples, d)`, or a `dict` with keys `"features"` and optionally `"sample_ids"`.

When both feature sets include `sample_ids`, the pipeline automatically finds the intersection and aligns rows. Without `sample_ids`, sample counts must match.

---

## Running Tests

```bash
cd representation_subsumption
pytest tests/ -v
```

To run a specific test file:

```bash
pytest tests/test_linear_metrics.py -v
pytest tests/test_geometry_metrics.py -v
pytest tests/test_subspace_metrics.py -v
pytest tests/test_feature_io.py -v
```

---

## Metric Interpretations

### Linear Containment (`metrics_linear.py`)

- **R²_L→S**: How well large-model features linearly predict small-model features (on held-out test set). High value means Large contains enough information to reconstruct Small.
- **R²_S→L**: Reverse direction. High value means Small can reconstruct Large.
- **Directional Gap** = R²_L→S − R²_S→L: Positive gap indicates Large subsumes Small asymmetrically.
- **Interpretation**:
  - Gap > 0.3: Clear subsumption (Large contains Small's information, but not vice versa).
  - Both R² > 0.8: Near-isomorphism (models encode similar information).
  - Both R² < 0.5: Little linear overlap.

### Geometric Alignment (`metrics_geometry.py`)

- **CKA** (Centered Kernel Alignment, Kornblith et al. 2019): Invariant to orthogonal transforms and isotropic scaling. CKA ∈ [0, 1]; 1 = identical geometry.
- **RSA** (Representational Similarity Analysis): Spearman correlation of upper-triangle pairwise distance matrices. Captures rank-order geometry.
- **Mutual kNN@k**: Fraction of each sample's k nearest neighbors shared between both spaces, averaged over all samples. Measures local neighborhood consistency.

### Subspace Containment (`metrics_subspace.py`)

- Uses SVD to extract top-r principal directions from each model's feature matrix.
- **Containment(S in L)** = `||P_L V_S||_F² / ||V_S||_F²`: How much of Small's top-r directions project onto Large's top-r subspace.
- **Containment(L in S)**: Reverse direction.
- **Gap** = Containment(S in L) − Containment(L in S): Positive = Small's structure is more explained by Large than vice versa.
- Value of 1.0 = complete containment; 0.0 = orthogonal subspaces.

---

## References

- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *ICML*.
- Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. *ICML*.
- Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis – connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*.
