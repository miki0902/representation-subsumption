# Representation Subsumption / Feature Hierarchy 分析ツール

マルチスケールニューラルネットワークモデルにおいて、**Large モデル**の表現が **Small モデル**の表現を包含（含有）しているかどうかを定量化する研究用ツールキットです。線形包含・幾何的整合・CCA / Regularized CCA の3系統の指標を実装しています。

---

## 目的

ニューラルネットワークをスケールアップするとき、「大きなモデルは小さなモデルが知っていることをすべて知っているうえで、さらに多くを知っているのか？」という問いが自然に生じます。これが *Representation Subsumption（表現包含）* です。具体的には：

- **線形包含**: Small モデルの特徴は Large モデルの特徴から線形予測できるか（その逆は難しいか）？
- **幾何的整合**: サンプル間の関係構造（距離・近傍）はモデル間で共通しているか？
- **CCA / Regularized CCA**: Large と Small をそれぞれ線形写像で共通空間（merge 空間）へ写像したとき、両モデルで共有される潜在構造はどれだけ強いか？

なぜ部分空間包含ではなく CCA を使うか:
  d_L != d_S の場合、Small の主成分方向と Large の主成分方向は異なる環境空間に存在するため、直接的な部分空間包含は定義できません。CCA は両モデルにとってフェアな共有潜在構造を評価できます。高次元特徴（d >> n）では Regularized CCA が自動的に選択され、数値的安定性を確保します。

本コードベースはこの3分析を統一パイプラインで提供し、再現可能な設定ファイルと構造化 Markdown レポートを出力します。

---

## インストール

```bash
cd representation_subsumption
pip install -r requirements.txt
```

Python 3.10 以上が必要です（`X | Y` 型ヒントを使用）。

---

## 使い方

### マルチファイルモード（推奨）

```bash
# configs/experiment.yaml を編集して特徴量ファイルのパスを設定してから実行
python scripts/run_analysis.py --config configs/experiment.yaml

# デバッグログあり
python scripts/run_analysis.py --config configs/experiment.yaml --debug
```

### 単一ファイルモード（提出用・パッケージ不要環境向け）

すべてのロジックを `scripts/run_single_file.py` にインライン化しています。`requirements.txt` 以外のパッケージインストール不要です。

```bash
python scripts/run_single_file.py \
    --large features/large_features.npy \
    --small features/small_features.npy \
    --large-model resnet50 \
    --small-model resnet18 \
    --knn-k 5 10 20 \
    --n-components 16 \
    --r-values 4 8 16 \
    --output-dir results \
    --figures

# Ridge 回帰を使う場合
python scripts/run_single_file.py \
    --large features/large.pt \
    --small features/small.pt \
    --ridge --ridge-alpha 10.0

# Regularized CCA を強制使用する場合
python scripts/run_single_file.py \
    --large features/large.npy \
    --small features/small.npy \
    --regularized --lambda-l 1e-2 --lambda-s 1e-2
```

---

## ディレクトリ構成

```
representation_subsumption/
  README.md               # 本ファイル
  requirements.txt        # Python 依存パッケージ
  pyproject.toml          # pytest 設定
  configs/
    experiment.yaml       # 実験設定ファイル
  src/
    __init__.py
    feature_io.py         # 特徴量の読み込み・アライメント（.pt, .npy, .npz）
    metrics_linear.py     # 双方向 R² / MSE
    metrics_geometry.py   # CKA, RSA, mutual kNN
    metrics_cca.py        # CCA / Regularized CCA による merge 空間分析
    report.py             # Markdown レポート生成
    utils.py              # ロギング・シード・YAML 読み込み
  scripts/
    run_analysis.py       # 設定ファイル駆動のマルチファイルパイプライン
    run_single_file.py    # 自己完結型単一ファイル版
  tests/
    test_feature_io.py
    test_linear_metrics.py
    test_geometry_metrics.py
    test_cca_metrics.py
  results/
    figures/              # PNG 図表
    tables/               # CSV テーブル（将来利用）
    reports/              # Markdown サマリーレポート
```

---

## 設定ファイルリファレンス（`configs/experiment.yaml`）

| キー | 説明 | デフォルト |
|------|------|-----------|
| `experiment.name` | レポートに記載する実験ラベル | `"representation_subsumption_analysis"` |
| `experiment.large_model` | Large モデルの名前ラベル | `"large_model"` |
| `experiment.small_model` | Small モデルの名前ラベル | `"small_model"` |
| `features.large_path` | Large モデル特徴量ファイルのパス | 必須 |
| `features.small_path` | Small モデル特徴量ファイルのパス | 必須 |
| `features.large_layers` | Large モデルの層別特徴量パスのリスト | `[]` |
| `features.small_layers` | Small モデルの層別特徴量パスのリスト | `[]` |
| `linear.test_size` | テスト用に保留するデータの割合 | `0.2` |
| `linear.random_state` | train/test split の乱数シード | `42` |
| `linear.use_ridge` | OLS の代わりに Ridge 回帰を使用する | `false` |
| `linear.ridge_alpha` | Ridge の正則化強度 α | `1.0` |
| `geometry.knn_k` | mutual kNN の k 値のリスト | `[5, 10, 20]` |
| `cca.n_components` | 正準成分数 | `16` |
| `cca.r_values` | meanCCA・sharedScore を計算する r 値リスト | `[4, 8, 16]` |
| `cca.knn_k` | merge 空間 mutual kNN の k 値リスト | `[5, 10, 20]` |
| `cca.test_size` | CCA の train/test 分割比率 | `0.2` |
| `cca.random_state` | CCA の train/test split の乱数シード | `42` |
| `cca.use_regularized` | `null` = 自動検出（n < d のとき Regularized CCA を使用） | `null` |
| `cca.lambda_L` | Regularized CCA の Large 側 ridge 係数 | `1.0e-3` |
| `cca.lambda_S` | Regularized CCA の Small 側 ridge 係数 | `1.0e-3` |
| `cca.standardize` | CCA 前に各次元を標準化するか | `true` |
| `output.results_dir` | 結果出力のルートディレクトリ | `"results"` |
| `output.figures_dir` | 図表の出力パス | `"results/figures"` |
| `output.tables_dir` | テーブルの出力パス | `"results/tables"` |
| `output.reports_dir` | レポートの出力パス | `"results/reports"` |

---

## 特徴量ファイルフォーマット

以下の形式に対応しています：

- **`.npy`**: `(n_samples, d)` の float32 配列をそのまま保存したもの。
- **`.npz`**: キー `"features"`（shape `(n_samples, d)`）と、オプションで `"sample_ids"`（shape `(n_samples,)`）を含む圧縮アーカイブ。
- **`.pt`**: PyTorch ファイル。`(n_samples, d)` の `Tensor`、またはキー `"features"` とオプションの `"sample_ids"` を持つ `dict`。

両方の特徴量セットに `sample_ids` が含まれている場合、パイプラインは自動的に共通部分を見つけて行を対応付けます。`sample_ids` がない場合はサンプル数が一致している必要があります。

---

## テスト実行

```bash
cd representation_subsumption
pytest tests/ -v
```

特定のテストファイルだけ実行する場合：

```bash
pytest tests/test_linear_metrics.py -v
pytest tests/test_geometry_metrics.py -v
pytest tests/test_cca_metrics.py -v
pytest tests/test_feature_io.py -v
```

---

## 指標の解釈

### 線形包含（`metrics_linear.py`）

- **R²_L→S**: Large モデルの特徴から Small モデルの特徴をどれだけ線形予測できるか（ホールドアウトテストセット上で評価）。高い値は Large が Small を再構成するのに十分な情報を持つことを意味します。
- **R²_S→L**: 逆方向。Small が Large を再構成できるか。
- **Directional Gap** = R²_L→S − R²_S→L: 正の Gap は Large が Small を非対称に包含していることを示します。
- **解釈の目安**：
  - Gap > 0.3: 明確な包含（Large は Small の情報を持つが逆は成り立たない）
  - 両方の R² > 0.8: ほぼ同型（両モデルが似た情報を符号化している）
  - 両方の R² < 0.5: 線形的な重複がほとんどない

### 幾何的整合（`metrics_geometry.py`）

- **CKA**（Centered Kernel Alignment, Kornblith et al. 2019）: 直交変換・等方スケーリングに対して不変。CKA ∈ [0, 1]、1 = 完全一致。
- **RSA**（Representational Similarity Analysis）: ペアワイズ距離行列の上三角部分の Spearman 相関。ランク順の幾何構造を捉えます。
- **Mutual kNN@k**: 各サンプルの k 近傍がモデル間で共有される割合の平均。局所的な近傍構造の一致を測定します。

### CCA / Regularized CCA（`metrics_cca.py`）

- **正準相関 ρ_i**: i 番目の正準成分ペア（Z_L の i 列、Z_S の i 列）の Pearson 相関。高いほど共有構造が強い。
- **meanCCA@r** = 上位 r 次元の平均正準相関: 共有構造の「密度」を示す代表値。
- **sharedScore@r** = Σ_{i=1}^{r} ρ_i²: 上位 r 成分で共有される分散の総量。
- **normalizedSharedScore@r** = sharedScore@r / r: [0, 1] に正規化。1 = 上位 r 成分すべてで完全相関。
- **Train/Test Gap**: 訓練相関 - テスト相関。大きな Gap は過学習を示し、λ を大きくすることで改善できます。
- **Merge 空間 CKA/RSA/kNN**: CCA 写像後の Z_L と Z_S の幾何的類似度。

#### 自動 Regularized CCA の選択

`use_regularized: null`（デフォルト）のとき、n_train < d_L または n_train < d_S であれば自動的に Regularized CCA が使用されます。この場合、共分散行列に λI が加算されて数値的安定性が確保されます。

---

## 参考文献

- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *ICML*.
- Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. *ICML*.
- Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis – connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*.
- Hotelling, H. (1936). Relations Between Two Sets of Variates. *Biometrika*.
