#!/usr/bin/env bash
# =============================================================================
# VLM Representation Subsumption パイプライン 実行スクリプト
#
# 使い方:
#   bash scripts/run_vlm_pipeline.sh [STEP] [OPTIONS]
#
# STEP:
#   all            全ステップを順に実行 (デフォルト)
#   extract        ① 特徴量抽出のみ
#   analyze        ② 表現包摂分析のみ
#   tasks          ③ タスク評価のみ
#   compare        ③ タスク評価（2モデル比較）
#   dry-run        各ステップを少数サンプルで動作確認
#
# 例:
#   bash scripts/run_vlm_pipeline.sh all
#   bash scripts/run_vlm_pipeline.sh extract --n-samples 500
#   bash scripts/run_vlm_pipeline.sh dry-run
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# デフォルト設定（環境変数で上書き可能）
# ---------------------------------------------------------------------------
LARGE_MODEL="${LARGE_MODEL:-Qwen/Qwen2-VL-7B-Instruct}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"

DATASET="${DATASET:-HuggingFaceM4/NoCaps}"
DATASET_SPLIT="${DATASET_SPLIT:-validation}"
LAYER="${LAYER:-vision_encoder}"     # vision_encoder | llm_last

N_SAMPLES="${N_SAMPLES:-1000}"       # 特徴量抽出サンプル数
N_TASK_SAMPLES="${N_TASK_SAMPLES:-500}"  # タスク評価サンプル数
BATCH_SIZE="${BATCH_SIZE:-8}"
DTYPE="${DTYPE:-bfloat16}"

FEATURES_DIR="${FEATURES_DIR:-features/vlm}"
OUTPUT_DIR="${OUTPUT_DIR:-results/vlm}"
TASK_OUTPUT_DIR="${TASK_OUTPUT_DIR:-results/task_eval}"

LARGE_FEAT="${FEATURES_DIR}/large_${LAYER}.npy"
SMALL_FEAT="${FEATURES_DIR}/small_${LAYER}.npy"

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

step_banner() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

check_deps() {
    python3 -c "import vllm" 2>/dev/null        || warn "vllm が見つかりません。pip install -r requirements_vlm.txt を実行してください"
    python3 -c "import transformers" 2>/dev/null || error "transformers が見つかりません"
    python3 -c "import datasets" 2>/dev/null     || error "datasets が見つかりません"
}

# ---------------------------------------------------------------------------
# ① 特徴量抽出
# ---------------------------------------------------------------------------
step_extract() {
    local n_samples="${1:-$N_SAMPLES}"

    step_banner "① 特徴量抽出  (n=$n_samples, layer=$LAYER)"
    mkdir -p "$FEATURES_DIR"

    info "Large モデル: $LARGE_MODEL"
    python3 scripts/extract_features_vlm.py \
        --model  "$LARGE_MODEL" \
        --dataset "$DATASET" \
        --split  "$DATASET_SPLIT" \
        --n-samples "$n_samples" \
        --layer  "$LAYER" \
        --batch-size "$BATCH_SIZE" \
        --dtype  "$DTYPE" \
        --output "$LARGE_FEAT"
    success "Large 特徴量 → $LARGE_FEAT"

    info "Small モデル: $SMALL_MODEL"
    python3 scripts/extract_features_vlm.py \
        --model  "$SMALL_MODEL" \
        --dataset "$DATASET" \
        --split  "$DATASET_SPLIT" \
        --n-samples "$n_samples" \
        --layer  "$LAYER" \
        --batch-size "$BATCH_SIZE" \
        --dtype  "$DTYPE" \
        --output "$SMALL_FEAT"
    success "Small 特徴量 → $SMALL_FEAT"
}

# ---------------------------------------------------------------------------
# ② 表現包摂分析
# ---------------------------------------------------------------------------
step_analyze() {
    step_banner "② 表現包摂分析  (CKA / CCA / 線形 / mutual kNN)"

    [[ -f "$LARGE_FEAT" ]] || error "特徴量が見つかりません: $LARGE_FEAT  (先に extract を実行)"
    [[ -f "$SMALL_FEAT" ]] || error "特徴量が見つかりません: $SMALL_FEAT  (先に extract を実行)"

    python3 scripts/run_single_file.py \
        --large          "$LARGE_FEAT" \
        --small          "$SMALL_FEAT" \
        --large-model    "$(basename $LARGE_MODEL)" \
        --small-model    "$(basename $SMALL_MODEL)" \
        --output-dir     "$OUTPUT_DIR" \
        --figures \
        --ridge \
        --n-components   16 \
        --r-values       4 8 16 \
        --knn-k          5 10 20

    success "レポート → $OUTPUT_DIR/reports/summary.md"
    success "図       → $OUTPUT_DIR/figures/"
}

# ---------------------------------------------------------------------------
# ③ タスク評価（単一モデル）
# ---------------------------------------------------------------------------
step_tasks() {
    local model="${1:-$LARGE_MODEL}"
    local n="${2:-$N_TASK_SAMPLES}"

    step_banner "③ タスク評価  model=$(basename $model)  n=$n"

    python3 scripts/evaluate_tasks_vlm.py \
        --model  "$model" \
        --tasks  captioning vqa multiple_choice binary \
        --n-samples "$n" \
        --output-dir "$TASK_OUTPUT_DIR"

    success "タスク結果 → $TASK_OUTPUT_DIR/$(basename $model)/"
}

# ---------------------------------------------------------------------------
# ③' タスク評価（2モデル比較）
# ---------------------------------------------------------------------------
step_compare() {
    local n="${1:-$N_TASK_SAMPLES}"

    step_banner "③ タスク評価（Large vs Small 比較）  n=$n"

    python3 scripts/evaluate_tasks_vlm.py \
        --model-pair "$LARGE_MODEL" "$SMALL_MODEL" \
        --tasks  captioning vqa multiple_choice binary \
        --n-samples "$n" \
        --output-dir "$TASK_OUTPUT_DIR"

    success "比較レポート → $TASK_OUTPUT_DIR/comparison_summary.md"
}

# ---------------------------------------------------------------------------
# dry-run（少数サンプルで動作確認）
# ---------------------------------------------------------------------------
step_dry_run() {
    step_banner "dry-run  (n=20 で全ステップを確認)"

    info "--- 特徴量抽出 (n=20) ---"
    mkdir -p "$FEATURES_DIR"
    python3 scripts/extract_features_vlm.py \
        --model  "$LARGE_MODEL" \
        --dataset "$DATASET" \
        --split  "$DATASET_SPLIT" \
        --n-samples 20 \
        --layer  "$LAYER" \
        --batch-size 4 \
        --output "${FEATURES_DIR}/dry_large.npy"

    python3 scripts/extract_features_vlm.py \
        --model  "$SMALL_MODEL" \
        --dataset "$DATASET" \
        --split  "$DATASET_SPLIT" \
        --n-samples 20 \
        --layer  "$LAYER" \
        --batch-size 4 \
        --output "${FEATURES_DIR}/dry_small.npy"

    info "--- 表現包摂分析 (dry-run) ---"
    python3 scripts/run_single_file.py \
        --large "${FEATURES_DIR}/dry_large.npy" \
        --small "${FEATURES_DIR}/dry_small.npy" \
        --large-model "$(basename $LARGE_MODEL)" \
        --small-model "$(basename $SMALL_MODEL)" \
        --output-dir "${OUTPUT_DIR}/dry" \
        --dry-run --dry-run-samples 20 \
        --figures

    info "--- タスク評価 (n=10) ---"
    python3 scripts/evaluate_tasks_vlm.py \
        --model  "$LARGE_MODEL" \
        --tasks  captioning vqa multiple_choice binary \
        --n-samples 10 \
        --output-dir "${TASK_OUTPUT_DIR}/dry"

    success "dry-run 完了"
}

# ---------------------------------------------------------------------------
# ヘルプ表示
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
使い方: bash scripts/run_vlm_pipeline.sh [STEP] [OPTIONS]

STEP:
  all        全ステップ実行: extract → analyze → compare  (デフォルト)
  extract    ① 特徴量抽出のみ
  analyze    ② 表現包摂分析のみ
  tasks      ③ タスク評価（Large モデルのみ）
  compare    ③ タスク評価（Large vs Small 比較）
  dry-run    n=20 で全ステップの動作確認

環境変数で設定を上書き可能:
  LARGE_MODEL   (デフォルト: Qwen/Qwen2-VL-7B-Instruct)
  SMALL_MODEL   (デフォルト: Qwen/Qwen2-VL-2B-Instruct)
  DATASET       (デフォルト: nlphuji/flickr30k)
  LAYER         (デフォルト: vision_encoder)  vision_encoder | llm_last
  N_SAMPLES     (デフォルト: 1000)  特徴量抽出サンプル数
  N_TASK_SAMPLES(デフォルト: 500)   タスク評価サンプル数
  BATCH_SIZE    (デフォルト: 8)
  DTYPE         (デフォルト: bfloat16)
  FEATURES_DIR  (デフォルト: features/vlm)
  OUTPUT_DIR    (デフォルト: results/vlm)

例:
  # フルパイプライン
  bash scripts/run_vlm_pipeline.sh all

  # 特徴量を LLM 最終層から抽出
  LAYER=llm_last bash scripts/run_vlm_pipeline.sh extract

  # カスタムモデルで比較
  LARGE_MODEL=llava-hf/llava-1.5-13b-hf \\
  SMALL_MODEL=llava-hf/llava-1.5-7b-hf  \\
  bash scripts/run_vlm_pipeline.sh all

  # 動作確認
  bash scripts/run_vlm_pipeline.sh dry-run
EOF
}

# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------
STEP="${1:-all}"
shift || true   # 残引数を後続に渡す（現状未使用、拡張用）

# help は deps チェック不要
case "$STEP" in
    help|-h|--help) usage; exit 0 ;;
esac

check_deps

case "$STEP" in
    all)
        step_extract
        step_analyze
        step_compare
        echo ""
        success "全ステップ完了"
        echo -e "  表現包摂レポート : ${CYAN}$OUTPUT_DIR/reports/summary.md${NC}"
        echo -e "  タスク比較レポート: ${CYAN}$TASK_OUTPUT_DIR/comparison_summary.md${NC}"
        ;;
    extract)   step_extract ;;
    analyze)   step_analyze ;;
    tasks)     step_tasks ;;
    compare)   step_compare ;;
    dry-run)   step_dry_run ;;
    *) error "不明な STEP: $STEP  (help で使い方を確認)" ;;
esac
