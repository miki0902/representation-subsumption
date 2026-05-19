"""
VLM タスク評価スクリプト。
Captioning / VQA / MultipleChoice / Binary の 4 タスクを実行して
results/task_eval/ に結果を保存する。

使い方:
  # 単一モデルの評価
  python scripts/evaluate_tasks_vlm.py \\
    --model Qwen/Qwen2-VL-7B-Instruct \\
    --tasks captioning vqa multiple_choice binary \\
    --n-samples 200 \\
    --output-dir results/task_eval

  # モデルペアの比較評価 (representation subsumption 用)
  python scripts/evaluate_tasks_vlm.py \\
    --model-pair Qwen/Qwen2-VL-7B-Instruct Qwen/Qwen2-VL-2B-Instruct \\
    --tasks captioning vqa \\
    --n-samples 200 \\
    --output-dir results/task_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# 対応タスク名のリスト
_SUPPORTED_TASKS = ["captioning", "vqa", "multiple_choice", "binary"]


# ---------------------------------------------------------------------------
# タスク実行
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    model_name: str,
    n_samples: int | None,
    output_dir: Path,
    dtype: str,
    device: str,
) -> dict:
    """1 タスクを実行して結果を保存し、メトリクス辞書を返す。

    Args:
        task_name:  実行するタスク名
        model_name: モデル ID
        n_samples:  評価サンプル数
        output_dir: 出力ディレクトリ (モデル名サブディレクトリを作成)
        dtype:      モデル dtype
        device:     推論デバイス

    Returns:
        タスクのメトリクス辞書
    """
    # src が import できるようにパスを通す
    repo_root = Path(__file__).parent.parent
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.task_eval import (
        CaptioningEvaluator,
        VQAEvaluator,
        MultipleChoiceEvaluator,
        BinaryEvaluator,
    )

    evaluator_map = {
        "captioning": CaptioningEvaluator,
        "vqa": VQAEvaluator,
        "multiple_choice": MultipleChoiceEvaluator,
        "binary": BinaryEvaluator,
    }

    if task_name not in evaluator_map:
        raise ValueError(
            f"非対応のタスク: {task_name}. 対応: {list(evaluator_map.keys())}"
        )

    EvaluatorClass = evaluator_map[task_name]
    evaluator = EvaluatorClass(model_name=model_name, device=device, dtype=dtype)

    logger.info(f"[{task_name}] データセットをロード中 (n_samples={n_samples})")
    samples = evaluator.load_dataset(n_samples=n_samples)

    logger.info(f"[{task_name}] 評価開始 ({len(samples)} サンプル)")
    result = evaluator.evaluate(samples)

    # 出力ディレクトリの準備
    safe_model_name = model_name.replace("/", "_")
    model_dir = output_dir / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 詳細結果を JSON で保存する
    result_dict = {
        "task_name": result.task_name,
        "model_name": result.model_name,
        "dataset_name": result.dataset_name,
        "n_samples": result.n_samples,
        "metrics": result.metrics,
        "per_sample": result.per_sample,
    }
    result_path = model_dir / f"{task_name}_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"[{task_name}] 結果を保存しました: {result_path}")

    # メトリクスをコンソールに表示する
    print(f"\n=== {task_name.upper()} ({model_name}) ===")
    for metric_name, value in result.metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    return result.metrics


# ---------------------------------------------------------------------------
# サマリー生成
# ---------------------------------------------------------------------------

def save_summary(
    model_name: str,
    task_metrics: dict[str, dict[str, float]],
    output_dir: Path,
) -> Path:
    """評価結果のサマリーを Markdown テーブルとして保存する。

    Args:
        model_name:   モデル ID
        task_metrics: タスク名 → メトリクス辞書
        output_dir:   出力ディレクトリ

    Returns:
        保存したファイルのパス
    """
    safe_model_name = model_name.replace("/", "_")
    model_dir = output_dir / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_path = model_dir / "summary.md"

    lines = [
        f"# Task Evaluation Summary",
        f"",
        f"**Model**: `{model_name}`",
        f"",
        f"## Results",
        f"",
        f"| Task | Metric | Value |",
        f"|------|--------|-------|",
    ]

    for task_name, metrics in sorted(task_metrics.items()):
        for metric_name, value in sorted(metrics.items()):
            if value is None:
                val_str = "N/A"
            else:
                val_str = f"{value:.4f}"
            lines.append(f"| {task_name} | {metric_name} | {val_str} |")

    lines.append("")
    content = "\n".join(lines)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"サマリーを保存しました: {summary_path}")
    return summary_path


def save_comparison_summary(
    model_names: list[str],
    all_metrics: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
) -> Path:
    """2 モデルの比較サマリーを Markdown テーブルとして保存する。

    Args:
        model_names:  モデル ID のリスト (2 要素)
        all_metrics:  モデル名 → タスク名 → メトリクス辞書
        output_dir:   出力ディレクトリ

    Returns:
        保存したファイルのパス
    """
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_path = comparison_dir / "comparison_summary.md"

    model_a, model_b = model_names[0], model_names[1]
    safe_a = model_a.replace("/", "_")
    safe_b = model_b.replace("/", "_")

    lines = [
        f"# Model Comparison Summary",
        f"",
        f"**Model A**: `{model_a}`",
        f"**Model B**: `{model_b}`",
        f"",
        f"## Results",
        f"",
        f"| Task | Metric | {safe_a} | {safe_b} | Delta (B - A) |",
        f"|------|--------|{'-'*len(safe_a)}|{'-'*len(safe_b)}|--------------|",
    ]

    all_tasks = sorted(
        set(list(all_metrics.get(model_a, {}).keys()) + list(all_metrics.get(model_b, {}).keys()))
    )

    for task_name in all_tasks:
        metrics_a = all_metrics.get(model_a, {}).get(task_name, {})
        metrics_b = all_metrics.get(model_b, {}).get(task_name, {})
        all_metric_names = sorted(
            set(list(metrics_a.keys()) + list(metrics_b.keys()))
        )
        for metric_name in all_metric_names:
            val_a = metrics_a.get(metric_name)
            val_b = metrics_b.get(metric_name)

            str_a = f"{val_a:.4f}" if val_a is not None else "N/A"
            str_b = f"{val_b:.4f}" if val_b is not None else "N/A"

            if val_a is not None and val_b is not None:
                delta = val_b - val_a
                str_delta = f"{delta:+.4f}"
            else:
                str_delta = "N/A"

            lines.append(
                f"| {task_name} | {metric_name} | {str_a} | {str_b} | {str_delta} |"
            )

    lines.append("")
    content = "\n".join(lines)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"比較サマリーを保存しました: {summary_path}")
    return summary_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VLM タスク評価スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # モデル指定 (単一 or ペア)
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        type=str,
        help="評価するモデルの HuggingFace ID",
    )
    model_group.add_argument(
        "--model-pair",
        type=str,
        nargs=2,
        metavar=("MODEL_A", "MODEL_B"),
        help="比較評価する 2 つのモデルの HuggingFace ID",
    )

    p.add_argument(
        "--tasks",
        nargs="+",
        default=_SUPPORTED_TASKS,
        choices=_SUPPORTED_TASKS,
        help="実行するタスク",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="各タスクで評価するサンプル数 (None = 全サンプル)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/task_eval",
        help="結果を保存するディレクトリ",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="モデル重みの dtype",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="推論デバイス (cuda / cpu / mps / auto)",
    )
    p.add_argument("--debug", action="store_true", help="デバッグログを有効化")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 評価するモデルのリストを決定する
    if args.model:
        model_names = [args.model]
    else:
        model_names = args.model_pair

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for model_name in model_names:
        logger.info(f"モデル評価開始: {model_name}")
        task_metrics: dict[str, dict[str, float]] = {}

        for task_name in args.tasks:
            logger.info(f"タスク '{task_name}' を実行中...")
            try:
                metrics = run_task(
                    task_name=task_name,
                    model_name=model_name,
                    n_samples=args.n_samples,
                    output_dir=output_dir,
                    dtype=args.dtype,
                    device=args.device,
                )
                task_metrics[task_name] = metrics
            except Exception as e:
                logger.error(f"タスク '{task_name}' の実行中にエラーが発生しました: {e}", exc_info=True)
                task_metrics[task_name] = {"error": float("nan")}

        # 単一モデルのサマリーを保存する
        summary_path = save_summary(model_name, task_metrics, output_dir)
        print(f"\nサマリーを保存しました: {summary_path}")
        all_metrics[model_name] = task_metrics

    # モデルペアの比較サマリーを保存する
    if len(model_names) == 2:
        comparison_path = save_comparison_summary(model_names, all_metrics, output_dir)
        print(f"\n比較サマリーを保存しました: {comparison_path}")


if __name__ == "__main__":
    main()
