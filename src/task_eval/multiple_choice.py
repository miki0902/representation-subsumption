"""Multiple choice evaluator — ScienceQA dataset.

評価内容:
  - データセット: derek-thomas/ScienceQA (test split)
  - 各サンプルに対して質問と選択肢を提示し、対応する選択肢の文字 (A, B, C, ...) を回答させる
  - 画像なしのサンプルはフィルタリングして除外する

メトリクス:
  accuracy: 正解した割合 (選択肢の文字が正確に一致するか)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .base import BaseTaskEvaluator, TaskResult

logger = logging.getLogger(__name__)

# デフォルトのサンプル数
_DEFAULT_N_SAMPLES = 500

# データセット設定
_DATASET_NAME = "derek-thomas/ScienceQA"
_DATASET_SPLIT = "test"

# 選択肢に対応する文字列 (最大 26 選択肢を想定)
_CHOICE_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _format_choices(choices: list[str]) -> str:
    """選択肢をプロンプト用の文字列にフォーマットする。

    例:
      ["cat", "dog", "bird"] → "A) cat\nB) dog\nC) bird"
    """
    lines = []
    for i, choice in enumerate(choices):
        if i < len(_CHOICE_LABELS):
            lines.append(f"{_CHOICE_LABELS[i]}) {choice}")
    return "\n".join(lines)


def _build_prompt(question: str, choices: list[str]) -> str:
    """質問と選択肢からプロンプトを構築する。"""
    formatted_choices = _format_choices(choices)
    return (
        f"<image>\n{question}\n"
        f"{formatted_choices}\n"
        "Answer with the letter only (A, B, C, ...)."
    )


def _parse_answer(output: str) -> str | None:
    """モデルの出力から最初の大文字 (A–Z) を抽出する。

    Args:
        output: モデルが生成したテキスト

    Returns:
        大文字 1 文字または None (抽出できない場合)
    """
    # 先頭の大文字を探す
    match = re.search(r"\b([A-Z])\b", output.strip())
    if match:
        return match.group(1)
    # \\b が効かない場合のフォールバック: 最初の大文字
    for char in output.strip():
        if char.isalpha() and char.isupper():
            return char
    return None


class MultipleChoiceEvaluator(BaseTaskEvaluator):
    """ScienceQA を用いた多肢選択評価器。

    使い方:
        evaluator = MultipleChoiceEvaluator("Qwen/Qwen2-VL-7B-Instruct")
        samples = evaluator.load_dataset(n_samples=500)
        result = evaluator.evaluate(samples)
        print(result.metrics)  # {"accuracy": 0.72}
    """

    task_name = "multiple_choice"

    def load_dataset(
        self,
        n_samples: int | None = _DEFAULT_N_SAMPLES,
        split: str = _DATASET_SPLIT,
    ) -> list[dict]:
        """ScienceQA データセットを読み込む (画像ありサンプルのみ)。

        各サンプル辞書:
          - "image":    PIL.Image.Image
          - "question": str
          - "choices":  list[str]
          - "answer":   int  (正解の選択肢インデックス, 0-based)
          - "idx":      int  (データセット内のインデックス)

        Args:
            n_samples: 使用するサンプル数 (None = 全サンプル、画像ありフィルタ後の数)
            split:     データセット分割

        Returns:
            サンプル辞書のリスト
        """
        from datasets import load_dataset

        logger.info(f"データセットをロード中: {_DATASET_NAME} (split={split})")
        ds = load_dataset(_DATASET_NAME, split=split, trust_remote_code=True)
        logger.info(f"全サンプル数: {len(ds)}")

        # 画像ありサンプルのみフィルタリングする
        samples = []
        for i in range(len(ds)):
            row = ds[i]
            img = row.get("image")
            if img is None:
                continue

            choices = row.get("choices", [])
            if not choices:
                continue

            answer_idx = row.get("answer")
            if answer_idx is None:
                continue

            samples.append({
                "image": img.convert("RGB") if hasattr(img, "convert") else img,
                "question": str(row.get("question", "")),
                "choices": [str(c) for c in choices],
                "answer": int(answer_idx),
                "idx": i,
            })

        logger.info(f"画像ありサンプル数: {len(samples)}")

        if n_samples is not None:
            samples = samples[:n_samples]
            logger.info(f"サブサンプリング後: {len(samples)} サンプル")

        return samples

    def evaluate(
        self,
        samples: list[dict],
        max_new_tokens: int = 16,
    ) -> TaskResult:
        """ScienceQA サンプルに対して回答を生成し精度を計算する。

        Args:
            samples:        load_dataset が返したサンプルのリスト
            max_new_tokens: 生成する最大トークン数 (1 文字回答を想定するため小さく設定)

        Returns:
            TaskResult (metrics には "accuracy" が含まれる)
        """
        logger.info(f"多肢選択評価開始: {len(samples)} サンプル, モデル={self.model_name}")

        images = [s["image"] for s in samples]
        prompts = [
            _build_prompt(s["question"], s["choices"])
            for s in samples
        ]

        # vLLM で推論
        generated = self._generate(prompts, images, max_new_tokens=max_new_tokens)

        # メトリクス計算
        correct = 0
        per_sample = []
        for s, gen in zip(samples, generated):
            predicted_letter = _parse_answer(gen)
            answer_letter = _CHOICE_LABELS[s["answer"]] if s["answer"] < len(_CHOICE_LABELS) else None
            is_correct = (predicted_letter == answer_letter) if answer_letter is not None else False
            if is_correct:
                correct += 1

            per_sample.append({
                "idx": s["idx"],
                "question": s["question"],
                "choices": s["choices"],
                "answer_letter": answer_letter,
                "generated": gen,
                "predicted_letter": predicted_letter,
                "correct": is_correct,
            })

        accuracy = correct / len(samples) if samples else 0.0
        logger.info(f"多肢選択精度: {accuracy:.4f} ({correct}/{len(samples)})")

        return TaskResult(
            task_name=self.task_name,
            model_name=self.model_name,
            dataset_name=_DATASET_NAME,
            n_samples=len(samples),
            metrics={"accuracy": accuracy},
            per_sample=per_sample,
        )
