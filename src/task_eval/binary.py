"""Binary (Yes/No) evaluator — POPE dataset.

評価内容:
  - データセット: lmms-lab/POPE (test split)
    フォールバック: POPE が利用できない場合は VQA v2 のバイナリサブセットを使用する
  - Yes/No 二値回答を評価し、精度・F1・適合率・再現率を計算する

メトリクス:
  accuracy:  正解率
  f1:        F1 スコア (二値分類)
  precision: 適合率 (Yes と予測したうち実際に Yes の割合)
  recall:    再現率 (実際に Yes のうち Yes と予測した割合)

POPE について:
  POPE (Polling-based Object Probing Evaluation) は VLM の幻覚
  (hallucination) を評価するためのデータセット。
  「この画像に [オブジェクト] がありますか？」という形式の質問に対して
  Yes/No で回答させる。
"""

from __future__ import annotations

import logging
import re
import string
from typing import Any

from .base import BaseTaskEvaluator, TaskResult

logger = logging.getLogger(__name__)

# デフォルトのサンプル数
_DEFAULT_N_SAMPLES = 500

# データセット設定
_POPE_DATASET_NAME = "lmms-lab/POPE"
_VQA_DATASET_NAME = "HuggingFaceM4/VQAv2"
_DATASET_SPLIT = "test"

# バイナリサブセット抽出に使う質問の先頭語
_BINARY_QUESTION_PREFIXES = ("is ", "are ", "does ", "do ", "has ", "have ", "can ", "did ")


def _normalize_binary(text: str) -> str:
    """回答テキストを正規化して "yes" または "no" に変換する。

    "yes" / "no" 以外の場合は先頭の単語を確認し、それでも判定できない場合は "unknown" を返す。
    """
    text = text.lower().strip().rstrip(string.punctuation)
    first_word = text.split()[0] if text.split() else ""
    if first_word in {"yes", "yeah", "yep", "correct", "true", "right"}:
        return "yes"
    if first_word in {"no", "nope", "not", "false", "incorrect", "wrong"}:
        return "no"
    # テキスト全体に yes / no が含まれるか確認する
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return "unknown"


def _compute_binary_metrics(
    preds: list[str],
    labels: list[str],
) -> dict[str, float]:
    """二値分類メトリクスを計算する。

    "yes" を正例 (positive) として扱う。

    Args:
        preds:  予測ラベルのリスト ("yes" / "no" / "unknown")
        labels: 正解ラベルのリスト ("yes" / "no")

    Returns:
        {"accuracy": float, "f1": float, "precision": float, "recall": float}
    """
    tp = fp = tn = fn = 0
    for pred, label in zip(preds, labels):
        pred_yes = pred == "yes"
        label_yes = label.lower().strip() == "yes"
        if pred_yes and label_yes:
            tp += 1
        elif pred_yes and not label_yes:
            fp += 1
        elif not pred_yes and label_yes:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


class BinaryEvaluator(BaseTaskEvaluator):
    """POPE を用いた二値 (Yes/No) 視覚的質問応答評価器。

    POPE が利用できない場合は VQA v2 のバイナリサブセット
    (Is/Are/Does/Do/Has/Have/Can/Did で始まる質問) を使用する。

    使い方:
        evaluator = BinaryEvaluator("Qwen/Qwen2-VL-7B-Instruct")
        samples = evaluator.load_dataset(n_samples=500)
        result = evaluator.evaluate(samples)
        print(result.metrics)  # {"accuracy": 0.85, "f1": 0.84, ...}
    """

    task_name = "binary"

    def load_dataset(
        self,
        n_samples: int | None = _DEFAULT_N_SAMPLES,
        split: str = _DATASET_SPLIT,
    ) -> list[dict]:
        """POPE データセットを読み込む。利用できない場合は VQA v2 バイナリサブセットを使用する。

        各サンプル辞書:
          - "image":    PIL.Image.Image
          - "question": str
          - "answer":   str  ("yes" または "no")
          - "idx":      int

        Args:
            n_samples: 使用するサンプル数 (None = 全サンプル)
            split:     データセット分割

        Returns:
            サンプル辞書のリスト
        """
        samples = self._load_pope(n_samples, split)
        if samples is None:
            logger.warning(
                f"POPE データセット ({_POPE_DATASET_NAME}) のロードに失敗しました。"
                " VQA v2 のバイナリサブセットを使用します。"
            )
            samples = self._load_vqa_binary(n_samples)
        return samples

    def _load_pope(self, n_samples: int | None, split: str) -> list[dict] | None:
        """POPE データセットをロードする。失敗した場合は None を返す。"""
        try:
            from datasets import load_dataset

            logger.info(f"POPE データセットをロード中: {_POPE_DATASET_NAME} (split={split})")
            ds = load_dataset(_POPE_DATASET_NAME, split=split, trust_remote_code=True)

            if n_samples is not None:
                n_samples = min(n_samples, len(ds))
                ds = ds.select(range(n_samples))

            samples = []
            for i in range(len(ds)):
                row = ds[i]
                img = row.get("image")
                if img is None:
                    continue
                answer = str(row.get("answer", row.get("label", ""))).lower().strip()
                if answer not in {"yes", "no"}:
                    answer = "yes" if "yes" in answer else "no"

                samples.append({
                    "image": img.convert("RGB") if hasattr(img, "convert") else img,
                    "question": str(row.get("question", "")),
                    "answer": answer,
                    "idx": i,
                })

            logger.info(f"POPE ロード完了: {len(samples)} サンプル")
            return samples if samples else None

        except Exception as e:
            logger.warning(f"POPE のロードに失敗: {e}")
            return None

    def _load_vqa_binary(self, n_samples: int | None) -> list[dict]:
        """VQA v2 からバイナリ質問 (Is/Are/... で始まる) のサブセットを抽出する。"""
        from datasets import load_dataset

        logger.info(f"VQA v2 バイナリサブセットをロード中: {_VQA_DATASET_NAME}")
        # 十分な数を取得するために多めにロードする
        fetch_n = (n_samples * 10) if n_samples is not None else None
        ds = load_dataset(_VQA_DATASET_NAME, split="validation", trust_remote_code=True)

        if fetch_n is not None:
            ds = ds.select(range(min(fetch_n, len(ds))))

        samples = []
        for i in range(len(ds)):
            row = ds[i]
            question = str(row.get("question", "")).lower()
            if not question.startswith(_BINARY_QUESTION_PREFIXES):
                continue

            mc_answer = str(row.get("multiple_choice_answer", "")).lower().strip()
            if mc_answer not in {"yes", "no"}:
                continue

            img = row.get("image") or row.get("img")
            if img is None:
                continue

            samples.append({
                "image": img.convert("RGB") if hasattr(img, "convert") else img,
                "question": str(row.get("question", "")),
                "answer": mc_answer,
                "idx": i,
            })

            if n_samples is not None and len(samples) >= n_samples:
                break

        logger.info(f"VQA v2 バイナリサブセット: {len(samples)} サンプル")
        return samples

    def evaluate(
        self,
        samples: list[dict],
        max_new_tokens: int = 16,
    ) -> TaskResult:
        """バイナリ QA サンプルに対して回答を生成しメトリクスを計算する。

        Args:
            samples:        load_dataset が返したサンプルのリスト
            max_new_tokens: 生成する最大トークン数

        Returns:
            TaskResult (metrics には "accuracy", "f1", "precision", "recall" が含まれる)
        """
        logger.info(f"バイナリ評価開始: {len(samples)} サンプル, モデル={self.model_name}")

        images = [s["image"] for s in samples]
        prompts = [
            f"<image>\nQuestion: {s['question']}\nAnswer yes or no."
            for s in samples
        ]

        # vLLM で推論
        generated = self._generate(prompts, images, max_new_tokens=max_new_tokens)

        # 回答を正規化する
        preds = [_normalize_binary(g) for g in generated]
        labels = [s["answer"] for s in samples]

        # メトリクス計算
        metrics = _compute_binary_metrics(preds, labels)
        logger.info(
            f"バイナリ精度: {metrics['accuracy']:.4f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}"
        )

        # サンプルごとの結果
        per_sample = [
            {
                "idx": s["idx"],
                "question": s["question"],
                "generated": gen,
                "predicted": pred,
                "label": s["answer"],
                "correct": pred == s["answer"],
            }
            for s, gen, pred in zip(samples, generated, preds)
        ]

        # データセット名 (実際にロードしたもの)
        dataset_name = _POPE_DATASET_NAME

        return TaskResult(
            task_name=self.task_name,
            model_name=self.model_name,
            dataset_name=dataset_name,
            n_samples=len(samples),
            metrics=metrics,
            per_sample=per_sample,
        )
