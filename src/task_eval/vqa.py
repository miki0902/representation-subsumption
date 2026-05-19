"""VQA evaluator — VQA v2 dataset, soft accuracy.

評価内容:
  - データセット: HuggingFaceM4/VQAv2 (validation split)
  - 各サンプルに対して質問を提示し、1 語または短いフレーズで回答させる
  - VQA ソフト精度: 各回答が参照回答群に含まれる割合で精度を計算する

VQA ソフト精度の計算:
  VQA v2 の公式メトリクスでは、10 名のアノテーターが付けた回答のうち
  予測と一致するものが何件あるかで精度を求める:
    accuracy_i = min(count(pred_i in references_i) / 3, 1.0)
    accuracy   = mean(accuracy_i)

  参照回答 (answers カラム) が利用できない場合は
  multiple_choice_answer (最多数回答) との完全一致で代替する。
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
_DATASET_NAME = "HuggingFaceM4/VQAv2"
_DATASET_SPLIT = "validation"


def _normalize_answer(ans: str) -> str:
    """回答を正規化する (小文字化・句読点除去・冠詞除去・空白正規化)。

    VQA 公式評価スクリプトと同様の前処理を行う。
    """
    ans = ans.lower().strip()
    # 句読点を除去
    ans = ans.translate(str.maketrans("", "", string.punctuation))
    # 冠詞を除去
    articles = {"a", "an", "the"}
    tokens = [t for t in ans.split() if t not in articles]
    return " ".join(tokens).strip()


def _vqa_accuracy(
    pred: str,
    references: list[str] | None,
    mc_answer: str | None,
) -> float:
    """1 サンプルの VQA 精度を計算する。

    Args:
        pred:       モデルが生成した回答
        references: アノテーターの回答リスト (最大 10 件)。None の場合は mc_answer を使用
        mc_answer:  最多数回答 (multiple_choice_answer)

    Returns:
        0.0–1.0 のスコア
    """
    pred_norm = _normalize_answer(pred)

    if references:
        refs_norm = [_normalize_answer(r) for r in references]
        count = sum(1 for r in refs_norm if r == pred_norm)
        return min(count / 3.0, 1.0)

    if mc_answer is not None:
        return 1.0 if pred_norm == _normalize_answer(mc_answer) else 0.0

    return 0.0


class VQAEvaluator(BaseTaskEvaluator):
    """VQA v2 を用いた視覚的質問応答評価器。

    使い方:
        evaluator = VQAEvaluator("Qwen/Qwen2-VL-7B-Instruct")
        samples = evaluator.load_dataset(n_samples=500)
        result = evaluator.evaluate(samples)
        print(result.metrics)  # {"accuracy": 0.72}
    """

    task_name = "vqa"

    def load_dataset(
        self,
        n_samples: int | None = _DEFAULT_N_SAMPLES,
        split: str = _DATASET_SPLIT,
    ) -> list[dict]:
        """VQA v2 データセットを読み込む。

        各サンプル辞書:
          - "image":       PIL.Image.Image
          - "question":    str
          - "mc_answer":   str  (最多数回答)
          - "answers":     list[str] または None (アノテーター回答)
          - "question_id": int

        Args:
            n_samples: 使用するサンプル数 (None = 全サンプル)
            split:     データセット分割

        Returns:
            サンプル辞書のリスト
        """
        from datasets import load_dataset

        logger.info(f"データセットをロード中: {_DATASET_NAME} (split={split})")
        ds = load_dataset(_DATASET_NAME, split=split, trust_remote_code=True)

        if n_samples is not None:
            n_samples = min(n_samples, len(ds))
            ds = ds.select(range(n_samples))

        logger.info(f"ロード完了: {len(ds)} サンプル")

        samples = []
        for i in range(len(ds)):
            row = ds[i]
            img = row.get("image") or row.get("img")
            if img is None:
                logger.warning(f"サンプル {i}: 画像が見つかりません。スキップします。")
                continue

            # answers カラムは辞書リストの場合がある: [{"answer": "yes", ...}, ...]
            raw_answers = row.get("answers", None)
            if raw_answers is not None:
                if isinstance(raw_answers, list) and len(raw_answers) > 0:
                    if isinstance(raw_answers[0], dict):
                        answers = [a["answer"] for a in raw_answers if "answer" in a]
                    else:
                        answers = [str(a) for a in raw_answers]
                else:
                    answers = None
            else:
                answers = None

            samples.append({
                "image": img.convert("RGB") if hasattr(img, "convert") else img,
                "question": str(row.get("question", "")),
                "mc_answer": str(row.get("multiple_choice_answer", "")),
                "answers": answers,
                "question_id": int(row.get("question_id", i)),
            })

        return samples

    def evaluate(
        self,
        samples: list[dict],
        max_new_tokens: int = 32,
    ) -> TaskResult:
        """VQA v2 サンプルに対して回答を生成し精度を計算する。

        Args:
            samples:        load_dataset が返したサンプルのリスト
            max_new_tokens: 生成する最大トークン数 (短い回答を想定するため小さく設定)

        Returns:
            TaskResult (metrics には "accuracy" が含まれる)
        """
        logger.info(f"VQA 評価開始: {len(samples)} サンプル, モデル={self.model_name}")

        images = [s["image"] for s in samples]
        prompts = [
            f"<image>\nQuestion: {s['question']}\nAnswer in one word or short phrase."
            for s in samples
        ]

        # vLLM で推論
        generated = self._generate(prompts, images, max_new_tokens=max_new_tokens)

        # メトリクス計算
        scores = [
            _vqa_accuracy(gen, s.get("answers"), s.get("mc_answer"))
            for gen, s in zip(generated, samples)
        ]
        accuracy = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"VQA 精度: {accuracy:.4f}")

        # サンプルごとの結果
        per_sample = [
            {
                "question_id": s["question_id"],
                "question": s["question"],
                "generated": gen,
                "mc_answer": s["mc_answer"],
                "score": score,
            }
            for s, gen, score in zip(samples, generated, scores)
        ]

        return TaskResult(
            task_name=self.task_name,
            model_name=self.model_name,
            dataset_name=_DATASET_NAME,
            n_samples=len(samples),
            metrics={"accuracy": accuracy},
            per_sample=per_sample,
        )
