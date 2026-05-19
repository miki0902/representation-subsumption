"""Image captioning evaluator — NoCaps dataset, BLEU-4 + CIDEr metrics.

評価内容:
  - データセット: HuggingFaceM4/NoCaps (validation split, 4500 画像)
  - 各画像に対して 11 の参照キャプションが用意されている
  - 生成したキャプションを参照キャプションと比較して BLEU-4 と CIDEr を計算する

メトリクス:
  BLEU-4: nltk を用いた corpus-level BLEU スコア (常に計算)
  CIDEr:  pycocoevalcap が利用可能な場合に計算。インストールされていない場合は
          "cider": null としてスキップする。
          インストール: pip install pycocoevalcap

プロンプト形式:
  Qwen2-VL などチャットテンプレートに対応するモデルはチャット形式を使用する。
  それ以外は "<image>\\nDescribe this image in one sentence." を使用する。
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
# HuggingFaceM4/NoCaps は Parquet 形式で trust_remote_code 不要
_DATASET_NAME = "HuggingFaceM4/NoCaps"
_DATASET_SPLIT = "validation"


def _normalize_text(text: str) -> str:
    """テキストを小文字化し句読点を除去して正規化する。"""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def _tokenize(text: str) -> list[str]:
    """シンプルな空白区切りトークン化。"""
    return _normalize_text(text).split()


def _compute_bleu4(hypotheses: list[list[str]], references_list: list[list[list[str]]]) -> float:
    """corpus-level BLEU-4 スコアを計算する。

    Args:
        hypotheses:       生成されたキャプションのトークンリスト
        references_list:  各サンプルの参照キャプションのトークンリスト (複数参照)

    Returns:
        BLEU-4 スコア (0.0–1.0)
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    smoothie = SmoothingFunction().method1
    score = corpus_bleu(
        references_list,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )
    return float(score)


def _compute_cider(
    hypotheses: list[str],
    references_list: list[list[str]],
) -> float | None:
    """CIDEr スコアを計算する。

    pycocoevalcap が利用可能な場合のみ計算する。
    利用できない場合は None を返す。

    Args:
        hypotheses:      生成されたキャプション (文字列)
        references_list: 各サンプルの参照キャプション (文字列リスト)

    Returns:
        CIDEr スコア (0.0 以上) または None
    """
    try:
        from pycocoevalcap.cider.cider import Cider
    except ImportError:
        logger.warning(
            "pycocoevalcap が見つかりません。CIDEr の計算をスキップします。"
            " インストール: pip install pycocoevalcap"
        )
        return None

    # pycocoevalcap が期待する辞書形式に変換する
    gts: dict[int, list[str]] = {}
    res: dict[int, list[str]] = {}
    for i, (hyp, refs) in enumerate(zip(hypotheses, references_list)):
        gts[i] = refs
        res[i] = [hyp]

    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return float(score)


def _build_prompt(model_name: str, use_chat_template: bool = False) -> str | dict:
    """モデルに応じたプロンプト形式を返す。

    Qwen2-VL などチャットテンプレートを使うモデルは辞書形式を返す。
    それ以外は文字列プロンプトを返す。
    """
    text = "Describe this image in one sentence."
    if use_chat_template:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
    return f"<image>\n{text}"


class CaptioningEvaluator(BaseTaskEvaluator):
    """Flickr30k を用いた画像キャプション評価器。

    使い方:
        evaluator = CaptioningEvaluator("Qwen/Qwen2-VL-7B-Instruct")
        samples = evaluator.load_dataset(n_samples=500)
        result = evaluator.evaluate(samples)
        print(result.metrics)
    """

    task_name = "captioning"

    def load_dataset(
        self,
        n_samples: int | None = _DEFAULT_N_SAMPLES,
        split: str = _DATASET_SPLIT,
    ) -> list[dict]:
        """Flickr30k データセットを読み込む。

        各サンプル辞書:
          - "image":    PIL.Image.Image
          - "captions": list[str]  (5 つの参照キャプション)
          - "idx":      int        (データセット内のインデックス)

        Args:
            n_samples: 使用するサンプル数 (None = 全サンプル)
            split:     データセット分割

        Returns:
            サンプル辞書のリスト
        """
        from datasets import load_dataset

        # Parquet 形式の代替候補（スクリプト形式は datasets>=2.x 非対応）
        _CANDIDATES = [
            (_DATASET_NAME, split),
            ("lmms-lab/coco2014_cap_val", "val"),
            ("lmms-lab/POPE", "test"),        # 画像のみ流用、キャプションなし
        ]
        ds = None
        for ds_name, ds_split in _CANDIDATES:
            try:
                logger.info(f"データセットをロード中: {ds_name} (split={ds_split})")
                ds = load_dataset(ds_name, split=ds_split)
                if ds_name != _DATASET_NAME:
                    logger.warning(
                        f"'{_DATASET_NAME}' はスクリプト形式のため代替 '{ds_name}' を使用"
                    )
                break
            except RuntimeError as e:
                if "Dataset scripts are no longer supported" in str(e):
                    logger.warning(f"'{ds_name}': スクリプト形式 — 次の候補へ")
                    continue
                raise
        if ds is None:
            raise RuntimeError(
                "キャプションデータセットのロードに失敗。\n"
                "lmms-lab/coco2014_cap_val が利用できるか確認してください。"
            )

        if n_samples is not None:
            n_samples = min(n_samples, len(ds))
            ds = ds.select(range(n_samples))

        logger.info(f"ロード完了: {len(ds)} サンプル  カラム: {ds.column_names}")

        samples = []
        for i in range(len(ds)):
            row = ds[i]
            img = row["image"]

            # NoCaps: annotations_captions (list[str])
            # Flickr30k 系: caption (list[str] or str)
            raw_cap = (
                row.get("annotations_captions")
                or row.get("caption")
                or row.get("captions")
                or []
            )
            if isinstance(raw_cap, str):
                captions = [raw_cap]
            else:
                captions = [c for c in raw_cap if isinstance(c, str)]

            if not captions:
                logger.debug(f"サンプル {i}: キャプションなし — スキップ")
                continue

            samples.append({
                "image": img.convert("RGB") if hasattr(img, "convert") else img,
                "captions": captions,
                "idx": i,
            })

        return samples

    def evaluate(
        self,
        samples: list[dict],
        max_new_tokens: int = 64,
        use_chat_template: bool | None = None,
    ) -> TaskResult:
        """サンプルに対してキャプション生成を実行し BLEU-4 と CIDEr を計算する。

        Args:
            samples:           load_dataset が返したサンプルのリスト
            max_new_tokens:    生成する最大トークン数
            use_chat_template: チャットテンプレートを使うか否か。
                               None の場合はモデル名から自動検出する

        Returns:
            TaskResult (metrics には "bleu4" と "cider" が含まれる)
        """
        if use_chat_template is None:
            lower = self.model_name.lower()
            use_chat_template = "qwen" in lower or "llava" in lower

        logger.info(f"キャプション評価開始: {len(samples)} サンプル, モデル={self.model_name}")

        images = [s["image"] for s in samples]

        # プロンプト構築
        if use_chat_template:
            # チャットテンプレートを使う場合はモデルの processor で変換する
            try:
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                template_msgs = _build_prompt(self.model_name, use_chat_template=True)
                text_prompt = processor.apply_chat_template(
                    template_msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"チャットテンプレートの適用に失敗しました: {e}. 通常プロンプトを使用します。")
                text_prompt = "<image>\nDescribe this image in one sentence."
        else:
            text_prompt = "<image>\nDescribe this image in one sentence."

        prompts = [text_prompt] * len(samples)

        # vLLM で推論
        generated = self._generate(prompts, images, max_new_tokens=max_new_tokens)

        # メトリクス計算
        hypotheses_tok = [_tokenize(h) for h in generated]
        references_tok = [[_tokenize(c) for c in s["captions"]] for s in samples]

        bleu4 = _compute_bleu4(hypotheses_tok, references_tok)
        logger.info(f"BLEU-4: {bleu4:.4f}")

        references_str = [s["captions"] for s in samples]
        cider = _compute_cider(generated, references_str)
        if cider is not None:
            logger.info(f"CIDEr: {cider:.4f}")

        metrics: dict[str, float] = {"bleu4": bleu4}
        if cider is not None:
            metrics["cider"] = cider

        # サンプルごとの結果
        per_sample = [
            {
                "idx": s["idx"],
                "generated": gen,
                "references": s["captions"],
            }
            for s, gen in zip(samples, generated)
        ]

        return TaskResult(
            task_name=self.task_name,
            model_name=self.model_name,
            dataset_name=_DATASET_NAME,
            n_samples=len(samples),
            metrics=metrics,
            per_sample=per_sample,
        )
