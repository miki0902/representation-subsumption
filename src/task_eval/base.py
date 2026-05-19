"""BaseTaskEvaluator: 全タスク共通インターフェース。

設計方針:
  - ABC を継承することで各タスク評価器が必ず load_dataset / evaluate を実装することを強制する。
  - vLLM の LLM インスタンスは初回アクセス時に生成し (_get_llm) 以降はキャッシュする。
    これにより複数タスクを評価する際にモデルの再ロードを避けられる。
  - _generate はプロンプトと PIL 画像のリストを受け取り vLLM でバッチ推論を行う。

注意:
  vLLM のバージョンによって multimodal 入力の API が異なる場合がある。
  vllm>=0.4.0 では multi_modal_data={"image": pil_image} の形式を想定しているが、
  新しいバージョンでは vllm.inputs.MultiModalData の形式が変わる可能性がある。
  その場合は _generate メソッドの inputs 構築部分を適宜修正すること。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """タスク評価の結果を格納するデータクラス。

    Attributes:
        task_name: タスクの識別名 (captioning / vqa / multiple_choice / binary)
        model_name: 評価に使用したモデルの HuggingFace ID
        dataset_name: 評価に使用したデータセット名
        n_samples: 評価したサンプル数
        metrics: タスク固有のメトリクス辞書 (例: {"bleu4": 0.28, "cider": 0.95})
        per_sample: サンプルごとの詳細結果 (省略可能)
    """

    task_name: str
    model_name: str
    dataset_name: str
    n_samples: int
    metrics: dict[str, float]
    per_sample: list[dict] = field(default_factory=list, repr=False)


class BaseTaskEvaluator(ABC):
    """全タスク評価器の基底クラス。

    サブクラスは load_dataset と evaluate を実装する必要がある。
    vLLM による推論は _generate を通じて行い、モデルインスタンスは
    _get_llm でキャッシュされる。
    """

    task_name: str = ""

    def __init__(self, model_name: str, device: str = "auto", dtype: str = "bfloat16") -> None:
        """評価器を初期化する。

        Args:
            model_name: HuggingFace モデル ID (例: Qwen/Qwen2-VL-7B-Instruct)
            device:     推論デバイス。"auto" の場合は vLLM が自動検出する
            dtype:      モデル重みの dtype (bfloat16 / float16 / float32)
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self._llm: Any = None  # lazy init

    @abstractmethod
    def load_dataset(self, n_samples: int | None = None, split: str = "test") -> list[dict]:
        """データセットを読み込み、サンプルのリストを返す。

        各サンプルは少なくとも以下のキーを含む辞書であること:
          - "image": PIL.Image.Image
          - タスク固有のキー (例: "question", "answer", "choices" など)

        Args:
            n_samples: 使用するサンプル数 (None = 全サンプル)
            split:     データセット分割 ("train" / "validation" / "test")

        Returns:
            サンプル辞書のリスト
        """

    @abstractmethod
    def evaluate(self, samples: list[dict], **kwargs) -> TaskResult:
        """推論を実行して TaskResult を返す。

        Args:
            samples: load_dataset が返したサンプルのリスト
            **kwargs: タスク固有の追加引数

        Returns:
            評価結果を含む TaskResult
        """

    def _get_llm(self):
        """vLLM LLM インスタンスをキャッシュして返す (lazy init)。

        なぜ lazy init か:
          サブクラスのインスタンス化時点ではまだモデルのロードが不要なため。
          evaluate が呼ばれた時点でロードすることでメモリを節約できる。

        Returns:
            vllm.LLM インスタンス
        """
        if self._llm is None:
            from vllm import LLM

            # vLLM は dtype に "bfloat16" / "half" (=float16) / "float" (=float32) を受け取る
            dtype_map = {
                "bfloat16": "bfloat16",
                "float16": "half",
                "float32": "float",
            }
            vllm_dtype = dtype_map.get(self.dtype, "bfloat16")

            logger.info(f"vLLM でモデルをロード中: {self.model_name} (dtype={vllm_dtype})")
            self._llm = LLM(
                model=self.model_name,
                dtype=vllm_dtype,
                trust_remote_code=True,
                max_model_len=4096,
            )
            logger.info("モデルのロード完了")
        return self._llm

    def _generate(
        self,
        prompts: list[str],
        images: list,
        max_new_tokens: int = 256,
    ) -> list[str]:
        """vLLM でバッチ推論を実行する。

        Args:
            prompts:        テキストプロンプトのリスト (len == len(images))
            images:         PIL.Image のリスト
            max_new_tokens: 生成する最大トークン数

        Returns:
            生成されたテキストのリスト (strip 済み)

        Note:
            vLLM のマルチモーダル入力 API は vllm>=0.4.0 を想定している。
            バージョンによっては multi_modal_data の構造が変わる可能性があるため、
            エラーが発生した場合は vllm のリリースノートを参照して修正すること。
        """
        from vllm import SamplingParams

        if len(prompts) != len(images):
            raise ValueError(
                f"prompts と images の長さが一致しません: {len(prompts)} vs {len(images)}"
            )

        llm = self._get_llm()
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

        # vLLM のマルチモーダル入力形式
        # vllm>=0.4.0: multi_modal_data={"image": pil_image}
        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }
            for prompt, image in zip(prompts, images)
        ]

        logger.debug(f"vLLM バッチ推論: {len(inputs)} サンプル")
        outputs = llm.generate(inputs, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]
