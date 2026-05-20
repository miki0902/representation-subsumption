"""
VLM ビジョンエンコーダから特徴量を抽出し .npy で保存する。
抽出した特徴量は run_single_file.py / run_analysis.py にそのまま渡せる。

対応モデル:
  - Qwen2-VL  (Qwen/Qwen2-VL-2B-Instruct, Qwen/Qwen2-VL-7B-Instruct)
  - LLaVA-1.5 (llava-hf/llava-1.5-7b-hf, llava-hf/llava-1.5-13b-hf)

使い方:
  python scripts/extract_features_vlm.py \\
    --model Qwen/Qwen2-VL-7B-Instruct \\
    --dataset nlphuji/flickr30k \\
    --split test \\
    --n-samples 1000 \\
    --output features/qwen2vl_7b_flickr30k.npy \\
    --layer vision_encoder   # or llm_last

特徴量の種類 (--layer):
  vision_encoder: ビジョンエンコーダ最終層の mean pooling (画像トークン平均)
  llm_last:       LLM バックボーン最終層の画像トークン平均
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# モデルファミリー検出
# ---------------------------------------------------------------------------

def _detect_model_family(model_name: str) -> str:
    """モデル名からファミリーを検出する。

    戻り値:
      "qwen25vl" : Qwen2.5-VL 系 (Qwen2_5_VLForConditionalGeneration)
      "qwen2vl"  : Qwen2-VL 系   (Qwen2VLForConditionalGeneration)
      "llava"    : LLaVA-1.5 系
      "generic"  : それ以外（LLM 最終層からの汎用抽出）
    """
    lower = model_name.lower()
    # Qwen2.5-VL は Qwen2-VL より先にチェックする（部分文字列の包含関係に注意）
    if "qwen2.5-vl" in lower or "qwen2_5_vl" in lower or "qwen2.5vl" in lower:
        return "qwen25vl"
    if "qwen2-vl" in lower or "qwen2vl" in lower:
        return "qwen2vl"
    if "llava" in lower:
        return "llava"
    return "generic"


# ---------------------------------------------------------------------------
# dtype ユーティリティ
# ---------------------------------------------------------------------------

def _resolve_dtype(dtype_str: str):
    """文字列から torch dtype を解決する。"""
    import torch
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"非対応の dtype: {dtype_str}. 対応: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def _build_quantization_config(quantize: str, compute_dtype=None):
    """量子化設定を返す。

    Args:
        quantize: "none" / "8bit" / "4bit"
        compute_dtype: 4bit 時の計算精度 (torch.bfloat16 など)

    Returns:
        BitsAndBytesConfig または None
    """
    if quantize == "none":
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning("BitsAndBytesConfig が見つかりません。量子化をスキップします。")
        return None

    if quantize == "8bit":
        logger.info("8-bit 量子化を有効化 (bitsandbytes)")
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "4bit":
        import torch
        bnb_dtype = compute_dtype if compute_dtype is not None else torch.bfloat16
        logger.info(f"4-bit 量子化を有効化 (bitsandbytes, compute_dtype={bnb_dtype})")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=bnb_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError(f"非対応の quantize: {quantize}. 対応: none, 8bit, 4bit")


def _resolve_device(device_str: str) -> str:
    """デバイス文字列を解決する。"auto" のときは CUDA → MPS → CPU の順に試みる。"""
    if device_str != "auto":
        return device_str
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Qwen2-VL 特徴量抽出
# ---------------------------------------------------------------------------

def _extract_qwen2vl(
    model_name: str,
    images: list,
    layer: str,
    device: str,
    dtype,
    batch_size: int,
    quantize: str = "none",
) -> np.ndarray:
    """Qwen2-VL のビジョンエンコーダまたは LLM 最終層から特徴量を抽出する。

    なぜ model.visual を使うか:
      Qwen2VLForConditionalGeneration の visual サブモジュールは
      画像パッチを受け取り固定次元の埋め込みを返す独立したエンコーダである。
      このモジュールを直接呼び出すことで LLM 部分を無視した
      純粋なビジョン特徴量を得ることができる。
    """
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None
        logger.warning(
            "qwen-vl-utils が見つかりません。pip install qwen-vl-utils を検討してください。"
        )

    quantization_config = _build_quantization_config(quantize, compute_dtype=dtype)

    logger.info(f"Qwen2-VL モデルをロード中: {model_name}")
    load_kwargs: dict = dict(
        device_map="auto" if quantization_config is not None else device,
        trust_remote_code=True,
    )
    if quantization_config is not None:
        # 量子化時は torch_dtype を BitsAndBytesConfig 側で管理する
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["torch_dtype"] = dtype

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        **load_kwargs,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # バージョン差を吸収: model.visual または model.model.visual
    visual_module = _resolve_visual_module(model)
    logger.info(f"ビジョンエンコーダを解決: {type(visual_module).__name__}")

    all_features: list[np.ndarray] = []
    n_batches = (len(images) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_imgs = images[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # チャットテンプレートを構築する
        messages_list = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            for img in batch_imgs
        ]

        batch_features: list[np.ndarray] = []

        if layer == "vision_encoder":
            # ビジョンエンコーダの出力を直接取得する
            for msgs, img in zip(messages_list, batch_imgs):
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[img],
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

                with torch.no_grad():
                    # pixel_values を visual モジュールに直接渡してビジョン特徴を取得する
                    pixel_values = inputs.get("pixel_values")
                    image_grid_thw = inputs.get("image_grid_thw")
                    if pixel_values is None:
                        logger.warning("pixel_values が見つかりません。空の特徴量をスキップします。")
                        continue
                    if image_grid_thw is not None:
                        vision_out = visual_module(pixel_values.to(dtype), grid_thw=image_grid_thw)
                    else:
                        vision_out = visual_module(pixel_values.to(dtype))
                    # Qwen2-VL はテンソル、Qwen2.5-VL は BaseModelOutputWithPooling
                    vision_tensor = _vision_out_to_tensor(vision_out)
                    feat = vision_tensor.mean(dim=0).float().cpu().numpy()
                    batch_features.append(feat)

        elif layer == "llm_last":
            # LLM 最終層の画像トークン位置を平均する
            for msgs, img in zip(messages_list, batch_imgs):
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[img],
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, d)
                    # input_ids から画像トークンの位置を特定する
                    # Qwen2-VL の画像トークン id は processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                    try:
                        img_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                    except Exception:
                        img_token_id = None

                    input_ids = inputs["input_ids"][0]
                    if img_token_id is not None:
                        mask = input_ids == img_token_id
                        if mask.sum() > 0:
                            feat = last_hidden[0][mask].mean(dim=0).float().cpu().numpy()
                        else:
                            feat = last_hidden[0].mean(dim=0).float().cpu().numpy()
                    else:
                        feat = last_hidden[0].mean(dim=0).float().cpu().numpy()
                    batch_features.append(feat)
        else:
            raise ValueError(f"非対応の layer: {layer}. 対応: vision_encoder, llm_last")

        if batch_features:
            all_features.extend(batch_features)

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            logger.info(f"  バッチ {batch_idx + 1}/{n_batches} 完了 ({len(all_features)} サンプル抽出済み)")

    return np.stack(all_features, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Qwen2.x-VL 共通ユーティリティ
# ---------------------------------------------------------------------------

def _vision_out_to_tensor(vision_out):
    """ビジョンエンコーダの出力を 2D テンソル (num_tokens, hidden_dim) に変換する。

    Qwen2-VL   : visual() が tensor (num_tokens, d) を直接返す
    Qwen2.5-VL : visual() が BaseModelOutputWithPooling を返す
                 → last_hidden_state (1, num_tokens, d) または (num_tokens, d)
    """
    import torch
    if isinstance(vision_out, torch.Tensor):
        return vision_out  # Qwen2-VL: すでにテンソル

    # BaseModelOutputWithPooling などの dataclass 系出力
    if hasattr(vision_out, "last_hidden_state") and vision_out.last_hidden_state is not None:
        t = vision_out.last_hidden_state
    elif hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
        t = vision_out.pooler_output
    else:
        raise ValueError(
            f"ビジョンエンコーダ出力から特徴量テンソルを取り出せません: {type(vision_out)}\n"
            f"属性: {[k for k in vision_out.keys() if vision_out[k] is not None]}"
        )

    # バッチ次元 (1, num_tokens, d) → (num_tokens, d)
    if t.dim() == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def _resolve_visual_module(model):
    """Qwen 系モデルのビジョンエンコーダモジュールを解決する。

    Qwen2-VL   : model.visual          (ForConditionalGeneration 直下)
    Qwen2.5-VL : model.model.visual    (inner Qwen2_5_VLModel 下に移動)

    どちらにも対応できるよう順番に試す。
    """
    # 直下にある場合（Qwen2-VL）
    visual = getattr(model, "visual", None)
    if visual is not None:
        return visual
    # inner model 下にある場合（Qwen2.5-VL）
    inner = getattr(model, "model", None)
    if inner is not None:
        visual = getattr(inner, "visual", None)
        if visual is not None:
            return visual
    raise AttributeError(
        f"{type(model).__name__} に 'visual' モジュールが見つかりません。"
        f"利用可能な属性: {[n for n, _ in model.named_children()]}"
    )


def _extract_qwen25vl(
    model_name: str,
    images: list,
    layer: str,
    device: str,
    dtype,
    batch_size: int,
    quantize: str = "none",
) -> np.ndarray:
    """Qwen2.5-VL のビジョンエンコーダまたは LLM 最終層から特徴量を抽出する。

    Qwen2.5-VL (Qwen2_5_VLForConditionalGeneration) では
    ビジョンエンコーダが model.model.visual に移動している。
    _resolve_visual_module() で自動解決する。
    """
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        logger.warning(
            "Qwen2_5_VLForConditionalGeneration が見つかりません。"
            "transformers を最新版にアップデートしてください: pip install -U transformers"
        )
        raise

    quantization_config = _build_quantization_config(quantize, compute_dtype=dtype)

    logger.info(f"Qwen2.5-VL モデルをロード中: {model_name}")
    load_kwargs: dict = dict(
        device_map="auto" if quantization_config is not None else device,
        trust_remote_code=True,
    )
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["torch_dtype"] = dtype

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        **load_kwargs,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Qwen2.5-VL では model.model.visual にエンコーダがある
    visual_module = _resolve_visual_module(model)
    logger.info(f"ビジョンエンコーダを解決: {type(visual_module).__name__}")

    all_features: list[np.ndarray] = []
    n_batches = (len(images) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_imgs = images[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        messages_list = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            for img in batch_imgs
        ]

        batch_features: list[np.ndarray] = []

        if layer == "vision_encoder":
            for msgs, img in zip(messages_list, batch_imgs):
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[img],
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

                with torch.no_grad():
                    pixel_values = inputs.get("pixel_values")
                    image_grid_thw = inputs.get("image_grid_thw")
                    if pixel_values is None:
                        logger.warning("pixel_values が見つかりません。スキップします。")
                        continue
                    if image_grid_thw is not None:
                        vision_out = visual_module(pixel_values.to(dtype), grid_thw=image_grid_thw)
                    else:
                        vision_out = visual_module(pixel_values.to(dtype))
                    feat = vision_out.mean(dim=0).float().cpu().numpy()
                    batch_features.append(feat)

        elif layer == "llm_last":
            for msgs, img in zip(messages_list, batch_imgs):
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[img],
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    last_hidden = outputs.hidden_states[-1]
                    try:
                        img_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                    except Exception:
                        img_token_id = None

                    input_ids = inputs["input_ids"][0]
                    if img_token_id is not None:
                        mask = input_ids == img_token_id
                        feat = (
                            last_hidden[0][mask].mean(dim=0).float().cpu().numpy()
                            if mask.sum() > 0
                            else last_hidden[0].mean(dim=0).float().cpu().numpy()
                        )
                    else:
                        feat = last_hidden[0].mean(dim=0).float().cpu().numpy()
                    batch_features.append(feat)
        else:
            raise ValueError(f"非対応の layer: {layer}. 対応: vision_encoder, llm_last")

        if batch_features:
            all_features.extend(batch_features)

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            logger.info(f"  バッチ {batch_idx + 1}/{n_batches} 完了 ({len(all_features)} サンプル抽出済み)")

    return np.stack(all_features, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# LLaVA-1.5 特徴量抽出
# ---------------------------------------------------------------------------

def _extract_llava(
    model_name: str,
    images: list,
    layer: str,
    device: str,
    dtype,
    batch_size: int,
    quantize: str = "none",
) -> np.ndarray:
    """LLaVA-1.5 のビジョンタワーまたは LLM 最終層から特徴量を抽出する。

    なぜ model.vision_tower を使うか:
      LlavaForConditionalGeneration の vision_tower は CLIP ViT であり、
      入力画像のパッチ特徴量を出力する。このモジュールを直接呼び出すことで
      LLM を経由しない純粋なビジョン特徴量を効率的に取得できる。
    """
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    quantization_config = _build_quantization_config(quantize, compute_dtype=dtype)

    logger.info(f"LLaVA モデルをロード中: {model_name}")
    load_kwargs: dict = dict(
        device_map="auto" if quantization_config is not None else device,
        trust_remote_code=True,
    )
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["torch_dtype"] = dtype

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        **load_kwargs,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    all_features: list[np.ndarray] = []
    n_batches = (len(images) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_imgs = images[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        batch_features: list[np.ndarray] = []

        if layer == "vision_encoder":
            for img in batch_imgs:
                # ビジョンタワーのみを使用して特徴量を抽出する
                inputs = processor(
                    text="<image>\nDescribe this image.",
                    images=img,
                    return_tensors="pt",
                )
                pixel_values = inputs["pixel_values"].to(model.device, dtype=dtype)

                with torch.no_grad():
                    # vision_tower は (1, num_patches, d) を返す
                    vision_out = model.vision_tower(pixel_values, output_hidden_states=False)
                    # last_hidden_state: (1, num_patches, d)
                    if hasattr(vision_out, "last_hidden_state"):
                        feats = vision_out.last_hidden_state[0]  # (num_patches, d)
                    else:
                        feats = vision_out[0]

                    # CLS トークンがあれば使用し、なければ mean pool する
                    # CLIP ViT は先頭が CLS トークン
                    feat = feats[0].float().cpu().numpy()  # CLS token
                batch_features.append(feat)

        elif layer == "llm_last":
            for img in batch_imgs:
                prompt = "USER: <image>\nDescribe this image. ASSISTANT:"
                inputs = processor(text=prompt, images=img, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
                pixel_values = inputs.get("pixel_values")
                if pixel_values is not None:
                    inputs["pixel_values"] = pixel_values.to(dtype)

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                    last_hidden = outputs.hidden_states[-1]  # (1, seq_len, d)

                    # image_token_index を使って画像トークン位置を特定する
                    img_token_id = getattr(model.config, "image_token_index", None)
                    if img_token_id is not None:
                        input_ids = inputs["input_ids"][0]
                        mask = input_ids == img_token_id
                        if mask.sum() > 0:
                            feat = last_hidden[0][mask].mean(dim=0).float().cpu().numpy()
                        else:
                            feat = last_hidden[0].mean(dim=0).float().cpu().numpy()
                    else:
                        feat = last_hidden[0].mean(dim=0).float().cpu().numpy()
                batch_features.append(feat)
        else:
            raise ValueError(f"非対応の layer: {layer}. 対応: vision_encoder, llm_last")

        all_features.extend(batch_features)

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            logger.info(f"  バッチ {batch_idx + 1}/{n_batches} 完了 ({len(all_features)} サンプル抽出済み)")

    return np.stack(all_features, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 汎用 (generic) 特徴量抽出
# ---------------------------------------------------------------------------

def _extract_generic(
    model_name: str,
    images: list,
    layer: str,
    device: str,
    dtype,
    batch_size: int,
    quantize: str = "none",
) -> np.ndarray:
    """汎用 VLM から LLM 最終層の hidden states を抽出する。

    Qwen2-VL / LLaVA 以外のモデルに対するフォールバック実装。
    AutoModelForVision2Seq または AutoModelForCausalLM + AutoProcessor を試みる。
    """
    import torch
    from transformers import AutoProcessor

    quantization_config = _build_quantization_config(quantize, compute_dtype=dtype)
    load_kwargs: dict = dict(
        device_map="auto" if quantization_config is not None else device,
        trust_remote_code=True,
    )
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["torch_dtype"] = dtype

    logger.info(f"汎用モデルとしてロード中: {model_name}")
    try:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            **load_kwargs,
        )
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )

    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    all_features: list[np.ndarray] = []
    n_batches = (len(images) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_imgs = images[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        for img in batch_imgs:
            try:
                inputs = processor(
                    text="Describe this image.",
                    images=img,
                    return_tensors="pt",
                )
            except Exception:
                inputs = processor(text="Describe this image.", return_tensors="pt")

            inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1]  # (1, seq_len, d)
                feat = last_hidden[0].mean(dim=0).float().cpu().numpy()
            all_features.append(feat)

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            logger.info(f"  バッチ {batch_idx + 1}/{n_batches} 完了 ({len(all_features)} サンプル抽出済み)")

    return np.stack(all_features, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# データセット読み込み
# ---------------------------------------------------------------------------

def load_images(dataset_name: str, split: str, n_samples: int | None) -> list:
    """HuggingFace datasets から PIL 画像のリストを返す。

    ローディングスクリプト形式（.py）のデータセットは datasets>=2.x で非対応。
    エラー時は代替候補を順に試す。
    """
    from datasets import load_dataset

    # 旧スクリプト形式データセットから Parquet 形式への代替候補
    # split も合わせて調整（datasets によって split 名が異なる）
    _FALLBACKS: dict[str, list[tuple[str, str]]] = {
        "nlphuji/flickr30k":     [("lmms-lab/POPE", "test")],
        "HuggingFaceM4/NoCaps":  [("lmms-lab/POPE", "test")],
    }

    candidates: list[tuple[str, str]] = [(dataset_name, split)]
    candidates += _FALLBACKS.get(dataset_name, [])

    ds = None
    for ds_name, ds_split in candidates:
        try:
            logger.info(f"データセットをロード中: {ds_name} (split={ds_split})")
            ds = load_dataset(ds_name, split=ds_split)
            if ds_name != dataset_name:
                logger.warning(
                    f"'{dataset_name}' はスクリプト形式のため非対応。"
                    f"代替 '{ds_name}' (split={ds_split}) を使用します。"
                )
            break
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                logger.warning(
                    f"'{ds_name}': ローディングスクリプト形式は非対応 — 次の候補へ"
                )
                continue
            raise

    if ds is None:
        tried = [f"{n} (split={s})" for n, s in candidates]
        raise RuntimeError(
            "全ての候補データセットが失敗しました:\n  " + "\n  ".join(tried) + "\n"
            "Parquet 形式のデータセットを --dataset で指定してください。\n"
            "動作確認済みの例:\n"
            "  lmms-lab/POPE            (split=test)\n"
            "  lmms-lab/flickr30k       (split=test) ※存在する場合\n"
        )

    if n_samples is not None:
        n_samples = min(n_samples, len(ds))
        ds = ds.select(range(n_samples))
        logger.info(f"サブサンプリング: {n_samples} サンプルを使用")

    # 画像カラム名を自動検出する
    image_col = None
    for col in ["image", "img", "pixel_values"]:
        if col in ds.column_names:
            image_col = col
            break
    if image_col is None:
        raise ValueError(
            f"画像カラムが見つかりません。利用可能なカラム: {ds.column_names}"
        )

    logger.info(f"画像カラム '{image_col}' を使用 ({len(ds)} 枚)")
    images = [ds[i][image_col] for i in range(len(ds))]

    # PIL Image でない場合は変換を試みる
    from PIL import Image
    converted = []
    for img in images:
        if not isinstance(img, Image.Image):
            if hasattr(img, "convert"):
                converted.append(img)
            else:
                converted.append(Image.fromarray(img))
        else:
            converted.append(img.convert("RGB"))
    return converted


# ---------------------------------------------------------------------------
# メタデータ保存
# ---------------------------------------------------------------------------

def save_metadata(
    output_path: Path,
    model_name: str,
    dataset_name: str,
    split: str,
    layer: str,
    n_samples: int,
    feature_shape: tuple[int, int],
    elapsed_sec: float,
    quantize: str = "none",
) -> None:
    """特徴量抽出の設定と結果をメタデータとして JSON に保存する。"""
    meta = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "split": split,
        "layer": layer,
        "n_samples": n_samples,
        "feature_shape": list(feature_shape),
        "elapsed_sec": round(elapsed_sec, 2),
        "quantize": quantize,
        "output_path": str(output_path),
    }
    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"メタデータを保存しました: {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VLM ビジョンエンコーダ特徴量抽出スクリプト",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help="HuggingFace モデルID (例: Qwen/Qwen2-VL-7B-Instruct)",
    )
    p.add_argument(
        "--dataset",
        default="lmms-lab/POPE",
        help="HuggingFace datasets のデータセット名 (Parquet形式のみ対応)",
    )
    p.add_argument("--split", default="test", help="データセット分割")
    p.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="使用するサンプル数 (None = 全サンプル)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="出力ファイルパス (.npy)。同名 .json にメタデータも保存される",
    )
    p.add_argument(
        "--layer",
        default="vision_encoder",
        choices=["vision_encoder", "llm_last"],
        help="抽出する特徴量の種類",
    )
    p.add_argument("--batch-size", type=int, default=8, help="バッチサイズ")
    p.add_argument(
        "--device",
        default="auto",
        help="使用デバイス (cuda / cpu / mps / auto)",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="モデルの重みの dtype",
    )
    p.add_argument(
        "--quantize",
        default="none",
        choices=["none", "8bit", "4bit"],
        help=(
            "bitsandbytes による量子化 (VRAM 不足時に使用)。"
            "8bit: ~8GB, 4bit: ~5GB で 7B モデルが動作。"
            "要: pip install bitsandbytes"
        ),
    )
    p.add_argument("--debug", action="store_true", help="デバッグログを有効化")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    logger.info(f"デバイス: {device}, dtype: {args.dtype}, quantize: {args.quantize}")

    # 画像の読み込み
    images = load_images(args.dataset, args.split, args.n_samples)
    logger.info(f"読み込み完了: {len(images)} 枚")

    # モデルファミリーの検出と特徴量抽出
    family = _detect_model_family(args.model)
    logger.info(f"モデルファミリー: {family}")

    t0 = time.time()
    if family == "qwen25vl":
        features = _extract_qwen25vl(
            args.model, images, args.layer, device, dtype, args.batch_size,
            quantize=args.quantize,
        )
    elif family == "qwen2vl":
        features = _extract_qwen2vl(
            args.model, images, args.layer, device, dtype, args.batch_size,
            quantize=args.quantize,
        )
    elif family == "llava":
        features = _extract_llava(
            args.model, images, args.layer, device, dtype, args.batch_size,
            quantize=args.quantize,
        )
    else:
        features = _extract_generic(
            args.model, images, args.layer, device, dtype, args.batch_size,
            quantize=args.quantize,
        )
    elapsed = time.time() - t0

    logger.info(f"特徴量抽出完了: shape={features.shape}, 所要時間={elapsed:.1f}s")

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), features)
    logger.info(f"特徴量を保存しました: {output_path}")

    save_metadata(
        output_path=output_path,
        model_name=args.model,
        dataset_name=args.dataset,
        split=args.split,
        layer=args.layer,
        n_samples=len(images),
        feature_shape=features.shape,
        elapsed_sec=elapsed,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()
