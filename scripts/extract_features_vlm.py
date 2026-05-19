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
      "qwen2vl"  : Qwen2-VL 系
      "llava"    : LLaVA-1.5 系
      "generic"  : それ以外（LLM 最終層からの汎用抽出）
    """
    lower = model_name.lower()
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

    logger.info(f"Qwen2-VL モデルをロード中: {model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

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
                        vision_out = model.visual(pixel_values.to(dtype), grid_thw=image_grid_thw)
                    else:
                        vision_out = model.visual(pixel_values.to(dtype))
                    # vision_out: (num_image_tokens, d) → mean pool → (d,)
                    feat = vision_out.mean(dim=0).float().cpu().numpy()
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
# LLaVA-1.5 特徴量抽出
# ---------------------------------------------------------------------------

def _extract_llava(
    model_name: str,
    images: list,
    layer: str,
    device: str,
    dtype,
    batch_size: int,
) -> np.ndarray:
    """LLaVA-1.5 のビジョンタワーまたは LLM 最終層から特徴量を抽出する。

    なぜ model.vision_tower を使うか:
      LlavaForConditionalGeneration の vision_tower は CLIP ViT であり、
      入力画像のパッチ特徴量を出力する。このモジュールを直接呼び出すことで
      LLM を経由しない純粋なビジョン特徴量を効率的に取得できる。
    """
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    logger.info(f"LLaVA モデルをロード中: {model_name}")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
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
) -> np.ndarray:
    """汎用 VLM から LLM 最終層の hidden states を抽出する。

    Qwen2-VL / LLaVA 以外のモデルに対するフォールバック実装。
    AutoModelForVision2Seq または AutoModelForCausalLM + AutoProcessor を試みる。
    """
    import torch
    from transformers import AutoProcessor

    logger.info(f"汎用モデルとしてロード中: {model_name}")
    try:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
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

    なぜ image カラムを使うか:
      VLM は画像入力を前提としており、datasets ライブラリの
      image カラムは PIL.Image オブジェクトとして自動デコードされるため
      前処理の手間が省ける。
    """
    from datasets import load_dataset

    logger.info(f"データセットをロード中: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

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
        default="nlphuji/flickr30k",
        help="HuggingFace datasets のデータセット名",
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
    logger.info(f"デバイス: {device}, dtype: {args.dtype}")

    # 画像の読み込み
    images = load_images(args.dataset, args.split, args.n_samples)
    logger.info(f"読み込み完了: {len(images)} 枚")

    # モデルファミリーの検出と特徴量抽出
    family = _detect_model_family(args.model)
    logger.info(f"モデルファミリー: {family}")

    t0 = time.time()
    if family == "qwen2vl":
        features = _extract_qwen2vl(
            args.model, images, args.layer, device, dtype, args.batch_size
        )
    elif family == "llava":
        features = _extract_llava(
            args.model, images, args.layer, device, dtype, args.batch_size
        )
    else:
        features = _extract_generic(
            args.model, images, args.layer, device, dtype, args.batch_size
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
    )


if __name__ == "__main__":
    main()
