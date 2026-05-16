"""
特徴量抽出スクリプト: torchvision モデルと標準データセットから特徴量を抽出して保存する。

なぜこのスクリプトが必要か:
  本リポジトリの分析コードは shape (n, d) の事前抽出済み特徴量ファイルを入力として受け取る。
  このスクリプトはその「前処理」として、
  モデルの中間層出力を npz ファイルに書き出す役割を担う。

対応モデル:
  resnet18, resnet50, resnet101,
  vit_b_16, vit_s_16 (timm 必要),
  efficientnet_b0, efficientnet_b4

対応データセット:
  cifar10, cifar100, stl10, imagenet

使い方:
  # CIFAR-10 test セットから ResNet50 の avgpool 特徴を抽出する
  python scripts/extract_features.py \\
      --model resnet50 \\
      --dataset cifar10 \\
      --split test \\
      --layer avgpool \\
      --output features/cifar10_resnet50_avgpool.npz

  # 複数層を一括抽出する（層別分析用）
  python scripts/extract_features.py \\
      --model resnet50 \\
      --dataset cifar10 \\
      --split test \\
      --layer layer1 layer2 layer3 layer4 avgpool \\
      --output-dir features/cifar10_resnet50

  # dry-run: 32 サンプルで動作確認する
  python scripts/extract_features.py \\
      --model resnet18 \\
      --dataset cifar10 \\
      --split test \\
      --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# モデル構築
# ---------------------------------------------------------------------------

# 対応モデルの (builder, デフォルト特徴次元) の対応表
_RESNET_FEATURE_DIMS = {
    "layer1": {"resnet18": 64, "resnet50": 256, "resnet101": 256},
    "layer2": {"resnet18": 128, "resnet50": 512, "resnet101": 512},
    "layer3": {"resnet18": 256, "resnet50": 1024, "resnet101": 1024},
    "layer4": {"resnet18": 512, "resnet50": 2048, "resnet101": 2048},
    "avgpool": {"resnet18": 512, "resnet50": 2048, "resnet101": 2048},
}


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """torchvision の事前学習済みモデルを構築する。

    なぜ weights="IMAGENET1K_V1" を使うか:
      複数バージョンある場合に再現性を確保するため。
      最新重みは精度が高いが再現性が下がる。
    """
    weights_map = {
        "resnet18":  models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50":  models.ResNet50_Weights.IMAGENET1K_V1,
        "resnet101": models.ResNet101_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b4": models.EfficientNet_B4_Weights.IMAGENET1K_V1,
    }

    if model_name not in weights_map and not model_name.startswith("vit"):
        raise ValueError(
            f"非対応のモデル: {model_name}。"
            f"対応モデル: {list(weights_map.keys()) + ['vit_b_16', 'vit_s_16']}"
        )

    if model_name.startswith("vit"):
        return _build_vit(model_name, pretrained)

    builder = getattr(models, model_name)
    weights = weights_map[model_name] if pretrained else None
    model = builder(weights=weights)
    model.eval()
    return model


def _build_vit(model_name: str, pretrained: bool) -> nn.Module:
    """ViT モデルを構築する（timm を使用）。

    なぜ timm を使うか:
      torchvision の ViT は ViT-B/16 のみ対応で、
      ViT-S/16 は timm 経由でないと取得できないため。
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "ViT モデルには timm が必要です: pip install timm"
        )
    name_map = {"vit_b_16": "vit_base_patch16_224", "vit_s_16": "vit_small_patch16_224"}
    timm_name = name_map.get(model_name, model_name)
    model = timm.create_model(timm_name, pretrained=pretrained)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 特徴抽出フック
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """forward hook を使って指定層の出力を捕捉する。

    なぜ hook を使うか:
      モデル本体を改変せずに任意の中間層出力を取得できるため。
      モデルの forward 処理はそのまま走るので、他の層への影響がない。
    """

    def __init__(self, model: nn.Module, layer_name: str) -> None:
        self.model = model
        self.layer_name = layer_name
        self._output: torch.Tensor | None = None
        self._hook = self._get_layer(layer_name).register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self._output = output.detach()

    def _get_layer(self, layer_name: str) -> nn.Module:
        """モデルから指定レイヤーを取得する。ネストされた名前（"layer1.0"）も対応。"""
        parts = layer_name.split(".")
        m = self.model
        for p in parts:
            m = getattr(m, p)
        return m

    def extract(self, x: torch.Tensor) -> np.ndarray:
        """入力 x を forward して指定層の出力を (n, d) の numpy 配列で返す。"""
        with torch.no_grad():
            self.model(x)
        feat = self._output
        if feat is None:
            raise RuntimeError(f"層 '{self.layer_name}' から出力を取得できませんでした")
        # (n, c, h, w) → global average pooling → (n, c)
        if feat.ndim == 4:
            feat = feat.mean(dim=(2, 3))
        # (n, c, 1, 1) → squeeze → (n, c)
        elif feat.ndim == 3:
            feat = feat.squeeze(-1)
        # (n, seq, d) の ViT などは [CLS] トークン（先頭）を使う
        elif feat.ndim == 3:
            feat = feat[:, 0, :]
        return feat.float().cpu().numpy()

    def remove(self) -> None:
        """hook を解除する。使用後は必ず呼ぶこと。"""
        self._hook.remove()


def extract_vit_cls(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """ViT の [CLS] トークンを特徴量として抽出する。

    なぜ [CLS] を使うか:
      ViT の [CLS] トークンは全パッチ情報を集約した大域的な表現であり、
      ResNet の avgpool に相当する位置づけを持つ。
    """
    with torch.no_grad():
        out = model.forward_features(x)
        # timm の ViT は (n, seq_len, d) を返し、先頭が [CLS]
        if out.ndim == 3:
            out = out[:, 0, :]
    return out.float().cpu().numpy()


# ---------------------------------------------------------------------------
# データセット構築
# ---------------------------------------------------------------------------

# CIFAR-10 / CIFAR-100 は 32×32 だが ResNet は 224×224 で事前学習されている。
# Resize して ImageNet 正規化を適用することで転移学習済み特徴を正しく取得する。
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRANSFORMS = {
    "cifar10":  T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]),
    "cifar100": T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]),
    "stl10":    T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]),
    "imagenet": T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]),
}


def build_dataset(dataset_name: str, split: str, data_root: str):
    """指定データセットの split を構築する。

    なぜ transform に Resize(224) を入れるか:
      ResNet / ViT は ImageNet（224×224）で学習されており、
      CIFAR-10（32×32）をそのまま入力すると特徴の品質が著しく落ちる。
    """
    transform = _TRANSFORMS.get(dataset_name)
    if transform is None:
        raise ValueError(f"非対応のデータセット: {dataset_name}。対応: {list(_TRANSFORMS.keys())}")

    root = Path(data_root)

    if dataset_name == "cifar10":
        train = split == "train"
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)

    if dataset_name == "cifar100":
        train = split == "train"
        return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)

    if dataset_name == "stl10":
        # stl10 の split は "train" / "test" / "unlabeled"
        return datasets.STL10(root=root, split=split, download=True, transform=transform)

    if dataset_name == "imagenet":
        # ImageNet は手動ダウンロードが必要なため data_root/imagenet/{train,val} を想定する
        split_dir = root / "imagenet" / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"ImageNet の {split} セットが見つかりません: {split_dir}\n"
                "ImageNet は手動ダウンロードが必要です。"
            )
        return datasets.ImageFolder(root=str(split_dir), transform=transform)

    raise ValueError(f"非対応のデータセット: {dataset_name}")


# ---------------------------------------------------------------------------
# 特徴量抽出メインロジック
# ---------------------------------------------------------------------------

def extract_features(
    model: nn.Module,
    loader: DataLoader,
    layer_name: str,
    device: torch.device,
    model_name: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """データローダー全体を走査して特徴量と sample_ids を返す。

    戻り値:
      features: shape (n, d)
      sample_ids: shape (n,) — データセット内のインデックス
    """
    model = model.to(device)
    model.eval()

    is_vit = model_name.startswith("vit")

    if not is_vit:
        extractor = FeatureExtractor(model, layer_name)
    else:
        extractor = None  # ViT は forward_features で直接取得する

    all_features = []
    all_ids = []
    offset = 0

    for batch_x, _ in loader:
        batch_x = batch_x.to(device)
        if is_vit:
            feat = extract_vit_cls(model, batch_x)
        else:
            feat = extractor.extract(batch_x)
        all_features.append(feat)
        all_ids.append(np.arange(offset, offset + len(feat)))
        offset += len(feat)
        logger.debug(f"抽出済み: {offset} サンプル")

    if extractor is not None:
        extractor.remove()

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    sample_ids = np.concatenate(all_ids, axis=0)
    logger.info(f"特徴量抽出完了: shape={features.shape}")
    return features, sample_ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="特徴量抽出スクリプト")
    p.add_argument("--model", required=True,
                   help="モデル名 (resnet18 / resnet50 / resnet101 / vit_b_16 / vit_s_16 / efficientnet_b0 / efficientnet_b4)")
    p.add_argument("--dataset", required=True,
                   help="データセット名 (cifar10 / cifar100 / stl10 / imagenet)")
    p.add_argument("--split", default="test",
                   help="データ分割 (train / test / val)")
    p.add_argument("--layer", nargs="+", default=["avgpool"],
                   help="抽出する層名（複数指定可）。ResNet 系: layer1 layer2 layer3 layer4 avgpool")
    p.add_argument("--output", default=None,
                   help="単一層時の出力ファイルパス (.npz)。未指定時は --output-dir を使用する")
    p.add_argument("--output-dir", default="features",
                   help="複数層抽出時の出力ディレクトリ")
    p.add_argument("--data-root", default="./data",
                   help="データセットのルートディレクトリ")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true",
                   help="事前学習なしのランダム重みを使用する（デバッグ用）")
    p.add_argument("--device", default=None,
                   help="使用デバイス (cpu / cuda / mps)。未指定時は自動検出")
    p.add_argument("--dry-run", action="store_true",
                   help="dry-run: 最初の --dry-run-samples サンプルのみ抽出して動作確認する")
    p.add_argument("--dry-run-samples", type=int, default=32,
                   help="dry-run 時のサンプル数（デフォルト: 32）")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _resolve_device(device_str: str | None) -> torch.device:
    """使用デバイスを解決する。"""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    device = _resolve_device(args.device)
    logger.info(f"使用デバイス: {device}")

    # モデルとデータセットの構築
    model = build_model(args.model, pretrained=not args.no_pretrained)
    dataset = build_dataset(args.dataset, args.split, args.data_root)
    logger.info(f"データセット: {args.dataset} {args.split}, n={len(dataset)}")

    # dry-run: 先頭 N サンプルのみ使用する
    if args.dry_run:
        n = min(args.dry_run_samples, len(dataset))
        dataset = Subset(dataset, list(range(n)))
        logger.warning(f"[dry-run] {n} サンプルで動作確認を実行します")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,       # sample_ids との対応を保つため shuffle しない
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 出力先の決定
    # 層が 1 つなら --output、複数なら --output-dir/{layer}.npz に保存する
    if len(args.layer) == 1 and args.output:
        output_paths = {args.layer[0]: Path(args.output)}
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {
            layer: out_dir / f"{args.dataset}_{args.model}_{layer}.npz"
            for layer in args.layer
        }

    # 層ごとに特徴量を抽出して保存する
    for layer_name, out_path in output_paths.items():
        logger.info(f"層 '{layer_name}' の特徴量を抽出しています...")
        features, sample_ids = extract_features(
            model, loader, layer_name, device, model_name=args.model
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path),
            features=features,
            sample_ids=sample_ids,
        )
        logger.info(f"保存しました: {out_path}  shape={features.shape}")


if __name__ == "__main__":
    main()
