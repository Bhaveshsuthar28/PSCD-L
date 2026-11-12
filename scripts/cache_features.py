
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.backbone import DINOv2Backbone


DATASET_ROOT = ROOT / "Dataset" / "VL-CMU-CD-binary255"
CACHE_ROOT = ROOT / "data" / "processed" / "vl_cmu_cd_v2"


def load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = TF.to_tensor(image)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    height, width = tensor.shape[-2:]
    pad_h = (-height) % 14
    pad_w = (-width) % 14

    # Pad only the bottom and right edges.
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)

    return tensor.unsqueeze(0)



def get_pairs(split: str) -> list[tuple[Path, Path, str]]:
    t0_dir = DATASET_ROOT / split / "t0"
    t1_dir = DATASET_ROOT / split / "t1"

    if not t0_dir.is_dir() or not t1_dir.is_dir():
        raise FileNotFoundError(f"Missing t0/t1 folders for split: {split}")

    pairs = []
    for t0_path in sorted(t0_dir.glob("*.png")):
        t1_path = t1_dir / t0_path.name
        if t1_path.is_file():
            pairs.append((t0_path, t1_path, t0_path.stem))

    if not pairs:
        raise RuntimeError(f"No paired PNG files found for split: {split}")

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    pairs = get_pairs(args.split)
    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    if args.limit:
        pairs = pairs[:args.limit]

    output_dir = CACHE_ROOT / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Split: {args.split}")
    print(f"Pairs selected: {len(pairs)}")
    print(f"Output: {output_dir}")

    model = DINOv2Backbone()
    saved = skipped = failed = 0

    try:
        for index, (t0_path, t1_path, sample_id) in enumerate(pairs, start=1):
            cache_path = output_dir / f"{sample_id}.pt"

            
            if cache_path.exists():
                try:
                    cached = torch.load(
                        cache_path,
                        map_location="cpu",
                        weights_only=True,
                    )

                    valid = (
                        cached.get("sample") == sample_id
                        and cached.get("dataset") == "VL-CMU-CD-binary255"
                        and cached.get("feature_model") == "facebook/dinov2-base"
                        and cached.get("feature_dim") == 768
                        and cached.get("patch_size") == 14
                        and cached.get("preprocessing")
                        == "rgb_imagenet_normalize_pad_bottom_right_to_multiple_14_v1"
                        and cached.get("original_size") == [512, 512]
                        and cached.get("padded_size") == [518, 518]
                        and isinstance(cached.get("t0"), torch.Tensor)
                        and isinstance(cached.get("t1"), torch.Tensor)
                        and tuple(cached["t0"].shape) == (1, 768, 37, 37)
                        and tuple(cached["t1"].shape) == (1, 768, 37, 37)
                        and torch.isfinite(cached["t0"]).all().item()
                        and torch.isfinite(cached["t1"]).all().item()
                    )

                    if valid:
                        skipped += 1
                        print(f"[{index}/{len(pairs)}] cached: {sample_id}")
                        continue

                    print(
                        f"[{index}/{len(pairs)}] cache invalid or outdated; "
                        f"regenerating: {sample_id}"
                    )

                except Exception as exc:
                    print(
                        f"[{index}/{len(pairs)}] cache unreadable; "
                        f"regenerating {sample_id}: {exc}"
                    )


            try:
                f0 = model(load_image(t0_path)).cpu()
                f1 = model(load_image(t1_path)).cpu()

                if f0.shape != f1.shape:
                    raise ValueError(f"Feature shape mismatch for {sample_id}")

                if not torch.isfinite(f0).all() or not torch.isfinite(f1).all():
                    raise ValueError(f"Non-finite features for {sample_id}")

                payload = {
                    "sample": sample_id,
                    "dataset": "VL-CMU-CD-binary255",
                    "split": args.split,
                    "feature_model": "facebook/dinov2-base",
                    "feature_dim": 768,
                    "patch_size": 14,
                    "preprocessing": "rgb_imagenet_normalize_pad_bottom_right_to_multiple_14_v1",
                    "original_size": [512, 512],
                    "padded_size": [518, 518],
                    "t0": f0,
                    "t1": f1,
                }

                temporary_path = cache_path.with_suffix(".tmp")
                torch.save(payload, temporary_path)
                temporary_path.replace(cache_path)

                saved += 1
                print(f"[{index}/{len(pairs)}] saved: {sample_id}")

                del f0, f1

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                failed += 1
                print(f"[{index}/{len(pairs)}] FAILED: {sample_id}: {exc}")

    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nSummary")
    print(f"Saved: {saved}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

    if failed:
        raise RuntimeError(f"{failed} feature pairs failed")


if __name__ == "__main__":
    main()
