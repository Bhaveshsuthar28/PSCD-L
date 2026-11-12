
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.backbone import DINOv2Backbone


DATASET = ROOT / "Dataset" / "VL-CMU-CD-binary255"
SPLIT = "test"
SAMPLE = "000_1_00_0.png"


def load_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = TF.resize(
        image,
        [518, 518],
        interpolation=InterpolationMode.BICUBIC,
    )

    tensor = TF.to_tensor(image)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    return ((tensor - mean) / std).unsqueeze(0)


def main() -> None:
    t0_path = DATASET / SPLIT / "t0" / SAMPLE
    t1_path = DATASET / SPLIT / "t1" / SAMPLE

    for path in (t0_path, t1_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"Sample: {SAMPLE}")
    print("Loading DINOv2...")

    model = DINOv2Backbone()
    features = []

    for label, path in (("t0", t0_path), ("t1", t1_path)):
        image = load_image(path)
        feature = model(image)
        features.append(feature.cpu())
        del image, feature

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"{label}: extracted")

    f0, f1 = features

    print(f"t0 feature shape: {tuple(f0.shape)}")
    print(f"t1 feature shape: {tuple(f1.shape)}")
    print(f"Features finite: {torch.isfinite(f0).all() and torch.isfinite(f1).all()}")
    print(f"Matching shapes: {f0.shape == f1.shape}")

    assert f0.shape == f1.shape == (1, 768, 37, 37)
    assert torch.isfinite(f0).all()
    assert torch.isfinite(f1).all()

    difference = (f1.float() - f0.float()).abs()
    print(f"Mean absolute feature difference: {difference.mean().item():.6f}")

    assert difference.mean().item() > 0
    print("PAIRED FEATURE TEST PASSED")


if __name__ == "__main__":
    main()
