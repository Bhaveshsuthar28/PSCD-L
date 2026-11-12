
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.models.backbone import DINOv2Backbone

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Dataset" / "VL-CMU-CD-binary255"

image_paths = sorted(
    path
    for path in DATASET.rglob("*.png")
    if path.parent.name.lower() != "mask"
    and "mask" not in path.stem.lower()
)

if not image_paths:
    raise FileNotFoundError(
        f"No PNG images found under {DATASET}"
    )

image_path = image_paths[0]
print(f"Test image: {image_path}")

image = Image.open(image_path).convert("RGB")
image = TF.resize(
    image,
    [518, 518],
    interpolation=InterpolationMode.BICUBIC,
)

tensor = TF.to_tensor(image)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
tensor = ((tensor - mean) / std).unsqueeze(0)

backbone = DINOv2Backbone()
features = backbone(tensor)

print(f"Feature shape: {tuple(features.shape)}")
print(f"Feature dtype: {features.dtype}")
print(f"Feature device: {features.device}")
print(f"All finite: {torch.isfinite(features).all().item()}")
print(
    "All parameters frozen:",
    all(not p.requires_grad for p in backbone.parameters()),
)

assert features.shape == (1, 768, 37, 37)
assert torch.isfinite(features).all()
assert all(not p.requires_grad for p in backbone.parameters())

print("DINOv2 smoke test PASSED")
