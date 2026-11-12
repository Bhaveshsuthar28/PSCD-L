
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
CACHE_DIR = ROOT / "data" / "processed" / "vl_cmu_cd" / "test"
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


def extract(model: DINOv2Backbone, path: Path) -> torch.Tensor:
    return model(load_image(path)).cpu()


def main() -> None:
    t0_path = DATASET / "test" / "t0" / SAMPLE
    t1_path = DATASET / "test" / "t1" / SAMPLE

    if not t0_path.is_file() or not t1_path.is_file():
        raise FileNotFoundError("The paired images were not found.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{Path(SAMPLE).stem}.pt"

    model = DINOv2Backbone()

    try:
        f0 = extract(model, t0_path)
        f1 = extract(model, t1_path)

        assert f0.shape == f1.shape == (1, 768, 37, 37)
        assert torch.isfinite(f0).all() and torch.isfinite(f1).all()

        payload = {
            "sample": SAMPLE,
            "dataset": "VL-CMU-CD-binary255",
            "split": "test",
            "feature_model": "facebook/dinov2-base",
            "feature_dim": 768,
            "patch_size": 14,
            "t0": f0,
            "t1": f1,
        }

        torch.save(payload, cache_path)
        print(f"Saved: {cache_path}")
        print(f"File size: {cache_path.stat().st_size / 1024:.1f} KB")

        loaded = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        assert loaded["sample"] == SAMPLE
        assert loaded["t0"].shape == f0.shape
        assert loaded["t1"].shape == f1.shape
        assert torch.equal(loaded["t0"], f0)
        assert torch.equal(loaded["t1"], f1)

        print("Cache reload: passed")
        print("Tensor equality: passed")
        print("FEATURE CACHE TEST PASSED")

    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
