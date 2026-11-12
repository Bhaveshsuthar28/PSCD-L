
from pathlib import Path
import argparse
import torch

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description="Validate cached VL-CMU-CD DINOv2 features."
)
parser.add_argument(
    "--split",
    choices=("train", "test"),
    default="train",
    help="Dataset split to validate (default: train).",
)
args = parser.parse_args()

CACHE_DIR = (
    ROOT / "data" / "processed" / "vl_cmu_cd_v2" / args.split
)

EXPECTED_SHAPE = (1, 768, 37, 37)
EXPECTED_PREPROCESSING = (
    "rgb_imagenet_normalize_pad_bottom_right_to_multiple_14_v1"
)

files = sorted(CACHE_DIR.glob("*.pt"))

if not files:
    raise RuntimeError(f"No cache files found in {CACHE_DIR}")

errors = []

for path in files:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)

        if payload.get("dataset") != "VL-CMU-CD-binary255":
            errors.append(f"{path.name}: dataset metadata mismatch")

        if payload.get("feature_model") != "facebook/dinov2-base":
            errors.append(f"{path.name}: feature_model metadata mismatch")

        if payload.get("feature_dim") != 768:
            errors.append(f"{path.name}: feature_dim metadata mismatch")

        if payload.get("patch_size") != 14:
            errors.append(f"{path.name}: patch_size metadata mismatch")

        if payload.get("split") != args.split:
            errors.append(f"{path.name}: split metadata mismatch")

        if payload.get("preprocessing") != EXPECTED_PREPROCESSING:
            errors.append(
                f"{path.name}: preprocessing metadata mismatch: "
                f"{payload.get('preprocessing')!r}"
            )
            continue

        if payload.get("original_size") != [512, 512]:
            errors.append(f"{path.name}: incorrect original_size")

        if payload.get("padded_size") != [518, 518]:
            errors.append(f"{path.name}: incorrect padded_size")

        for key in ("t0", "t1"):
            tensor = payload.get(key)

            if not isinstance(tensor, torch.Tensor):
                errors.append(
                    f"{path.name}: {key} is missing or not a tensor"
                )
                continue

            if tuple(tensor.shape) != EXPECTED_SHAPE:
                errors.append(
                    f"{path.name}: {key} shape is {tuple(tensor.shape)}"
                )

            if not torch.isfinite(tensor).all().item():
                errors.append(
                    f"{path.name}: {key} contains non-finite values"
                )

        print(f"Checked: {path.name}")

    except Exception as exc:
        errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

print(f"\nSplit: {args.split}")
print(f"Files checked: {len(files)}")

if errors:
    print("VALIDATION FAILED:")
    for error in errors:
        print(f"- {error}")
    raise SystemExit(1)

print("FEATURE CACHE VALIDATION PASSED")
