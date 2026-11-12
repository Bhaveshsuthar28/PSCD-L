import importlib.util
import platform
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

print(f"Project root: {ROOT}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("\nDependencies:")
for package in ("torchvision", "transformers", "PIL", "numpy"):
    print(f"{package}: {'installed' if importlib.util.find_spec(package) else 'missing'}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\nProject folders:")
for folder in (
    "Dataset",
    "configs",
    "data",
    "src/models",
    "src/pipeline",
    "scripts",
    "tests",
    "outputs",
):
    path = ROOT / folder
    print(f"{folder}: {'exists' if path.is_dir() else 'missing'}")

print("\nDataset folders:")
dataset_root = ROOT / "Dataset"

if dataset_root.is_dir():
    for path in sorted(dataset_root.iterdir()):
        if path.is_dir():
            print(f"{path.name}/")
else:
    print("Dataset/ not found; no files were modified.")