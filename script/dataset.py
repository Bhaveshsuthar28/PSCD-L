import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ChangeDetectionDataset(Dataset):

    def __init__(self, index_path, image_size=518):
        with open(index_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def _resize_image(self, image):
        scale = self.image_size / min(image.size)

        width = round(image.width * scale)
        height = round(image.height * scale)

        return image.resize(
            (width, height),
            Image.Resampling.BICUBIC
        )

    def _resize_mask(self, mask):
        scale = self.image_size / min(mask.size)

        width = round(mask.width * scale)
        height = round(mask.height * scale)

        return mask.resize(
            (width, height),
            Image.Resampling.NEAREST
        )

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        image = self._resize_image(image)

        array = np.asarray(image, dtype=np.float32)
        array = array / 255.0

        tensor = torch.from_numpy(array)
        tensor = tensor.permute(2, 0, 1)

        return tensor

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        mask = self._resize_mask(mask)

        array = np.asarray(mask, dtype=np.float32)
        array = (array > 0).astype(np.float32)

        tensor = torch.from_numpy(array)
        tensor = tensor.unsqueeze(0)

        return tensor

    def __getitem__(self, index):
        sample = self.samples[index]

        t0 = self._load_image(sample["t0_path"])
        t1 = self._load_image(sample["t1_path"])
        mask = self._load_mask(sample["mask_path"])

        return {
            "t0": t0,
            "t1": t1,
            "mask": mask,
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
        }