from PIL import Image
import numpy as np


def load_image(path):
    image = Image.open(path).convert("RGB")
    return image


def load_mask(path, rule):
    mask = Image.open(path).convert("L")
    array = np.asarray(mask)

    if rule == "binary_0_255":
        array = (array > 0).astype(np.float32)

    elif rule == "mask > 0":
        array = (array > 0).astype(np.float32)

    else:
        raise ValueError(f"Unknown mask rule: {rule}")

    return array