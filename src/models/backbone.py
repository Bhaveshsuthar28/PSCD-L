
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Dinov2Model


class DINOv2Backbone(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str | None = None,
    ) -> None:
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device_name = device
        self.patch_size = 14
        self.feature_dim = 768

        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.to(device)

        if device.startswith("cuda"):
            self.model.to(dtype=torch.float16)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected [B, 3, H, W], got {tuple(images.shape)}"
            )

        _, _, height, width = images.shape

        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Dimensions must be divisible by {self.patch_size}; "
                f"got {height}x{width}"
            )

        images = images.to(
            device=self.device_name,
            dtype=next(self.model.parameters()).dtype,
        )

        outputs = self.model(
            pixel_values=images,
            interpolate_pos_encoding=True,
        )

        tokens = outputs.last_hidden_state[:, 1:, :]
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        return tokens.transpose(1, 2).reshape(
            images.shape[0],
            self.feature_dim,
            grid_h,
            grid_w,
        )
