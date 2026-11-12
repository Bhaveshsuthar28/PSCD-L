
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONVOLUTION BLOCK
# ============================================================

class ConvBlock(nn.Module):
    """
    Two convolution layers with GroupNorm and GELU.

    GroupNorm is used instead of BatchNorm because training
    may use small batch sizes on the RTX 3050.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

        self.dropout = (
            nn.Dropout2d(dropout)
            if dropout > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))


# ============================================================
# CHANGE DETECTION DECODER
# ============================================================

class ChangeDetectionDecoder(nn.Module):
    """
    Lightweight DINOv2 feature-fusion decoder.

    Inputs:
        t0: [B, 768, 37, 37]
        t1: [B, 768, 37, 37]

    Processing:
        1. Absolute feature difference.
        2. Cosine distance map.
        3. Feature projection.
        4. Feature fusion.
        5. Progressive upsampling.

    Output:
        logits: [B, 1, 512, 512]

    Note:
        Output contains raw logits.
        Apply sigmoid only for inference or visualization.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        feature_channels: int = 64,
        output_channels: int = 1,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim

        # ----------------------------------------------------
        # FEATURE PROJECTION
        # ----------------------------------------------------

        self.feature_projection = nn.Sequential(
            nn.Conv2d(
                feature_dim,
                feature_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(8, feature_channels),
            nn.GELU(),
        )

        # Input to fusion:
        # 64 projected difference channels + 1 cosine channel.
        fusion_channels = feature_channels + 1

        # ----------------------------------------------------
        # INITIAL FUSION
        # ----------------------------------------------------

        self.fusion = ConvBlock(
            in_channels=fusion_channels,
            out_channels=128,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # PROGRESSIVE UPSAMPLING
        # ----------------------------------------------------

        self.decoder_74 = ConvBlock(
            in_channels=128,
            out_channels=96,
            dropout=dropout,
        )

        self.decoder_148 = ConvBlock(
            in_channels=96,
            out_channels=64,
            dropout=dropout,
        )

        self.decoder_296 = ConvBlock(
            in_channels=64,
            out_channels=32,
            dropout=dropout,
        )

        self.decoder_512 = ConvBlock(
            in_channels=32,
            out_channels=16,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # PREDICTION HEAD
        # ----------------------------------------------------

        self.prediction_head = nn.Sequential(
            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                16,
                output_channels,
                kernel_size=1,
            ),
        )

        self._initialize_weights()

    # ========================================================
    # WEIGHT INITIALIZATION
    # ========================================================

    def _initialize_weights(self) -> None:
        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ========================================================
    # COSINE DISTANCE
    # ========================================================

    @staticmethod
    def cosine_distance(
        t0: torch.Tensor,
        t1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate pixel-wise cosine distance.

        Input:
            [B, C, H, W]

        Output:
            [B, 1, H, W]

        Formula:
            distance = 1 - cosine_similarity
        """

        similarity = F.cosine_similarity(
            t0.float(),
            t1.float(),
            dim=1,
            eps=1e-6,
        )

        distance = 1.0 - similarity

        return distance.unsqueeze(1)

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def _validate_inputs(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
    ) -> None:

        if t0.ndim != 4 or t1.ndim != 4:
            raise ValueError(
                "Expected 4D tensors [B, C, H, W]."
            )

        if t0.shape != t1.shape:
            raise ValueError(
                f"Feature shape mismatch: "
                f"{tuple(t0.shape)} vs {tuple(t1.shape)}"
            )

        if t0.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} channels, "
                f"received {t0.shape[1]}."
            )

        if t0.shape[-2:] != (37, 37):
            raise ValueError(
                f"Expected 37x37 feature maps, "
                f"received {tuple(t0.shape[-2:])}."
            )

        if t0.device != t1.device:
            raise ValueError(
                "Both feature tensors must be on the same device."
            )

        if not torch.is_floating_point(t0):
            raise TypeError("t0 must be a floating-point tensor.")

        if not torch.is_floating_point(t1):
            raise TypeError("t1 must be a floating-point tensor.")

    # ========================================================
    # UPSAMPLING
    # ========================================================

    @staticmethod
    def upsample(
        x: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:

        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
    ) -> torch.Tensor:

        self._validate_inputs(t0, t1)

        # ----------------------------------------------------
        # 1. Absolute feature difference
        # ----------------------------------------------------

        difference = torch.abs(
            t0.float() - t1.float()
        )

        # ----------------------------------------------------
        # 2. Cosine distance
        # ----------------------------------------------------

        cosine_map = self.cosine_distance(
            t0,
            t1,
        )

        # ----------------------------------------------------
        # 3. Project feature difference
        # ----------------------------------------------------

        projected_difference = self.feature_projection(
            difference
        )

        # ----------------------------------------------------
        # 4. Feature fusion
        # ----------------------------------------------------

        fused = torch.cat(
            [
                projected_difference,
                cosine_map,
            ],
            dim=1,
        )

        x = self.fusion(fused)

        # ----------------------------------------------------
        # 5. Progressive decoder
        # ----------------------------------------------------

        x = self.upsample(x, (74, 74))
        x = self.decoder_74(x)

        x = self.upsample(x, (148, 148))
        x = self.decoder_148(x)

        x = self.upsample(x, (296, 296))
        x = self.decoder_296(x)

        x = self.upsample(x, (512, 512))
        x = self.decoder_512(x)

        # ----------------------------------------------------
        # 6. Change prediction logits
        # ----------------------------------------------------

        logits = self.prediction_head(x)

        return logits


# ============================================================
# MODEL SUMMARY
# ============================================================

def count_parameters(
    model: nn.Module,
) -> dict[str, int]:

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
    }
