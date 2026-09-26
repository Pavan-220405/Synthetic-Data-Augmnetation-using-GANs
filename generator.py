"""2D conditional NPPGAN/GliGAN generator.

This is the earlier convolutional NPPGAN-style generator adapted to the
current GliGAN training pipeline.

Current interface:
    noisy RGB image: [B, 3, H, W]
    RGB label:       [B, 3, H, W]
    conditioning:    [B, 6, H, W]
    generated image: [B, 3, H, W]

The generator keeps the original convolutional/residual/attention topology:
    7x7 conv -> downsample -> 4 residual blocks -> self-attention ->
    spatial attention -> upsample -> 7x7 conv.

The current dataset uses RGB tensors normalized to [0, 1], so the final
activation is Sigmoid rather than the old Tanh output in [-1, 1].

The constructor intentionally accepts the same arguments expected by the
current train.py (feature_size, use_checkpoint, image_size), even though
feature_size/use_checkpoint are not architectural controls for this
convolutional generator.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def concatenate_condition(image: Tensor, label: Tensor) -> Tensor:
    """Concatenate the RGB image and RGB label into a 6-channel condition."""

    if image.ndim != 4 or label.ndim != 4:
        raise ValueError(
            "Expected image and label tensors with shape [B, C, H, W]."
        )

    if image.shape[0] != label.shape[0] or image.shape[2:] != label.shape[2:]:
        raise ValueError("Image and label batch/spatial dimensions must match.")

    return torch.cat((image, label), dim=1)


class ResidualBlock(nn.Module):
    """Two 3x3 convolution blocks with an identity skip connection."""

    def __init__(self, channels: int = 256, eps: float = 1e-5) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels, eps=eps, affine=True),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels, eps=eps, affine=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class SelfAttention(nn.Module):
    """Spatial self-attention applied at the compact bottleneck."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        attention_channels = max(1, channels // 8)
        self.query = nn.Conv2d(channels, attention_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, attention_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        locations = height * width

        query = self.query(x).view(batch, -1, locations).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, locations)
        attention = self.softmax(torch.bmm(query, key))

        value = self.value(x).view(batch, channels, locations)
        attended = torch.bmm(value, attention.permute(0, 2, 1))
        attended = attended.view(batch, channels, height, width)

        return x + self.gamma * attended


class SpatialAttention(nn.Module):
    """Channel-pooled spatial attention gate."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SpatialAttention kernel_size must be odd.")

        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        average_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        attention = self.sigmoid(
            self.conv(torch.cat((average_map, max_map), dim=1))
        )
        return x * attention


class NPPGenerator(nn.Module):
    """Conditional 2D NPPGAN generator used by the current GliGAN pipeline.

    Accepts either ``generator(noisy_image, label)`` or an already
    concatenated six-channel condition.

    The topology is resolution-flexible for compatible even spatial sizes;
    the current training pipeline uses 96x96.
    """

    def __init__(
        self,
        image_channels: int = 3,
        label_channels: int = 3,
        out_channels: int = 3,
        feature_size: int = 48,
        use_checkpoint: bool = False,
        image_size: int | tuple[int, int] = 96,
    ) -> None:
        super().__init__()

        if image_channels <= 0:
            raise ValueError("image_channels must be positive.")
        if label_channels <= 0:
            raise ValueError("label_channels must be positive.")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        self.image_channels = image_channels
        self.label_channels = label_channels
        self.in_channels = image_channels + label_channels
        self.out_channels = out_channels

        # Kept for compatibility with the existing TrainConfig interface.
        # The NPPGAN convolutional architecture uses fixed 64/128/256 widths.
        self.feature_size = feature_size
        self.use_checkpoint = use_checkpoint
        self.image_size = image_size

        # c7s1-64
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                self.in_channels,
                64,
                kernel_size=7,
                stride=1,
                padding=0,
            ),
            nn.InstanceNorm2d(64, eps=1e-5, affine=True),
            nn.ReLU(inplace=False),
        )

        # d128 -> d256
        # Gives 96 -> 48 -> 24 for the current training resolution.
        self.downsample = nn.Sequential(
            nn.ReflectionPad2d((0, 1, 0, 1)),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(128, eps=1e-5, affine=True),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d((0, 1, 0, 1)),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(256, eps=1e-5, affine=True),
            nn.ReLU(inplace=False),
        )

        # Four residual blocks as in the earlier generator.
        self.residual_blocks = nn.Sequential(
            *(ResidualBlock(256) for _ in range(4))
        )

        self.attention = SelfAttention(256)
        self.spatial_attention = SpatialAttention()

        # u128 -> u64. output_padding=1 gives 24 -> 48 -> 96.
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(128, eps=1e-5, affine=True),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.InstanceNorm2d(64, eps=1e-5, affine=True),
            nn.ReLU(inplace=False),
        )

        # Dataset tensors are in [0, 1], so Sigmoid keeps G's output in the
        # same range as the target image and the current L1 losses.
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                64,
                self.out_channels,
                kernel_size=7,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        image_or_condition: Tensor,
        label: Tensor | None = None,
    ) -> Tensor:
        """Generate an RGB image from noisy RGB image and RGB label."""

        if label is not None:
            if image_or_condition.ndim != 4 or label.ndim != 4:
                raise ValueError(
                    "Expected image and label tensors with shape [B, C, H, W]."
                )

            if image_or_condition.shape[1] != self.image_channels:
                raise ValueError(
                    f"Expected image with {self.image_channels} channels, "
                    f"got {image_or_condition.shape[1]}."
                )

            if label.shape[1] != self.label_channels:
                raise ValueError(
                    f"Expected label with {self.label_channels} channels, "
                    f"got {label.shape[1]}."
                )

            condition = concatenate_condition(image_or_condition, label)
        else:
            condition = image_or_condition

            if condition.ndim != 4:
                raise ValueError(
                    "Expected concatenated condition with shape [B, C, H, W]."
                )

            if condition.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected concatenated input with {self.in_channels} "
                    f"channels, got {condition.shape[1]}."
                )

        x = self.initial(condition)
        x = self.downsample(x)
        x = self.residual_blocks(x)
        x = self.attention(x)
        x = self.spatial_attention(x)
        x = self.upsample(x)
        return self.final(x)


# Generic name expected by train.py.
Generator = NPPGenerator


def build_generator(**kwargs) -> Generator:
    """Factory compatible with the current generator interface."""
    return Generator(**kwargs)


__all__ = [
    "NPPGenerator",
    "Generator",
    "ResidualBlock",
    "SelfAttention",
    "SpatialAttention",
    "build_generator",
    "concatenate_condition",
]


if __name__ == "__main__":
    model = NPPGenerator(
        image_channels=3,
        label_channels=3,
        out_channels=3,
        feature_size=48,
        use_checkpoint=False,
        image_size=96,
    )

    noisy = torch.randn(2, 3, 96, 96)
    label = torch.randn(2, 3, 96, 96)

    with torch.no_grad():
        output = model(noisy, label)

    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"noisy:  {tuple(noisy.shape)}")
    print(f"label:  {tuple(label.shape)}")
    print(f"output: {tuple(output.shape)}")
    print(
        f"output range: [{output.min().item():.4f}, "
        f"{output.max().item():.4f}]"
    )
