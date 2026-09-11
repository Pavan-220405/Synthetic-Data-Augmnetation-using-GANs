"""2D conditional GliGAN discriminator."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.utils import spectral_norm



def _concatenate_condition(image: Tensor, label: Tensor) -> Tensor:
    if image.ndim != 4 or label.ndim != 4:
        raise ValueError("Expected image and label tensors with shape [B, C, H, W].")
    if image.shape[0] != label.shape[0] or image.shape[2:] != label.shape[2:]:
        raise ValueError("Image and label batch/spatial dimensions must match.")
    return torch.cat((image, label), dim=1)


class Discriminator(nn.Module):
    """Ferreira-style five-layer spectral-normalized 2D discriminator.

    The discriminator receives the image and RGB label concatenated as
    channels. Its final output is one channel: a real/fake score. With the
    Set ``use_sigmoid=False`` for AMP-safe training with
    ``BCEWithLogitsLoss``. Enable sigmoid only when probabilities are needed
    for inference or explicit non-AMP ``BCELoss`` experiments.
    """

    def __init__(
        self,
        image_channels: int = 3,
        label_channels: int = 3,
        channel: int = 768,
        use_sigmoid: bool = False,
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.label_channels = label_channels
        self.in_channels = image_channels + label_channels
        self.channel = channel
        self.use_sigmoid = use_sigmoid

        self.conv1 = spectral_norm(
            nn.Conv2d(self.in_channels, channel // 16, kernel_size=4, stride=2, padding=1)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(channel // 16, channel // 8, kernel_size=4, stride=2, padding=1)
        )
        self.conv3 = spectral_norm(
            nn.Conv2d(channel // 8, channel // 4, kernel_size=4, stride=2, padding=1)
        )
        self.conv4 = spectral_norm(
            nn.Conv2d(channel // 4, channel // 2, kernel_size=4, stride=2, padding=1)
        )
        self.conv5 = spectral_norm(
            nn.Conv2d(channel // 2, channel, kernel_size=4, stride=2, padding=1)
        )
        # Ferreira uses a 3x3x3 final kernel in 3D; this adaptation is 3x3.
        self.conv6 = nn.Conv2d(channel, 1, kernel_size=3, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_or_condition: Tensor, label: Optional[Tensor] = None) -> Tensor:
        if label is not None:
            condition = _concatenate_condition(image_or_condition, label)
        else:
            condition = image_or_condition
            if condition.ndim != 4 or condition.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected concatenated input [B, {self.in_channels}, H, W]."
                )

        h = torch.nn.functional.leaky_relu(self.conv1(condition), negative_slope=0.2)
        h = torch.nn.functional.leaky_relu(self.conv2(h), negative_slope=0.2)
        h = torch.nn.functional.leaky_relu(self.conv3(h), negative_slope=0.2)
        h = torch.nn.functional.leaky_relu(self.conv4(h), negative_slope=0.2)
        h = torch.nn.functional.leaky_relu(self.conv5(h), negative_slope=0.2)
        output = self.conv6(h)
        return self.sigmoid(output) if self.use_sigmoid else output


def build_discriminator(**kwargs) -> Discriminator:
    return Discriminator(**kwargs)


__all__ = ["Discriminator", "build_discriminator"]
