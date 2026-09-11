"""2D GliGAN generator.

The generator follows Ferreira et al.'s SwinUNETR-based generator, with the
3D spatial dimension changed to 2D. Conditioning is explicit: the RGB image and
RGB label are concatenated along the channel dimension before entering the
network.
"""

from __future__ import annotations

import inspect
from typing import Optional

import torch
from torch import Tensor, nn

def concatenate_condition(image: Tensor, label: Tensor) -> Tensor:
    """Return the explicitly channel-concatenated image/label condition."""

    if image.ndim != 4 or label.ndim != 4:
        raise ValueError(
            "Expected image and label tensors with shape [B, C, H, W]."
        )
    if image.shape[0] != label.shape[0] or image.shape[2:] != label.shape[2:]:
        raise ValueError("Image and label batch/spatial dimensions must match.")
    return torch.cat((image, label), dim=1)


class Generator(nn.Module):
    """SwinUNETR generator for 3-channel images and 3-channel RGB labels."""

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
        self.image_channels = image_channels
        self.label_channels = label_channels
        self.in_channels = image_channels + label_channels
        self.out_channels = out_channels

        try:
            from monai.networks.nets import SwinUNETR
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "Generator requires MONAI. Install it with `pip install monai`."
            ) from exc

        network_kwargs = dict(
            spatial_dims=2,
            in_channels=self.in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )
        # MONAI versions before the API change require img_size; newer
        # versions infer the spatial size and reject that argument.
        if "img_size" in inspect.signature(SwinUNETR).parameters:
            size = (image_size, image_size) if isinstance(image_size, int) else image_size
            network_kwargs["img_size"] = size
        self.network = SwinUNETR(**network_kwargs)

    def forward(self, image_or_condition: Tensor, label: Optional[Tensor] = None) -> Tensor:
        """Generate an image from ``[image, label]`` or an already-concatenated input."""

        if label is not None:
            condition = concatenate_condition(image_or_condition, label)
        else:
            condition = image_or_condition
            if condition.ndim != 4 or condition.shape[1] != self.in_channels:
                raise ValueError(
                    f"Expected concatenated input [B, {self.in_channels}, H, W]."
                )
        return self.network(condition)


def build_generator(**kwargs) -> Generator:
    """Factory kept small so notebooks and training scripts can share setup."""

    return Generator(**kwargs)


__all__ = ["Generator", "build_generator", "concatenate_condition"]
