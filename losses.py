"""GliGAN losses for the new 2D conditional pipeline.

Losses:
    - Hinge adversarial loss
    - Cell-masked L1 reconstruction loss
    - Background-masked L1 reconstruction loss

The small RGB label mask is NOT used here. It remains conditioning
information for G and D. The cell mask is the flood-fill mask:
1 = cell, 0 = black background.
"""

from __future__ import annotations

import torch
from torch import Tensor


def masked_l1(
    generated: Tensor,
    target: Tensor,
    mask: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Normalized L1 error inside a spatial mask."""
    if generated.shape != target.shape:
        raise ValueError(
            f"generated and target must have the same shape, "
            f"got {generated.shape} and {target.shape}"
        )

    if mask.ndim != 4:
        raise ValueError(
            f"mask must be [B,1,H,W] or [B,C,H,W], got {mask.shape}"
        )

    if mask.shape[0] != generated.shape[0]:
        raise ValueError("mask batch size must match generated/target")

    if mask.shape[-2:] != generated.shape[-2:]:
        raise ValueError("mask spatial size must match generated/target")

    mask = mask.to(device=generated.device, dtype=generated.dtype)

    # Broadcast [B,1,H,W] mask over all image channels.
    if mask.shape[1] == 1 and generated.shape[1] != 1:
        mask = mask.expand(-1, generated.shape[1], -1, -1)
    elif mask.shape[1] != generated.shape[1]:
        raise ValueError(
            f"mask channels must be 1 or {generated.shape[1]}, "
            f"got {mask.shape[1]}"
        )

    error = torch.abs(generated - target) * mask
    denominator = mask.sum().clamp_min(eps)

    return error.sum() / denominator


class GliGANLoss:
    """Hinge GAN + cell/background masked L1.

    Generator:
        L_G = lambda_adv * L_adv
            + lambda_cell * L_cell
            + lambda_bg * L_bg

    L_adv  = -mean(D(fake, label))
    L_cell = masked L1(fake, target, cell_mask)
    L_bg   = masked L1(fake, target, 1 - cell_mask)

    The RGB label remains a conditioning input to G and D.
    """

    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_cell: float = 1.0,
        lambda_bg: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        if lambda_adv < 0:
            raise ValueError("lambda_adv must be non-negative.")
        if lambda_cell < 0:
            raise ValueError("lambda_cell must be non-negative.")
        if lambda_bg < 0:
            raise ValueError("lambda_bg must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.lambda_adv = float(lambda_adv)
        self.lambda_cell = float(lambda_cell)
        self.lambda_bg = float(lambda_bg)
        self.eps = float(eps)

    def discriminator(
        self,
        real_scores: Tensor,
        fake_scores: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Hinge discriminator loss."""
        real_loss = torch.relu(1.0 - real_scores).mean()
        fake_loss = torch.relu(1.0 + fake_scores).mean()
        total = real_loss + fake_loss

        return total, real_loss, fake_loss

    def generator(
        self,
        generated: Tensor,
        original: Tensor,
        fake_scores: Tensor,
        cell_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generator loss.

        Args:
            generated: Generated image [B,C,H,W].
            original: Target/original image [B,C,H,W].
            fake_scores: D(generated, label).
            cell_mask: Flood-fill mask [B,1,H,W].
                       1 = cell, 0 = black background.

        Returns:
            total_loss, cell_loss, background_loss, adversarial_loss
        """
        if cell_mask.ndim != 4:
            raise ValueError(
                f"cell_mask must be [B,1,H,W], got {cell_mask.shape}"
            )

        cell_mask = cell_mask.to(
            device=generated.device,
            dtype=generated.dtype,
        )

        # Accept either binary [0,1] or image-style [0,255] masks.
        if cell_mask.max().detach() > 1.0:
            cell_mask = cell_mask / 255.0

        cell_mask = (cell_mask > 0.5).to(generated.dtype)
        background_mask = 1.0 - cell_mask

        cell_loss = masked_l1(
            generated, original, cell_mask, eps=self.eps
        )

        background_loss = masked_l1(
            generated, original, background_mask, eps=self.eps
        )

        # Hinge generator objective.
        adversarial_loss = -fake_scores.mean()

        total = (
            self.lambda_adv * adversarial_loss
            + self.lambda_cell * cell_loss
            + self.lambda_bg * background_loss
        )

        return total, cell_loss, background_loss, adversarial_loss


__all__ = ["GliGANLoss", "masked_l1"]