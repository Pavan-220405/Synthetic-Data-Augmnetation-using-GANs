"""Ferreira-style GliGAN losses."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def progressive_reconstruction_weight(
    epoch: int,
    maximum_weight: float,
    progression_epochs: int = 1000,
) -> float:
    """Linearly increase L1 weight from 1 to ``maximum_weight``."""

    if maximum_weight < 1:
        raise ValueError("maximum_weight must be at least 1.")
    if progression_epochs <= 0:
        return float(maximum_weight)
    weight = 1.0 + ((maximum_weight - 1.0) / progression_epochs) * epoch
    return min(float(maximum_weight), weight)


class GliGANLoss:
    """BCE adversarial losses plus L1 reconstruction loss.

    The default weighting follows Ferreira's first-stage setup. Set
    ``reconstruction_max_weight=100`` for the second-stage schedule.
    """

    def __init__(
        self,
        reconstruction_max_weight: float = 5.0,
        progression_epochs: int = 1000,
        from_logits: bool = False,
    ) -> None:
        if reconstruction_max_weight < 1:
            raise ValueError("reconstruction_max_weight must be at least 1.")
        if progression_epochs < 0:
            raise ValueError("progression_epochs cannot be negative.")
        self.adversarial = nn.BCEWithLogitsLoss() if from_logits else nn.BCELoss()
        self.reconstruction = nn.L1Loss()
        self.reconstruction_max_weight = reconstruction_max_weight
        self.progression_epochs = progression_epochs
        self.from_logits = from_logits

    def discriminator(
        self,
        real_scores: Tensor,
        fake_scores: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        real_loss = self.adversarial(real_scores, torch.ones_like(real_scores))
        fake_loss = self.adversarial(fake_scores, torch.zeros_like(fake_scores))
        return real_loss + fake_loss, real_loss, fake_loss

    def generator(
        self,
        generated: Tensor,
        original: Tensor,
        fake_scores: Tensor,
        epoch: int,
    ) -> tuple[Tensor, Tensor, Tensor, float]:
        reconstruction_loss = self.reconstruction(generated, original)
        adversarial_loss = self.adversarial(fake_scores, torch.ones_like(fake_scores))
        weight = progressive_reconstruction_weight(
            epoch,
            maximum_weight=self.reconstruction_max_weight,
            progression_epochs=self.progression_epochs,
        )
        total = reconstruction_loss * weight + adversarial_loss / weight
        return total, reconstruction_loss, adversarial_loss, weight


__all__ = ["GliGANLoss", "progressive_reconstruction_weight"]
