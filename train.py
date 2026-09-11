"""Training loop for the 2D conditional GliGAN baseline.

The dataloader is intentionally supplied by the caller. A batch may be a
mapping with ``original_image``, ``noisy_image`` and ``label`` keys, or a tuple
``(original_image, noisy_image, label)``. This keeps YOLO preprocessing
independent from the GAN implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn

from discriminator import Discriminator
from generator import Generator
from losses import GliGANLoss


@dataclass
class TrainConfig:
    image_channels: int = 3
    label_channels: int = 3
    generator_in_channels: int = 6
    generator_out_channels: int = 3
    discriminator_in_channels: int = 6
    image_size: int = 96
    feature_size: int = 48
    discriminator_channels: int = 768
    use_checkpoint: bool = False
    # Keep discriminator outputs as logits during training. This pairs with
    # BCEWithLogitsLoss and is safe under CUDA autocast/AMP.
    use_sigmoid: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple[float, float] = (0.5, 0.999)
    epochs: int = 100
    generator_updates: int = 2
    discriminator_updates: int = 1
    reconstruction_max_weight: float = 5.0
    reconstruction_progression_epochs: int = 1000
    amp: bool = True
    show_progress: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        if self.image_channels <= 0 or self.label_channels <= 0:
            raise ValueError("image_channels and label_channels must be positive.")
        expected_condition_channels = self.image_channels + self.label_channels
        if self.generator_in_channels != expected_condition_channels:
            raise ValueError(
                "generator_in_channels must equal image_channels + label_channels."
            )
        if self.discriminator_in_channels != expected_condition_channels:
            raise ValueError(
                "discriminator_in_channels must equal image_channels + label_channels."
            )
        if self.generator_out_channels != self.image_channels:
            raise ValueError("generator_out_channels must match image_channels.")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32 for SwinUNETR.")
        if self.discriminator_channels < 16 or self.discriminator_channels % 16 != 0:
            raise ValueError("discriminator_channels must be at least 16 and divisible by 16.")
        if self.generator_updates <= 0 or self.discriminator_updates <= 0:
            raise ValueError("generator_updates and discriminator_updates must be positive.")
        if not 0 < self.learning_rate:
            raise ValueError("learning_rate must be positive.")
        if self.use_sigmoid:
            raise ValueError(
                "Training must use discriminator logits: set use_sigmoid=False. "
                "This keeps BCEWithLogitsLoss safe under AMP on Colab/Kaggle."
            )


def build_models(config: TrainConfig) -> tuple[Generator, Discriminator]:
    generator = Generator(
        image_channels=config.image_channels,
        label_channels=config.label_channels,
        out_channels=config.generator_out_channels,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        image_size=config.image_size,
    )
    discriminator = Discriminator(
        image_channels=config.image_channels,
        label_channels=config.label_channels,
        channel=config.discriminator_channels,
        use_sigmoid=config.use_sigmoid,
    )
    return generator, discriminator


def _first(batch: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in batch:
            return batch[name]
    raise KeyError(f"Batch is missing one of: {', '.join(names)}")


def unpack_batch(batch: Any) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize supported batch formats to ``(original, noisy, label)``."""

    if isinstance(batch, Mapping):
        original = _first(
            batch,
            ("original_image", "original", "image", "real_image", "scan_t1ce_crop_pad"),
        )
        noisy = _first(
            batch,
            ("noisy_image", "noised_image", "noisy", "input_image", "scan_t1ce_noisy"),
        )
        label = _first(batch, ("label", "rgb_label", "target_label", "label_crop_pad"))
    elif isinstance(batch, (tuple, list)) and len(batch) == 3:
        original, noisy, label = batch
    else:
        raise TypeError(
            "Each batch must be a mapping or a tuple/list: "
            "(original_image, noisy_image, label)."
        )

    return original.float(), noisy.float(), label.float()


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _autocast_context(enabled: bool, device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return torch.autocast(device_type=device.type, enabled=False)


def validate_training_batch(
    original: Tensor,
    noisy: Tensor,
    label: Tensor,
    image_channels: int = 3,
    label_channels: int = 3,
    image_size: Optional[int] = 96,
) -> None:
    """Validate the tensor contract used by the paper-style pipeline.

    This checks tensor shapes and finite values only. Image normalization is
    intentionally left to the future data pipeline.
    """

    tensors = (original, noisy, label)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("Expected [B, C, H, W] tensors for original, noisy, and label.")
    if original.shape != noisy.shape:
        raise ValueError(f"Original/noisy shapes must match, got {original.shape} and {noisy.shape}.")
    if original.shape[1] != image_channels:
        raise ValueError(f"Expected {image_channels} image channels, got {original.shape[1]}.")
    if label.shape[0] != original.shape[0] or label.shape[1] != label_channels:
        raise ValueError(f"Expected label shape [B, {label_channels}, H, W], got {label.shape}.")
    if label.shape[2:] != original.shape[2:]:
        raise ValueError("Label and image spatial dimensions must match.")
    if image_size is not None and original.shape[2:] != (image_size, image_size):
        raise ValueError(
            f"Expected spatial size {(image_size, image_size)} for SwinUNETR/discriminator, "
            f"got {tuple(original.shape[2:])}."
        )
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("Training tensors contain NaN or infinite values.")


def _progress_bar(iterable, *, enabled: bool, **kwargs):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def _write_progress(message: str, *, enabled: bool) -> None:
    if enabled:
        try:
            from tqdm.auto import tqdm

            tqdm.write(message)
            return
        except ImportError:
            pass
    print(message)


def _format_epoch_summary(
    epoch: int,
    total_epochs: int,
    metrics: Mapping[str, float],
) -> str:
    return (
        f"Epoch {epoch + 1:03d}/{total_epochs:03d} | "
        f"loss_G={metrics['loss_G']:.6f} | "
        f"loss_D={metrics['loss_D']:.6f} | "
        f"loss_recons={metrics['loss_recons']:.6f} | "
        f"loss_adv_G={metrics['loss_adv_G']:.6f}"
    )


def train_one_epoch(
    loader,
    generator: Generator,
    discriminator: Discriminator,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    loss_fn: GliGANLoss,
    device: torch.device,
    epoch: int,
    generator_updates: int = 2,
    discriminator_updates: int = 1,
    scaler: Optional[torch.amp.GradScaler] = None,
    amp: bool = True,
    image_size: Optional[int] = 96,
    show_progress: bool = True,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    totals = {"loss_G": 0.0, "loss_D": 0.0, "loss_recons": 0.0, "loss_adv_G": 0.0}
    batches = 0

    progress = _progress_bar(
        loader,
        enabled=show_progress,
        desc=f"Epoch {epoch + 1}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in progress:
        original, noisy, label = (item.to(device, non_blocking=True) for item in unpack_batch(batch))
        validate_training_batch(
            original,
            noisy,
            label,
            image_channels=generator.image_channels,
            label_channels=generator.label_channels,
            image_size=image_size,
        )

        for _ in range(discriminator_updates):
            d_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                generated = generator(noisy, label)
                if generated.shape != original.shape:
                    raise RuntimeError(
                        f"Generator output must match original shape {original.shape}, "
                        f"got {generated.shape}."
                    )
            real_scores = discriminator(original, label)
            fake_scores = discriminator(generated.detach(), label)
            loss_d, _, _ = loss_fn.discriminator(real_scores, fake_scores)
            if scaler is not None and amp and device.type == "cuda":
                scaler.scale(loss_d).backward()
                scaler.step(d_optimizer)
                scaler.update()
            else:
                loss_d.backward()
                d_optimizer.step()

        for _ in range(generator_updates):
            g_optimizer.zero_grad(set_to_none=True)
            _set_requires_grad(discriminator, False)
            with _autocast_context(amp, device):
                generated = generator(noisy, label)
                if generated.shape != original.shape:
                    raise RuntimeError(
                        f"Generator output must match original shape {original.shape}, "
                        f"got {generated.shape}."
                    )
                fake_scores = discriminator(generated, label)
                loss_g, loss_recons, loss_adv_g, _ = loss_fn.generator(
                    generated, original, fake_scores, epoch
                )
            if scaler is not None and amp and device.type == "cuda":
                scaler.scale(loss_g).backward()
                scaler.step(g_optimizer)
                scaler.update()
            else:
                loss_g.backward()
                g_optimizer.step()
            _set_requires_grad(discriminator, True)

        totals["loss_G"] += float(loss_g.detach().cpu())
        totals["loss_D"] += float(loss_d.detach().cpu())
        totals["loss_recons"] += float(loss_recons.detach().cpu())
        totals["loss_adv_G"] += float(loss_adv_g.detach().cpu())
        batches += 1
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                loss_G=totals["loss_G"] / batches,
                loss_D=totals["loss_D"] / batches,
                recons=totals["loss_recons"] / batches,
                adv_G=totals["loss_adv_G"] / batches,
            )

    if batches == 0:
        raise ValueError("The training loader produced no batches.")
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(
    path: str | Path,
    epoch: int,
    generator: Generator,
    discriminator: Discriminator,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    metrics: Mapping[str, float],
    config: TrainConfig,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "metrics": dict(metrics),
            "config": asdict(config),
        },
        path,
    )


def fit(loader, config: Optional[TrainConfig] = None) -> tuple[Generator, Discriminator, list[dict[str, float]]]:
    config = config or TrainConfig()
    device = torch.device(config.device)
    generator, discriminator = build_models(config)
    generator.to(device)
    discriminator.to(device)

    g_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    d_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    loss_fn = GliGANLoss(
        reconstruction_max_weight=config.reconstruction_max_weight,
        progression_epochs=config.reconstruction_progression_epochs,
        from_logits=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    history = []

    epoch_progress = _progress_bar(
        range(config.epochs),
        enabled=config.show_progress,
        desc="Training",
        dynamic_ncols=True,
    )

    for epoch in epoch_progress:
        metrics = train_one_epoch(
            loader,
            generator,
            discriminator,
            g_optimizer,
            d_optimizer,
            loss_fn,
            device,
            epoch,
            generator_updates=config.generator_updates,
            discriminator_updates=config.discriminator_updates,
            scaler=scaler,
            amp=config.amp,
            image_size=config.image_size,
            show_progress=config.show_progress,
        )
        history.append(metrics)
        if hasattr(epoch_progress, "set_postfix"):
            epoch_progress.set_postfix(**metrics)
        _write_progress(
            _format_epoch_summary(epoch, config.epochs, metrics),
            enabled=config.show_progress,
        )
        if config.checkpoint_every and (epoch + 1) % config.checkpoint_every == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                checkpoint_path,
                epoch + 1,
                generator,
                discriminator,
                g_optimizer,
                d_optimizer,
                metrics,
                config,
            )
            _write_progress(
                f"Saved checkpoint: {checkpoint_path}",
                enabled=config.show_progress,
            )

    return generator, discriminator, history


__all__ = [
    "TrainConfig",
    "build_models",
    "fit",
    "save_checkpoint",
    "train_one_epoch",
    "unpack_batch",
    "validate_training_batch",
]
