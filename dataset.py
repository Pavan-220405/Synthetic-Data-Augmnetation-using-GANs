"""Dataset helpers for prepared 2D GliGAN RGB samples.

The default project layout is a flat prepared-data directory:

    data/
      images/
      noised_images/
      masks/

Each sample is matched by filename across those three folders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def load_rgb_tensor(path: str | Path) -> Tensor:
    """Load a PNG as a float RGB tensor with shape [3, H, W]."""

    array = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class GliGANDataset(Dataset):
    """Read prepared RGB image/noisy-image/RGB-label samples.

    The preferred layout is ``root/images``, ``root/noised_images``, and
    ``root/masks``. The older nested ``root/*/*/images`` layout is still
    accepted as a fallback so existing prepared exports remain usable.
    """

    def __init__(
        self,
        root: str | Path = "data",
        transform: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"No prepared GliGAN samples found under {self.root.resolve()}.")

    def _collect_samples(self) -> list[tuple[Path, Path, Path, str]]:
        flat_samples = self._collect_flat_samples()
        if flat_samples:
            return flat_samples
        return self._collect_nested_samples()

    def _collect_flat_samples(self) -> list[tuple[Path, Path, Path, str]]:
        samples = []
        images_dir = self.root / "images"
        noised_dir = self.root / "noised_images"
        labels_dir = self.root / "masks"
        if not images_dir.is_dir() or not noised_dir.is_dir() or not labels_dir.is_dir():
            return samples

        for image_path in sorted(images_dir.glob("*.png")):
            noised_path = noised_dir / image_path.name
            label_path = labels_dir / image_path.name
            if noised_path.exists() and label_path.exists():
                samples.append((image_path, noised_path, label_path, self.root.name))
        return samples

    def _collect_nested_samples(self) -> list[tuple[Path, Path, Path, str]]:
        samples = []
        for images_dir in sorted(self.root.glob("*/*/images")):
            sample_root = images_dir.parent
            noised_dir = sample_root / "noised_images"
            labels_dir = sample_root / "masks"
            class_name = sample_root.parent.name

            if not noised_dir.is_dir() or not labels_dir.is_dir():
                continue

            for image_path in sorted(images_dir.glob("*.png")):
                noised_path = noised_dir / image_path.name
                label_path = labels_dir / image_path.name
                if noised_path.exists() and label_path.exists():
                    samples.append((image_path, noised_path, label_path, class_name))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        image_path, noised_path, label_path, class_name = self.samples[index]
        item: dict[str, Tensor | str] = {
            "original_image": load_rgb_tensor(image_path),
            "noisy_image": load_rgb_tensor(noised_path),
            "label": load_rgb_tensor(label_path),
            "class_name": class_name,
        }

        if self.transform is not None:
            tensor_item = {
                key: value for key, value in item.items() if isinstance(value, Tensor)
            }
            item.update(self.transform(tensor_item))

        return item


def build_dataloader(
    root: str | Path = "data",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = GliGANDataset(root=root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


__all__ = ["GliGANDataset", "build_dataloader", "load_rgb_tensor"]
