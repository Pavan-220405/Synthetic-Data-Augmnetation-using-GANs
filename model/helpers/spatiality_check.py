from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from model.v1.generator import Generator


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 96

TEST_DATA_DIR = Path("model_data/test_data")
LABEL_DIR = TEST_DATA_DIR / "labels"
CELL_MASK_DIR = TEST_DATA_DIR / "cell_masks"

DEFAULT_CHECKPOINT = Path("model/checkpoints/epoch_0028.pt")

# Same noise settings used in preprocess_data.py
GAUSSIAN_STD = 65.0
CIRCULAR_DIAMETER_SCALE = 1.5
EXTRA_REGION_NOISE_PROBABILITY = 0.5

FUSION_CLASS_ID = 3


# ============================================================
# FILE HELPERS
# ============================================================

def load_rgb(path: Path) -> np.ndarray:
    """Load an image as RGB uint8."""

    image = np.array(
        Image.open(path).convert("RGB"),
        dtype=np.uint8,
    )

    if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} image:\n"
            f"{path}\n"
            f"Got {image.shape[:2]}"
        )

    return image


def find_cell_mask(original_path: Path) -> Path:
    """
    Find the cell mask corresponding to the original image.

    Expected:
        test_data/cell_masks/<original_stem>.png
    """

    exact = CELL_MASK_DIR / f"{original_path.stem}.png"

    if exact.exists():
        return exact

    # Fallback: search for the same stem with any extension.
    matches = list(CELL_MASK_DIR.glob(f"{original_path.stem}.*"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find cell mask for:\n"
            f"{original_path.name}\n\n"
            f"Expected something like:\n"
            f"{exact}"
        )

    return matches[0]


def load_cell_mask(path: Path) -> np.ndarray:
    """
    Load cell mask as boolean [H, W].

    The project dataset supports binary, 0/255 and RGB masks.
    """

    mask = np.array(
        Image.open(path).convert("L"),
        dtype=np.uint8,
    )

    if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Cell mask must be {IMAGE_SIZE}x{IMAGE_SIZE}, "
            f"got {mask.shape}: {path}"
        )

    return mask > 0


# ============================================================
# YOLO LABEL
# ============================================================

def read_fusion_bbox(label_path: Path):
    """
    Read the first Fusion annotation (class 3).

    YOLO format:
        class_id x_center y_center width height

    Coordinates are normalized to [0, 1].
    """

    if not label_path.exists():
        raise FileNotFoundError(
            f"Target label file not found:\n{label_path}"
        )

    fusion_annotations = []

    with label_path.open("r") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"WARNING: invalid line {line_number} "
                    f"in {label_path.name}; skipping."
                )
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(
                    float,
                    parts[1:],
                )
            except ValueError:
                print(
                    f"WARNING: non-numeric line {line_number} "
                    f"in {label_path.name}; skipping."
                )
                continue

            if class_id != FUSION_CLASS_ID:
                continue

            fusion_annotations.append(
                {
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }
            )

    if not fusion_annotations:
        raise ValueError(
            f"No Fusion (class {FUSION_CLASS_ID}) annotation found in:\n"
            f"{label_path}"
        )

    if len(fusion_annotations) > 1:
        print(
            f"WARNING: {len(fusion_annotations)} Fusion boxes found. "
            f"Using the first one."
        )

    return fusion_annotations[0]


def yolo_to_xyxy(annotation):
    """
    Convert normalized YOLO bbox to pixel coordinates.

    Returns:
        x1, y1, x2, y2
    """

    xc = annotation["x_center"] * IMAGE_SIZE
    yc = annotation["y_center"] * IMAGE_SIZE

    bw = annotation["width"] * IMAGE_SIZE
    bh = annotation["height"] * IMAGE_SIZE

    x1 = int(round(xc - bw / 2))
    y1 = int(round(yc - bh / 2))
    x2 = int(round(xc + bw / 2))
    y2 = int(round(yc + bh / 2))

    # Clip to image boundaries.
    x1 = max(0, min(x1, IMAGE_SIZE))
    y1 = max(0, min(y1, IMAGE_SIZE))
    x2 = max(0, min(x2, IMAGE_SIZE))
    y2 = max(0, min(y2, IMAGE_SIZE))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid Fusion bbox after conversion: "
            f"{(x1, y1, x2, y2)}"
        )

    return x1, y1, x2, y2


# ============================================================
# TARGET LABEL MOVEMENT
# ============================================================

def create_spatially_moved_label(
    target_image: np.ndarray,
    target_bbox,
    click_x: float,
    click_y: float,
):
    """
    Extract the target bbox RGB region from the TARGET IMAGE
    and translate it so that its bbox center is exactly at
    the clicked point.

    The bbox width and height are preserved.

    Output:
        moved_label: 96x96 RGB conditioning label
        moved_bbox: translated bbox coordinates
    """

    tx1, ty1, tx2, ty2 = target_bbox

    bbox_width = tx2 - tx1
    bbox_height = ty2 - ty1

    # --------------------------------------------------------
    # Extract RGB pixels inside the target bbox.
    #
    # This follows the project's RGB-label definition:
    # keep pixels inside bbox, black everywhere else.
    # --------------------------------------------------------

    target_patch = target_image[ty1:ty2, tx1:tx2].copy()

    # --------------------------------------------------------
    # New bbox centered exactly on user's click.
    # --------------------------------------------------------

    new_x1 = int(round(click_x - bbox_width / 2.0))
    new_y1 = int(round(click_y - bbox_height / 2.0))

    new_x2 = new_x1 + bbox_width
    new_y2 = new_y1 + bbox_height

    # --------------------------------------------------------
    # Create empty 96x96 RGB label.
    # --------------------------------------------------------

    moved_label = np.zeros_like(target_image)

    # --------------------------------------------------------
    # Handle clipping if the moved bbox reaches an image edge.
    #
    # The mathematical bbox center is still the click point.
    # --------------------------------------------------------

    dst_x1 = max(0, new_x1)
    dst_y1 = max(0, new_y1)
    dst_x2 = min(IMAGE_SIZE, new_x2)
    dst_y2 = min(IMAGE_SIZE, new_y2)

    src_x1 = max(0, -new_x1)
    src_y1 = max(0, -new_y1)

    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        moved_label[
            dst_y1:dst_y2,
            dst_x1:dst_x2
        ] = target_patch[
            src_y1:src_y2,
            src_x1:src_x2
        ]

    moved_bbox = (
        new_x1,
        new_y1,
        new_x2,
        new_y2,
    )

    return moved_label, moved_bbox


# ============================================================
# NOISE GENERATION
# ============================================================

def create_spatially_aligned_noise(
    original_image: np.ndarray,
    cell_mask: np.ndarray,
    bbox_width: int,
    bbox_height: int,
    click_x: float,
    click_y: float,
    rng: np.random.Generator,
):
    """
    Generate the same circular Gaussian-noise geometry used
    during preprocessing, but centered at the USER CLICK.

    The size of the noise region comes from the TARGET bbox
    dimensions.

    Noise is restricted to the ORIGINAL image's cell mask.
    """

    bbox_width = max(1, bbox_width)
    bbox_height = max(1, bbox_height)

    # Same geometry as preprocess_data.py:
    # the bbox diagonal defines the inner circle diameter.
    bbox_diagonal = np.sqrt(
        bbox_width ** 2 +
        bbox_height ** 2
    )

    inner_radius = max(
        0.5,
        bbox_diagonal / 2.0,
    )

    outer_radius = max(
        inner_radius,
        CIRCULAR_DIAMETER_SCALE * bbox_diagonal / 2.0,
    )

    yy, xx = np.indices(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    distance_squared = (
        (xx - click_x) ** 2 +
        (yy - click_y) ** 2
    )

    inner_region = (
        distance_squared <= inner_radius ** 2
    )

    extra_region = (
        (distance_squared > inner_radius ** 2)
        &
        (distance_squared <= outer_radius ** 2)
    )

    sparse_extra_region = (
        extra_region
        &
        (
            rng.random(
                (IMAGE_SIZE, IMAGE_SIZE)
            )
            <= EXTRA_REGION_NOISE_PROBABILITY
        )
    )

    # IMPORTANT:
    # Noise only occurs inside the original cell.
    noise_region = (
        inner_region |
        sparse_extra_region
    ) & cell_mask

    noisy_image = original_image.copy()

    gaussian = rng.normal(
        loc=127.5,
        scale=GAUSSIAN_STD,
        size=original_image.shape,
    )

    noisy_image[noise_region] = gaussian[noise_region]

    noisy_image = np.clip(
        noisy_image,
        0,
        255,
    ).astype(np.uint8)

    return noisy_image, noise_region


# ============================================================
# CLICK HANDLING
# ============================================================

def get_click_point(original_image: np.ndarray):
    """
    Display original image and wait for one mouse click.
    """

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.imshow(original_image)
    ax.set_title(
        "Click the desired Fusion-cell location"
    )
    ax.set_xlim(-0.5, IMAGE_SIZE - 0.5)
    ax.set_ylim(IMAGE_SIZE - 0.5, -0.5)
    ax.set_aspect("equal")

    ax.set_xticks(range(0, IMAGE_SIZE, 8))
    ax.set_yticks(range(0, IMAGE_SIZE, 8))

    print("\nClick inside the desired cell.")
    print("Close the window after selecting the point.")

    points = plt.ginput(
        n=1,
        timeout=-1,
    )

    plt.close(fig)

    if not points:
        raise RuntimeError(
            "No point was selected."
        )

    x, y = points[0]

    # Keep click inside image coordinates.
    x = float(np.clip(x, 0, IMAGE_SIZE - 1))
    y = float(np.clip(y, 0, IMAGE_SIZE - 1))

    return x, y


# ============================================================
# MODEL
# ============================================================

def load_generator(
    checkpoint_path: Path,
    device: torch.device,
):
    """
    Load the generator using the configuration stored in the
    checkpoint.

    This is important because feature_size / architecture must
    match the checkpoint exactly.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    print("\nLoading checkpoint:")
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "generator" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain a 'generator' state_dict."
        )

    config = checkpoint.get("config", {})

    image_channels = int(
        config.get("image_channels", 3)
    )

    label_channels = int(
        config.get("label_channels", 3)
    )

    out_channels = int(
        config.get("generator_out_channels", 3)
    )

    feature_size = int(
        config.get("feature_size", 48)
    )

    use_checkpoint = bool(
        config.get("use_checkpoint", False)
    )

    image_size = int(
        config.get("image_size", IMAGE_SIZE)
    )

    if image_size != IMAGE_SIZE:
        raise ValueError(
            f"Checkpoint expects image_size={image_size}, "
            f"but this spatiality test uses {IMAGE_SIZE}x{IMAGE_SIZE}."
        )

    print("\nCheckpoint configuration:")
    print(f"  image_channels : {image_channels}")
    print(f"  label_channels : {label_channels}")
    print(f"  output_channels: {out_channels}")
    print(f"  feature_size   : {feature_size}")
    print(f"  image_size     : {image_size}")
    print(f"  device         : {device}")

    generator = Generator(
        image_channels=image_channels,
        label_channels=label_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        image_size=image_size,
    )

    generator.load_state_dict(
        checkpoint["generator"]
    )

    generator = generator.to(device)
    generator.eval()

    print("\nGenerator loaded successfully.")

    return generator


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    generator,
    noisy_image: np.ndarray,
    moved_label: np.ndarray,
    device: torch.device,
):
    """
    Run:
        noisy_image + spatially moved RGB label
        -> Generator
    """

    noisy_tensor = (
        torch.from_numpy(
            noisy_image.astype(np.float32) / 255.0
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    label_tensor = (
        torch.from_numpy(
            moved_label.astype(np.float32) / 255.0
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        generated = generator(
            noisy_tensor,
            label_tensor,
        )

    generated = (
        generated
        .squeeze(0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )

    generated = np.clip(
        generated,
        0,
        1,
    )

    return generated


# ============================================================
# DISPLAY
# ============================================================

def display_intermediate(
    noisy_image,
    moved_label,
    click_x,
    click_y,
    moved_bbox,
    noise_region,
):
    """
    Display the intermediate spatial alignment before inference.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
    )

    axes[0].imshow(noisy_image)
    axes[0].scatter(
        [click_x],
        [click_y],
        marker="x",
        s=80,
        linewidths=2,
    )
    axes[0].set_title(
        "Noised Original Image"
    )

    axes[1].imshow(moved_label)
    axes[1].scatter(
        [click_x],
        [click_y],
        marker="x",
        s=80,
        linewidths=2,
    )

    x1, y1, x2, y2 = moved_bbox

    # Draw the translated bbox.
    axes[1].plot(
        [x1, x2, x2, x1, x1],
        [y1, y1, y2, y2, y1],
        linewidth=1.5,
    )

    axes[1].set_title(
        "Spatially Moved Target Label"
    )

    for ax in axes:
        ax.set_xlim(-0.5, IMAGE_SIZE - 0.5)
        ax.set_ylim(IMAGE_SIZE - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        f"Click = ({click_x:.1f}, {click_y:.1f})\n"
        f"Noise pixels = {int(noise_region.sum())}"
    )

    plt.tight_layout()

    print(
        "\nIntermediate result displayed."
    )
    print(
        "Close the window to run model inference."
    )

    plt.show()


def display_final(
    original_image,
    noisy_image,
    moved_label,
    generated,
    click_x,
    click_y,
):
    """
    Final four-panel spatiality-check result.
    """

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original")

    axes[1].imshow(noisy_image)
    axes[1].set_title("Original Noised")

    axes[2].imshow(moved_label)
    axes[2].set_title(
        "Spatially Moved Target Label"
    )

    axes[3].imshow(generated)
    axes[3].set_title("Generated")

    for ax in axes:
        ax.set_xlim(-0.5, IMAGE_SIZE - 0.5)
        ax.set_ylim(IMAGE_SIZE - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        f"Spatiality Check | "
        f"Target Fusion bbox moved to "
        f"({click_x:.1f}, {click_y:.1f})"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="GliGAN spatial conditioning / spatiality check"
    )

    parser.add_argument(
        "--original",
        required=True,
        type=Path,
        help="Path to the ORIGINAL 96x96 image.",
    )

    parser.add_argument(
        "--target",
        required=True,
        type=Path,
        help="Path to the TARGET 96x96 image whose Fusion label will be used.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained generator checkpoint.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Inference device.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Gaussian noise.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    original_path = args.original
    target_path = args.target

    label_path = (
        LABEL_DIR /
        f"{target_path.stem}.txt"
    )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )
        device = torch.device("cuda")

    elif args.device == "cpu":
        device = torch.device("cpu")

    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print("=" * 70)
    print("GliGAN SPATIALITY CHECK")
    print("=" * 70)

    print(f"Original image : {original_path}")
    print(f"Target image   : {target_path}")
    print(f"Target label   : {label_path}")
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Device         : {device}")

    # --------------------------------------------------------
    # Load original and target images
    # --------------------------------------------------------

    original_image = load_rgb(
        original_path
    )

    target_image = load_rgb(
        target_path
    )

    # --------------------------------------------------------
    # Load original image's cell mask
    # --------------------------------------------------------

    cell_mask_path = find_cell_mask(
        original_path
    )

    cell_mask = load_cell_mask(
        cell_mask_path
    )

    print(f"Cell mask      : {cell_mask_path}")

    # --------------------------------------------------------
    # Read TARGET Fusion bbox
    # --------------------------------------------------------

    annotation = read_fusion_bbox(
        label_path
    )

    target_bbox = yolo_to_xyxy(
        annotation
    )

    tx1, ty1, tx2, ty2 = target_bbox

    bbox_width = tx2 - tx1
    bbox_height = ty2 - ty1

    print("\nTarget Fusion annotation:")
    print(
        f"  class_id = {annotation['class_id']}"
    )
    print(
        f"  normalized bbox = "
        f"({annotation['x_center']:.6f}, "
        f"{annotation['y_center']:.6f}, "
        f"{annotation['width']:.6f}, "
        f"{annotation['height']:.6f})"
    )
    print(
        f"  pixel bbox = {target_bbox}"
    )
    print(
        f"  bbox dimensions = "
        f"{bbox_width} x {bbox_height}"
    )

    # --------------------------------------------------------
    # User clicks location in ORIGINAL image
    # --------------------------------------------------------

    click_x, click_y = get_click_point(
        original_image
    )

    print(
        f"\nSelected point: "
        f"({click_x:.2f}, {click_y:.2f})"
    )

    # --------------------------------------------------------
    # Generate noise using TARGET bbox dimensions
    # but centered at ORIGINAL click.
    # --------------------------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    noisy_image, noise_region = (
        create_spatially_aligned_noise(
            original_image=original_image,
            cell_mask=cell_mask,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            click_x=click_x,
            click_y=click_y,
            rng=rng,
        )
    )

    # --------------------------------------------------------
    # Move TARGET RGB label to the same location.
    # --------------------------------------------------------

    moved_label, moved_bbox = (
        create_spatially_moved_label(
            target_image=target_image,
            target_bbox=target_bbox,
            click_x=click_x,
            click_y=click_y,
        )
    )

    moved_center_x = (
        moved_bbox[0] +
        moved_bbox[2]
    ) / 2.0

    moved_center_y = (
        moved_bbox[1] +
        moved_bbox[3]
    ) / 2.0

    print("\nSpatial alignment:")
    print(
        f"  Click point      = "
        f"({click_x:.2f}, {click_y:.2f})"
    )
    print(
        f"  Noise center     = "
        f"({click_x:.2f}, {click_y:.2f})"
    )
    print(
        f"  Label bbox center= "
        f"({moved_center_x:.2f}, {moved_center_y:.2f})"
    )
    print(
        f"  Target bbox size = "
        f"{bbox_width} x {bbox_height}"
    )
    print(
        f"  Noise pixels     = "
        f"{int(noise_region.sum())}"
    )

    # --------------------------------------------------------
    # INTERMEDIATE DISPLAY
    # --------------------------------------------------------

    display_intermediate(
        noisy_image=noisy_image,
        moved_label=moved_label,
        click_x=click_x,
        click_y=click_y,
        moved_bbox=moved_bbox,
        noise_region=noise_region,
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    generator = load_generator(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    print("\nRunning generator inference...")

    generated = run_inference(
        generator=generator,
        noisy_image=noisy_image,
        moved_label=moved_label,
        device=device,
    )

    print("Inference complete.")

    # --------------------------------------------------------
    # FINAL DISPLAY
    # --------------------------------------------------------

    display_final(
        original_image=original_image,
        noisy_image=noisy_image,
        moved_label=moved_label,
        generated=generated,
        click_x=click_x,
        click_y=click_y,
    )


if __name__ == "__main__":
    main()