
from pathlib import Path
import numpy as np
from PIL import Image
import shutil
from scipy import ndimage


# ============================================================
# CONFIGURATION
# ============================================================

# Folder containing sets such as 6291_SJ, 6888_SJ, etc.
INPUT_ROOT = Path("YOLO_TRAIN_DATA")

# New dataset created by this script.
OUTPUT_ROOT = Path("GLIGAN_DATA")

# Remove the previous generated dataset before writing new samples.
CLEAR_OUTPUT = True

# Every selected object becomes a 96x96 training sample.
CROP_SIZE = 96

# YOLO classes in the dataset.
CLASSES = {
    0: "HER2_cluster3",
    1: "HER2_cluster6",
    2: "HER2_cluster12",
    3: "fusion",
    4: "pink",
    5: "black",
}

# Keep ONLY these classes.
TARGET_CLASSES = {
    0: "Cluster3",
    3: "Fusion",
}

# Gaussian-noise settings.
GAUSSIAN_STD = 65.0

# Circular noise region around the YOLO bbox.
# The circle first fully encloses the rectangular bbox using its diagonal.
# Then it is expanded by 40%, so the noisy region is clearly larger than
# the bbox while remaining circular.
CIRCULAR_DIAMETER_SCALE = 1.5

# Only this fraction of the extra +40% ring is noised. The core circle that
# encloses the bbox remains densely noised.
EXTRA_REGION_NOISE_PROBABILITY = 0.5

# Pixels darker than this are treated as background while estimating the
# target-cell boundary. With YOLO-only labels this is an approximation;
# exact boundaries require segmentation labels.
FOREGROUND_THRESHOLD = 8

# Expand the YOLO bbox by 30% on each side when estimating the target-cell
# boundary. This expansion is ONLY for boundary estimation, not the final
# circular noise region.
BOUNDARY_MARGIN = 1.7


# ============================================================
# YOLO LABEL READING
# ============================================================

def read_yolo_labels(label_path):
    """
    Read YOLO annotations of the form:

        class_id x_center y_center width height

    The four coordinates are normalized to [0, 1].

    Returns only Cluster3 (class 0) and Fusion (class 3).
    """
    records = []

    with label_path.open("r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                print(
                    f"WARNING: {label_path} line {line_number} "
                    f"does not contain 5 values. Skipping."
                )
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError:
                print(
                    f"WARNING: {label_path} line {line_number} "
                    f"contains non-numeric values. Skipping."
                )
                continue

            if not (
                0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 <= width <= 1.0
                and 0.0 <= height <= 1.0
            ):
                print(
                    f"WARNING: {label_path} line {line_number} "
                    f"has YOLO coordinates outside [0, 1]. Skipping."
                )
                continue

            if class_id in TARGET_CLASSES:
                records.append(
                    {
                        "class_id": class_id,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                    }
                )

    return records


# ============================================================
# YOLO -> PIXEL BBOX
# ============================================================

def yolo_to_xyxy(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height,
):
    """
    Convert normalized YOLO coordinates to pixel coordinates.

    Returns:
        x1, y1, x2, y2

    x2 and y2 are exclusive.
    """
    xc = x_center * image_width
    yc = y_center * image_height
    bw = width * image_width
    bh = height * image_height

    x1 = int(round(xc - bw / 2))
    y1 = int(round(yc - bh / 2))
    x2 = int(round(xc + bw / 2))
    y2 = int(round(yc + bh / 2))

    # Clip to the image.
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    return x1, y1, x2, y2


# ============================================================
# 96x96 CROP AROUND ONE BBOX
# ============================================================

def make_centered_crop(image, bbox, crop_size=96):
    """
    Make a crop_size x crop_size RGB crop centred on one bbox.

    The bbox itself is NOT cropped to its exact dimensions.
    Surrounding image context is retained.

    If the crop reaches outside the source image, the missing
    area is padded with black pixels.

    Returns:
        crop:       [96, 96, 3] RGB uint8 image
        crop_bbox:  bbox coordinates relative to the crop
    """
    image_height, image_width = image.shape[:2]

    x1, y1, x2, y2 = bbox

    bbox_center_x = (x1 + x2) / 2.0
    bbox_center_y = (y1 + y2) / 2.0

    crop_x1 = int(round(bbox_center_x - crop_size / 2))
    crop_y1 = int(round(bbox_center_y - crop_size / 2))

    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    # Intersection between crop and source image.
    src_x1 = max(0, crop_x1)
    src_y1 = max(0, crop_y1)
    src_x2 = min(image_width, crop_x2)
    src_y2 = min(image_height, crop_y2)

    # Corresponding coordinates inside the output crop.
    dst_x1 = src_x1 - crop_x1
    dst_y1 = src_y1 - crop_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    crop = np.zeros(
        (crop_size, crop_size, 3),
        dtype=np.uint8,
    )

    crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[
        src_y1:src_y2,
        src_x1:src_x2,
    ]

    # Convert original bbox coordinates to crop coordinates.
    bx1 = max(0, x1 - crop_x1)
    by1 = max(0, y1 - crop_y1)
    bx2 = min(crop_size, x2 - crop_x1)
    by2 = min(crop_size, y2 - crop_y1)

    crop_bbox = (bx1, by1, bx2, by2)

    return crop, crop_bbox


# ============================================================
# TARGET IMAGE / BOUNDARY ESTIMATE
# ============================================================

def estimate_target_boundary(crop, bbox):
    """
    Estimate the target-cell boundary using a search region expanded by
    30% on each side of the YOLO bbox.

    The expanded region is used only to find the target cell. The final
    returned mask contains the selected cell component, not the expanded
    rectangle.
    """
    x1, y1, x2, y2 = bbox
    target_mask = np.zeros(crop.shape[:2], dtype=bool)

    if x2 <= x1 or y2 <= y1:
        return target_mask

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Expand bbox by 30% of its width/height on EACH side.
    search_x1 = max(0, int(round(x1 - BOUNDARY_MARGIN * bbox_width)))
    search_y1 = max(0, int(round(y1 - BOUNDARY_MARGIN * bbox_height)))
    search_x2 = min(crop.shape[1], int(round(x2 + BOUNDARY_MARGIN * bbox_width)))
    search_y2 = min(crop.shape[0], int(round(y2 + BOUNDARY_MARGIN * bbox_height)))

    search_region = crop[search_y1:search_y2, search_x1:search_x2]
    foreground = np.any(search_region > FOREGROUND_THRESHOLD, axis=2)

    if not np.any(foreground):
        target_mask[y1:y2, x1:x2] = True
        return target_mask

    foreground = ndimage.binary_closing(foreground, structure=np.ones((3, 3)))
    foreground = ndimage.binary_fill_holes(foreground)

    labels, num_labels = ndimage.label(foreground)
    if num_labels == 0:
        target_mask[y1:y2, x1:x2] = True
        return target_mask

    # Anchor selection to the center of the ORIGINAL YOLO bbox.
    center_x = int(round((x1 + x2) / 2.0)) - search_x1
    center_y = int(round((y1 + y2) / 2.0)) - search_y1
    center_x = min(max(center_x, 0), labels.shape[1] - 1)
    center_y = min(max(center_y, 0), labels.shape[0] - 1)
    center_label = labels[center_y, center_x]

    if center_label:
        selected = labels == center_label
    else:
        # If bbox center is background, choose the component with the
        # greatest overlap with the ORIGINAL YOLO bbox.
        bx1 = max(0, x1 - search_x1)
        by1 = max(0, y1 - search_y1)
        bx2 = min(labels.shape[1], x2 - search_x1)
        by2 = min(labels.shape[0], y2 - search_y1)

        overlap_scores = []
        for label_id in range(1, num_labels + 1):
            component = labels == label_id
            overlap_scores.append(component[by1:by2, bx1:bx2].sum())

        selected_label = int(np.argmax(overlap_scores)) + 1
        selected = labels == selected_label

    # Map selected component back to full crop coordinates.
    target_mask[search_y1:search_y2, search_x1:search_x2] = selected
    return target_mask


def make_bbox_target_image(crop, bbox):
    """
    Keep the original RGB pixels inside the YOLO bbox and black out everything
    around it.

    This is the label image requested for this 2D pipeline: not a binary mask,
    but a 3-channel RGB image with the same spatial size as the training crop.
    """
    target = np.zeros_like(crop)
    x1, y1, x2, y2 = bbox
    if x2 > x1 and y2 > y1:
        target[y1:y2, x1:x2] = crop[y1:y2, x1:x2]
    return target


# ============================================================
# NOISING
# ============================================================

def add_circular_gaussian_noise(image, target_mask, bbox, rng=None):
    """
    Add Gaussian noise in a circular region centered on the YOLO bbox.

    The circle is defined from the bbox dimensions rather than using the
    rectangular bbox itself. The core noise region is:

        inner circular region AND target_mask

    The extra +40% ring is sparsely sampled:

        outer ring AND target_mask AND random <= EXTRA_REGION_NOISE_PROBABILITY

    Therefore the Gaussian noise remains inside the estimated target-cell
    boundary and cannot overwrite neighboring cells, while the expansion area
    is visibly lighter/sparser than the bbox-enclosing core.
    """
    rng = rng or np.random.default_rng()

    if not np.any(target_mask):
        return image.copy()

    x1, y1, x2, y2 = bbox
    bbox_width = max(1, x2 - x1)
    bbox_height = max(1, y2 - y1)

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    # Minimum circle that contains the ENTIRE rectangular bbox:
    # its diameter is the bbox diagonal.
    bbox_diagonal = np.sqrt(
        bbox_width ** 2 + bbox_height ** 2
    )

    inner_radius = max(0.5, bbox_diagonal / 2.0)

    # Expand that enclosing circle by 40%; only this extra ring is sparse.
    outer_radius = max(inner_radius, CIRCULAR_DIAMETER_SCALE * bbox_diagonal / 2.0)

    height, width = image.shape[:2]
    yy, xx = np.indices((height, width))

    distance_squared = (
        (xx - center_x) ** 2 +
        (yy - center_y) ** 2
    )

    inner_region = distance_squared <= inner_radius ** 2
    extra_region = (
        (distance_squared > inner_radius ** 2) &
        (distance_squared <= outer_radius ** 2)
    )
    sparse_extra_region = extra_region & (
        rng.random((height, width)) <= EXTRA_REGION_NOISE_PROBABILITY
    )

    # Most important constraint: noise can only occur inside the estimated
    # target-cell boundary.
    noise_region = (inner_region | sparse_extra_region) & target_mask

    noisy = image.copy()
    gaussian = rng.normal(
        loc=127.5,
        scale=GAUSSIAN_STD,
        size=image.shape,
    )

    noisy[noise_region] = gaussian[noise_region]

    return np.clip(noisy, 0, 255).astype(np.uint8)


# ============================================================
# SAVE
# ============================================================

def save_png(array, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if array.ndim == 2:
        Image.fromarray(array, mode="L").save(path)
    else:
        Image.fromarray(array, mode="RGB").save(path)


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n"
            f"{INPUT_ROOT.resolve()}"
        )

    if CLEAR_OUTPUT and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    rng = np.random.default_rng()

    total_samples = 0
    total_cluster3 = 0
    total_fusion = 0

    # Every immediate directory is one set.
    set_dirs = sorted(
        p for p in INPUT_ROOT.iterdir()
        if p.is_dir()
    )

    print(f"Input : {INPUT_ROOT.resolve()}")
    print(f"Output: {OUTPUT_ROOT.resolve()}")
    print(f"Sets  : {len(set_dirs)}")
    print()

    for set_dir in set_dirs:

        images_dir = set_dir / "images"
        labels_dir = set_dir / "labels"

        if not images_dir.is_dir():
            print(
                f"Skipping {set_dir.name}: "
                f"images/ directory not found."
            )
            continue

        if not labels_dir.is_dir():
            print(
                f"Skipping {set_dir.name}: "
                f"labels/ directory not found."
            )
            continue

        print(f"Processing {set_dir.name}...")

        image_paths = sorted(
            images_dir.glob("*.png")
        )

        for image_path in image_paths:

            label_path = (
                labels_dir /
                f"{image_path.stem}.txt"
            )

            if not label_path.exists():
                print(
                    f"  WARNING: label missing for "
                    f"{image_path.name}"
                )
                continue

            annotations = read_yolo_labels(
                label_path
            )

            # No Cluster3/Fusion annotation in this image.
            if not annotations:
                continue

            image = np.array(
                Image.open(image_path).convert("RGB")
            )

            image_height, image_width = image.shape[:2]

            # ------------------------------------------------
            # IMPORTANT:
            # Every selected bbox becomes its OWN sample.
            # ------------------------------------------------
            for object_index, annotation in enumerate(
                annotations,
                start=1,
            ):

                class_id = annotation["class_id"]

                bbox = yolo_to_xyxy(
                    annotation["x_center"],
                    annotation["y_center"],
                    annotation["width"],
                    annotation["height"],
                    image_width,
                    image_height,
                )

                x1, y1, x2, y2 = bbox

                if x2 <= x1 or y2 <= y1:
                    print(
                        f"  WARNING: invalid bbox in "
                        f"{label_path.name}; skipping."
                    )
                    continue

                # 96x96 RGB crop centred on THIS bbox.
                crop, crop_bbox = make_centered_crop(
                    image,
                    bbox,
                    CROP_SIZE,
                )

                # RGB bbox-label image in the same spatial location as the
                # crop. It is saved under masks/ for compatibility with the
                # existing dataset layout, but it is not a binary mask.
                target_image = make_bbox_target_image(
                    crop,
                    crop_bbox,
                )

                # Estimate the target-cell boundary using a 30%-expanded YOLO search region.
                # The 30% expansion is used only for finding the boundary.
                target_boundary = estimate_target_boundary(
                    crop,
                    crop_bbox,
                )

                if not np.any(target_boundary):
                    print(
                        f"  WARNING: could not estimate target boundary in "
                        f"{label_path.name}; using full bbox for noise."
                    )
                    target_boundary = np.any(target_image > FOREGROUND_THRESHOLD, axis=2)

                # Apply circular Gaussian noise centered on the YOLO bbox.
                # The circle fully encloses the bbox and is expanded by 40%.
                # It is clipped only by the estimated target-cell boundary,
                # preventing noise from spilling into neighboring cells.
                noisy_crop = add_circular_gaussian_noise(
                    crop,
                    target_boundary,
                    crop_bbox,
                    rng,
                )

                class_name = TARGET_CLASSES[class_id]

                output_base = (
                    OUTPUT_ROOT /
                    class_name /
                    set_dir.name
                )

                # Unique name for multiple objects in one source image.
                sample_name = (
                    f"{image_path.stem}_"
                    f"{class_name.lower()}_"
                    f"{object_index}"
                )

                save_png(
                    crop,
                    output_base /
                    "images" /
                    f"{sample_name}.png",
                )

                save_png(
                    target_image,
                    output_base /
                    "masks" /
                    f"{sample_name}.png",
                )

                save_png(
                    noisy_crop,
                    output_base /
                    "noised_images" /
                    f"{sample_name}.png",
                )

                total_samples += 1

                if class_id == 0:
                    total_cluster3 += 1
                elif class_id == 3:
                    total_fusion += 1

        print()

    print("=" * 60)
    print("DONE")
    print(f"Total samples : {total_samples}")
    print(f"Cluster3      : {total_cluster3}")
    print(f"Fusion        : {total_fusion}")
    print(f"Output        : {OUTPUT_ROOT.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
