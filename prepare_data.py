from pathlib import Path
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_ROOT = Path("GLIGAN_DATA")
OUTPUT_ROOT = Path("data")

IMAGE_DIR = OUTPUT_ROOT / "images"
MASK_DIR = OUTPUT_ROOT / "masks"
NOISED_DIR = OUTPUT_ROOT / "noised_images"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)
NOISED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COLLECT SAMPLES
# ============================================================

total = 0
skipped = 0

# Find every subject folder containing images/
for images_dir in SOURCE_ROOT.glob("*/*/images"):

    subject_dir = images_dir.parent
    subject_name = subject_dir.name

    masks_dir = subject_dir / "masks"
    noised_images_dir = subject_dir / "noised_images"

    if not masks_dir.is_dir() or not noised_images_dir.is_dir():
        print(f"Skipping {subject_dir}: missing masks/noised_images")
        continue

    for image_path in sorted(images_dir.glob("*.png")):

        filename = image_path.name

        mask_path = masks_dir / filename
        noised_path = noised_images_dir / filename

        # ----------------------------------------------------
        # Make sure all three corresponding files exist
        # ----------------------------------------------------

        if not mask_path.exists() or not noised_path.exists():
            print(f"Skipping incomplete sample: {subject_name}/{filename}")
            skipped += 1
            continue

        # ----------------------------------------------------
        # Keep the same filename across all three directories
        # ----------------------------------------------------

        destination_image = IMAGE_DIR / filename
        destination_mask = MASK_DIR / filename
        destination_noised = NOISED_DIR / filename

        # ----------------------------------------------------
        # Prevent accidental overwriting if filenames collide
        # between subjects.
        # ----------------------------------------------------

        if (
            destination_image.exists()
            or destination_mask.exists()
            or destination_noised.exists()
        ):
            stem = image_path.stem
            suffix = image_path.suffix

            new_filename = f"{subject_name}_{stem}{suffix}"

            destination_image = IMAGE_DIR / new_filename
            destination_mask = MASK_DIR / new_filename
            destination_noised = NOISED_DIR / new_filename

        # ----------------------------------------------------
        # Copy the three corresponding files
        # ----------------------------------------------------

        shutil.copy2(image_path, destination_image)
        shutil.copy2(mask_path, destination_mask)
        shutil.copy2(noised_path, destination_noised)

        total += 1

        print(f"Copied: {subject_name}/{filename}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print(f"Samples copied : {total}")
print(f"Samples skipped: {skipped}")

print(f"\nImages       : {IMAGE_DIR.resolve()}")
print(f"Masks        : {MASK_DIR.resolve()}")
print(f"Noised images: {NOISED_DIR.resolve()}")