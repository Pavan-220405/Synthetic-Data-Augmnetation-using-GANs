import cv2
import numpy as np
from collections import deque
from pathlib import Path
import argparse


def create_mask(image_path, threshold=100):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape

    corners = [
        (0, 0),
        (0, W - 1),
        (H - 1, 0),
        (H - 1, W - 1)
    ]

    visited = np.zeros((H, W), dtype=np.uint8)

    dx = [1, 0, -1, 0]
    dy = [0, -1, 0, 1]

    # --------------------------------------------------
    # BFS from each corner, one by one
    # --------------------------------------------------
    for start_y, start_x in corners:

        # Already reached by a previous BFS
        if visited[start_y, start_x]:
            continue

        # Corner itself must be dark/background
        if gray[start_y, start_x] >= threshold:
            continue

        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = 1

        while queue:

            y, x = queue.popleft()

            for k in range(4):

                ny = y + dy[k]
                nx = x + dx[k]

                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue

                if visited[ny, nx]:
                    continue

                # Bright region = cell
                if gray[ny, nx] >= threshold:
                    continue

                visited[ny, nx] = 1
                queue.append((ny, nx))

    # --------------------------------------------------
    # Create binary masks
    #
    # background = 255
    # cell       = 255
    # --------------------------------------------------
    background_mask = visited * 255
    cell_mask = (1 - visited) * 255

    return cell_mask, background_mask


def process_folder(input_dir, output_dir, threshold=100):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Image extensions
    extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    }

    image_paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    print(f"Found {len(image_paths)} images.")

    for i, image_path in enumerate(image_paths, 1):

        try:
            cell_mask, background_mask = create_mask(
                image_path,
                threshold
            )

            # --------------------------------------------------
            # Save using SAME filename
            #
            # data/images/image.png
            #       ↓
            # data/cell_masks/image.png
            # --------------------------------------------------
            output_path = output_dir / image_path.name

            # For now, save the CELL mask.
            # White = cell
            # Black = background
            cv2.imwrite(
                str(output_path),
                cell_mask
            )

            print(
                f"[{i}/{len(image_paths)}] "
                f"{image_path.name}"
            )

        except Exception as e:
            print(
                f"[ERROR] {image_path.name}: {e}"
            )

    print("\nDone.")
    print(f"Masks saved to: {output_dir}")


def main():

    parser = argparse.ArgumentParser(
        description="Generate cell masks using BFS flood fill."
    )

    parser.add_argument(
        "--input_dir",
        default="data/images",
        help="Input directory (default: data/images)"
    )

    parser.add_argument(
        "--output_dir",
        default="data/cell_masks",
        help="Output directory (default: data/cell_masks)"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Grayscale threshold (default: 100)"
    )

    args = parser.parse_args()

    process_folder(
        args.input_dir,
        args.output_dir,
        args.threshold
    )


if __name__ == "__main__":
    main()