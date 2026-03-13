"""Precompute sRGB grid color vectors for all poster images.

Each 230x345 poster is divided into a 10x15 grid (23x23px cells).
The average RGB color of each cell is stored as a 450-dimensional
feature vector (150 cells x 3 channels) per poster.

Usage:
    python precompute.py --images path/to/posters --output grid_data.npz
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

GRID_COLS = 10
GRID_ROWS = 15
POSTER_W = 230
POSTER_H = 345
CELL_W = POSTER_W // GRID_COLS  # 23
CELL_H = POSTER_H // GRID_ROWS  # 23


def poster_to_vector(img: Image.Image) -> np.ndarray:
    """Convert a 230x345 poster image to a 450D sRGB feature vector.

    Returns:
        1D array of shape (450,) with float32 sRGB values (0-255 range).
    """
    pixels = np.array(img, dtype=np.float32)  # (345, 230, 3)

    # Reshape into grid of cells: (rows, cell_h, cols, cell_w, 3)
    grid = pixels.reshape(GRID_ROWS, CELL_H, GRID_COLS, CELL_W, 3)

    # Average each cell in sRGB space: (15, 10, 3)
    cell_avgs = grid.mean(axis=(1, 3))

    return cell_avgs.reshape(-1).astype(np.float32)  # (450,)


def main():
    parser = argparse.ArgumentParser(description="Precompute sRGB grid vectors for poster images.")
    parser.add_argument(
        "--images",
        default=os.environ.get("MOSAIC_IMAGES_DIR", "images"),
        help="Directory containing 230x345 poster JPEGs (default: $MOSAIC_IMAGES_DIR or 'images/')",
    )
    parser.add_argument(
        "--output",
        default="grid_data.npz",
        help="Output file path (default: grid_data.npz)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.images):
        print(f"Error: image directory not found: {args.images}", file=sys.stderr)
        sys.exit(1)

    # Collect all JPEG files
    files = sorted(
        f for f in os.listdir(args.images)
        if f.lower().endswith((".jpg", ".jpeg"))
    )
    print(f"Found {len(files)} images in {args.images}")

    vectors = []
    filenames = []
    skipped = 0

    for fname in tqdm(files, desc="Processing posters"):
        path = os.path.join(args.images, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            skipped += 1
            continue

        w, h = img.size
        if w != POSTER_W or h != POSTER_H:
            skipped += 1
            continue

        vec = poster_to_vector(img)
        vectors.append(vec)
        filenames.append(fname)

    if not vectors:
        print("Error: no valid 230x345 images found.", file=sys.stderr)
        sys.exit(1)

    vectors_arr = np.stack(vectors)  # (N, 450) float32
    filenames_arr = np.array(filenames)

    np.savez_compressed(args.output, vectors=vectors_arr, filenames=filenames_arr)
    print(f"Saved {len(filenames)} poster vectors to {args.output}")
    if skipped:
        print(f"Skipped {skipped} files (wrong size or unreadable)")


if __name__ == "__main__":
    main()
