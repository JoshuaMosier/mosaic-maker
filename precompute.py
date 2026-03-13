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
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

GRID_COLS = 10
GRID_ROWS = 15
POSTER_W = 230
POSTER_H = 345
CELL_W = POSTER_W // GRID_COLS  # 23
CELL_H = POSTER_H // GRID_ROWS  # 23


def process_one(args_tuple):
    """Load and compute vector for a single image. Returns (fname, vector) or None."""
    images_dir, fname = args_tuple
    path = os.path.join(images_dir, fname)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w != POSTER_W or h != POSTER_H:
        return None
    # cv2 loads BGR; convert to RGB then compute vector
    pixels = img[:, :, ::-1].astype(np.float32)  # (345, 230, 3) RGB
    grid = pixels.reshape(GRID_ROWS, CELL_H, GRID_COLS, CELL_W, 3)
    cell_avgs = grid.mean(axis=(1, 3))
    return fname, cell_avgs.reshape(-1).astype(np.float32)


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

    num_workers = min(os.cpu_count() or 4, 16)
    work_items = [(args.images, f) for f in files]

    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(
            executor.map(process_one, work_items),
            total=len(files),
            desc="Processing posters",
        ):
            if result is not None:
                fname, vec = result
                results[fname] = vec

    if not results:
        print("Error: no valid 230x345 images found.", file=sys.stderr)
        sys.exit(1)

    # Sort by filename to maintain deterministic order
    filenames = sorted(results.keys())
    vectors = [results[f] for f in filenames]

    vectors_arr = np.stack(vectors)  # (N, 450) float32
    filenames_arr = np.array(filenames)

    np.savez_compressed(args.output, vectors=vectors_arr, filenames=filenames_arr)
    print(f"Saved {len(filenames)} poster vectors to {args.output}")
    skipped = len(files) - len(filenames)
    if skipped:
        print(f"Skipped {skipped} files (wrong size or unreadable)")


if __name__ == "__main__":
    main()
