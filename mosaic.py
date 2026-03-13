"""Build a photomosaic from a reference image using precomputed poster data.

Divides the reference image into a grid of cells, matches each cell to
the most visually similar poster using sRGB color vectors and Euclidean
distance, then assembles the matched posters into a high-resolution mosaic.

Usage:
    python mosaic.py --reference input.jpg --data grid_data.npz --images path/to/posters --cells 30
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

GRID_COLS = 10
GRID_ROWS = 15
POSTER_W = 230
POSTER_H = 345


def crop_to_aspect(img: Image.Image, aspect_w: int, aspect_h: int) -> Image.Image:
    """Center-crop an image to the given aspect ratio."""
    w, h = img.size
    target_ratio = aspect_w / aspect_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif current_ratio < target_ratio:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    return img


def image_to_vector(img: Image.Image) -> np.ndarray:
    """Convert an image (any size) to a 450D sRGB vector.

    Resizes to 230x345, divides into 10x15 grid, averages each cell.
    """
    resized = img.resize((POSTER_W, POSTER_H), Image.LANCZOS)
    pixels = np.array(resized, dtype=np.float32)  # (345, 230, 3)

    cell_h = POSTER_H // GRID_ROWS  # 23
    cell_w = POSTER_W // GRID_COLS  # 23

    grid = pixels.reshape(GRID_ROWS, cell_h, GRID_COLS, cell_w, 3)
    cell_avgs = grid.mean(axis=(1, 3))  # (15, 10, 3)

    return cell_avgs.reshape(-1).astype(np.float32)  # (450,)


def compute_all_cell_vectors(reference: Image.Image, cols: int, rows: int) -> np.ndarray:
    """Precompute all cell vectors from the reference image."""
    ref_w, ref_h = reference.size
    cell_w = ref_w / cols
    cell_h = ref_h / rows

    vectors = np.empty((rows * cols, 450), dtype=np.float32)

    for row in tqdm(range(rows), desc="Computing cell vectors"):
        for col in range(cols):
            left = int(col * cell_w)
            top = int(row * cell_h)
            right = int((col + 1) * cell_w)
            bottom = int((row + 1) * cell_h)
            cell_img = reference.crop((left, top, right, bottom))
            vectors[row * cols + col] = image_to_vector(cell_img)

    return vectors


def build_mosaic(
    reference: Image.Image,
    vectors: np.ndarray,
    filenames: np.ndarray,
    images_dir: str,
    cells: int,
    rows_override: int | None = None,
) -> Image.Image:
    """Build the mosaic image."""
    cols = cells
    rows = rows_override if rows_override is not None else round(cells * 1.5)

    n_cells = cols * rows
    n_posters = len(filenames)

    # Phase 1: Compute all cell vectors
    print(f"Matching {cols}x{rows} = {n_cells} cells...")
    cell_vectors = compute_all_cell_vectors(reference, cols, rows)

    # Phase 2: Compute squared Euclidean distances via BLAS matmul
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    print("Computing distances...")
    q_sq = np.sum(cell_vectors ** 2, axis=1, keepdims=True)  # (n_cells, 1)
    v_sq = np.sum(vectors ** 2, axis=1, keepdims=True).T     # (1, n_posters)
    dot = cell_vectors @ vectors.T                            # (n_cells, n_posters)
    dists = q_sq + v_sq - 2 * dot                            # (n_cells, n_posters)

    # Sort all candidates by distance for each cell
    print("Sorting candidates...")
    all_sorted = np.argsort(dists, axis=1)  # (n_cells, n_posters)

    # Phase 3: Sequential greedy assignment
    used = set()
    tile_assignments = []

    print("Assigning tiles...")
    for i in range(n_cells):
        indices = all_sorted[i]
        chosen = None
        for idx in indices:
            if idx not in used:
                chosen = idx
                used.add(idx)
                break
        if chosen is None:
            chosen = indices[0]
        tile_assignments.append(chosen)

    # Phase 4: Assemble mosaic using cv2 for fast JPEG loading
    mosaic_w = cols * POSTER_W
    mosaic_h = rows * POSTER_H
    mosaic_arr = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    print(f"Assembling {mosaic_w}x{mosaic_h} mosaic...")
    for i, idx in enumerate(tqdm(tile_assignments, desc="Placing tiles")):
        fname = filenames[idx]
        path = os.path.join(images_dir, fname)
        tile = cv2.imread(path, cv2.IMREAD_COLOR)
        if tile is None:
            continue
        tile_rgb = tile[:, :, ::-1]  # BGR -> RGB
        row = i // cols
        col = i % cols
        y = row * POSTER_H
        x = col * POSTER_W
        mosaic_arr[y:y + POSTER_H, x:x + POSTER_W] = tile_rgb

    return Image.fromarray(mosaic_arr)


def main():
    parser = argparse.ArgumentParser(description="Build a photomosaic from a reference image.")
    parser.add_argument("--reference", required=True, help="Path to the reference image")
    parser.add_argument("--data", default="grid_data.npz", help="Precomputed grid data (default: grid_data.npz)")
    parser.add_argument(
        "--images",
        default=os.environ.get("MOSAIC_IMAGES_DIR", "images"),
        help="Directory containing poster images",
    )
    parser.add_argument("--cells", type=int, default=30, help="Number of columns in the mosaic grid")
    parser.add_argument("--rows", type=int, default=None, help="Number of rows (default: auto-calculated for 2:3 tiles)")
    parser.add_argument("--output", default="mosaic.jpg", help="Output file path (default: mosaic.jpg)")
    args = parser.parse_args()

    # Load precomputed data
    print(f"Loading precomputed data from {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    vectors = data["vectors"]
    filenames = data["filenames"]
    print(f"Loaded {len(filenames)} poster vectors ({vectors.shape[1]}D)")

    # Load and crop reference image
    ref = Image.open(args.reference).convert("RGB")
    ref = crop_to_aspect(ref, 2, 3)
    print(f"Reference image: {ref.size[0]}x{ref.size[1]} (cropped to 2:3)")

    # Build mosaic
    mosaic = build_mosaic(ref, vectors, filenames, args.images, args.cells, args.rows)

    # Save
    mosaic.save(args.output, quality=95)
    print(f"Saved mosaic to {args.output} ({mosaic.size[0]}x{mosaic.size[1]})")


if __name__ == "__main__":
    main()
