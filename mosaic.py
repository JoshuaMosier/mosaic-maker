"""Build a photomosaic from a reference image using precomputed poster data.

Divides the reference image into a grid of cells, matches each cell to
the most visually similar poster using sRGB color vectors and a KD-tree,
then assembles the matched posters into a high-resolution mosaic.

Usage:
    python mosaic.py --reference input.jpg --data grid_data.npz --images path/to/posters --cells 30
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial import KDTree
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


def build_mosaic(
    reference: Image.Image,
    tree: KDTree,
    filenames: np.ndarray,
    images_dir: str,
    cells: int,
    rows_override: int | None = None,
) -> Image.Image:
    """Build the mosaic image.

    Args:
        reference: Reference image (already cropped to 2:3).
        tree: KDTree built from precomputed sRGB vectors.
        filenames: Array of poster filenames matching tree indices.
        images_dir: Path to poster image directory.
        cells: Number of columns in the mosaic grid.
        rows_override: Explicit row count, or None to auto-calculate for 2:3 tiles.

    Returns:
        Assembled mosaic as a PIL Image.
    """
    ref_w, ref_h = reference.size
    cols = cells
    rows = rows_override if rows_override is not None else round(cells * 1.5)

    cell_w = ref_w / cols
    cell_h = ref_h / rows

    used = set()
    tile_assignments = []

    print(f"Matching {cols}x{rows} = {cols * rows} cells...")
    for row in tqdm(range(rows), desc="Matching rows"):
        for col in range(cols):
            left = int(col * cell_w)
            top = int(row * cell_h)
            right = int((col + 1) * cell_w)
            bottom = int((row + 1) * cell_h)
            cell_img = reference.crop((left, top, right, bottom))

            query = image_to_vector(cell_img)

            # Find nearest unused poster
            dists, indices = tree.query(query, k=min(len(filenames), 100))
            chosen = None
            for idx in indices:
                if idx not in used:
                    chosen = idx
                    used.add(idx)
                    break

            if chosen is None:
                _, all_indices = tree.query(query, k=len(filenames))
                for idx in all_indices:
                    if idx not in used:
                        chosen = idx
                        used.add(idx)
                        break

            if chosen is None:
                chosen = indices[0]

            tile_assignments.append(filenames[chosen])

    # Assemble the mosaic
    mosaic_w = cols * POSTER_W
    mosaic_h = rows * POSTER_H
    mosaic = Image.new("RGB", (mosaic_w, mosaic_h))

    print(f"Assembling {mosaic_w}x{mosaic_h} mosaic...")
    for i, fname in enumerate(tqdm(tile_assignments, desc="Placing tiles")):
        row = i // cols
        col = i % cols
        path = os.path.join(images_dir, fname)
        try:
            tile = Image.open(path).convert("RGB")
        except Exception:
            continue
        mosaic.paste(tile, (col * POSTER_W, row * POSTER_H))

    return mosaic


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

    # Build KD-tree
    print("Building KD-tree...")
    tree = KDTree(vectors)

    # Load and crop reference image
    ref = Image.open(args.reference).convert("RGB")
    ref = crop_to_aspect(ref, 2, 3)
    print(f"Reference image: {ref.size[0]}x{ref.size[1]} (cropped to 2:3)")

    # Build mosaic
    mosaic = build_mosaic(ref, tree, filenames, args.images, args.cells, args.rows)

    # Save
    mosaic.save(args.output, quality=95)
    print(f"Saved mosaic to {args.output} ({mosaic.size[0]}x{mosaic.size[1]})")


if __name__ == "__main__":
    main()
