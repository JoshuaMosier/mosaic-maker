"""Build a single metadata file for all downloaded poster images.

Merges existing metadata sources (bulk_movie_metadata.json, new_metadata.json)
and optionally backfills missing entries from Letterboxd film pages.

Usage:
    python scripts/build_metadata.py                    # Merge existing only
    python scripts/build_metadata.py --backfill         # Also fetch missing from Letterboxd
    python scripts/build_metadata.py --backfill --limit 1000  # Backfill up to 1000
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

IMAGES_DIR = "images"
OUTPUT_PATH = "data/poster_metadata.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

JSON_LD_RE = re.compile(
    r'<script\s+type="application/ld\+json">\s*'
    r'(?:/\*\s*<!\[CDATA\[\s*\*/\s*)?'
    r'(.*?)'
    r'(?:\s*/\*\s*\]\]>\s*\*/\s*)?'
    r'</script>',
    re.DOTALL,
)

_thread_local = threading.local()


def get_thread_session(pool_size: int = 20) -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers["User-Agent"] = UA
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _thread_local.session = s
    return _thread_local.session


def fetch_lb_metadata(session: requests.Session, slug: str, max_retries: int = 4) -> dict | None:
    """Fetch metadata from a Letterboxd film page. Returns None on failure."""
    # Skip TMDB-only slugs
    if slug.startswith("tmdb-"):
        return None

    url = f"https://letterboxd.com/film/{slug}/"
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=10)
        except requests.RequestException:
            return None

        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code != 200:
            return None

        match = JSON_LD_RE.search(resp.text)
        if not match:
            return None

        try:
            data = json.loads(match.group(1).strip())
            aggregate = data.get("aggregateRating", {})
            directors = data.get("director", [])
            return {
                "slug": slug,
                "name": data.get("name", ""),
                "director": directors[0]["name"] if directors else "",
                "year": data.get("releasedEvent", [{}])[0].get("startDate", ""),
                "genre": data.get("genre", []),
                "rating": aggregate.get("ratingValue", ""),
                "num_ratings": aggregate.get("ratingCount", 0),
                "source": "letterboxd",
            }
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return None

    return None


def main():
    parser = argparse.ArgumentParser(description="Build unified poster metadata.")
    parser.add_argument("--backfill", action="store_true", help="Fetch missing metadata from Letterboxd")
    parser.add_argument("--limit", type=int, default=None, help="Max films to backfill")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers for backfill (default: 32)")
    args = parser.parse_args()

    # Step 1: Scan all images
    print(f"Scanning {IMAGES_DIR}...")
    image_index = {}  # slug -> file_index
    for f in os.listdir(IMAGES_DIR):
        if f.endswith(".jpg"):
            parts = f.split("_", 1)
            if len(parts) == 2:
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                slug = parts[1].replace(".jpg", "")
                image_index[slug] = idx
    print(f"  {len(image_index)} images")

    # Step 2: Load existing metadata sources
    metadata = {}  # slug -> dict

    # Old bulk metadata
    old_path = "data/bulk_movie_metadata.json"
    if os.path.isfile(old_path):
        with open(old_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        for m in old_data:
            if not isinstance(m, dict) or "Film URL" not in m:
                continue
            slug = m["Film URL"].replace("https://letterboxd.com/film/", "").rstrip("/")
            if slug in image_index:
                metadata[slug] = {
                    "slug": slug,
                    "file_index": image_index[slug],
                    "name": m.get("Film Name", ""),
                    "director": m.get("Director", ""),
                    "year": str(m.get("Release Date", "")),
                    "genre": m.get("Genre", []),
                    "country": m.get("Country of Origin", []),
                    "rating": m.get("Rating", ""),
                    "num_ratings": m.get("Number of Ratings", 0),
                    "source": "letterboxd",
                }
        print(f"  From bulk_movie_metadata.json: {len(metadata)}")

    # New metadata
    new_path = "data/new_metadata.json"
    if os.path.isfile(new_path):
        with open(new_path, "r", encoding="utf-8") as f:
            new_data = json.load(f)
        added = 0
        for m in new_data:
            slug = m.get("slug", "")
            if slug in image_index and slug not in metadata:
                m["file_index"] = image_index[slug]
                if "source" not in m:
                    m["source"] = "letterboxd"
                metadata[slug] = m
                added += 1
        print(f"  From new_metadata.json: {added} new")

    # Existing poster_metadata.json (from previous runs)
    if os.path.isfile(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
        added = 0
        for m in existing:
            slug = m.get("slug", "")
            if slug in image_index and slug not in metadata:
                metadata[slug] = m
                added += 1
        print(f"  From previous poster_metadata.json: {added} new")

    # Ensure file_index is set for all
    for slug, m in metadata.items():
        if "file_index" not in m:
            m["file_index"] = image_index.get(slug, -1)

    missing = [slug for slug in image_index if slug not in metadata]
    print(f"\n  Total with metadata: {len(metadata)}")
    print(f"  Missing metadata: {len(missing)}")

    # Step 3: Backfill from Letterboxd
    if args.backfill and missing:
        # Skip tmdb- slugs for backfill (no Letterboxd page)
        backfill_slugs = [s for s in missing if not s.startswith("tmdb-")]
        tmdb_only = len(missing) - len(backfill_slugs)
        if tmdb_only:
            print(f"  Skipping {tmdb_only} tmdb-only slugs (no Letterboxd page)")

        if args.limit:
            backfill_slugs = backfill_slugs[:args.limit]
        print(f"\n  Backfilling {len(backfill_slugs)} from Letterboxd ({args.workers} workers)...")

        succeeded = 0
        failed = 0
        status_counts = Counter()

        def fetch_one(slug):
            s = get_thread_session(pool_size=args.workers)
            return slug, fetch_lb_metadata(s, slug)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(fetch_one, s): s for s in backfill_slugs}
            with tqdm(total=len(backfill_slugs), desc="Backfilling metadata") as pbar:
                for future in as_completed(futures):
                    slug, result = future.result()
                    if result is not None:
                        result["file_index"] = image_index[slug]
                        metadata[slug] = result
                        succeeded += 1
                    else:
                        failed += 1
                    pbar.update(1)

                    # Periodic save
                    if succeeded > 0 and succeeded % 2000 == 0:
                        _save(metadata, OUTPUT_PATH)

        print(f"  Backfilled: {succeeded}")
        print(f"  Failed: {failed}")

    # Step 4: Save
    _save(metadata, OUTPUT_PATH)

    # Summary
    sources = Counter(m.get("source", "unknown") for m in metadata.values())
    print(f"\nSaved {len(metadata)} entries to {OUTPUT_PATH}")
    print(f"  Sources: {dict(sources)}")
    print(f"  Still missing: {len(image_index) - len(metadata)}")


def _save(metadata: dict, path: str):
    """Save metadata dict to JSON, sorted by file_index."""
    entries = sorted(metadata.values(), key=lambda m: m.get("file_index", 0))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
