"""Fetch new movie poster images from Letterboxd or TMDB.

Sources films from full_data.json (cross-referenced lists), Letterboxd list
URLs, or TMDB discover API (by year). Skips posters already downloaded.

Usage:
    python fetch_new_posters.py [--source lists|url|tmdb] [options]

Examples:
    python fetch_new_posters.py --source lists --min-lists 2
    python fetch_new_posters.py --source url --url https://letterboxd.com/user/list/my-list/
    python fetch_new_posters.py --source tmdb --year 2024 2025 --tmdb-key YOUR_KEY
"""

import argparse
import io
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
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm

SINGLE_LIST_URL = "https://letterboxd.com/sprudelheinz/list/all-the-movies-sorted-by-movie-posters-1/"
FULL_DATA_PATH = os.path.join(os.path.dirname(__file__), "full_data.json")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
POSTER_W, POSTER_H = 230, 345
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

# Characters illegal in Windows filenames
UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')

# Regex to extract JSON-LD from film pages (avoids parsing full HTML with BS4)
JSON_LD_RE = re.compile(
    r'<script\s+type="application/ld\+json">\s*'
    r'(?:/\*\s*<!\[CDATA\[\s*\*/\s*)?'
    r'(.*?)'
    r'(?:\s*/\*\s*\]\]>\s*\*/\s*)?'
    r'</script>',
    re.DOTALL,
)

# Thread-local storage for reusing sessions (HTTP keep-alive)
_thread_local = threading.local()


def safe_filename(slug: str) -> str:
    """Sanitize a slug for use in a Windows filename."""
    return UNSAFE_CHARS.sub("_", slug)


def get_thread_session(pool_size: int = 20) -> requests.Session:
    """Return a per-thread session with connection pooling and keep-alive."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers["User-Agent"] = UA
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _thread_local.session = s
    return _thread_local.session


def get_existing_slugs(images_dir: str) -> set[str]:
    """Get the set of film slugs already downloaded (from filenames like 123_slug.jpg)."""
    slugs = set()
    for f in os.listdir(images_dir):
        if f.endswith(".jpg"):
            parts = f.split("_", 1)
            if len(parts) == 2:
                slugs.add(parts[1].replace(".jpg", ""))
    return slugs


def get_next_index(images_dir: str) -> int:
    """Find the next available file index."""
    max_idx = 0
    for f in os.listdir(images_dir):
        if f.endswith(".jpg"):
            try:
                idx = int(f.split("_", 1)[0])
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
    return max_idx + 1


def load_films_from_full_data(min_lists: int) -> list[dict]:
    """Load films from full_data.json, ranked by how many lists they appear in."""
    if not os.path.isfile(FULL_DATA_PATH):
        print(f"Error: {FULL_DATA_PATH} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {FULL_DATA_PATH}...")
    with open(FULL_DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    slug_counts = Counter()
    slug_ids = {}
    for lst in data:
        for film in lst.get("films", []):
            slug = film["film_slug"]
            slug_counts[slug] += 1
            if slug not in slug_ids:
                slug_ids[slug] = film.get("film_id", "")

    print(f"  {len(data)} lists, {len(slug_counts)} unique films")

    # Filter and sort by popularity
    films = [
        {"slug": slug, "film_id": slug_ids[slug], "list_count": count}
        for slug, count in slug_counts.items()
        if count >= min_lists
    ]
    films.sort(key=lambda x: -x["list_count"])
    print(f"  {len(films)} films appear in {min_lists}+ lists")
    return films


def scrape_list_url(session: requests.Session, list_url: str, max_pages: int | None = None) -> list[dict]:
    """Scrape any Letterboxd list URL for films."""
    # Normalize URL: ensure trailing slash
    if not list_url.endswith("/"):
        list_url += "/"

    print(f"\nScraping list: {list_url}")
    for attempt in range(3):
        resp = session.get(list_url)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        break
    if resp.status_code != 200:
        print(f"  Skipping — HTTP {resp.status_code}")
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    pages_el = soup.select("li.paginate-page a")
    total_pages = int(pages_el[-1].text) if pages_el else 1
    if max_pages:
        total_pages = min(total_pages, max_pages)
    print(f"  {total_pages} pages")

    films = []
    for page in tqdm(range(1, total_pages + 1), desc="Scraping list pages"):
        url = list_url if page == 1 else f"{list_url}page/{page}/"
        r = None
        for attempt in range(3):
            try:
                r = session.get(url)
                if r.status_code == 429 or r.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    print(f"\n  Warning: page {page} failed: {e}")
                time.sleep(2 ** attempt)
        if r is None or r.status_code != 200:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for el in soup.select('div.react-component[data-component-class="LazyPoster"]'):
            slug = el.get("data-item-slug")
            film_id = el.get("data-film-id")
            if slug and film_id:
                films.append({"slug": slug, "film_id": film_id})
        time.sleep(0.1)

    print(f"  {len(films)} films found")
    return films


def fetch_film_data(session: requests.Session, slug: str, max_retries: int = 4) -> tuple[dict | None, str]:
    """Fetch poster URL and metadata from a film's Letterboxd page.

    Uses regex to extract JSON-LD (avoids full HTML parse with BS4).
    Retries with exponential backoff on rate limits (429) and server errors (5xx).
    Returns (data_dict, status) where status is 'ok', 'no_poster', 'http_NNN', or 'error'.
    """
    url = f"https://letterboxd.com/film/{slug}/"

    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=10)
        except requests.RequestException:
            return None, "error"

        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2 ** attempt  # 1, 2, 4, 8 seconds
            time.sleep(wait)
            continue

        if resp.status_code == 404:
            return None, "http_404"

        if resp.status_code != 200:
            return None, f"http_{resp.status_code}"

        match = JSON_LD_RE.search(resp.text)
        if not match:
            return None, "no_poster"

        try:
            data = json.loads(match.group(1).strip())
            poster_url = data.get("image")
            if not poster_url or "empty-poster" in poster_url:
                return None, "no_poster"

            aggregate = data.get("aggregateRating", {})
            directors = data.get("director", [])

            return {
                "slug": slug,
                "poster_url": poster_url,
                "name": data.get("name", ""),
                "director": directors[0]["name"] if directors else "",
                "year": data.get("releasedEvent", [{}])[0].get("startDate", ""),
                "genre": data.get("genre", []),
                "rating": aggregate.get("ratingValue", ""),
                "num_ratings": aggregate.get("ratingCount", 0),
            }, "ok"
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            return None, "parse_error"

    return None, "rate_limited"


def _title_to_slug(title: str) -> str:
    """Convert a film title to a likely Letterboxd slug."""
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return slug


def discover_tmdb_films(api_key: str, years: list[int], max_films: int | None = None,
                        max_per_year: int | None = None, min_votes: int = 0) -> list[dict]:
    """Query TMDB discover API for films by year, sorted by popularity.

    Each film includes a tmdb_poster_url (fallback) and candidate Letterboxd
    slugs to try. The download step will attempt Letterboxd first for the
    normalized 230x345 poster and metadata, falling back to TMDB if needed.
    """
    # Fetch genre mapping
    r = requests.get(f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}")
    r.raise_for_status()
    genre_map = {g["id"]: g["name"] for g in r.json()["genres"]}

    films = []
    for year in years:
        print(f"\nDiscovering TMDB films for {year}...")
        page = 1
        year_count = 0
        while True:
            params = {
                "api_key": api_key,
                "primary_release_year": year,
                "sort_by": "popularity.desc",
                "page": page,
            }
            if min_votes > 0:
                params["vote_count.gte"] = min_votes
            r = requests.get("https://api.themoviedb.org/3/discover/movie", params=params)
            r.raise_for_status()
            data = r.json()

            for f in data["results"]:
                if not f.get("poster_path"):
                    continue
                title = f.get("title", "")
                release_year = f.get("release_date", "")[:4]
                base_slug = _title_to_slug(title)

                # Candidate Letterboxd slugs: try without year, then with year
                lb_slugs = [base_slug]
                if release_year:
                    lb_slugs.append(f"{base_slug}-{release_year}")

                films.append({
                    "slug": f"tmdb-{f['id']}",
                    "tmdb_poster_url": f"{TMDB_IMG_BASE}{f['poster_path']}",
                    "lb_slugs": lb_slugs,
                    "name": title,
                    "year": release_year,
                    "genre": [genre_map.get(gid, "") for gid in f.get("genre_ids", [])],
                    "rating": f.get("vote_average", ""),
                    "num_ratings": f.get("vote_count", 0),
                    "tmdb_popularity": f.get("popularity", 0),
                })
                year_count += 1

            total_pages = min(data["total_pages"], 500)  # TMDB caps at 500
            if page >= total_pages:
                break
            if max_per_year and year_count >= max_per_year:
                break
            page += 1

        print(f"  {year}: {year_count} films")

    films.sort(key=lambda x: -x.get("tmdb_popularity", 0))
    if max_films:
        films = films[:max_films]
    print(f"\n  Total TMDB films: {len(films)}")
    return films


def download_tmdb_poster(session: requests.Session, url: str, path: str) -> bool:
    """Download a TMDB poster and resize to 230x345 only if aspect ratio is 2:3.

    Skips posters that aren't 2:3 aspect ratio.
    Returns True on success.
    """
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))

        # Check aspect ratio: allow small tolerance for rounding
        w, h = img.size
        expected_h = w * 3 / 2
        if abs(h - expected_h) > 2:
            return False

        if (w, h) != (POSTER_W, POSTER_H):
            img = img.resize((POSTER_W, POSTER_H), Image.LANCZOS)

        img.save(path, "JPEG", quality=90)
        return True
    except (requests.RequestException, OSError):
        return False


def download_poster(session: requests.Session, url: str, path: str) -> bool:
    """Download a poster image. Returns True on success."""
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        return True
    except requests.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch new posters from Letterboxd.")
    parser.add_argument(
        "--source", choices=["lists", "url", "tmdb"], default="lists",
        help="Film source: 'lists' uses full_data.json, 'url' scrapes Letterboxd list URL(s), "
             "'tmdb' uses TMDB discover API by year (default: lists)",
    )
    parser.add_argument(
        "--url", nargs="+", default=[SINGLE_LIST_URL],
        help="Letterboxd list URL(s) to scrape (used with --source url)",
    )
    parser.add_argument("--min-lists", type=int, default=2, help="Min list appearances for --source lists (default: 2)")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages per list for --source url")
    parser.add_argument(
        "--tmdb-key", default=os.environ.get("TMDB_API_KEY"),
        help="TMDB API key (or set TMDB_API_KEY env var)",
    )
    parser.add_argument(
        "--year", nargs="+", type=int, default=None,
        help="Release year(s) for --source tmdb (e.g., --year 2024 2025)",
    )
    parser.add_argument(
        "--max-per-year", type=int, default=None,
        help="Max films to discover per year from TMDB (default: all, up to 10K)",
    )
    parser.add_argument(
        "--min-votes", type=int, default=0,
        help="Min TMDB vote count to include a film (default: 0)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be downloaded")
    parser.add_argument("--workers", type=int, default=32, help="Parallel workers (default: 32)")
    parser.add_argument("--limit", type=int, default=None, help="Max new films to download")
    args = parser.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = UA

    # Step 1: Find what we already have
    print(f"Scanning existing images in {IMAGES_DIR}...")
    existing_slugs = get_existing_slugs(IMAGES_DIR)
    existing_count = len(existing_slugs)
    print(f"  Found {existing_count} existing posters")

    next_idx = get_next_index(IMAGES_DIR)

    is_tmdb = args.source == "tmdb"

    # Step 2: Gather film candidates from chosen source
    all_films = []
    if args.source == "lists":
        all_films.extend(load_films_from_full_data(args.min_lists))
    elif args.source == "url":
        for list_url in args.url:
            all_films.extend(scrape_list_url(session, list_url, args.max_pages))
    elif args.source == "tmdb":
        if not args.tmdb_key:
            print("Error: --tmdb-key required (or set TMDB_API_KEY env var)", file=sys.stderr)
            sys.exit(1)
        if not args.year:
            print("Error: --year required with --source tmdb", file=sys.stderr)
            sys.exit(1)
        all_films.extend(discover_tmdb_films(args.tmdb_key, args.year, args.limit, args.max_per_year, args.min_votes))

    # Deduplicate, keeping first occurrence (lists source is pre-sorted by popularity)
    seen = set()
    new_films = []
    for f in all_films:
        slug = f["slug"]
        safe = safe_filename(slug)
        # For TMDB films, also check candidate Letterboxd slugs against existing
        lb_slugs = f.get("lb_slugs", [])
        lb_exists = any(s in existing_slugs or safe_filename(s) in existing_slugs for s in lb_slugs)
        if slug not in seen and slug not in existing_slugs and safe not in existing_slugs and not lb_exists:
            seen.add(slug)
            # Also mark candidate lb_slugs as seen to prevent duplicates
            for s in lb_slugs:
                seen.add(s)
            new_films.append(f)

    print(f"\n  New films to fetch: {len(new_films)}")

    if args.limit:
        new_films = new_films[:args.limit]
        print(f"  Limited to: {len(new_films)}")

    if not new_films:
        print("Nothing new to download!")
        return

    if args.dry_run:
        print(f"\nDry run — would fetch {len(new_films)} new posters. First 20:")
        for f in new_films[:20]:
            extra = f"  ({f['list_count']} lists)" if "list_count" in f else ""
            print(f"  {f['slug']}{extra}")
        return

    # Step 3: Fetch poster URLs, metadata, and download
    print(f"\nFetching poster URLs and downloading ({args.workers} workers)...")
    metadata_path = os.path.join(os.path.dirname(__file__), "new_metadata.json")

    def fetch_and_download(film):
        slug = film["slug"]
        s = get_thread_session(pool_size=args.workers)

        if is_tmdb:
            return _fetch_tmdb_film(s, film)

        film_data, status = fetch_film_data(s, slug)
        if film_data is None:
            return slug, status, None

        path = os.path.join(IMAGES_DIR, f"__temp_{safe_filename(slug)}.jpg")
        if download_poster(s, film_data["poster_url"], path):
            return slug, path, film_data
        return slug, "download_failed", None

    def _fetch_tmdb_film(s, film):
        """Try Letterboxd first for poster + metadata, fall back to TMDB poster."""
        slug = film["slug"]

        # Try each candidate Letterboxd slug
        for lb_slug in film.get("lb_slugs", []):
            lb_data, status = fetch_film_data(s, lb_slug)
            if lb_data is not None:
                # Got Letterboxd data — download the normalized poster
                path = os.path.join(IMAGES_DIR, f"__temp_{safe_filename(lb_slug)}.jpg")
                if download_poster(s, lb_data["poster_url"], path):
                    return lb_slug, path, lb_data
                return slug, "download_failed", None

        # Letterboxd lookup failed — fall back to TMDB poster
        tmdb_url = film.get("tmdb_poster_url")
        if not tmdb_url:
            return slug, "no_poster", None

        path = os.path.join(IMAGES_DIR, f"__temp_{safe_filename(slug)}.jpg")
        if download_tmdb_poster(s, tmdb_url, path):
            # Use TMDB metadata (no Letterboxd enrichment)
            meta = {
                "slug": slug,
                "poster_url": tmdb_url,
                "name": film.get("name", ""),
                "year": film.get("year", ""),
                "genre": film.get("genre", []),
                "rating": film.get("rating", ""),
                "num_ratings": film.get("num_ratings", 0),
                "source": "tmdb",
            }
            return slug, path, meta
        return slug, "bad_aspect_ratio", None

    succeeded = 0
    failed = 0
    status_counts = Counter()
    all_metadata = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_and_download, f): f for f in new_films}
        with tqdm(total=len(new_films), desc="Downloading posters") as pbar:
            for future in as_completed(futures):
                slug, result, film_data = future.result()
                if film_data is None:
                    status_counts[result] += 1
                    if result not in ("no_poster", "http_404", "bad_aspect_ratio"):
                        failed += 1
                else:
                    final_path = os.path.join(IMAGES_DIR, f"{next_idx}_{safe_filename(slug)}.jpg")
                    try:
                        os.rename(result, final_path)
                        film_data["file_index"] = next_idx
                        all_metadata.append(film_data)
                        next_idx += 1
                        succeeded += 1
                    except OSError:
                        # Clean up temp file on rename failure
                        try:
                            os.remove(result)
                        except OSError:
                            pass
                        failed += 1
                pbar.update(1)

                # Periodic metadata save every 500 downloads
                if succeeded > 0 and succeeded % 500 == 0:
                    with open(metadata_path, "w", encoding="utf-8") as mf:
                        json.dump(all_metadata, mf, indent=2, ensure_ascii=False)

    # Clean up any leftover temp files
    for f in os.listdir(IMAGES_DIR):
        if f.startswith("__temp_"):
            os.remove(os.path.join(IMAGES_DIR, f))

    # Final metadata save
    if all_metadata:
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(all_metadata, mf, indent=2, ensure_ascii=False)
        print(f"\n  Metadata saved to {metadata_path} ({len(all_metadata)} films)")

    print(f"\nDone!")
    print(f"  Downloaded: {succeeded}")
    print(f"  Skipped (no poster/404): {status_counts.get('no_poster', 0) + status_counts.get('http_404', 0)}")
    print(f"  Failed: {failed}")
    if status_counts:
        print(f"  Status breakdown: {dict(status_counts)}")
    print(f"  Total images now: {existing_count + succeeded}")


if __name__ == "__main__":
    main()
