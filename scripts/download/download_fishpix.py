#!/usr/bin/env python3
"""
Download fish images from FishPix (fishpix.kahaku.go.jp).

FishPix is a database of 217,476 fish photographs maintained by the
Kanagawa Prefectural Museum of Natural History and the National Museum
of Nature and Science, Japan.

This script:
1. Searches FishPix by family name to discover available images
2. Downloads images with proper attribution
3. Tracks which images appeared in the Miyazawa (2020) study
4. Creates a manifest CSV for the inventory system

URL pattern for images:
  https://fishpix.kahaku.go.jp/photos/NR{prefix}/{id}AF.{ext}
  where prefix = int(id) // 1000, zero-padded to 4 digits

Usage:
    python download_fishpix.py --family Chaetodontidae
    python download_fishpix.py --family Chaetodontidae --dry-run
    python download_fishpix.py --family Chaetodontidae --miyazawa-only
"""

import os
import re
import csv
import time
import json
import argparse
import urllib.request
import urllib.error
from urllib.parse import urlencode, quote
from collections import defaultdict

import openpyxl


# ---------- paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MIYAZAWA_XLSX = os.path.join(PROJECT_DIR, "papers", "abb9107_data_file_s1.xlsx")

# Output paths (will be set based on family)
IMAGE_DIR_TEMPLATE = os.path.join(PROJECT_DIR, "images_fishpix")
MANIFEST_CSV_TEMPLATE = os.path.join(DATA_DIR, "fishpix", "fishpix_download_manifest.csv")
CACHE_DIR = os.path.join(DATA_DIR, "fishpix", "cache")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (compatible; Academic-Fish-Research)"
)

# FishPix base URLs
# Search uses GET: /fishimage-e/search?FAMILY=Chaetodontidae&FAMILY_OPT=0&START=1
FISHPIX_SEARCH_URL = "https://fishpix.kahaku.go.jp/fishimage-e/search"
FISHPIX_DETAIL_URL = "https://fishpix.kahaku.go.jp/fishimage-e/detail.php"
FISHPIX_IMAGE_BASE = "https://fishpix.kahaku.go.jp/photos"

# Results per page (FishPix default)
RESULTS_PER_PAGE = 20


# ---------- Miyazawa data loader ----------
def load_miyazawa_fishpix_images(xlsx_path, family_filter=None):
    """Load FishPix images from Miyazawa (2020) study.

    Args:
        xlsx_path: Path to Miyazawa supplementary data Excel file
        family_filter: If provided, only return images from this family

    Returns dict mapping img_file -> species for FishPix images.
    """
    if not os.path.exists(xlsx_path):
        print(f"  WARNING: Miyazawa data file not found: {xlsx_path}")
        return {}

    miyazawa_images = {}
    try:
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        ws = wb["A_FishPatterns_img"]

        for row in ws.iter_rows(min_row=2, values_only=True):
            img_file = row[0]
            db = row[1]
            family = row[2]
            species = row[4]

            if db == "FishPix":
                if family_filter is None or family == family_filter:
                    miyazawa_images[img_file] = species

        wb.close()
    except Exception as e:
        print(f"  WARNING: Could not load Miyazawa data: {e}")

    return miyazawa_images


# ---------- URL construction ----------
def fishpix_image_url(photo_id, nr_prefix=None, ext="jpg"):
    """Construct FishPix image URL from photo ID.

    Photo IDs follow pattern: 12345 or 12345AF
    Thumbnails: NR0012/12345AI.jpg (I = thumbnail)
    Full images: NR0012/12345AF.jpg (F = full)
    URL: https://fishpix.kahaku.go.jp/photos/NR0012/12345AF.jpg

    Args:
        photo_id: Numeric photo ID (e.g., "12345" or "12345AF")
        nr_prefix: If known, the NR prefix (e.g., "NR0012")
        ext: File extension (default: jpg)

    Returns tuple: (url, filename)
    """
    # Extract numeric ID, stripping any suffixes like AF
    numeric_str = re.sub(r'[^\d]', '', str(photo_id))
    numeric_id = int(numeric_str) if numeric_str else 0

    # Use provided prefix or calculate
    if nr_prefix:
        prefix = nr_prefix
    else:
        prefix = f"NR{numeric_id // 1000:04d}"

    filename = f"{numeric_id}AF.{ext}"
    url = f"{FISHPIX_IMAGE_BASE}/{prefix}/{filename}"
    return url, filename


def parse_fishpix_search_results(html_content):
    """Parse FishPix search results HTML to extract image info.

    FishPix search results have this structure for each image:
    - Thumbnail: ../photos/NR0010/10004AI.jpg (AI = thumbnail)
    - Full image: ../photos/NR0010/10004AF.jpg (AF = full)
    - Catalog: [KPM-NR 10004]
    - Species: Chaetodon vagabundus Linnaeus, 1758 (plain text after image)

    Returns list of dicts with pic_id, nr_prefix, species.
    """
    results = []

    # Pattern to find thumbnail image references
    # ../photos/NR0010/10004AI.jpg
    thumb_pattern = re.compile(
        r'\.\./photos/(NR\d{4})/(\d+)AI\.jpg',
        re.IGNORECASE
    )

    # Species name pattern: extract from <SPAN class="result">Genus species Author</SPAN>
    # The species text SPAN comes after the image SPAN for each entry.
    # Handles both "Genus species Author, Year" and "Genus species (Author, Year)"
    species_span_pattern = re.compile(
        r'<SPAN\s+class="result">\s*([A-Z][a-z]+)\s+([a-z]+)\s',
        re.IGNORECASE
    )

    # Find all thumbnails and extract associated species
    for match in thumb_pattern.finditer(html_content):
        nr_prefix = match.group(1)  # e.g., NR0010
        pic_id = match.group(2)      # e.g., 10004

        # Look for species name in a <SPAN class="result"> AFTER this thumbnail
        after_pos = match.end()
        end_pos = min(len(html_content), after_pos + 600)
        context_after = html_content[after_pos:end_pos]

        species_match = species_span_pattern.search(context_after)

        if species_match:
            genus = species_match.group(1)
            epithet = species_match.group(2)
            species = f"{genus} {epithet}"
        else:
            species = "Unknown"

        results.append({
            'pic_id': pic_id,
            'nr_prefix': nr_prefix,
            'species': species,
        })

    return results


def extract_total_hits(html_content):
    """Extract total number of hits from search results page.

    FishPix shows "4903 hits" or similar at top of results.
    """
    # Pattern: "4903 hits" or "4,903 hits"
    hits_pattern = re.compile(r'([\d,]+)\s*hits?', re.IGNORECASE)
    match = hits_pattern.search(html_content)
    if match:
        return int(match.group(1).replace(',', ''))
    return 0


def fetch_url(url, timeout=30):
    """Fetch URL content with proper headers."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return None


def search_fishpix_by_family(family_name, cache_file=None, delay=0.5):
    """Search FishPix for all images in a family.

    FishPix search uses GET requests:
    https://fishpix.kahaku.go.jp/fishimage-e/search?FAMILY=Chaetodontidae&FAMILY_OPT=0&START=1

    Pagination via START parameter (1, 21, 41, 61, ...)
    Returns 20 results per page.

    Returns list of image records.
    """
    # Check cache first
    if cache_file and os.path.exists(cache_file):
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    print(f"  Searching FishPix for family: {family_name}")

    all_results = []
    total_hits = 0

    # First request to get total count
    first_url = f"{FISHPIX_SEARCH_URL}?FAMILY={quote(family_name)}&FAMILY_OPT=0&START=1"
    html = fetch_url(first_url)

    if not html:
        print(f"  ERROR: Could not fetch search results")
        return []

    # Check for no results
    if 'No data found' in html or '0 hits' in html.lower():
        print(f"  No results found for family {family_name}")
        return []

    # Extract total hits
    total_hits = extract_total_hits(html)
    print(f"  Total images available: {total_hits}")

    # Parse first page
    results = parse_fishpix_search_results(html)
    all_results.extend(results)
    print(f"  Page 1: {len(results)} images")

    # Calculate number of pages needed
    if total_hits > RESULTS_PER_PAGE:
        n_pages = (total_hits + RESULTS_PER_PAGE - 1) // RESULTS_PER_PAGE
        print(f"  Need to fetch {n_pages} pages total")

        # Fetch remaining pages
        for page in range(2, n_pages + 1):
            start = (page - 1) * RESULTS_PER_PAGE + 1
            page_url = f"{FISHPIX_SEARCH_URL}?FAMILY={quote(family_name)}&FAMILY_OPT=0&START={start}"

            time.sleep(delay)  # Be polite to server

            html = fetch_url(page_url)
            if not html:
                print(f"    Page {page} (START={start}): FAILED")
                continue

            page_results = parse_fishpix_search_results(html)
            all_results.extend(page_results)

            # Progress update every 10 pages
            if page % 10 == 0 or page == n_pages:
                print(f"    Pages 1-{page}: {len(all_results)} images collected")

    # Deduplicate by pic_id (in case of any overlap)
    seen = set()
    unique_results = []
    for r in all_results:
        if r['pic_id'] not in seen:
            seen.add(r['pic_id'])
            unique_results.append(r)

    print(f"  Total unique images: {len(unique_results)}")

    # Save to cache
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(unique_results, f, indent=2)
        print(f"  Cached results to {cache_file}")

    return unique_results


def download_image(url, dest_path, retries=3, timeout=30):
    """Download a single image with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                with open(dest_path, "wb") as f:
                    f.write(data)
                return True, len(data)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                return False, str(e)
    return False, "max retries"


def safe_filename(species, pic_id):
    """Build a safe local filename: Genus_species_FishPix_ID.jpg"""
    safe_sp = species.strip().replace(" ", "_")
    return f"{safe_sp}_FishPix_{pic_id}.jpg"


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Download fish images from FishPix database"
    )
    ap.add_argument("--family", type=str, required=True,
                    help="Family name to search for (e.g., Chaetodontidae)")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="Seconds between downloads (default: 1.0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be downloaded without downloading")
    ap.add_argument("--limit", type=int, default=0,
                    help="Download only first N images (0 = all)")
    ap.add_argument("--miyazawa-only", action="store_true",
                    help="Only download images from Miyazawa (2020) study")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-download even if file exists")
    ap.add_argument("--no-cache", action="store_true",
                    help="Don't use cached search results")
    ap.add_argument("--image-dir", type=str, default=None,
                    help="Output directory (default: images_fishpix/)")
    args = ap.parse_args()

    family = args.family
    image_dir = args.image_dir or IMAGE_DIR_TEMPLATE
    cache_file = None if args.no_cache else os.path.join(
        CACHE_DIR, f"fishpix_search_{family.lower()}.json"
    )
    manifest_csv = MANIFEST_CSV_TEMPLATE

    print("=" * 64)
    print("FishPix Image Downloader")
    print("=" * 64)
    print(f"  Family:     {family}")
    print(f"  Output:     {image_dir}")
    print(f"  Delay:      {args.delay}s between requests")
    if args.miyazawa_only:
        print(f"  MODE:       Miyazawa (2020) images only")
    if args.dry_run:
        print(f"  MODE:       DRY RUN (no downloads)")
    if args.limit:
        print(f"  Limit:      first {args.limit} images only")
    print()

    # Load Miyazawa data (filtered by family)
    print(f"[1] Loading Miyazawa (2020) FishPix images for {family}...")
    miyazawa_images = load_miyazawa_fishpix_images(MIYAZAWA_XLSX, family_filter=family)
    print(f"    {len(miyazawa_images)} FishPix images in Miyazawa study for {family}")

    # Search FishPix or use Miyazawa list
    if args.miyazawa_only:
        print("\n[2] Using Miyazawa image list only...")
        search_results = []
        for img_file, species in miyazawa_images.items():
            # Extract pic_id from filename like "12345AF.jpg"
            pic_id = img_file.replace(".jpg", "").replace(".JPG", "")
            search_results.append({
                'pic_id': pic_id,
                'species': species,
                'in_miyazawa': True,
            })
        print(f"    {len(search_results)} images to download")
    else:
        print("\n[2] Searching FishPix database...")
        search_results = search_fishpix_by_family(family, cache_file)
        print(f"    {len(search_results)} total images found")

        # Mark Miyazawa images
        for result in search_results:
            pic_id = result['pic_id']
            # Miyazawa filenames are like "12345AF.jpg"
            img_file = f"{pic_id}AF.jpg"
            result['in_miyazawa'] = img_file in miyazawa_images
            if result['in_miyazawa']:
                # Use species from Miyazawa if available (more reliable)
                result['species'] = miyazawa_images.get(img_file, result.get('species', 'Unknown'))

        n_miyazawa = sum(1 for r in search_results if r.get('in_miyazawa'))
        print(f"    {n_miyazawa} of these are in Miyazawa (2020)")

    if not search_results:
        print("\nNo images found. Exiting.")
        return

    if args.limit:
        search_results = search_results[:args.limit]
        print(f"\n    Limited to first {args.limit} images")

    # Create directories
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_csv), exist_ok=True)

    if args.dry_run:
        print(f"\n[DRY RUN] Would download {len(search_results)} images:")
        for i, result in enumerate(search_results[:20]):
            pic_id = result['pic_id']
            species = result.get('species', 'Unknown')
            url, _ = fishpix_image_url(pic_id)
            local = safe_filename(species, pic_id)
            miyazawa_flag = " [MIYAZAWA]" if result.get('in_miyazawa') else ""
            print(f"  {species}: {pic_id}{miyazawa_flag}")
            print(f"    {url} -> {local}")
        if len(search_results) > 20:
            print(f"  ... and {len(search_results) - 20} more")
        return

    # Download
    print(f"\n[3] Downloading {len(search_results)} images...")
    manifest = []
    n_downloaded = 0
    n_skipped = 0
    n_failed = 0
    total_bytes = 0

    for i, result in enumerate(search_results):
        pic_id = result['pic_id']
        species = result.get('species', 'Unknown')
        in_miyazawa = result.get('in_miyazawa', False)
        nr_prefix = result.get('nr_prefix')  # Use if available from search

        url, original_filename = fishpix_image_url(pic_id, nr_prefix=nr_prefix)
        local_filename = safe_filename(species, pic_id)
        dest_path = os.path.join(image_dir, local_filename)

        miyazawa_flag = " [MIYAZAWA]" if in_miyazawa else ""
        print(f"  [{i+1:4d}/{len(search_results)}] {species}{miyazawa_flag}", end="", flush=True)

        # Skip if exists
        if os.path.exists(dest_path) and not args.overwrite:
            size = os.path.getsize(dest_path)
            print(f" -> exists ({size:,} bytes)")
            n_skipped += 1
            manifest.append({
                "species": species,
                "pic_id": pic_id,
                "original_filename": original_filename,
                "local_filename": local_filename,
                "url": url,
                "family": family,
                "in_miyazawa": "yes" if in_miyazawa else "no",
                "download_status": "exists",
                "file_size": size,
            })
            continue

        # Download
        success, result_data = download_image(url, dest_path)

        if success:
            n_downloaded += 1
            total_bytes += result_data
            print(f" -> OK ({result_data:,} bytes)")
            manifest.append({
                "species": species,
                "pic_id": pic_id,
                "original_filename": original_filename,
                "local_filename": local_filename,
                "url": url,
                "family": family,
                "in_miyazawa": "yes" if in_miyazawa else "no",
                "download_status": "success",
                "file_size": result_data,
            })
        else:
            n_failed += 1
            print(f" -> FAILED: {result_data}")
            manifest.append({
                "species": species,
                "pic_id": pic_id,
                "original_filename": original_filename,
                "local_filename": local_filename,
                "url": url,
                "family": family,
                "in_miyazawa": "yes" if in_miyazawa else "no",
                "download_status": f"failed: {result_data}",
                "file_size": 0,
            })

        time.sleep(args.delay)

    # Write manifest
    print(f"\n[4] Writing download manifest...")
    manifest_fields = [
        "species", "pic_id", "original_filename", "local_filename",
        "url", "family", "in_miyazawa", "download_status", "file_size",
    ]
    with open(manifest_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        w.writerows(manifest)
    print(f"    {manifest_csv}")

    # Summary
    species_set = {r["species"] for r in manifest if r["species"] != "Unknown"}
    n_miyazawa = sum(1 for r in manifest if r["in_miyazawa"] == "yes")

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Total images:        {len(search_results)}")
    print(f"  Downloaded:          {n_downloaded}")
    print(f"  Already existed:     {n_skipped}")
    print(f"  Failed:              {n_failed}")
    print(f"  Total bytes:         {total_bytes:,}")
    print(f"  Total MB:            {total_bytes / 1024 / 1024:.1f}")
    print(f"  Species covered:     {len(species_set)}")
    print(f"  In Miyazawa (2020):  {n_miyazawa}")
    print(f"\n  Images:   {image_dir}/")
    print(f"  Manifest: {manifest_csv}")

    if n_failed:
        print(f"\n  WARNING: {n_failed} downloads failed.")
        print(f"  Re-run the script to retry (existing files will be skipped).")


if __name__ == "__main__":
    main()
