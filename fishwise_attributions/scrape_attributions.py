#!/usr/bin/env python3
"""
Gently scrape FishWise Professional image attributions for Chaetodontidae.

Uses the SID cache (from update_fishwise_sid_cache.py) to request each
species gallery page once via /Pictures/?SID={sid}&Find=0. Extracts
species, photographer, filename, and URLs from the HTML. Does NOT
download images.

Output: attributions.csv in this folder.

Usage:
  python scrape_attributions.py [--delay 2.5] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

# Paths relative to this script's directory
FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parent
SID_CACHE = PROJECT / "data" / "fishwise" / "fishwise_sid_cache.json"
SPECIES_CSV = PROJECT / "data" / "species_image_summary.csv"
OUTPUT_CSV = FOLDER / "attributions.csv"

BASE_URL = "https://www.fishwisepro.com"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (compatible; Chaetodontidae-attribution-scrape)"


def load_sid_cache() -> dict[str, int]:
    with open(SID_CACHE) as f:
        raw = json.load(f)
    return {k: int(v) for k, v in (raw or {}).items()}


def load_target_species() -> list[str]:
    """Species we want (from summary CSV) so we only scrape those in cache."""
    out = []
    seen = set()
    with open(SPECIES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            sp = (row.get("species") or "").strip()
            if sp and sp not in seen:
                out.append(sp)
                seen.add(sp)
    return out


def parse_images_from_html(html: str) -> list[dict]:
    """
    Extract image entries from page HTML.
    Pattern: <img src="/pics/{JPG|GIF}/TN/TN{filename}" alt="Species - Photographer">
    """
    results = []
    pattern = (
        r'src="/pics/(JPG|GIF)/TN/TN([^"]+\.(?:jpg|gif))"'
        r'[^>]*alt="([^"]*)"'
    )
    for match in re.finditer(pattern, html, re.I):
        fmt = match.group(1).upper()
        filename = match.group(2)
        alt = match.group(3)

        species = ""
        photographer = ""
        if " - " in alt:
            parts = alt.rsplit(" - ", 1)
            species = parts[0].strip()
            photographer = parts[1].strip()
        else:
            species = alt.strip()

        sid = None
        sid_match = re.match(r"(\d{6})F", filename)
        if sid_match:
            sid = int(sid_match.group(1))

        results.append({
            "species": species,
            "photographer": photographer,
            "filename": filename,
            "full_url": f"{BASE_URL}/pics/{fmt}/{filename}",
            "thumb_url": f"{BASE_URL}/pics/{fmt}/TN/TN{filename}",
            "format": fmt,
            "sid": sid,
        })
    return results


def fetch_page(sid: int, timeout: int = 30):
    import urllib.request
    import urllib.error

    url = f"{BASE_URL}/Pictures/?SID={sid}&Find=0"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description="Scrape FishWise attributions (no image download)")
    ap.add_argument("--delay", type=float, default=2.5, help="Seconds between requests")
    ap.add_argument("--dry-run", action="store_true", help="Print URLs only, do not request")
    ap.add_argument("--timeout", type=int, default=30, help="Request timeout (seconds)")
    args = ap.parse_args()

    print("FishWise Pro — Chaetodontidae attribution scrape")
    print("  SID cache:  ", SID_CACHE)
    print("  Output:     ", OUTPUT_CSV)
    print("  Delay:      ", args.delay, "s")
    print("  No images downloaded.\n")

    target = set(load_target_species())
    cache = load_sid_cache()
    # Only species that are in our target list and have a SID
    to_scrape = [(sp, cache[sp]) for sp in sorted(cache) if sp in target]
    print(f"Species to scrape: {len(to_scrape)}\n")

    if args.dry_run:
        for sp, sid in to_scrape[:5]:
            print(f"  {sp} -> {BASE_URL}/Pictures/?SID={sid}&Find=0")
        if len(to_scrape) > 5:
            print(f"  ... and {len(to_scrape) - 5} more")
        return

    rows = []
    for i, (sp, sid) in enumerate(to_scrape, start=1):
        print(f"  [{i:3d}/{len(to_scrape)}] {sp} (SID={sid})", end="", flush=True)
        try:
            html = fetch_page(sid, timeout=args.timeout)
            images = parse_images_from_html(html)
            for img in images:
                if not img["species"]:
                    img["species"] = sp
                rows.append({
                    "species": img["species"],
                    "photographer": img["photographer"],
                    "filename": img["filename"],
                    "thumb_url": img["thumb_url"],
                    "full_url": img["full_url"],
                    "format": img["format"],
                    "sid": img["sid"] or sid,
                })
            print(f" -> {len(images)} images")
        except Exception as e:
            print(f" ERROR: {e}")
        time.sleep(args.delay)

    # Write
    if rows:
        fieldnames = ["species", "photographer", "filename", "thumb_url", "full_url", "format", "sid"]
        with open(OUTPUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
    else:
        print("\nNo rows to write.")


if __name__ == "__main__":
    main()
