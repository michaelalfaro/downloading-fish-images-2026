#!/usr/bin/env python3
"""
Download Chaetodontidae fish images from the Bishop Museum
John E. Randall Fish Photo collection.

Source: https://pbs.bishopmuseum.org/images/JER/images.asp
Search: Family = Chaetodontidae

This script scrapes the gallery page for all Chaetodontidae entries,
downloads the large-format images, and produces a CSV file recording
the species name, filename, locality, and tree match status.
"""

import os
import re
import csv
import time
import urllib.request
import urllib.error


# ---------- paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TREE_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "chaets-divergence-2026",
    "rooted-chaet-tree-vis",
    "Butterflyfish_concat_final.tre",
)
IMAGE_DIR = os.path.join(SCRIPT_DIR, "images_bishop")
CSV_PATH = os.path.join(SCRIPT_DIR, "bishop_museum_image_data.csv")

GALLERY_URL = (
    "https://pbs.bishopmuseum.org/images/JER/images.asp"
    "?nm=Chaetodontidae&loc=&size=s&cols=2"
)
BASE_URL = "https://pbs.bishopmuseum.org/images/JER"


# ---------- 1. Parse tree ----------
def parse_tree_species(tree_path):
    with open(tree_path) as f:
        content = f.read()

    taxa_match = re.search(r"taxlabels\s+(.*?)\s*;", content, re.DOTALL)
    if not taxa_match:
        raise ValueError("Could not find taxlabels in tree file")

    taxa = taxa_match.group(1).strip().split()

    chaet_genera = {
        "Amphichaetodon", "Chaetodon", "Chelmon", "Chelmonops",
        "Coradion", "Forcipiger", "Hemitaurichthys", "Heniochus",
        "Johnrandallia", "Parachaetodon", "Prognathodes", "Roa",
    }

    species_set = set()
    for taxon in taxa:
        parts = taxon.split("_")
        genus = parts[0].capitalize()
        if genus not in chaet_genera or len(parts) < 2:
            continue
        epithet = parts[1]
        species_name = f"{genus} {epithet}"
        species_set.add(species_name)

    return species_set


# ---------- 2. Scrape gallery ----------
def scrape_gallery(gallery_url):
    """Fetch the gallery page and extract all image entries."""
    req = urllib.request.Request(gallery_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        html = response.read().decode("latin-1")

    # Pattern: <img src="small/ID.jpg" alt="Locality\nDate" ...> ... <i>Species</i>
    pattern = (
        r'<img src="small/([^"]+)\.jpg"[^>]*alt="([^"]*)"[^>]*>'
        r'.*?<br><i>([^<]+)</i>'
    )
    matches = re.findall(pattern, html, re.DOTALL)

    records = []
    for img_id, alt_text, species_name in matches:
        species_name = species_name.strip()
        # Extract locality from alt text (first line)
        locality = alt_text.split("\n")[0].strip() if alt_text else ""
        records.append({
            "species": species_name,
            "img_id": img_id,
            "locality": locality,
        })

    return records


# ---------- 3. Download ----------
def download_image(url, dest_path, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
                with open(dest_path, "wb") as f:
                    f.write(data)
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"  FAILED: {url} -> {e}")
                return False
    return False


# ---------- 4. Match to tree ----------
def match_to_tree(species_name, tree_species):
    norm = species_name.strip().lower()
    for ts in tree_species:
        if ts.lower() == norm:
            return True

    # Known variants
    variants = {
        "chaetodon auriga": "chaetodon auriga2",
        "roa modesta": "roa modestua",
        "chaetodon zanzibariensis": "chaetodon zanzibarensis",
        "chaetodon excelsa": "roa excelsa",
    }
    if norm in variants:
        for ts in tree_species:
            if ts.lower() == variants[norm]:
                return True

    return False


# ---------- main ----------
def main():
    print("=" * 60)
    print("Bishop Museum Chaetodontidae Image Downloader")
    print("=" * 60)

    # Parse tree
    print("\n[1] Parsing phylogenetic tree...")
    tree_species = parse_tree_species(TREE_PATH)
    print(f"    Found {len(tree_species)} Chaetodontidae species in tree")

    # Scrape gallery
    print("\n[2] Scraping Bishop Museum gallery...")
    records = scrape_gallery(GALLERY_URL)
    print(f"    Found {len(records)} image records")
    unique_sp = {r["species"] for r in records}
    print(f"    Covering {len(unique_sp)} unique species")

    # Filter out unidentified species (e.g. "Chaetodon sp.", "Heniochus sp.")
    identified = [r for r in records if " sp." not in r["species"]]
    unidentified = [r for r in records if " sp." in r["species"]]
    if unidentified:
        print(f"    Skipping {len(unidentified)} unidentified (sp.) records")

    # Create image directory
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Download images
    print(f"\n[3] Downloading images to {IMAGE_DIR}/ ...")
    csv_rows = []
    for i, rec in enumerate(identified):
        species = rec["species"]
        img_id = rec["img_id"]
        locality = rec["locality"]

        # Build filename
        safe_sp = species.replace(" ", "_")
        local_filename = f"{safe_sp}_Bishop_{img_id}.jpg"
        dest_path = os.path.join(IMAGE_DIR, local_filename)

        # Large image URL
        url = f"{BASE_URL}/large/{img_id}.jpg"
        in_tree = match_to_tree(species, tree_species)

        print(f"  [{i+1:3d}/{len(identified)}] {species} (ID={img_id})", end=" ... ")

        if os.path.exists(dest_path):
            print("already exists")
            success = True
        else:
            success = download_image(url, dest_path)
            if success:
                print("OK")
            time.sleep(0.3)

        csv_rows.append({
            "species": species,
            "filename": local_filename,
            "source": "BishopMuseum",
            "image_url": url,
            "locality": locality,
            "in_tree": "yes" if in_tree else "no",
            "download_success": "yes" if success else "no",
        })

    # Write CSV
    print(f"\n[4] Writing CSV to {CSV_PATH} ...")
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "species", "filename", "source", "image_url",
            "locality", "in_tree", "download_success",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Summary
    n_downloaded = sum(1 for r in csv_rows if r["download_success"] == "yes")
    n_in_tree = len({r["species"] for r in csv_rows if r["in_tree"] == "yes"})
    n_not_in_tree = len({r["species"] for r in csv_rows if r["in_tree"] == "no"})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total image records:        {len(csv_rows)}")
    print(f"  Successfully downloaded:    {n_downloaded}")
    print(f"  Unique species (identified): {len({r['species'] for r in csv_rows})}")
    print(f"  Species matched to tree:    {n_in_tree}")
    print(f"  Species NOT in tree:        {n_not_in_tree}")
    print(f"\n  CSV written to: {CSV_PATH}")
    print(f"  Images saved to: {IMAGE_DIR}/")


if __name__ == "__main__":
    main()
