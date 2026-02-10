#!/usr/bin/env python3
"""
Download Chaetodontidae fish images from FishBase and FishPix databases.

This script reads the Miyazawa (2020) supplementary data file to identify
all Chaetodontidae species with annotated color pattern images, downloads
the images from their respective databases (FishBase or FishPix), and
creates a CSV file that records the species name, downloaded filename,
image source, and whether the species matches a tip in the Butterflyfish
phylogenetic tree from the chaets-divergence-2026 project.

Data sources:
  - Miyazawa S (2020) Science Advances 6: eabb9107
    Supplementary Data File S1: abb9107_data_file_s1.xlsx
  - FishBase: https://www.fishbase.org/
  - FishPix: http://fishpix.kahaku.go.jp/fishimage-e/
  - Phylogenetic tree: Butterflyfish_concat_final.tre
"""

import os
import re
import csv
import time
import urllib.request
import urllib.error
import openpyxl


# ---------- paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(SCRIPT_DIR, "papers", "abb9107_data_file_s1.xlsx")
TREE_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR),
    "chaets-divergence-2026",
    "rooted-chaet-tree-vis",
    "Butterflyfish_concat_final.tre",
)
IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")
CSV_PATH = os.path.join(SCRIPT_DIR, "chaetodontidae_image_data.csv")


# ---------- 1. Parse tree to get Chaetodontidae species ----------
def parse_tree_species(tree_path):
    """Extract species names from the NEXUS tree file.

    Returns a set of canonical species names (e.g. 'Chaetodon auriga')
    extracted from the taxlabels block. Specimen/voucher suffixes are
    stripped, and genus names are title-cased.
    """
    with open(tree_path) as f:
        content = f.read()

    taxa_match = re.search(r"taxlabels\s+(.*?)\s*;", content, re.DOTALL)
    if not taxa_match:
        raise ValueError("Could not find taxlabels in tree file")

    taxa = taxa_match.group(1).strip().split()

    # Chaetodontidae genera present in the tree
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


# ---------- 2. Read Chaetodontidae from Excel ----------
def read_chaetodontidae(xlsx_path):
    """Read image-level annotation data for family Chaetodontidae.

    Returns a list of dicts with keys: species, img_file, db (source).
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["A_FishPatterns_img"]

    records = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[2] == "Chaetodontidae":
            records.append({
                "species": row[4],
                "img_file": row[0],
                "db": row[1],
            })
    wb.close()
    return records


# ---------- 3. Build download URLs ----------
def fishpix_url(img_file):
    """Construct FishPix image URL from filename.

    Filenames follow the pattern <ID>AF.jpg.
    Images are stored under /photos/NR<prefix>/<filename>
    where prefix = int(ID) // 1000, zero-padded to 4 digits.
    """
    photo_id = img_file.replace("AF.jpg", "")
    nr_prefix = f"NR{int(photo_id) // 1000:04d}"
    return f"https://fishpix.kahaku.go.jp/photos/{nr_prefix}/{img_file}"


def fishbase_url(img_file):
    """Construct FishBase image URL from filename."""
    return f"https://www.fishbase.se/images/species/{img_file}"


def get_image_url(img_file, db):
    if db == "FishPix":
        return fishpix_url(img_file)
    else:
        return fishbase_url(img_file)


# ---------- 4. Download images ----------
def download_image(url, dest_path, retries=3):
    """Download a single image with retry logic."""
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


# ---------- 5. Match species to tree ----------
def normalize_species(name):
    """Normalize a species name for matching.

    Handles cases like 'Chaetodon auriga' vs 'Chaetodon auriga2' in the tree,
    and 'Roa modesta' vs 'Roa modestua' (spelling variants).
    """
    return name.strip().lower()


def match_to_tree(species_name, tree_species):
    """Check if a species from the dataset matches a tip in the tree.

    Uses exact match first, then fuzzy matching for known variants.
    """
    norm = normalize_species(species_name)

    # Direct match
    for ts in tree_species:
        if normalize_species(ts) == norm:
            return True

    # Known synonyms / spelling variants between dataset and tree
    variants = {
        "chaetodon auriga": "chaetodon auriga2",
        "roa modesta": "roa modestua",
    }
    if norm in variants:
        for ts in tree_species:
            if normalize_species(ts) == variants[norm]:
                return True

    return False


# ---------- main ----------
def main():
    print("=" * 60)
    print("Chaetodontidae Image Downloader")
    print("=" * 60)

    # Parse tree
    print("\n[1] Parsing phylogenetic tree...")
    tree_species = parse_tree_species(TREE_PATH)
    print(f"    Found {len(tree_species)} Chaetodontidae species in tree")

    # Read Excel data
    print("\n[2] Reading Miyazawa (2020) dataset...")
    records = read_chaetodontidae(XLSX_PATH)
    print(f"    Found {len(records)} Chaetodontidae image records")
    unique_sp = {r["species"] for r in records}
    print(f"    Covering {len(unique_sp)} unique species")

    # Create image directory
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Download images and build CSV data
    print(f"\n[3] Downloading images to {IMAGE_DIR}/ ...")
    csv_rows = []
    for i, rec in enumerate(records):
        species = rec["species"]
        img_file = rec["img_file"]
        db = rec["db"]

        # Build a clean filename: Genus_species_source_originalname
        sp_parts = species.split()
        safe_sp = "_".join(sp_parts)
        local_filename = f"{safe_sp}_{db}_{img_file}"
        dest_path = os.path.join(IMAGE_DIR, local_filename)

        url = get_image_url(img_file, db)
        in_tree = match_to_tree(species, tree_species)

        print(f"  [{i+1:3d}/{len(records)}] {species} ({db})", end=" ... ")

        if os.path.exists(dest_path):
            print("already exists")
            success = True
        else:
            success = download_image(url, dest_path)
            if success:
                print("OK")
            time.sleep(0.5)  # be polite to servers

        csv_rows.append({
            "species": species,
            "filename": local_filename,
            "source": db,
            "image_url": url,
            "in_tree": "yes" if in_tree else "no",
            "download_success": "yes" if success else "no",
        })

    # Write CSV
    print(f"\n[4] Writing CSV to {CSV_PATH} ...")
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "species", "filename", "source", "image_url", "in_tree", "download_success",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Summary
    n_downloaded = sum(1 for r in csv_rows if r["download_success"] == "yes")
    n_in_tree = sum(1 for r in csv_rows if r["in_tree"] == "yes")
    n_not_in_tree = sum(1 for r in csv_rows if r["in_tree"] == "no")
    unique_sp_in_tree = len({r["species"] for r in csv_rows if r["in_tree"] == "yes"})
    unique_sp_not_in_tree = len({r["species"] for r in csv_rows if r["in_tree"] == "no"})

    # Species in tree but NOT in Miyazawa dataset
    dataset_species = {r["species"] for r in csv_rows}
    tree_only = set()
    for ts in tree_species:
        matched = False
        for ds in dataset_species:
            if match_to_tree(ds, {ts}):
                matched = True
                break
        if not matched:
            tree_only.add(ts)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total image records:        {len(csv_rows)}")
    print(f"  Successfully downloaded:    {n_downloaded}")
    print(f"  Unique species in dataset:  {len(unique_sp)}")
    print(f"  Species in tree:            {unique_sp_in_tree}")
    print(f"  Species NOT in tree:        {unique_sp_not_in_tree}")
    print(f"  Tree species not in dataset: {len(tree_only)}")
    if tree_only:
        print("    " + ", ".join(sorted(tree_only)))
    print(f"\n  CSV written to: {CSV_PATH}")
    print(f"  Images saved to: {IMAGE_DIR}/")


if __name__ == "__main__":
    main()
