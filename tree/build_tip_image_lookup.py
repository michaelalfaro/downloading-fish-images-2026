#!/usr/bin/env python3
"""
Build tip_image_lookup.csv mapping tree tip labels to image file paths.

For each of the 111 Chaetodontidae tree tips, assigns up to 3 images
ranked by source priority:
  1. Bishop Museum (Randall photos, highest quality museum specimens)
  2. iNaturalist (research-grade, in-situ photos)
  3. FishPix (curated collection)
  4. FishBase (original Miyazawa dataset images)
  5. FishBase_extra (additional API-verified FishBase images)

When multiple images from the same source exist, uses different
individual photos for columns 2 and 3 (never repeats an image).

Output: tip_image_lookup.csv with columns:
  tip_label, img_path, img_path_2, img_path_3
"""

import os
import re
import csv
import hashlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

TREE_PATH = os.path.join(SCRIPT_DIR, "Butterflyfish_concat_final.tre")
INVENTORY_CSV = os.path.join(PROJECT_DIR, "all_images_inventory.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "tip_image_lookup.csv")

# Image directories (absolute paths)
IMAGE_DIRS = {
    "images": os.path.join(PROJECT_DIR, "images"),
    "images_bishop": os.path.join(PROJECT_DIR, "images_bishop"),
    "images_fishbase_extra": os.path.join(PROJECT_DIR, "images_fishbase_extra"),
    "images_inaturalist": os.path.join(PROJECT_DIR, "images_inaturalist"),
}

# Source priority (lower = better)
SOURCE_PRIORITY = {
    "BishopMuseum": 0,
    "iNaturalist": 1,
    "FishPix": 2,
    "FishBase": 3,      # original Miyazawa
    "FishBase_extra": 4,
}

# Species name variants: tree name -> dataset name
# (the tree sometimes uses different spellings)
TREE_TO_DATASET = {
    "Chaetodon auriga2": "Chaetodon auriga",
    "Roa modestua": "Roa modesta",
    "Chaetodon zanzibarensis": "Chaetodon zanzibarensis",
}

# Also handle Roa excelsa (tree) vs Chaetodon excelsa (some databases)
DATASET_TO_TREE_VARIANTS = {
    "Chaetodon auriga": ["Chaetodon auriga", "Chaetodon auriga2"],
    "Roa modesta": ["Roa modesta", "Roa modestua"],
    "Roa excelsa": ["Roa excelsa", "Chaetodon excelsa"],
    "Chaetodon zanzibarensis": ["Chaetodon zanzibarensis", "Chaetodon zanzibariensis"],
}


def parse_tree_tips(tree_path):
    """Extract all tip labels from NEXUS tree."""
    with open(tree_path) as f:
        content = f.read()

    taxa_match = re.search(r"taxlabels\s+(.*?)\s*;", content, re.DOTALL)
    if not taxa_match:
        raise ValueError("Could not find taxlabels in tree file")

    return taxa_match.group(1).strip().split()


def is_chaetodontidae(tip_label):
    """Check if a tip label belongs to Chaetodontidae."""
    chaet_genera = {
        "amphichaetodon", "chaetodon", "chelmon", "chelmonops",
        "coradion", "forcipiger", "hemitaurichthys", "heniochus",
        "johnrandallia", "parachaetodon", "prognathodes", "roa",
    }
    parts = tip_label.split("_")
    return parts[0].lower() in chaet_genera


def tip_to_species(tip_label):
    """Convert tree tip label to species name.

    e.g., 'Chaetodon_auriga2' -> 'Chaetodon auriga2'
          'Chaetodon_burgessi_JRH80' -> 'Chaetodon burgessi'
    """
    parts = tip_label.split("_")
    if len(parts) >= 2:
        # Handle lowercase tips like 'chaetodon_trifascialis'
        genus = parts[0].capitalize()
        epithet = parts[1]
        return f"{genus} {epithet}"
    return tip_label


def build_species_image_index(inventory_csv):
    """Build index: species -> list of (source, directory, filename, priority)."""
    index = {}

    with open(inventory_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["is_duplicate"] == "yes":
                continue

            species = row["species"]
            source = row["source"]
            directory = row["directory"]
            filename = row["filename"]

            # Determine source category
            if source == "FishBase" and directory == "images_fishbase_extra":
                src_cat = "FishBase_extra"
            elif source == "BishopMuseum":
                src_cat = "BishopMuseum"
            elif source == "FishPix":
                src_cat = "FishPix"
            elif source == "FishBase":
                src_cat = "FishBase"
            elif source == "iNaturalist":
                src_cat = "iNaturalist"
            else:
                src_cat = source

            priority = SOURCE_PRIORITY.get(src_cat, 99)
            filepath = os.path.join(IMAGE_DIRS.get(directory, directory), filename)

            if species not in index:
                index[species] = []
            index[species].append({
                "source": src_cat,
                "directory": directory,
                "filename": filename,
                "filepath": filepath,
                "priority": priority,
            })

    # Sort each species's images by priority
    for sp in index:
        index[sp].sort(key=lambda x: (x["priority"], x["filename"]))

    return index


def get_images_for_species(index, species_name, n=3):
    """Get up to n unique images for a species, trying different sources.

    Returns list of absolute file paths.
    Strategy: pick from different sources first, then different images
    from the same source.
    """
    # Try the exact species name first
    candidates = index.get(species_name, [])

    # Also try variants
    if not candidates:
        for variant_sp, variant_list in DATASET_TO_TREE_VARIANTS.items():
            for variant in variant_list:
                if variant.lower() == species_name.lower():
                    candidates = index.get(variant_sp, [])
                    if candidates:
                        break
            if candidates:
                break

    # Still no candidates? Try the tree-to-dataset mapping
    if not candidates:
        mapped = TREE_TO_DATASET.get(species_name)
        if mapped:
            candidates = index.get(mapped, [])

    if not candidates:
        return [None] * n

    # Pick images: prefer different sources, then different files within source
    selected = []
    used_sources = set()
    used_files = set()

    # First pass: one from each source (best priority first)
    for img in candidates:
        if img["source"] not in used_sources and os.path.exists(img["filepath"]):
            selected.append(img["filepath"])
            used_sources.add(img["source"])
            used_files.add(img["filepath"])
            if len(selected) >= n:
                break

    # Second pass: fill remaining slots with any unused images
    if len(selected) < n:
        for img in candidates:
            if img["filepath"] not in used_files and os.path.exists(img["filepath"]):
                selected.append(img["filepath"])
                used_files.add(img["filepath"])
                if len(selected) >= n:
                    break

    # Pad with None if needed
    while len(selected) < n:
        selected.append(None)

    return selected


def main():
    print("Building tip image lookup with 3 columns...")

    # Parse tree tips
    all_tips = parse_tree_tips(TREE_PATH)
    chaet_tips = [t for t in all_tips if is_chaetodontidae(t)]
    print(f"  {len(chaet_tips)} Chaetodontidae tips in tree")

    # Build image index
    index = build_species_image_index(INVENTORY_CSV)
    print(f"  {len(index)} species in image inventory")

    # Map tips to images
    rows = []
    n_with_1 = 0
    n_with_2 = 0
    n_with_3 = 0

    for tip in chaet_tips:
        species = tip_to_species(tip)
        images = get_images_for_species(index, species, n=3)

        row = {
            "tip_label": tip,
            "img_path": images[0] if images[0] else "NA",
            "img_path_2": images[1] if images[1] else "NA",
            "img_path_3": images[2] if images[2] else "NA",
        }
        rows.append(row)

        if images[0]:
            n_with_1 += 1
        if images[1]:
            n_with_2 += 1
        if images[2]:
            n_with_3 += 1

    # Write output
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tip_label", "img_path", "img_path_2", "img_path_3"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Results:")
    print(f"    Tips with >= 1 image: {n_with_1}/{len(chaet_tips)}")
    print(f"    Tips with >= 2 images: {n_with_2}/{len(chaet_tips)}")
    print(f"    Tips with >= 3 images: {n_with_3}/{len(chaet_tips)}")
    print(f"\n  Output: {OUTPUT_CSV}")

    # Report any tips with no image
    no_img = [r for r in rows if r["img_path"] == "NA"]
    if no_img:
        print(f"\n  Tips with NO image:")
        for r in no_img:
            print(f"    {r['tip_label']}")


if __name__ == "__main__":
    main()
