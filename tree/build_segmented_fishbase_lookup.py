#!/usr/bin/env python3
"""
Build tip_image_lookup_segmented.csv using SEGMENTED images from FishBase only.

For the phylogenetic tree visualization, uses:
  1. Segmented fish images (background removed, on gray)
  2. FishBase images only (including Randall photos now in FishBase)
  3. Color-corrected FishPix images where available

Output: tip_image_lookup_segmented.csv with columns:
  tip_label, img_path
"""

import os
import re
import csv
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Add project dir to path for imports
sys.path.insert(0, PROJECT_DIR)
from analysis_utils import (
    SEGMENTED_DIR, GMM_DIR, load_inventory, species_to_dirname
)

TREE_PATH = os.path.join(SCRIPT_DIR, "Butterflyfish_concat_final.tre")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "tip_image_lookup_segmented.csv")

# Color-corrected FishPix directory
CORRECTED_DIR = os.path.join(GMM_DIR, "color_corrected")

# Source priority for FishBase-focused tree (prefer museum-quality images)
# Note: "FishBase" includes Randall duplicates that are now credited to FishBase
SOURCE_PRIORITY = {
    "BishopMuseum": 0,   # Randall photos - highest quality
    "FishBase": 1,       # FishBase (includes some Randall)
    "FishPix": 2,        # Japanese museum, color-corrected
}

# Species name variants: tree name -> dataset name
TREE_TO_DATASET = {
    "Chaetodon auriga2": "Chaetodon auriga",
    "Roa modestua": "Roa modesta",
}

DATASET_TO_TREE_VARIANTS = {
    "Chaetodon auriga": ["Chaetodon auriga", "Chaetodon auriga2"],
    "Roa modesta": ["Roa modesta", "Roa modestua"],
    "Roa excelsa": ["Roa excelsa", "Chaetodon excelsa"],
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
    """Convert tree tip label to species name."""
    parts = tip_label.split("_")
    if len(parts) >= 2:
        genus = parts[0].capitalize()
        epithet = parts[1]
        return f"{genus} {epithet}"
    return tip_label


def load_review_exclusions():
    """Load excluded images from review_state.csv."""
    excluded = set()
    review_path = os.path.join(GMM_DIR, "review_state.csv")
    if os.path.exists(review_path):
        with open(review_path) as f:
            for row in csv.DictReader(f):
                if row.get("action") in ("exclude", "alt_morph"):
                    excluded.add(row["filename"])
    return excluded


def build_species_segmented_index(inventory, excluded):
    """Build index: species -> list of segmented image paths with metadata."""
    index = {}

    for row in inventory:
        if row["is_duplicate"] == "yes":
            continue

        # Only use FishBase-related sources (Bishop, FishBase, FishPix)
        source = row["source"]
        if source not in SOURCE_PRIORITY:
            continue

        filename = row["filename"]
        if filename in excluded:
            continue

        species = row["species"]
        sp_dir = species_to_dirname(species)

        # Check for segmented image
        base_name = os.path.splitext(filename)[0] + ".png"

        # For FishPix, check if color-corrected version exists
        if source == "FishPix":
            corrected_path = os.path.join(CORRECTED_DIR, sp_dir, base_name)
            if os.path.exists(corrected_path):
                seg_path = corrected_path
            else:
                seg_path = os.path.join(SEGMENTED_DIR, sp_dir, base_name)
        else:
            seg_path = os.path.join(SEGMENTED_DIR, sp_dir, base_name)

        if not os.path.exists(seg_path):
            continue

        priority = SOURCE_PRIORITY[source]

        if species not in index:
            index[species] = []

        index[species].append({
            "source": source,
            "filename": filename,
            "seg_path": seg_path,
            "priority": priority,
        })

    # Sort each species by priority, then filename
    for sp in index:
        index[sp].sort(key=lambda x: (x["priority"], x["filename"]))

    return index


def get_best_image_for_species(index, species_name):
    """Get the best segmented image for a species."""
    # Try exact name
    candidates = index.get(species_name, [])

    # Try variants
    if not candidates:
        for variant_sp, variant_list in DATASET_TO_TREE_VARIANTS.items():
            for variant in variant_list:
                if variant.lower() == species_name.lower():
                    candidates = index.get(variant_sp, [])
                    if candidates:
                        break
            if candidates:
                break

    # Try tree-to-dataset mapping
    if not candidates:
        mapped = TREE_TO_DATASET.get(species_name)
        if mapped:
            candidates = index.get(mapped, [])

    if candidates:
        return candidates[0]["seg_path"]
    return None


def main():
    print("Building segmented FishBase-only tip image lookup...")

    # Parse tree tips
    all_tips = parse_tree_tips(TREE_PATH)
    chaet_tips = [t for t in all_tips if is_chaetodontidae(t)]
    print(f"  {len(chaet_tips)} Chaetodontidae tips in tree")

    # Load inventory and exclusions
    inventory = load_inventory()
    excluded = load_review_exclusions()
    print(f"  {len(excluded)} images excluded from review")

    # Build segmented image index
    index = build_species_segmented_index(inventory, excluded)
    print(f"  {len(index)} species with segmented FishBase images")

    # Map tips to images
    rows = []
    n_with_img = 0
    source_counts = {"BishopMuseum": 0, "FishBase": 0, "FishPix": 0}

    for tip in chaet_tips:
        species = tip_to_species(tip)
        img_path = get_best_image_for_species(index, species)

        row = {
            "tip_label": tip,
            "img_path": img_path if img_path else "NA",
        }
        rows.append(row)

        if img_path:
            n_with_img += 1
            # Track source
            for src in SOURCE_PRIORITY:
                if src.lower() in img_path.lower() or f"_{src}_" in img_path:
                    source_counts[src] += 1
                    break
            else:
                # Check Bishop by directory
                if "images_bishop" in img_path or "Bishop" in img_path:
                    source_counts["BishopMuseum"] += 1
                elif "FishPix" in img_path:
                    source_counts["FishPix"] += 1
                else:
                    source_counts["FishBase"] += 1

    # Write output
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tip_label", "img_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Results:")
    print(f"    Tips with image: {n_with_img}/{len(chaet_tips)}")
    print(f"    Source breakdown:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"      {src}: {count}")
    print(f"\n  Output: {OUTPUT_CSV}")

    # Report any tips with no image
    no_img = [r for r in rows if r["img_path"] == "NA"]
    if no_img:
        print(f"\n  Tips with NO image ({len(no_img)}):")
        for r in no_img:
            print(f"    {r['tip_label']}")


if __name__ == "__main__":
    main()
