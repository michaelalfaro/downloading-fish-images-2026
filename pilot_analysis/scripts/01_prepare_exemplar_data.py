#!/usr/bin/env python3
"""
01_prepare_exemplar_data.py

Prepare exemplar image inventory for pilot color pattern analysis.
- Loads species_exemplar.csv and species_gestalt_k.csv
- Merges to get gestalt k for each exemplar
- Resolves best image path (oriented > color_corrected > segmented)
- Generates grayscale versions of all exemplar images
- Exports exemplar_inventory.csv

This script is designed to be re-run as more exemplars are added.
"""

import os
import sys
import csv
from datetime import datetime

import numpy as np
from PIL import Image

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PILOT_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(PILOT_DIR)
sys.path.insert(0, PROJECT_DIR)

from analysis_utils import (
    GMM_DIR, SEGMENTED_DIR, species_to_dirname
)

# Directories
ORIENTED_DIR = os.path.join(GMM_DIR, "oriented")
CORRECTED_DIR = os.path.join(GMM_DIR, "color_corrected")
GRAYSCALE_DIR = os.path.join(PILOT_DIR, "grayscale_images")

# Input files
EXEMPLAR_CSV = os.path.join(GMM_DIR, "species_exemplar.csv")
GESTALT_CSV = os.path.join(GMM_DIR, "species_gestalt_k.csv")

# Output file
OUTPUT_CSV = os.path.join(PILOT_DIR, "data", "exemplar_inventory.csv")


def load_exemplars():
    """Load species exemplar selections."""
    exemplars = {}
    with open(EXEMPLAR_CSV, newline="") as f:
        for row in csv.DictReader(f):
            species = row["species"]
            exemplars[species] = {
                "filename": row["filename"],
                "png_name": row["png_name"],
                "source": row["source"],
            }
    return exemplars


def load_gestalt_k():
    """Load gestalt k values for species."""
    gestalt = {}
    if os.path.exists(GESTALT_CSV):
        with open(GESTALT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                species = row["species"]
                k = row.get("gestalt_k", "4")
                try:
                    gestalt[species] = int(k) if k else 4
                except ValueError:
                    gestalt[species] = 4
    return gestalt


def get_best_image_path(species, png_name, source):
    """
    Get the best available image path for an exemplar.
    Priority: oriented > color_corrected (for FishPix) > segmented
    """
    sp_dir = species_to_dirname(species)

    # Check oriented first (user-corrected orientation)
    oriented_path = os.path.join(ORIENTED_DIR, sp_dir, png_name)
    if os.path.exists(oriented_path):
        return oriented_path, "oriented"

    # For FishPix, check color-corrected
    if source == "FishPix":
        corrected_path = os.path.join(CORRECTED_DIR, sp_dir, png_name)
        if os.path.exists(corrected_path):
            return corrected_path, "color_corrected"

    # Fall back to segmented
    seg_path = os.path.join(SEGMENTED_DIR, sp_dir, png_name)
    if os.path.exists(seg_path):
        return seg_path, "segmented"

    return None, None


def create_grayscale(img_path, output_path):
    """
    Create a grayscale version of an image, preserving alpha channel.
    """
    img = Image.open(img_path).convert("RGBA")

    # Split into channels
    r, g, b, a = img.split()

    # Convert RGB to grayscale (standard luminance weights)
    gray = Image.merge("RGB", (r, g, b)).convert("L")

    # Merge grayscale with original alpha
    gray_rgba = Image.merge("RGBA", (gray, gray, gray, a))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gray_rgba.save(output_path, "PNG")
    return True


def main():
    print("=" * 60)
    print("Preparing Exemplar Data for Pilot Analysis")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    print("Loading exemplar selections...")
    exemplars = load_exemplars()
    print(f"  {len(exemplars)} species with exemplars")

    print("Loading gestalt k values...")
    gestalt_k = load_gestalt_k()
    print(f"  {len(gestalt_k)} species with gestalt k")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(GRAYSCALE_DIR, exist_ok=True)

    # Process each exemplar
    print("\nProcessing exemplars...")
    inventory = []
    n_found = 0
    n_grayscale = 0
    n_missing = 0

    for species, ex_info in sorted(exemplars.items()):
        png_name = ex_info["png_name"]
        source = ex_info["source"]

        # Get gestalt k (default to 4 if not set)
        k = gestalt_k.get(species, 4)

        # Find best image path
        img_path, img_type = get_best_image_path(species, png_name, source)

        if img_path is None:
            print(f"  WARNING: Image not found for {species}: {png_name}")
            n_missing += 1
            continue

        n_found += 1

        # Create grayscale version
        sp_dir = species_to_dirname(species)
        gray_path = os.path.join(GRAYSCALE_DIR, sp_dir, png_name)

        try:
            create_grayscale(img_path, gray_path)
            n_grayscale += 1
        except Exception as e:
            print(f"  WARNING: Failed to create grayscale for {species}: {e}")
            gray_path = ""

        # Add to inventory
        inventory.append({
            "species": species,
            "species_dir": sp_dir,
            "png_name": png_name,
            "original_source": source,
            "image_type": img_type,
            "image_path": img_path,
            "grayscale_path": gray_path,
            "gestalt_k": k,
        })

    # Write inventory CSV
    print(f"\nWriting inventory to {OUTPUT_CSV}...")
    fieldnames = [
        "species", "species_dir", "png_name", "original_source",
        "image_type", "image_path", "grayscale_path", "gestalt_k"
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inventory)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Exemplars found: {n_found}")
    print(f"  Grayscale created: {n_grayscale}")
    print(f"  Missing images: {n_missing}")
    print()

    # Image type breakdown
    type_counts = {}
    for row in inventory:
        t = row["image_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print("  Image type breakdown:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {count}")

    # Gestalt k distribution
    k_counts = {}
    for row in inventory:
        k = row["gestalt_k"]
        k_counts[k] = k_counts.get(k, 0) + 1

    print("\n  Gestalt k distribution:")
    for k, count in sorted(k_counts.items()):
        print(f"    k={k}: {count} species")

    print(f"\n  Output: {OUTPUT_CSV}")
    print(f"  Grayscale images: {GRAYSCALE_DIR}/")
    print()

    return len(inventory)


if __name__ == "__main__":
    n = main()
    print(f"Done! {n} exemplars ready for analysis.")
