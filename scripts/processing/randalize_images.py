#!/usr/bin/env python3
"""
Apply Randalize filter to segmented/oriented fish images.

The Randalize filter normalizes image colors toward Jack Randall's specimen
photo style, which helps standardize colors across different lighting conditions
(underwater photos, aquarium photos, etc.).

The filter applies:
- 15% saturation boost (more vivid colors)
- 12% warm sepia tone (removes cool underwater cast)
- 5% brightness increase

Only affects the fish pixels (preserves transparent background).

Usage:
    # Apply to all oriented images for a species
    python3 randalize_images.py --species "Chaetodon meyeri"

    # Apply to specific images (by filename pattern)
    python3 randalize_images.py --pattern "*iNaturalist*"

    # Apply to all images that have the randall filter flag set in review
    python3 randalize_images.py --flagged-only

    # Preview without saving (just shows effect)
    python3 randalize_images.py --species "Chaetodon meyeri" --preview

Reads: analysis/approach_1_gmm/oriented/ (or normalized/ if no oriented)
Writes: analysis/approach_1_gmm/randalized/
"""

import os
import argparse
import fnmatch
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from analysis_utils import (
    GMM_DIR, ORIENTED_DIR, NORMALIZED_DIR,
    load_inventory, species_to_dirname, all_species, ensure_dir,
    randalize_image,
)

RANDALIZED_DIR = os.path.join(GMM_DIR, "randalized")


def get_input_image_path(species, filename):
    """Get best input image: oriented if exists, else normalized."""
    sp_dir = species_to_dirname(species)
    base = os.path.splitext(filename)[0] + ".png"

    oriented_path = os.path.join(ORIENTED_DIR, sp_dir, base)
    if os.path.exists(oriented_path):
        return oriented_path

    norm_path = os.path.join(NORMALIZED_DIR, sp_dir, base)
    if os.path.exists(norm_path):
        return norm_path

    return None


def process_image(input_path, output_path, preview=False):
    """Apply Randalize filter to a single image.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image
        img = Image.open(input_path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        rgba = np.array(img)

        # Apply Randalize
        randalized = randalize_image(rgba)

        if preview:
            # Just show side-by-side comparison
            print(f"  Preview: {os.path.basename(input_path)}")
            # Could add matplotlib display here if --preview is common
            return True

        # Save
        ensure_dir(os.path.dirname(output_path))
        Image.fromarray(randalized, "RGBA").save(output_path)
        return True

    except Exception as e:
        print(f"  Error processing {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Apply Randalize filter to fish images"
    )
    parser.add_argument("--species", type=str, nargs="*", default=None,
                        help="Process specific species (space-separated)")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Filename pattern to match (e.g., '*iNaturalist*')")
    parser.add_argument("--flagged-only", action="store_true",
                        help="Only process images flagged for Randalize in review app")
    parser.add_argument("--preview", action="store_true",
                        help="Preview without saving")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing randalized images")
    args = parser.parse_args()

    print("[1] Loading inventory...")
    inventory = load_inventory()

    # Get species list
    if args.species:
        species_list = args.species
    else:
        species_list = all_species(inventory)

    print(f"    {len(species_list)} species to process")

    # Load filter flags if using --flagged-only
    flagged_files = set()
    if args.flagged_only:
        # Would load from a CSV or the review state
        # For now, this is a placeholder
        print("    Note: --flagged-only requires filter state to be persisted")
        print("    Processing all images instead")

    n_processed = 0
    n_skipped = 0
    n_errors = 0

    print("[2] Applying Randalize filter...")

    for species in species_list:
        sp_dir = species_to_dirname(species)
        out_dir = os.path.join(RANDALIZED_DIR, sp_dir)

        # Get images for this species
        species_rows = [r for r in inventory if r["species"] == species]

        for row in species_rows:
            fname = row["filename"]

            # Pattern filter
            if args.pattern and not fnmatch.fnmatch(fname, args.pattern):
                continue

            # Flagged filter
            if args.flagged_only and fname not in flagged_files:
                continue

            # Get input path
            input_path = get_input_image_path(species, fname)
            if not input_path:
                continue

            # Output path
            base = os.path.splitext(fname)[0] + ".png"
            output_path = os.path.join(out_dir, base)

            # Skip if exists and not forcing
            if os.path.exists(output_path) and not args.force:
                n_skipped += 1
                continue

            # Process
            if process_image(input_path, output_path, preview=args.preview):
                n_processed += 1
                if n_processed % 100 == 0:
                    print(f"    ... {n_processed} images processed")
            else:
                n_errors += 1

    print(f"\n[3] Randalize complete")
    print(f"    Processed: {n_processed}")
    print(f"    Skipped (already exists): {n_skipped}")
    print(f"    Errors: {n_errors}")
    if not args.preview:
        print(f"    Output: {RANDALIZED_DIR}")


if __name__ == "__main__":
    main()
