#!/usr/bin/env python3
"""
Apply Sea-thru color correction to segmented/oriented fish images.

The Sea-thru filter removes underwater color cast using physics-informed
white balance based on Akkaynak & Treibitz (CVPR 2019). It normalizes
each color channel by its 90th percentile to undo wavelength-dependent
absorption by water.

Only affects the fish pixels (preserves transparent background).

Usage:
    # Apply to all oriented images for a species
    python3 seathru_images.py --species "Chaetodon meyeri"

    # Apply only to images flagged as underwater
    python3 seathru_images.py --flagged-only

    # Also remove estimated backscatter (more aggressive correction)
    python3 seathru_images.py --remove-backscatter

    # Use depth-aware physics correction (requires estimate_depths.py first)
    python3 seathru_images.py --depth --flagged-only

    # Force overwrite existing output
    python3 seathru_images.py --force

Reads: analysis/approach_1_gmm/oriented/ (or normalized/ if no oriented)
       analysis/approach_1_gmm/depth_maps/ (when using --depth)
Writes: analysis/approach_1_gmm/seathru_corrected/
"""

import os
import csv
import argparse
import fnmatch
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from analysis_utils import (
    GMM_DIR, ORIENTED_DIR, NORMALIZED_DIR,
    load_inventory, get_image_path, species_to_dirname, all_species, ensure_dir,
    estimate_seathru_params, seathru_image, seathru_depth_correct, load_depth_map,
)

SEATHRU_DIR = os.path.join(GMM_DIR, "seathru_corrected")
UW_REPORT_CSV = os.path.join(GMM_DIR, "underwater_detection_report.csv")


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


def load_flagged_images():
    """Load filenames flagged as underwater from the detection report."""
    flagged = set()
    if not os.path.exists(UW_REPORT_CSV):
        return flagged
    with open(UW_REPORT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("whole_is_uw") == "True" or
                    row.get("body_is_uw") == "True"):
                flagged.add(row["filename"])
    return flagged


def process_image(input_path, output_path, orig_path=None,
                  remove_backscatter=False, preview=False,
                  use_depth=False, filename=None):
    """Apply Sea-thru correction to a single image.

    If orig_path is provided, estimates correction parameters from the full
    original image (better water column info) and applies them to the
    segmented fish. Otherwise estimates from the segmented image itself.

    If use_depth is True and a precomputed depth map exists, uses the full
    physics-based depth-aware correction instead of simple white balance.

    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(input_path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        rgba = np.array(img)

        # Try depth-aware correction first
        if use_depth and filename:
            depth_map = load_depth_map(filename)
            if depth_map is not None:
                # Resize depth map to match image if needed
                if depth_map.shape[:2] != rgba.shape[:2]:
                    depth_pil = Image.fromarray(depth_map)
                    depth_pil = depth_pil.resize(
                        (rgba.shape[1], rgba.shape[0]), Image.LANCZOS)
                    depth_map = np.array(depth_pil, dtype=np.float32)

                corrected = seathru_depth_correct(rgba, depth_map)

                if preview:
                    print(f"  Preview (depth): {os.path.basename(input_path)}")
                    return True

                ensure_dir(os.path.dirname(output_path))
                Image.fromarray(corrected, "RGBA").save(output_path)
                return True

        # Fall back to simple white balance correction
        params = None
        if orig_path and os.path.exists(orig_path):
            try:
                orig_img = Image.open(orig_path).convert("RGB")
                ow, oh = orig_img.size
                if max(ow, oh) > 512:
                    s = 512 / max(ow, oh)
                    orig_img = orig_img.resize(
                        (max(1, int(ow * s)), max(1, int(oh * s))),
                        Image.LANCZOS)
                orig_pixels = np.array(orig_img).reshape(-1, 3)
                params = estimate_seathru_params(orig_pixels,
                                                 remove_backscatter=remove_backscatter)
            except Exception:
                pass  # fall back to self-estimation

        # Apply Sea-thru
        corrected = seathru_image(rgba, params=params,
                                  remove_backscatter=remove_backscatter)

        if preview:
            print(f"  Preview: {os.path.basename(input_path)}")
            return True

        # Save
        ensure_dir(os.path.dirname(output_path))
        Image.fromarray(corrected, "RGBA").save(output_path)
        return True

    except Exception as e:
        print(f"  Error processing {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Apply Sea-thru color correction to fish images"
    )
    parser.add_argument("--species", type=str, nargs="*", default=None,
                        help="Process specific species (space-separated)")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Filename pattern to match (e.g., '*iNaturalist*')")
    parser.add_argument("--flagged-only", action="store_true",
                        help="Only process images flagged as underwater")
    parser.add_argument("--remove-backscatter", action="store_true",
                        help="Also subtract estimated backscatter (more aggressive)")
    parser.add_argument("--preview", action="store_true",
                        help="Preview without saving")
    parser.add_argument("--depth", action="store_true",
                        help="Use depth-aware physics correction (requires "
                             "precomputed depth maps from estimate_depths.py)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing corrected images")
    args = parser.parse_args()

    print("[1] Loading inventory...")
    inventory = load_inventory()

    # Get species list
    if args.species:
        species_list = args.species
    else:
        species_list = all_species(inventory)

    print(f"    {len(species_list)} species to process")

    # Load flagged images if using --flagged-only
    flagged_files = set()
    if args.flagged_only:
        flagged_files = load_flagged_images()
        if not flagged_files:
            print("    Warning: No flagged underwater images found.")
            print("    Run detect_underwater.py first, then retry with --flagged-only")
            return
        print(f"    {len(flagged_files)} images flagged as underwater")

    n_processed = 0
    n_skipped = 0
    n_errors = 0

    print("[2] Applying Sea-thru correction...")

    for species in species_list:
        sp_dir = species_to_dirname(species)
        out_dir = os.path.join(SEATHRU_DIR, sp_dir)

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

            # Get original image path for parameter estimation
            orig_path = get_image_path(row)

            # Process
            if process_image(input_path, output_path, orig_path=orig_path,
                             remove_backscatter=args.remove_backscatter,
                             preview=args.preview,
                             use_depth=args.depth, filename=fname):
                n_processed += 1
                if n_processed % 100 == 0:
                    print(f"    ... {n_processed} images processed")
            else:
                n_errors += 1

    print(f"\n[3] Sea-thru correction complete")
    print(f"    Processed: {n_processed}")
    print(f"    Skipped (already exists): {n_skipped}")
    print(f"    Errors: {n_errors}")
    if not args.preview:
        print(f"    Output: {SEATHRU_DIR}")


if __name__ == "__main__":
    main()
