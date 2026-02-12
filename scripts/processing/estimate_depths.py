#!/usr/bin/env python3
"""
Estimate depth maps for fish images using Depth Anything V2 (ViT-L).

Runs monocular depth estimation on the original full-resolution images and
saves both summary statistics (CSV) and per-image depth maps (.npz) for use
by the depth-aware Sea-thru correction pipeline.

Usage:
    # Estimate depths for all images
    python3 estimate_depths.py

    # Only underwater-flagged images
    python3 estimate_depths.py --flagged-only

    # Specific species, using GPU
    python3 estimate_depths.py --species "Chaetodon auriga" --device cuda

    # Force re-estimation of existing depth maps
    python3 estimate_depths.py --force

Reads:  data/all_images_inventory.csv, original image directories
Writes: analysis/approach_1_gmm/depth_estimates.csv
        analysis/approach_1_gmm/depth_maps/<filename>_depth.npz
"""

import os
import csv
import argparse
import fnmatch
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from analysis_utils import (
    GMM_DIR, SEGMENTED_DIR, DEPTH_CSV, DEPTH_MAPS_DIR,
    load_inventory, get_image_path, species_to_dirname, all_species,
    ensure_dir, load_depth_anything_model, estimate_depth_map,
    scale_depth_map, depth_map_stats, get_depth_map_path,
)

UW_REPORT_CSV = os.path.join(GMM_DIR, "underwater_detection_report.csv")


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


def get_body_mask(species, filename):
    """Load segmentation mask to get fish body pixels."""
    sp_dir = species_to_dirname(species)
    base = os.path.splitext(filename)[0] + ".png"
    seg_path = os.path.join(SEGMENTED_DIR, sp_dir, base)
    if not os.path.exists(seg_path):
        return None
    try:
        seg_img = Image.open(seg_path).convert("RGBA")
        alpha = np.array(seg_img)[:, :, 3]
        return alpha > 0
    except Exception:
        return None


def process_image(row, model, device, force=False, multiply=10.0,
                  additive=2.0):
    """Estimate depth for a single image.

    Returns:
        dict with depth statistics, or None on failure
    """
    fname = row["filename"]
    species = row["species"]

    # Check if depth map already exists
    npz_path = get_depth_map_path(fname)
    if os.path.exists(npz_path) and not force:
        # Load existing and compute stats
        data = np.load(npz_path)
        depth_raw = data["depth"]
        depth_scaled = scale_depth_map(depth_raw, multiply=multiply,
                                       additive=additive)
        stats = depth_map_stats(depth_scaled)

        # Body stats
        body_mask = get_body_mask(species, fname)
        if body_mask is not None and body_mask.shape == depth_scaled.shape:
            body_stats = depth_map_stats(depth_scaled, mask=body_mask)
        else:
            body_stats = {"mean_depth": "", "median_depth": ""}

        return {
            "filename": fname,
            "species": species,
            "source": row.get("source", ""),
            **{f"whole_{k}": v for k, v in stats.items()},
            "body_mean_depth": body_stats["mean_depth"],
            "body_median_depth": body_stats["median_depth"],
            "depth_range_m": stats.get("max_depth", ""),
        }

    # Load original image
    orig_path = get_image_path(row)
    if not orig_path or not os.path.exists(orig_path):
        return None

    try:
        img = Image.open(orig_path).convert("RGB")
        # Downsample large images for speed (keep aspect ratio)
        ow, oh = img.size
        max_dim = 1024
        if max(ow, oh) > max_dim:
            s = max_dim / max(ow, oh)
            img = img.resize((max(1, int(ow * s)), max(1, int(oh * s))),
                             Image.LANCZOS)

        img_rgb = np.array(img)

        # Estimate depth
        depth_raw = estimate_depth_map(model, img_rgb)

        # Save depth map
        ensure_dir(os.path.dirname(npz_path))
        np.savez_compressed(npz_path, depth=depth_raw)

        # Scale to meters
        depth_scaled = scale_depth_map(depth_raw, multiply=multiply,
                                       additive=additive)
        stats = depth_map_stats(depth_scaled)

        # Body-specific stats
        body_mask = get_body_mask(species, fname)
        if body_mask is not None:
            # Resize depth to match segmentation mask if needed
            if body_mask.shape != depth_scaled.shape:
                from PIL import Image as PILImage
                depth_img = PILImage.fromarray(depth_scaled)
                depth_img = depth_img.resize(
                    (body_mask.shape[1], body_mask.shape[0]),
                    PILImage.LANCZOS)
                depth_for_body = np.array(depth_img)
            else:
                depth_for_body = depth_scaled
            body_stats = depth_map_stats(depth_for_body, mask=body_mask)
        else:
            body_stats = {"mean_depth": "", "median_depth": ""}

        return {
            "filename": fname,
            "species": species,
            "source": row.get("source", ""),
            **{f"whole_{k}": v for k, v in stats.items()},
            "body_mean_depth": body_stats["mean_depth"],
            "body_median_depth": body_stats["median_depth"],
            "depth_range_m": stats.get("max_depth", ""),
        }

    except Exception as e:
        print(f"  Error processing {fname}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Estimate depth maps using Depth Anything V2"
    )
    parser.add_argument("--species", type=str, nargs="*", default=None,
                        help="Process specific species (space-separated)")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Filename pattern to match (e.g., '*iNaturalist*')")
    parser.add_argument("--flagged-only", action="store_true",
                        help="Only process images flagged as underwater")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing depth maps")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, mps, or cpu (auto-detected)")
    parser.add_argument("--multiply", type=float, default=10.0,
                        help="Depth range scaling factor (default 10.0)")
    parser.add_argument("--additive", type=float, default=2.0,
                        help="Minimum depth offset (default 2.0)")
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
            print("    Run detect_underwater.py first, then retry.")
            return
        print(f"    {len(flagged_files)} images flagged as underwater")

    print("[2] Loading Depth Anything V2 model...")
    model, device = load_depth_anything_model(device=args.device)
    print(f"    Model loaded on {device}")

    # Load existing results to append to
    existing_results = {}
    if os.path.exists(DEPTH_CSV) and not args.force:
        with open(DEPTH_CSV, newline="") as f:
            for row in csv.DictReader(f):
                existing_results[row["filename"]] = row

    n_processed = 0
    n_skipped = 0
    n_errors = 0
    results = dict(existing_results)  # start with existing

    print("[3] Estimating depth maps...")

    for species in species_list:
        species_rows = [r for r in inventory if r["species"] == species]

        for row in species_rows:
            fname = row["filename"]

            # Pattern filter
            if args.pattern and not fnmatch.fnmatch(fname, args.pattern):
                continue

            # Flagged filter
            if args.flagged_only and fname not in flagged_files:
                continue

            # Skip if already processed (unless forcing)
            if fname in existing_results and not args.force:
                n_skipped += 1
                continue

            result = process_image(row, model, device, force=args.force,
                                   multiply=args.multiply,
                                   additive=args.additive)
            if result:
                results[result["filename"]] = result
                n_processed += 1
                if n_processed % 50 == 0:
                    print(f"    ... {n_processed} images processed")
            else:
                n_errors += 1

    # Write CSV
    fieldnames = [
        "filename", "species", "source",
        "whole_mean_depth", "whole_median_depth", "whole_std_depth",
        "whole_min_depth", "whole_max_depth",
        "body_mean_depth", "body_median_depth", "depth_range_m",
    ]

    ensure_dir(os.path.dirname(DEPTH_CSV))
    with open(DEPTH_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for fname in sorted(results):
            writer.writerow(results[fname])

    print(f"\n[4] Depth estimation complete")
    print(f"    Processed: {n_processed}")
    print(f"    Skipped (already exists): {n_skipped}")
    print(f"    Errors: {n_errors}")
    print(f"    Output CSV: {DEPTH_CSV}")
    print(f"    Depth maps: {DEPTH_MAPS_DIR}")


if __name__ == "__main__":
    main()
