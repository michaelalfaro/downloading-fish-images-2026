#!/usr/bin/env python3
"""
L-channel normalization in CIELAB space for segmented fish images.

For each species, computes the mean luminance (L) across all body pixels
in all images, then shifts each image's L channel so its mean matches
the species mean. Preserves hue (a, b channels unchanged).

Reads: analysis/approach_1_gmm/segmented/
Writes: analysis/approach_1_gmm/normalized/
        analysis/approach_1_gmm/normalization_report.csv

Usage:
    python3 normalize_luminance.py
"""

import os
import csv
import numpy as np
from collections import defaultdict
from PIL import Image

from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR, NORMALIZED_DIR,
    load_inventory, load_annotations, species_to_dirname,
    load_segmented_image, body_pixels_to_lab, lab_to_rgb_pixels,
    all_species, ensure_dir,
)

REPORT_CSV = os.path.join(GMM_DIR, "normalization_report.csv")
REPORT_FIELDS = [
    "filename", "species", "original_mean_L", "species_mean_L",
    "L_shift", "n_body_pixels", "n_pixels_clamped",
]


def main():
    print("[1] Loading inventory and annotations...")
    inventory = load_inventory()
    annotations = load_annotations()

    # Group images by species, excluding larvae and manually excluded
    species_images = defaultdict(list)
    for row in inventory:
        fname = row["filename"]
        ann = annotations.get(fname)
        if ann and ann["annotation_type"] in ("larva", "manual_exclude"):
            continue
        species_images[row["species"]].append(row)

    print(f"    {len(species_images)} species")

    print("[2] Computing species-level mean L and normalizing...")
    report_rows = []
    n_processed = 0
    n_skipped = 0

    for i, (species, rows) in enumerate(sorted(species_images.items())):
        sp_dir = species_to_dirname(species)
        seg_dir = os.path.join(SEGMENTED_DIR, sp_dir)
        norm_dir = os.path.join(NORMALIZED_DIR, sp_dir)
        ensure_dir(norm_dir)

        # First pass: compute mean L for the species
        all_L_values = []
        image_data = []  # (filename, png_path, rgb, mask)

        for row in rows:
            png_name = os.path.splitext(row["filename"])[0] + ".png"
            png_path = os.path.join(seg_dir, png_name)
            if not os.path.exists(png_path):
                continue

            try:
                rgb, mask = load_segmented_image(png_path)
            except Exception:
                continue

            body_pixels = rgb[mask]
            if len(body_pixels) < 100:
                continue

            lab = body_pixels_to_lab(body_pixels)
            all_L_values.append(np.mean(lab[:, 0]))
            image_data.append((row["filename"], png_path, rgb, mask))

        if not image_data:
            n_skipped += 1
            continue

        species_mean_L = float(np.mean(all_L_values))

        # Second pass: normalize each image
        for fname, png_path, rgb, mask in image_data:
            body_pixels = rgb[mask]
            lab = body_pixels_to_lab(body_pixels)

            image_mean_L = float(np.mean(lab[:, 0]))
            L_shift = species_mean_L - image_mean_L

            # Shift L channel
            lab[:, 0] += L_shift

            # Clamp to valid range
            n_clamped = int(np.sum(lab[:, 0] < 0) + np.sum(lab[:, 0] > 100))
            lab[:, 0] = np.clip(lab[:, 0], 0, 100)

            # Convert back to RGB
            new_rgb = lab_to_rgb_pixels(lab)

            # Rebuild RGBA image
            h, w = rgb.shape[:2]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[mask, :3] = new_rgb
            rgba[:, :, 3] = (mask * 255).astype(np.uint8)

            # Save
            out_name = os.path.splitext(fname)[0] + ".png"
            out_path = os.path.join(norm_dir, out_name)
            Image.fromarray(rgba, "RGBA").save(out_path)

            report_rows.append({
                "filename": fname,
                "species": species,
                "original_mean_L": round(image_mean_L, 2),
                "species_mean_L": round(species_mean_L, 2),
                "L_shift": round(L_shift, 2),
                "n_body_pixels": len(body_pixels),
                "n_pixels_clamped": n_clamped,
            })
            n_processed += 1

        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(species_images)} species")

    # Write report
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\n[3] Normalization complete")
    print(f"    Images normalized: {n_processed}")
    print(f"    Species skipped (no segmented images): {n_skipped}")
    print(f"    Report: {REPORT_CSV}")


if __name__ == "__main__":
    main()
