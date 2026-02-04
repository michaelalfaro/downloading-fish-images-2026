#!/usr/bin/env python3
"""
Generate zone maps by applying species-specific GMM models to each image.

For each image, assigns every body pixel to its GMM cluster and produces:
1. A zone map PNG (categorical colors, transparent background)
2. Per-image cluster proportion data

Reads: analysis/approach_1_gmm/normalized/
       analysis/approach_1_gmm/k_selection/models/
Writes: analysis/approach_1_gmm/zone_maps/
        analysis/approach_1_gmm/color_composition/

Usage:
    python3 generate_zone_maps.py
    python3 generate_zone_maps.py --species "Chaetodon auriga"
"""

import os
import csv
import json
import argparse
import numpy as np
from collections import defaultdict
from PIL import Image

from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, NORMALIZED_DIR, ZONE_MAPS_DIR,
    COLOR_COMP_DIR, K_SELECTION_DIR,
    load_inventory, load_annotations, species_to_dirname,
    load_segmented_image, body_pixels_to_lab, lab_to_rgb_pixels,
    all_species, ensure_dir,
)

MODELS_DIR = os.path.join(K_SELECTION_DIR, "models")

# Distinct categorical colors for zone maps (up to 8 clusters)
ZONE_COLORS = np.array([
    [228,  26,  28],   # red
    [ 55, 126, 184],   # blue
    [ 77, 175,  74],   # green
    [152,  78, 163],   # purple
    [255, 127,   0],   # orange
    [255, 255,  51],   # yellow
    [166,  86,  40],   # brown
    [247, 129, 191],   # pink
], dtype=np.uint8)


def load_gmm_model(species):
    """Load a fitted GMM model for a species."""
    import joblib
    sp_safe = species_to_dirname(species)
    model_path = os.path.join(MODELS_DIR, f"{sp_safe}_gmm.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def generate_zone_map(rgb, mask, gmm):
    """Apply GMM to body pixels and create a zone map image.

    Returns (zone_rgba [H,W,4], labels [N], proportions dict).
    """
    h, w = rgb.shape[:2]
    body_pixels = rgb[mask]
    lab = body_pixels_to_lab(body_pixels)

    # Predict cluster assignments
    labels = gmm.predict(lab)
    k = gmm.n_components

    # Compute proportions
    proportions = {}
    for c in range(k):
        count = int(np.sum(labels == c))
        proportions[c] = count / len(labels) if len(labels) > 0 else 0

    # Build zone map image
    zone_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    body_coords = np.where(mask)

    for c in range(k):
        color = ZONE_COLORS[c % len(ZONE_COLORS)]
        cluster_mask = labels == c
        rows = body_coords[0][cluster_mask]
        cols = body_coords[1][cluster_mask]
        zone_rgba[rows, cols, :3] = color
        zone_rgba[rows, cols, 3] = 255

    return zone_rgba, labels, proportions


def main():
    parser = argparse.ArgumentParser(description="Generate zone maps")
    parser.add_argument("--species", type=str, default=None)
    args = parser.parse_args()

    print("[1] Loading inventory and annotations...")
    inventory = load_inventory()
    annotations = load_annotations()

    species_images = defaultdict(list)
    for row in inventory:
        species_images[row["species"]].append(row)

    if args.species:
        species_images = {args.species: species_images.get(args.species, [])}

    # Load k selection to know which species have models
    k_csv = os.path.join(K_SELECTION_DIR, "species_k_selection.csv")
    species_k = {}
    if os.path.exists(k_csv):
        with open(k_csv, newline="") as f:
            for row in csv.DictReader(f):
                species_k[row["species"]] = int(row["k_selected"])

    print(f"    {len(species_k)} species with GMM models")

    print("[2] Generating zone maps...")
    n_processed = 0
    n_skipped_species = 0
    all_composition_rows = []

    for i, (species, rows) in enumerate(sorted(species_images.items())):
        if species not in species_k:
            n_skipped_species += 1
            continue

        gmm = load_gmm_model(species)
        if gmm is None:
            n_skipped_species += 1
            continue

        sp_dir = species_to_dirname(species)
        norm_dir = os.path.join(NORMALIZED_DIR, sp_dir)
        zone_dir = os.path.join(ZONE_MAPS_DIR, sp_dir)
        ensure_dir(zone_dir)

        k = species_k[species]
        species_comp_rows = []

        for row in rows:
            fname = row["filename"]
            ann = annotations.get(fname)
            if ann and ann["annotation_type"] in ("larva", "manual_exclude"):
                continue

            png_name = os.path.splitext(fname)[0] + ".png"
            png_path = os.path.join(norm_dir, png_name)
            if not os.path.exists(png_path):
                continue

            try:
                rgb, mask = load_segmented_image(png_path)
            except Exception:
                continue

            body_pixels = rgb[mask]
            if len(body_pixels) < 100:
                continue

            zone_rgba, labels, proportions = generate_zone_map(rgb, mask, gmm)

            # Save zone map
            zone_name = os.path.splitext(fname)[0] + "_zones.png"
            zone_path = os.path.join(zone_dir, zone_name)
            Image.fromarray(zone_rgba, "RGBA").save(zone_path)

            # Composition data
            comp_row = {
                "filename": fname,
                "species": species,
                "k": k,
                "n_body_pixels": len(body_pixels),
            }
            # Cluster means from the GMM
            for c in range(k):
                mean = gmm.means_[c]
                comp_row[f"cluster_{c}_proportion"] = round(proportions.get(c, 0), 4)
                comp_row[f"cluster_{c}_lab_L"] = round(mean[0], 1)
                comp_row[f"cluster_{c}_lab_a"] = round(mean[1], 1)
                comp_row[f"cluster_{c}_lab_b"] = round(mean[2], 1)
                # Convert cluster center to hex for reference
                from skimage.color import lab2rgb
                lab_arr = np.array([[mean]])
                rgb_arr = lab2rgb(lab_arr)
                r, g, b = (np.clip(rgb_arr[0, 0] * 255, 0, 255)).astype(int)
                comp_row[f"cluster_{c}_hex"] = f"#{r:02x}{g:02x}{b:02x}"

            species_comp_rows.append(comp_row)
            n_processed += 1

        # Write per-species composition CSV
        if species_comp_rows:
            comp_path = os.path.join(COLOR_COMP_DIR,
                                     f"{sp_dir}_composition.csv")
            # Determine all fieldnames dynamically
            fieldnames = ["filename", "species", "k", "n_body_pixels"]
            for c in range(k):
                fieldnames.extend([
                    f"cluster_{c}_proportion",
                    f"cluster_{c}_lab_L", f"cluster_{c}_lab_a",
                    f"cluster_{c}_lab_b", f"cluster_{c}_hex",
                ])
            with open(comp_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(species_comp_rows)

            all_composition_rows.extend(species_comp_rows)

        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(species_images)} species")

    print(f"\n[3] Zone map generation complete")
    print(f"    Images processed: {n_processed}")
    print(f"    Species skipped (no model): {n_skipped_species}")
    print(f"    Zone maps: {ZONE_MAPS_DIR}")
    print(f"    Composition data: {COLOR_COMP_DIR}")


if __name__ == "__main__":
    main()
