#!/usr/bin/env python3
"""
Shared utilities for the Chaetodontidae color pattern analysis pipeline.
Imported by: segment_fish.py, detect_outliers.py, normalize_luminance.py,
             fit_gmm_species.py, generate_zone_maps.py
"""

import os
import csv
import numpy as np
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIRS = {
    "images": os.path.join(SCRIPT_DIR, "images"),
    "images_bishop": os.path.join(SCRIPT_DIR, "images_bishop"),
    "images_fishbase_extra": os.path.join(SCRIPT_DIR, "images_fishbase_extra"),
    "images_inaturalist": os.path.join(SCRIPT_DIR, "images_inaturalist"),
}

ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "analysis")
GMM_DIR = os.path.join(ANALYSIS_DIR, "approach_1_gmm")
SEGMENTED_DIR = os.path.join(GMM_DIR, "segmented")
NORMALIZED_DIR = os.path.join(GMM_DIR, "normalized")
ZONE_MAPS_DIR = os.path.join(GMM_DIR, "zone_maps")
K_SELECTION_DIR = os.path.join(GMM_DIR, "k_selection")
COLOR_COMP_DIR = os.path.join(GMM_DIR, "color_composition")

INVENTORY_CSV = os.path.join(SCRIPT_DIR, "all_images_inventory.csv")
ANNOTATIONS_CSV = os.path.join(ANALYSIS_DIR, "image_annotations.csv")


def ensure_dir(path):
    """Create directory and parents if they don't exist."""
    os.makedirs(path, exist_ok=True)


def species_to_dirname(species_name):
    """Convert 'Chaetodon auriga' to 'Chaetodon_auriga'."""
    return species_name.replace(" ", "_")


def dirname_to_species(dirname):
    """Convert 'Chaetodon_auriga' to 'Chaetodon auriga'."""
    return dirname.replace("_", " ", 1)  # only first underscore


def load_inventory(csv_path=None, exclude_duplicates=True):
    """Load all_images_inventory.csv, optionally filtering out duplicates.

    Returns list of dicts with keys:
        species, filename, source, directory, md5_hash, is_duplicate
    """
    if csv_path is None:
        csv_path = INVENTORY_CSV
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if exclude_duplicates and row.get("is_duplicate") == "yes":
                continue
            rows.append(row)
    return rows


def load_annotations(csv_path=None):
    """Load image_annotations.csv. Returns dict keyed by filename."""
    if csv_path is None:
        csv_path = ANNOTATIONS_CSV
    annotations = {}
    if not os.path.exists(csv_path):
        return annotations
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations[row["filename"]] = row
    return annotations


def save_annotations(annotations_list, csv_path=None):
    """Write list of annotation dicts to CSV."""
    if csv_path is None:
        csv_path = ANNOTATIONS_CSV
    fieldnames = [
        "filename", "species", "directory", "annotation_type",
        "annotation_detail", "n_fish_noted", "annotated_by", "date_annotated",
    ]
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(annotations_list)


def append_annotations(new_rows, csv_path=None):
    """Append annotation rows without overwriting existing ones."""
    if csv_path is None:
        csv_path = ANNOTATIONS_CSV
    existing = load_annotations(csv_path)
    # merge: new rows update or add
    for row in new_rows:
        existing[row["filename"]] = row
    all_rows = list(existing.values())
    save_annotations(all_rows, csv_path)


def get_image_path(row):
    """Get full path for an inventory row."""
    return os.path.join(SCRIPT_DIR, row["directory"], row["filename"])


def get_images_for_species(inventory, species, annotations=None,
                           exclude_larvae=True, exclude_multi_inat=True):
    """Filter inventory to images for a species, with annotation exclusions.

    Excludes:
      - larvae (annotation_type == "larva")
      - multi-fish iNaturalist images (annotation_type == "multi_fish_inat")
      - manually excluded (annotation_type == "manual_exclude")
    """
    exclude_types = set()
    if exclude_larvae:
        exclude_types.add("larva")
    if exclude_multi_inat:
        exclude_types.add("multi_fish_inat")
    exclude_types.add("manual_exclude")

    results = []
    for row in inventory:
        if row["species"] != species:
            continue
        if annotations:
            ann = annotations.get(row["filename"])
            if ann and ann["annotation_type"] in exclude_types:
                continue
        results.append(row)
    return results


def get_segmented_path(species, filename):
    """Get the output path for a segmented image."""
    sp_dir = species_to_dirname(species)
    base = os.path.splitext(filename)[0] + ".png"
    return os.path.join(SEGMENTED_DIR, sp_dir, base)


def get_normalized_path(species, filename):
    """Get the output path for a normalized image."""
    sp_dir = species_to_dirname(species)
    base = os.path.splitext(filename)[0] + ".png"
    return os.path.join(NORMALIZED_DIR, sp_dir, base)


def load_segmented_image(png_path):
    """Load an RGBA PNG. Returns (rgb_array [H,W,3], mask [H,W] bool)."""
    from PIL import Image
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3]
    mask = arr[:, :, 3] > 0
    return rgb, mask


def body_pixels_to_lab(rgb_pixels):
    """Convert Nx3 uint8 RGB array to Nx3 float CIELAB array."""
    from skimage.color import rgb2lab
    # rgb2lab expects (H, W, 3) float in [0,1]
    pixels_float = rgb_pixels.astype(np.float64) / 255.0
    # reshape to (N, 1, 3) so rgb2lab treats it as an image
    as_image = pixels_float.reshape(-1, 1, 3)
    lab_image = rgb2lab(as_image)
    return lab_image.reshape(-1, 3)


def lab_to_rgb_pixels(lab_pixels):
    """Convert Nx3 float CIELAB array to Nx3 uint8 RGB array."""
    from skimage.color import lab2rgb
    as_image = lab_pixels.reshape(-1, 1, 3)
    rgb_image = lab2rgb(as_image)
    rgb_uint8 = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)
    return rgb_uint8.reshape(-1, 3)


def all_species(inventory):
    """Get sorted unique species list from inventory."""
    return sorted(set(row["species"] for row in inventory))
