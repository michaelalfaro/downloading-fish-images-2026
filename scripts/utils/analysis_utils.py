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

# SCRIPT_DIR points to this file's location (scripts/utils/)
# REPO_DIR points to the repository root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

IMAGE_DIRS = {
    "images": os.path.join(REPO_DIR, "images"),
    "images_bishop": os.path.join(REPO_DIR, "images_bishop"),
    "images_fishbase_extra": os.path.join(REPO_DIR, "images_fishbase_extra"),
    "images_fishbase_usercontrib": os.path.join(REPO_DIR, "images_fishbase_usercontrib"),
    "images_inaturalist": os.path.join(REPO_DIR, "images_inaturalist"),
    "images_fishwise": os.path.join(REPO_DIR, "images_fishwise"),
}

ANALYSIS_DIR = os.path.join(REPO_DIR, "analysis")
GMM_DIR = os.path.join(ANALYSIS_DIR, "approach_1_gmm")
SEGMENTED_DIR = os.path.join(GMM_DIR, "segmented")
ORIENTED_DIR = os.path.join(GMM_DIR, "oriented")
NORMALIZED_DIR = os.path.join(GMM_DIR, "normalized")
ZONE_MAPS_DIR = os.path.join(GMM_DIR, "zone_maps")
K_SELECTION_DIR = os.path.join(GMM_DIR, "k_selection")
COLOR_COMP_DIR = os.path.join(GMM_DIR, "color_composition")

DATA_DIR = os.path.join(REPO_DIR, "data")
INVENTORY_CSV = os.path.join(DATA_DIR, "all_images_inventory.csv")
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
    return os.path.join(REPO_DIR, row["directory"], row["filename"])


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


def randalize_pixels(rgb_pixels, saturation_boost=1.15, sepia_amount=0.12,
                     brightness_boost=1.05):
    """Apply Randalize filter to RGB pixels (warm specimen photo style).

    This normalizes image colors toward Jack Randall's specimen photo style:
    - Slight saturation boost (more vivid colors)
    - Warm sepia tone (removes cool underwater cast)
    - Slight brightness increase

    Args:
        rgb_pixels: Nx3 uint8 RGB array (only non-transparent pixels)
        saturation_boost: Saturation multiplier (default 1.15 = 15% boost)
        sepia_amount: Sepia blend amount (default 0.12 = 12%)
        brightness_boost: Brightness multiplier (default 1.05 = 5% boost)

    Returns:
        Nx3 uint8 RGB array with Randalize effect applied
    """
    # Work in float
    rgb = rgb_pixels.astype(np.float32)

    # Saturation boost (increase distance from grayscale)
    gray = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    rgb[:, 0] = gray + (rgb[:, 0] - gray) * saturation_boost
    rgb[:, 1] = gray + (rgb[:, 1] - gray) * saturation_boost
    rgb[:, 2] = gray + (rgb[:, 2] - gray) * saturation_boost

    # Warm sepia tone (blend toward sepia color transformation)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    sepia_r = r * 0.393 + g * 0.769 + b * 0.189
    sepia_g = r * 0.349 + g * 0.686 + b * 0.168
    sepia_b = r * 0.272 + g * 0.534 + b * 0.131

    rgb[:, 0] = r * (1 - sepia_amount) + sepia_r * sepia_amount
    rgb[:, 1] = g * (1 - sepia_amount) + sepia_g * sepia_amount
    rgb[:, 2] = b * (1 - sepia_amount) + sepia_b * sepia_amount

    # Brightness boost
    rgb *= brightness_boost

    # Clamp and convert back to uint8
    return np.clip(rgb, 0, 255).astype(np.uint8)


def randalize_image(rgba_image):
    """Apply Randalize filter to an RGBA image, preserving transparency.

    Args:
        rgba_image: HxWx4 uint8 RGBA array

    Returns:
        HxWx4 uint8 RGBA array with Randalize effect on non-transparent pixels
    """
    h, w = rgba_image.shape[:2]
    result = rgba_image.copy()

    # Get mask of non-transparent pixels
    mask = rgba_image[:, :, 3] > 0

    if np.any(mask):
        # Extract RGB of visible pixels
        rgb_pixels = rgba_image[mask, :3]

        # Apply Randalize
        randalized = randalize_pixels(rgb_pixels)

        # Put back
        result[mask, :3] = randalized

    return result


def all_species(inventory):
    """Get sorted unique species list from inventory."""
    return sorted(set(row["species"] for row in inventory))


def get_oriented_path(species, filename):
    """Get the output path for an oriented image."""
    sp_dir = species_to_dirname(species)
    base = os.path.splitext(filename)[0] + ".png"
    return os.path.join(ORIENTED_DIR, sp_dir, base)


def get_best_segmented_path(species, filename):
    """Get best available segmented image: oriented if exists, else segmented.

    Use this for downstream analysis (pavo, color analysis, etc.) to ensure
    oriented images are used when available.

    Returns:
        Path to oriented image if it exists, otherwise path to segmented image.
        Returns None if neither exists.
    """
    oriented = get_oriented_path(species, filename)
    if os.path.exists(oriented):
        return oriented

    segmented = get_segmented_path(species, filename)
    if os.path.exists(segmented):
        return segmented

    return None


def get_included_images_for_species(inventory, species, annotations=None,
                                     exclude_larvae=True, exclude_multi_inat=True):
    """Get list of (filename, best_path) tuples for included images.

    This is the main function for downstream analysis. It:
    1. Filters out excluded images (larvae, multi-fish, manual excludes)
    2. Uses oriented images when available, otherwise segmented
    3. Only returns images that have a valid segmented/oriented file

    Returns:
        List of tuples: (original_filename, best_image_path)
    """
    # Get filtered inventory rows
    rows = get_images_for_species(
        inventory, species, annotations,
        exclude_larvae=exclude_larvae, exclude_multi_inat=exclude_multi_inat
    )

    results = []
    for row in rows:
        best_path = get_best_segmented_path(species, row["filename"])
        if best_path:
            results.append((row["filename"], best_path))

    return results


def count_oriented_images(species=None):
    """Count oriented images, optionally for a specific species.

    Returns:
        If species is None: dict mapping species to count
        If species is provided: int count for that species
    """
    counts = {}
    if not os.path.exists(ORIENTED_DIR):
        return 0 if species else counts

    for sp_dir in os.listdir(ORIENTED_DIR):
        sp_path = os.path.join(ORIENTED_DIR, sp_dir)
        if os.path.isdir(sp_path):
            sp_name = dirname_to_species(sp_dir)
            n = len([f for f in os.listdir(sp_path) if f.endswith('.png')])
            counts[sp_name] = n

    if species:
        return counts.get(species, 0)
    return counts


EXEMPLAR_CSV = os.path.join(GMM_DIR, "species_exemplar.csv")


def load_exemplars(csv_path=None):
    """Load species_exemplar.csv. Returns dict keyed by species.

    Each entry contains: species, filename, png_name, source, selected_at
    """
    if csv_path is None:
        csv_path = EXEMPLAR_CSV
    exemplars = {}
    if not os.path.exists(csv_path):
        return exemplars
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exemplars[row["species"]] = row
    return exemplars


def get_exemplar_path(species, exemplars=None):
    """Get the path to the exemplar image for a species.

    Uses oriented version if available, otherwise segmented.

    Returns:
        Path to exemplar image, or None if no exemplar set or file not found.
    """
    if exemplars is None:
        exemplars = load_exemplars()

    if species not in exemplars:
        return None

    png_name = exemplars[species].get("png_name", "")
    if not png_name:
        return None

    return get_best_segmented_path(species, png_name)
