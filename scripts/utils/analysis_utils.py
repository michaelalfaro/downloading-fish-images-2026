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


def detect_underwater_metrics(rgb_pixels):
    """Compute underwater detection metrics from an Nx3 uint8 RGB pixel array.

    Scores 4 indicators of underwater photography:
      1. Blue channel > Red channel * 1.2  (blue dominance)
      2. CIELAB b* < -5                     (blue-yellow axis shifted blue)
      3. CIELAB a* < 3                      (green-red axis shifted green)
      4. (B+G) / (2*R) > 1.3               (blue-green to red ratio)

    Returns:
        dict with mean R/G/B, mean L*/a*/b*, indicator scores, uw_score (0-4),
        and is_uw (True if score >= 2).
    """
    if len(rgb_pixels) == 0:
        return {"uw_score": 0, "is_uw": False}

    mean_r = float(np.mean(rgb_pixels[:, 0]))
    mean_g = float(np.mean(rgb_pixels[:, 1]))
    mean_b = float(np.mean(rgb_pixels[:, 2]))

    # Convert to CIELAB for perceptual color metrics
    lab = body_pixels_to_lab(rgb_pixels)
    mean_L = float(np.mean(lab[:, 0]))
    mean_a_lab = float(np.mean(lab[:, 1]))
    mean_b_lab = float(np.mean(lab[:, 2]))

    # Indicator 1: blue > red * 1.2
    blue_gt_red = mean_b > mean_r * 1.2
    # Indicator 2: b* < -5 (blue shift in CIELAB)
    b_lab_neg = mean_b_lab < -5
    # Indicator 3: a* < 3 (green shift in CIELAB)
    a_lab_low = mean_a_lab < 3
    # Indicator 4: (B+G)/(2R) > 1.3
    bg_to_r = (mean_b + mean_g) / (2 * mean_r) if mean_r > 0 else 999.0
    bg_ratio_high = bg_to_r > 1.3

    uw_score = sum([blue_gt_red, b_lab_neg, a_lab_low, bg_ratio_high])

    return {
        "mean_r": round(mean_r, 1),
        "mean_g": round(mean_g, 1),
        "mean_b": round(mean_b, 1),
        "mean_L": round(mean_L, 1),
        "mean_a_lab": round(mean_a_lab, 1),
        "mean_b_lab": round(mean_b_lab, 1),
        "blue_ratio": round(mean_b / mean_r if mean_r > 0 else 0, 3),
        "bg_to_r_ratio": round(bg_to_r, 3),
        "uw_score": uw_score,
        "is_uw": uw_score >= 2,
    }


MAX_GAIN = 3.0  # Per-channel gain cap to prevent over-correction

# ══════════════════════════════════════════════════════════════════════════════
# White Balance Methods for Underwater Color Correction
#
# Based on Akkaynak & Treibitz "Sea-thru" (CVPR 2019) and the hainh/sea-thru
# implementation. The paper notes that for scenes with diverse colors, the
# Gray World Hypothesis works well (Section 4.4.2).
#
# Available methods:
#   - "max_channel" (original): normalize to brightest channel's percentile
#   - "gray_world": assume average color should be neutral gray
#   - "gray_world_10p": use brightest 10% for more robust estimation
#   - "red_boost": always boost red channel (most attenuated underwater)
# ══════════════════════════════════════════════════════════════════════════════

WB_METHODS = ["max_channel", "gray_world", "gray_world_10p", "red_boost", "depth_seathru"]

DEPTH_CSV = os.path.join(GMM_DIR, "depth_estimates.csv")
DEPTH_MAPS_DIR = os.path.join(GMM_DIR, "depth_maps")


def estimate_wb_gray_world(pixels):
    """Gray World white balance: assume average color should be neutral.

    From the Sea-thru paper (Section 4.4.2): "for the scenes that contain a
    sufficiently diverse set of colors, we adopt the simple and fast Gray
    World Hypothesis."

    This method normalizes each channel so their means become equal.

    Args:
        pixels: Nx3 float64 array in [0, 1] range

    Returns:
        [R_gain, G_gain, B_gain] multipliers
    """
    means = np.mean(pixels, axis=0)
    if np.min(means) < 1e-6:
        return [1.0, 1.0, 1.0]

    # Target is the geometric mean of all channels (preserves overall brightness)
    target = np.power(np.prod(means), 1/3)

    gains = [min(float(target / means[ch]), MAX_GAIN) if means[ch] > 1e-6 else 1.0
             for ch in range(3)]
    return gains


def estimate_wb_gray_world_10p(pixels):
    """Gray World using brightest 10% of pixels for robustness.

    From hainh/sea-thru implementation: Uses the top 10% brightest pixels
    per channel instead of the full mean. More robust to dark regions and
    shadows that can skew the average.

    Args:
        pixels: Nx3 float64 array in [0, 1] range

    Returns:
        [R_gain, G_gain, B_gain] multipliers
    """
    # Get the mean of the top 10% brightest pixels per channel
    top_10p_means = np.zeros(3)
    for ch in range(3):
        threshold = np.percentile(pixels[:, ch], 90)
        bright_mask = pixels[:, ch] >= threshold
        if np.any(bright_mask):
            top_10p_means[ch] = np.mean(pixels[bright_mask, ch])
        else:
            top_10p_means[ch] = np.mean(pixels[:, ch])

    if np.min(top_10p_means) < 1e-6:
        return [1.0, 1.0, 1.0]

    # Target is the geometric mean (preserves brightness)
    target = np.power(np.prod(top_10p_means), 1/3)

    gains = [min(float(target / top_10p_means[ch]), MAX_GAIN)
             if top_10p_means[ch] > 1e-6 else 1.0
             for ch in range(3)]
    return gains


def estimate_wb_red_boost(pixels, percentile=90):
    """Red-boosting white balance for typical underwater blue/green cast.

    Underwater images lose red wavelengths most rapidly with depth. This
    method specifically boosts the red channel to match the green channel,
    which is the middle ground between over-attenuated red and dominant blue.

    The approach: normalize R and B toward G (the intermediate channel).

    Args:
        pixels: Nx3 float64 array in [0, 1] range
        percentile: percentile to use for channel estimation (default 90)

    Returns:
        [R_gain, G_gain, B_gain] multipliers
    """
    # Get percentile values per channel
    p_vals = np.array([np.percentile(pixels[:, ch], percentile) for ch in range(3)])

    if np.min(p_vals) < 1e-6:
        return [1.0, 1.0, 1.0]

    # Use green channel as target (middle ground for underwater)
    # Red gets boosted up, blue may get reduced or stay same
    target = p_vals[1]  # Green channel

    gains = [1.0, 1.0, 1.0]
    for ch in range(3):
        if p_vals[ch] > 1e-6:
            gains[ch] = min(float(target / p_vals[ch]), MAX_GAIN)

    return gains


def estimate_seathru_params(rgb_pixels, remove_backscatter=False, percentile=90,
                            filter_extremes=True, wb_method="max_channel"):
    """Estimate Sea-thru correction parameters from an Nx3 uint8 RGB pixel array.

    Should be called on the FULL original image (including background/water)
    to get the best estimate of underwater lighting conditions. The water
    column and background provide strong signal about wavelength-dependent
    absorption and scattering.

    Args:
        rgb_pixels: Nx3 uint8 RGB array (ideally from the full original photo)
        remove_backscatter: If True, also estimate backscatter from darkest 1%
        percentile: Which percentile to use for white balance estimation.
            Use 90 for full images with water background (default).
            Use 75 for body-only estimation where bright markings (white
            stripes, specular highlights) can make the 90th percentile
            appear neutral even in underwater images.
        filter_extremes: If True (default), filter out near-white (>=240) and
            near-black (<=10) pixels. Good for full images to remove white
            backgrounds and borders. Set False for segmented body pixels
            where dark markings carry important color cast information.
        wb_method: White balance method to use:
            - "max_channel": normalize to brightest channel (original Sea-thru)
            - "gray_world": assume average should be neutral (from paper Sec 4.4.2)
            - "gray_world_10p": gray world using top 10% brightest pixels
            - "red_boost": boost red toward green (counteract underwater absorption)

    Returns:
        dict with keys:
            wb_gains: [R_gain, G_gain, B_gain] — per-channel multipliers (capped at MAX_GAIN)
            backscatter: [R_bs, G_bs, B_bs] — per-channel offsets to subtract
                         (all zeros if remove_backscatter=False)
            white_fraction: fraction of input pixels that were near-white (>=240)
            wb_method: the method used for white balance
    """
    if len(rgb_pixels) == 0:
        return {"wb_gains": [1.0, 1.0, 1.0], "backscatter": [0.0, 0.0, 0.0],
                "white_fraction": 0.0, "wb_method": wb_method}

    # Compute white fraction regardless of filtering mode
    max_ch = rgb_pixels.max(axis=1)
    n_total = len(rgb_pixels)
    n_white = int(np.sum(max_ch >= 240))
    white_fraction = n_white / n_total if n_total > 0 else 0.0

    if filter_extremes:
        # Filter out near-white and near-black pixels before estimating.
        # White backgrounds, watermarks, and black borders can dominate
        # percentile estimates and mask the actual underwater color cast.
        min_ch = rgb_pixels.min(axis=1)
        content_mask = (max_ch < 240) & (min_ch > 10)
        content_pixels = rgb_pixels[content_mask]

        # Fall back to all pixels if filtering removes too many
        if len(content_pixels) < 100:
            content_pixels = rgb_pixels
    else:
        content_pixels = rgb_pixels

    pixels = content_pixels.astype(np.float64) / 255.0

    # Estimate backscatter from darkest 1% per channel
    backscatter = [0.0, 0.0, 0.0]
    if remove_backscatter:
        for ch in range(3):
            backscatter[ch] = float(np.percentile(pixels[:, ch], 1))
            pixels[:, ch] = pixels[:, ch] - backscatter[ch]
        pixels = np.clip(pixels, 0, None)

    # White balance: select method
    if wb_method == "gray_world":
        wb_gains = estimate_wb_gray_world(pixels)
    elif wb_method == "gray_world_10p":
        wb_gains = estimate_wb_gray_world_10p(pixels)
    elif wb_method == "red_boost":
        wb_gains = estimate_wb_red_boost(pixels, percentile=percentile)
    else:
        # Default: max_channel (original Sea-thru behavior)
        wb_targets = np.zeros(3)
        for ch in range(3):
            wb_targets[ch] = np.percentile(pixels[:, ch], percentile)

        target = np.max(wb_targets)
        wb_gains = [1.0, 1.0, 1.0]
        if target > 1e-6:
            for ch in range(3):
                if wb_targets[ch] > 1e-6:
                    wb_gains[ch] = min(float(target / wb_targets[ch]), MAX_GAIN)

    return {"wb_gains": wb_gains, "backscatter": backscatter,
            "white_fraction": white_fraction, "wb_method": wb_method}


def apply_seathru_params(rgb_pixels, params):
    """Apply pre-computed Sea-thru parameters to Nx3 uint8 RGB pixels.

    Use estimate_seathru_params() on the full original image to get params,
    then apply them here to the segmented fish pixels.

    Args:
        rgb_pixels: Nx3 uint8 RGB array (e.g., body-only pixels)
        params: dict from estimate_seathru_params()

    Returns:
        Nx3 uint8 RGB array with correction applied
    """
    if len(rgb_pixels) == 0:
        return rgb_pixels.copy()

    pixels = rgb_pixels.astype(np.float64) / 255.0

    # Subtract backscatter
    bs = params.get("backscatter", [0.0, 0.0, 0.0])
    for ch in range(3):
        if bs[ch] > 0:
            pixels[:, ch] = pixels[:, ch] - bs[ch]
    pixels = np.clip(pixels, 0, None)

    # Apply white balance gains
    gains = params.get("wb_gains", [1.0, 1.0, 1.0])
    for ch in range(3):
        pixels[:, ch] = pixels[:, ch] * gains[ch]

    pixels = np.clip(pixels, 0, 1)
    return (pixels * 255).astype(np.uint8)


def seathru_pixels(rgb_pixels, remove_backscatter=False):
    """Apply Sea-thru color correction to Nx3 uint8 RGB pixels.

    Convenience wrapper that estimates parameters from the same pixels it
    corrects. For better results, use estimate_seathru_params() on the full
    original image and apply_seathru_params() on the target pixels.

    Args:
        rgb_pixels: Nx3 uint8 RGB array
        remove_backscatter: If True, subtract estimated backscatter first

    Returns:
        Nx3 uint8 RGB array with underwater color cast removed
    """
    params = estimate_seathru_params(rgb_pixels, remove_backscatter=remove_backscatter)
    return apply_seathru_params(rgb_pixels, params)


def seathru_image(rgba_image, params=None, remove_backscatter=False,
                  wb_method="max_channel"):
    """Apply Sea-thru color correction to an RGBA image, preserving transparency.

    Args:
        rgba_image: HxWx4 uint8 RGBA array
        params: Pre-computed params from estimate_seathru_params(). If None,
                estimates from the image's own visible pixels.
        remove_backscatter: If True and params is None, also remove backscatter
        wb_method: White balance method (see estimate_seathru_params for options)

    Returns:
        HxWx4 uint8 RGBA array with Sea-thru correction on non-transparent pixels
    """
    result = rgba_image.copy()

    mask = rgba_image[:, :, 3] > 0

    if np.any(mask):
        rgb_pixels = rgba_image[mask, :3]

        if params is None:
            params = estimate_seathru_params(rgb_pixels,
                                             remove_backscatter=remove_backscatter,
                                             wb_method=wb_method)

        # If the original image had a large white border/background (>25%
        # near-white pixels), the full-image estimate is unreliable — the
        # white pixels mask the underwater color cast. Fall back to
        # estimating from the segmented fish body pixels instead, using
        # p75 (bright markings make p90 neutral) and no extreme-pixel
        # filtering (dark fish markings carry color cast info).
        if params.get("white_fraction", 0) > 0.25:
            params = estimate_seathru_params(rgb_pixels,
                                             remove_backscatter=remove_backscatter,
                                             percentile=75,
                                             filter_extremes=False,
                                             wb_method=wb_method)

        corrected = apply_seathru_params(rgb_pixels, params)
        result[mask, :3] = corrected

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


# ══════════════════════════════════════════════════════════════════════════════
# Depth Estimation via Depth Anything V2
#
# Uses the ViT-L variant of Depth Anything V2 (NeurIPS 2024) for monocular
# depth estimation. Produces relative depth maps that are then scaled to
# approximate metric depth using the hainh/sea-thru convention.
# ══════════════════════════════════════════════════════════════════════════════

_depth_model = None  # lazy-loaded singleton


def load_depth_anything_model(device=None):
    """Load Depth Anything V2 ViT-L model (singleton, lazy-loaded).

    Args:
        device: "cuda", "mps", or "cpu". Auto-detected if None.

    Returns:
        (model, device_str) tuple
    """
    global _depth_model
    import torch

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if _depth_model is not None:
        return _depth_model, device

    import sys
    models_dir = os.path.join(REPO_DIR, "models")
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)

    from depth_anything_v2 import DepthAnythingV2

    model_configs = {
        "vits": {"encoder": "vits", "features": 64,
                 "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128,
                 "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256,
                 "out_channels": [256, 512, 1024, 1024]},
    }

    encoder = "vitl"
    model = DepthAnythingV2(**model_configs[encoder])

    ckpt_path = os.path.join(models_dir, "checkpoints",
                             f"depth_anything_v2_{encoder}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Depth Anything V2 weights not found at {ckpt_path}. "
            "Download from: https://huggingface.co/depth-anything/"
            "Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
        )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    _depth_model = model
    return model, device


def estimate_depth_map(model, image_rgb, input_size=518):
    """Estimate relative depth map from an RGB image.

    Args:
        model: Loaded DepthAnythingV2 model
        image_rgb: HxWx3 uint8 numpy array (RGB order)
        input_size: Model input resolution (default 518 for ViT-L)

    Returns:
        HxW float32 numpy array of relative depth values (higher = farther)
    """
    import cv2
    # DepthAnythingV2.infer_image expects BGR input
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    depth = model.infer_image(bgr, input_size=input_size)
    return depth.astype(np.float32)


def scale_depth_map(depth_map, multiply=10.0, additive=2.0):
    """Scale relative depth to approximate metric depth.

    Uses the hainh/sea-thru convention:
        depth_scaled = multiply * (1 - normalized_depth) + additive

    This converts relative depth (where higher values are farther) to
    approximate depth in meters. The additive offset accounts for minimum
    scene depth (camera-to-nearest-object distance).

    Args:
        depth_map: HxW float32 relative depth (from estimate_depth_map)
        multiply: Depth range scaling factor (default 10.0 meters)
        additive: Minimum depth offset (default 2.0 meters)

    Returns:
        HxW float32 depth in approximate meters
    """
    d = depth_map.astype(np.float64)
    d_min, d_max = d.min(), d.max()
    if d_max - d_min < 1e-8:
        return np.full_like(depth_map, additive + multiply * 0.5)
    normalized = (d - d_min) / (d_max - d_min)
    scaled = multiply * (1.0 - normalized) + additive
    return scaled.astype(np.float32)


def depth_map_stats(depth_map, mask=None):
    """Compute summary statistics from a depth map.

    Args:
        depth_map: HxW float32 depth map (in meters after scaling)
        mask: Optional HxW bool mask (True = included). If None, uses all.

    Returns:
        dict with mean, median, std, min, max depth values
    """
    if mask is not None:
        vals = depth_map[mask]
    else:
        vals = depth_map.ravel()

    if len(vals) == 0:
        return {"mean_depth": "", "median_depth": "", "std_depth": "",
                "min_depth": "", "max_depth": ""}

    return {
        "mean_depth": round(float(np.mean(vals)), 2),
        "median_depth": round(float(np.median(vals)), 2),
        "std_depth": round(float(np.std(vals)), 2),
        "min_depth": round(float(np.min(vals)), 2),
        "max_depth": round(float(np.max(vals)), 2),
    }


def load_depth_estimates(csv_path=None):
    """Load depth estimates CSV into a dict keyed by filename.

    Returns:
        dict: filename -> {mean_depth, median_depth, std_depth, ...}
    """
    if csv_path is None:
        csv_path = DEPTH_CSV
    estimates = {}
    if not os.path.exists(csv_path):
        return estimates
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            estimates[row["filename"]] = row
    return estimates


def get_depth_map_path(filename):
    """Get the path to a precomputed depth map .npz file."""
    base = os.path.splitext(filename)[0]
    return os.path.join(DEPTH_MAPS_DIR, base + "_depth.npz")


def load_depth_map(filename):
    """Load a precomputed depth map from .npz file.

    Returns:
        HxW float32 array, or None if file doesn't exist
    """
    path = get_depth_map_path(filename)
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data["depth"]


# ══════════════════════════════════════════════════════════════════════════════
# Full Physics Sea-thru Pipeline (Depth-Aware)
#
# Ported from hainh/sea-thru (Akkaynak & Treibitz, CVPR 2019).
# Unlike the simple white-balance methods above, this uses per-pixel depth
# to model and remove backscatter and wavelength-dependent attenuation.
#
# Pipeline: backscatter estimation → illumination → attenuation → recovery
# ══════════════════════════════════════════════════════════════════════════════


def _find_backscatter_estimation_points(img, depths, num_bins=10,
                                         fraction=0.01, max_vals=20):
    """Find darkest pixels at each depth range for backscatter estimation.

    Backscatter (B) increases with depth and dominates in dark image regions.
    By finding the darkest pixels at each depth bin, we estimate B(z).

    Args:
        img: HxWx3 float64 image in [0, 1]
        depths: HxW float32 depth map (in meters)
        num_bins: Number of depth bins
        fraction: Fraction of darkest pixels per bin
        max_vals: Max sample points per channel per bin

    Returns:
        (pts_r, pts_g, pts_b): Each is a list of [depth, value] pairs
    """
    z_min, z_max = depths.min(), depths.max()
    if z_max - z_min < 1e-6:
        return [], [], []

    bins = np.linspace(z_min, z_max, num_bins + 1)
    pts = [[] for _ in range(3)]  # R, G, B

    h, w = depths.shape
    flat_depths = depths.ravel()
    flat_img = img.reshape(-1, 3)

    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == num_bins - 1:
            mask = (flat_depths >= lo) & (flat_depths <= hi)
        else:
            mask = (flat_depths >= lo) & (flat_depths < hi)

        indices = np.where(mask)[0]
        if len(indices) < 10:
            continue

        bin_pixels = flat_img[indices]
        bin_depths = flat_depths[indices]

        # Brightness = sum of RGB
        brightness = bin_pixels.sum(axis=1)
        n_dark = max(1, int(len(brightness) * fraction))
        dark_idx = np.argsort(brightness)[:n_dark]

        # Sample up to max_vals points
        if len(dark_idx) > max_vals:
            dark_idx = dark_idx[np.linspace(0, len(dark_idx) - 1,
                                            max_vals, dtype=int)]

        for ch in range(3):
            for j in dark_idx:
                pts[ch].append([bin_depths[j], bin_pixels[j, ch]])

    return [np.array(p) if p else np.empty((0, 2)) for p in pts]


def _fit_backscatter_model(points, depths):
    """Fit exponential backscatter model to estimation points.

    Model: B(z) = B_inf * (1 - exp(-beta_B * z)) + J * exp(-beta_D * z)

    Args:
        points: Nx2 array of [depth, value] pairs
        depths: HxW depth map for generating full backscatter image

    Returns:
        (backscatter_map, coefficients) or (zeros, None) on failure
    """
    from scipy.optimize import curve_fit

    if len(points) < 4:
        return np.zeros(depths.shape, dtype=np.float64), None

    z = points[:, 0]
    v = points[:, 1]

    def backscatter_func(z, B_inf, beta_B, J, beta_D):
        return B_inf * (1.0 - np.exp(-beta_B * z)) + J * np.exp(-beta_D * z)

    # Multiple random restarts for robustness
    best_coefs = None
    best_loss = np.inf

    for _ in range(10):
        try:
            p0 = [np.random.uniform(0, 0.5),
                  np.random.uniform(0.01, 2.0),
                  np.random.uniform(0, 0.5),
                  np.random.uniform(0.01, 2.0)]
            bounds = ([0, 0.001, 0, 0.001], [1.0, 10.0, 1.0, 10.0])
            coefs, _ = curve_fit(backscatter_func, z, v, p0=p0,
                                 bounds=bounds, maxfev=5000)
            predicted = backscatter_func(z, *coefs)
            loss = np.mean((predicted - v) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_coefs = coefs
        except (RuntimeError, ValueError):
            continue

    if best_coefs is None:
        return np.zeros(depths.shape, dtype=np.float64), None

    bs_map = backscatter_func(depths, *best_coefs)
    bs_map = np.clip(bs_map, 0, 1)
    return bs_map, best_coefs


def _construct_neighborhood_map(depths, epsilon_frac=0.05):
    """Segment image into regions of similar depth.

    Uses a simple quantization approach (faster than flood fill for our
    image sizes) that bins depth into discrete levels.

    Args:
        depths: HxW float32 depth map
        epsilon_frac: Fraction of depth range per neighborhood

    Returns:
        (nmap, n_neighborhoods): Integer-labeled map and count
    """
    z_min, z_max = depths.min(), depths.max()
    z_range = z_max - z_min
    if z_range < 1e-6:
        return np.ones(depths.shape, dtype=np.int32), 1

    epsilon = z_range * epsilon_frac
    n_bins = max(1, int(z_range / epsilon))
    n_bins = min(n_bins, 50)  # cap for performance

    # Quantize depth into bins
    normalized = (depths - z_min) / z_range
    nmap = np.clip((normalized * n_bins).astype(np.int32), 0, n_bins - 1) + 1
    return nmap, int(nmap.max())


def _estimate_illumination(img_channel, backscatter, nmap, n_neighborhoods,
                           p=0.5, f=2.0, max_iters=50, tol=1e-5):
    """Estimate illumination map for one channel using neighborhood averaging.

    The direct signal D = image - backscatter is modulated by illumination.
    We iteratively estimate illumination by averaging within depth
    neighborhoods (similar-depth pixels see similar illumination).

    Args:
        img_channel: HxW float64 single channel
        backscatter: HxW float64 estimated backscatter for this channel
        nmap: HxW int32 neighborhood labels
        n_neighborhoods: Number of distinct neighborhoods
        p: Balance between direct estimate and neighborhood average (0-1)
        f: Brightness amplification factor
        max_iters: Maximum optimization iterations
        tol: Convergence threshold

    Returns:
        HxW float64 illumination map
    """
    D = np.maximum(img_channel - backscatter, 1e-8)
    illum = D.copy()

    for _ in range(max_iters):
        prev = illum.copy()
        # Compute neighborhood means
        for label in range(1, n_neighborhoods + 1):
            mask = nmap == label
            if np.any(mask):
                neighborhood_mean = np.mean(illum[mask])
                illum[mask] = p * D[mask] + (1 - p) * neighborhood_mean
        if np.max(np.abs(illum - prev)) < tol:
            break

    return np.clip(illum * f, 1e-8, None)


def _estimate_attenuation(depths, illum, max_val=10.0):
    """Estimate wideband attenuation coefficient beta_D.

    From the physics model: illum = exp(-beta_D * z)
    Therefore: beta_D = -ln(illum) / z

    Args:
        depths: HxW float32 depth map
        illum: HxW float64 illumination estimate
        max_val: Maximum attenuation value (clips outliers)

    Returns:
        HxW float64 attenuation coefficient map
    """
    from skimage.morphology import disk, closing

    eps = 1e-8
    safe_depths = np.maximum(depths, eps)
    safe_illum = np.maximum(illum, eps)

    beta_D = np.minimum(max_val, -np.log(safe_illum) / safe_depths)

    # Clean up with morphological closing
    valid_mask = (depths > eps) & (illum > eps)
    beta_D = beta_D * valid_mask
    beta_D = np.clip(beta_D, 0, max_val)

    # Smooth with morphological closing (removes noise)
    try:
        beta_D = closing(np.maximum(0, beta_D), disk(3))
    except Exception:
        pass  # skip if closing fails on edge cases

    return beta_D


def _wbalance_no_red_10p(img):
    """White balance using top 10% brightest pixels (no-red variant).

    Normalizes green and blue channels relative to the green channel mean,
    leaving red unchanged (red is already boosted by the recovery step).
    """
    result = img.copy()
    for ch in range(3):
        ch_vals = result[:, :, ch]
        valid = ch_vals[ch_vals > 1e-6]
        if len(valid) == 0:
            continue
        threshold = np.percentile(valid, 90)
        bright = valid[valid >= threshold]
        if len(bright) == 0:
            continue
        ch_mean = np.mean(bright)
        if ch_mean > 1e-6:
            result[:, :, ch] = result[:, :, ch] / ch_mean
    return result


def _recover_image(img, depths, backscatter_maps, beta_D_maps, nmap):
    """Apply inverse degradation model to recover true colors.

    recovered = (img - B) * exp(beta_D * depth)

    Args:
        img: HxWx3 float64 image in [0, 1]
        depths: HxW float32 depth
        backscatter_maps: list of 3 HxW arrays (R, G, B backscatter)
        beta_D_maps: list of 3 HxW arrays (R, G, B attenuation)
        nmap: HxW int32 neighborhood map

    Returns:
        HxWx3 float64 recovered image clipped to [0, 1]
    """
    result = np.zeros_like(img)
    for ch in range(3):
        direct = img[:, :, ch] - backscatter_maps[ch]
        direct = np.maximum(direct, 0)
        result[:, :, ch] = direct * np.exp(beta_D_maps[ch] * depths)

    result = np.clip(result, 0, 1)

    # Set background (nmap == 0) to original
    bg_mask = nmap == 0
    result[bg_mask] = img[bg_mask]

    # White balance and scale
    result = _wbalance_no_red_10p(result)

    # Scale to full range
    for ch in range(3):
        ch_max = result[:, :, ch].max()
        if ch_max > 1e-6:
            result[:, :, ch] = result[:, :, ch] / ch_max

    result[bg_mask] = img[bg_mask]
    return np.clip(result, 0, 1)


def seathru_depth_pipeline(img_float, depth_map, p=0.01, f=2.0,
                           num_bins=10):
    """Run the full physics-based Sea-thru correction with depth.

    This is the core depth-aware correction pipeline ported from
    hainh/sea-thru. Unlike the simple white-balance methods, it models
    backscatter and wavelength-dependent attenuation using per-pixel depth.

    Args:
        img_float: HxWx3 float64 image in [0, 1] range
        depth_map: HxW float32 depth in meters (from scale_depth_map)
        p: Illumination locality parameter (0=neighborhood, 1=direct)
        f: Brightness amplification factor
        num_bins: Number of depth bins for backscatter estimation

    Returns:
        HxWx3 float64 corrected image in [0, 1]
    """
    # Step 1: Find backscatter estimation points
    pts_r, pts_g, pts_b = _find_backscatter_estimation_points(
        img_float, depth_map, num_bins=num_bins)

    # Step 2: Fit backscatter models per channel
    bs_maps = []
    for pts in [pts_r, pts_g, pts_b]:
        bs_map, _ = _fit_backscatter_model(pts, depth_map)
        bs_maps.append(bs_map)

    # Step 3: Construct neighborhood map
    nmap, n_neighborhoods = _construct_neighborhood_map(depth_map)

    # Step 4: Estimate illumination and attenuation per channel
    beta_maps = []
    for ch in range(3):
        illum = _estimate_illumination(img_float[:, :, ch], bs_maps[ch],
                                        nmap, n_neighborhoods, p=p, f=f)
        beta_D = _estimate_attenuation(depth_map, illum)
        beta_maps.append(beta_D)

    # Step 5: Recover image
    recovered = _recover_image(img_float, depth_map, bs_maps,
                                beta_maps, nmap)

    return recovered


def seathru_depth_correct(rgba_image, depth_map_raw, multiply=10.0,
                          additive=2.0):
    """Apply full depth-aware Sea-thru correction to an RGBA image.

    This is the top-level function for depth-aware correction. It:
    1. Scales the relative depth map to approximate meters
    2. Fills transparent background with median fish color so the pipeline's
       backscatter/illumination estimation isn't poisoned by black pixels
    3. Runs the full physics pipeline (backscatter, illumination, recovery)
    4. Applies correction only to opaque fish pixels, preserving alpha

    Args:
        rgba_image: HxWx4 uint8 RGBA array
        depth_map_raw: HxW float32 relative depth (from estimate_depth_map)
        multiply: Depth range scaling (default 10.0 meters)
        additive: Minimum depth offset (default 2.0 meters)

    Returns:
        HxWx4 uint8 RGBA array with depth-aware correction applied
    """
    result = rgba_image.copy()
    alpha = rgba_image[:, :, 3]
    mask = alpha > 0

    if not np.any(mask):
        return result

    # Scale depth to meters
    depth_scaled = scale_depth_map(depth_map_raw, multiply=multiply,
                                   additive=additive)

    # Convert image to float [0, 1]
    img_float = rgba_image[:, :, :3].astype(np.float64) / 255.0

    # Fill transparent pixels with median color of opaque pixels so the
    # pipeline's darkest-pixel backscatter estimation isn't confused by
    # black background regions
    if not np.all(mask):
        median_color = np.median(img_float[mask], axis=0)
        for ch in range(3):
            img_float[:, :, ch][~mask] = median_color[ch]

    # Run physics pipeline on the filled image
    corrected = seathru_depth_pipeline(img_float, depth_scaled)

    # Apply correction only to opaque fish pixels
    corrected_uint8 = (np.clip(corrected, 0, 1) * 255).astype(np.uint8)
    result[mask, :3] = corrected_uint8[mask]

    return result
