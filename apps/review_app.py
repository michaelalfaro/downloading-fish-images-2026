#!/usr/bin/env python3
"""
Fish image review & curation app for Chaetodontidae color pattern analysis.

Browse species, view segmented + normalized thumbnails, mark images for
exclusion, orientation fixes, or alternate morphs, and set gestalt k per species.

Usage:
    python3 review_app.py                    # start on port 5050
    python3 review_app.py --port 8080        # custom port

Then open http://localhost:5050 in a browser.

Features:
- Auto-excludes iNaturalist and FishBaseUser images by default (can be unchecked)
- Three action types: Exclude, Fix Orientation, Alternate Morph
- All changes save immediately
- Resume from previous session automatically
"""

import os
import csv
import io
import argparse
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from flask import (
    Flask, render_template_string, request, jsonify, send_file, redirect,
    url_for,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'utils'))
from analysis_utils import (
    SCRIPT_DIR, REPO_DIR, GMM_DIR, SEGMENTED_DIR, NORMALIZED_DIR, ANNOTATIONS_CSV,
    IMAGE_DIRS, load_inventory, load_annotations, append_annotations, save_annotations,
    species_to_dirname, dirname_to_species, all_species, ensure_dir,
    estimate_seathru_params, seathru_image, seathru_depth_correct,
    load_depth_map, load_depth_estimates, WB_METHODS, DEPTH_CSV,
)

app = Flask(__name__)

# ── Paths ──────────────────────────────────────────────────────
REVIEW_CSV = os.path.join(GMM_DIR, "review_state.csv")
GESTALT_CSV = os.path.join(GMM_DIR, "species_gestalt_k.csv")
EXEMPLAR_CSV = os.path.join(GMM_DIR, "species_exemplar.csv")
ORIENT_REVIEW_CSV = os.path.join(GMM_DIR, "orientation_review.csv")
ORIENT_LANDMARKS_CSV = os.path.join(GMM_DIR, "orientation_landmarks.csv")
FILTERS_CSV = os.path.join(GMM_DIR, "image_filters.csv")
ORIENTED_DIR = os.path.join(GMM_DIR, "oriented")
PERSPECTIVE_DIR = os.path.join(GMM_DIR, "perspective_corrected")  # For images with perspective correction
THUMB_HEIGHT = 200

# ── In-memory state (loaded on startup) ───────────────────────
_inventory = []
_species_list = []           # sorted species names
_annotations = {}            # filename -> annotation dict
_review_state = {}           # filename -> {species, action, reviewed_at}
_gestalt_k = {}              # species -> {gestalt_k, review_status, notes, reviewed_at}
_exemplars = {}              # species -> {filename, png_name, selected_at}
_orient_warnings = set()     # filenames with orientation uncertainty
_initialized_species = set() # species that have had iNat defaults applied
_orient_landmarks = {}       # filename -> {mouth_x, mouth_y, tail_x, tail_y, angle}
_duplicate_mapping = {}      # bishop_png -> list of equivalent fishbase_pngs
_fishbase_duplicates = set() # set of fishbase_pngs that are duplicates of bishop images
_phash_cache = {}            # filepath -> hash bytes (cached perceptual hashes)
_photographer_data = {}      # filename -> photographer name
_image_metadata = {}         # filename -> metadata dict (image_type, is_grayscale, etc.)
_miyazawa_images = {}        # img_file -> species (from Miyazawa 2020 study)
_image_filters = {}          # filename -> {filter_name: enabled} (e.g., {"randall": True})
_underwater_flags = {}       # filename -> {whole_is_uw, body_is_uw, whole_uw_score, ...}
_depth_estimates = {}        # filename -> {mean_depth, median_depth, ...}
_orig_image_index = {}       # png_basename (no ext) -> original image path


# ══════════════════════════════════════════════════════════════
#  Data persistence
# ══════════════════════════════════════════════════════════════

def _load_review_state():
    """Load review_state.csv into memory."""
    state = {}
    if not os.path.exists(REVIEW_CSV):
        return state
    with open(REVIEW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            state[row["filename"]] = row
    return state


def _save_review_state():
    """Write full review state to CSV."""
    fields = ["filename", "species", "action", "reviewed_at"]
    ensure_dir(os.path.dirname(REVIEW_CSV))
    with open(REVIEW_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(_review_state.values(), key=lambda r: r["filename"]):
            w.writerow(row)


def _load_gestalt_k():
    """Load species_gestalt_k.csv into memory."""
    data = {}
    if not os.path.exists(GESTALT_CSV):
        return data
    with open(GESTALT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            data[row["species"]] = row
    return data


def _save_gestalt_k():
    """Write gestalt k data to CSV."""
    fields = ["species", "gestalt_k", "review_status", "notes", "reviewed_at"]
    ensure_dir(os.path.dirname(GESTALT_CSV))
    with open(GESTALT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sp in sorted(_gestalt_k.keys()):
            w.writerow(_gestalt_k[sp])


def _load_orient_warnings():
    """Load orientation_review.csv filenames."""
    warns = set()
    if not os.path.exists(ORIENT_REVIEW_CSV):
        return warns
    with open(ORIENT_REVIEW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            warns.add(row["filename"])
    return warns


def _load_orient_landmarks():
    """Load orientation landmarks from CSV."""
    landmarks = {}
    if not os.path.exists(ORIENT_LANDMARKS_CSV):
        return landmarks
    with open(ORIENT_LANDMARKS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            landmarks[row["filename"]] = row
    return landmarks


def _load_exemplars():
    """Load species exemplar selections from CSV."""
    exemplars = {}
    if not os.path.exists(EXEMPLAR_CSV):
        return exemplars
    with open(EXEMPLAR_CSV, newline="") as f:
        for row in csv.DictReader(f):
            exemplars[row["species"]] = row
    return exemplars


def _save_exemplars():
    """Save species exemplar selections to CSV."""
    fields = ["species", "filename", "png_name", "source", "selected_at", "is_randall"]
    ensure_dir(os.path.dirname(EXEMPLAR_CSV))
    with open(EXEMPLAR_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for sp in sorted(_exemplars.keys()):
            w.writerow(_exemplars[sp])


def _save_orient_landmarks():
    """Save orientation landmarks to CSV."""
    fields = ["filename", "species", "mouth_x", "mouth_y", "tail_x", "tail_y",
              "original_angle", "angle", "flip_h", "flip_v", "dv_tilt", "ht_tilt",
              "status", "saved_at"]
    ensure_dir(os.path.dirname(ORIENT_LANDMARKS_CSV))
    with open(ORIENT_LANDMARKS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in sorted(_orient_landmarks.values(), key=lambda r: r["filename"]):
            w.writerow(row)


PHASH_CACHE_FILE = os.path.join(GMM_DIR, "phash_cache.pkl")


def _load_phash_cache():
    """Load cached perceptual hashes from disk."""
    if os.path.exists(PHASH_CACHE_FILE):
        try:
            import pickle
            with open(PHASH_CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  Warning: Could not load phash cache: {e}")
    return {}


def _save_phash_cache():
    """Save perceptual hash cache to disk."""
    try:
        import pickle
        ensure_dir(os.path.dirname(PHASH_CACHE_FILE))
        with open(PHASH_CACHE_FILE, "wb") as f:
            pickle.dump(_phash_cache, f)
    except Exception as e:
        print(f"  Warning: Could not save phash cache: {e}")


def _compute_phash(filepath):
    """Compute perceptual hash for an image, using cache when available."""
    global _phash_cache

    # Check cache first (keyed by filepath and mtime)
    try:
        mtime = os.path.getmtime(filepath)
        cache_key = (filepath, mtime)

        if cache_key in _phash_cache:
            return _phash_cache[cache_key]
    except Exception:
        cache_key = None

    # Compute hash
    try:
        img = cv2.imread(filepath)
        if img is None:
            return None
        img = cv2.resize(img, (16, 16))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        phash = (gray > avg).flatten().tobytes()

        # Cache it
        if cache_key is not None:
            _phash_cache[cache_key] = phash

        return phash
    except Exception:
        return None


def _load_miyazawa_images():
    """Load images from Miyazawa (2020) study.

    Returns dict mapping img_file -> species for images used in the study.
    These are the images with annotated color patterns from FishBase and FishPix.
    """
    miyazawa_xlsx = os.path.join(SCRIPT_DIR, '..', 'papers', 'abb9107_data_file_s1.xlsx')

    if not os.path.exists(miyazawa_xlsx):
        print(f"  Miyazawa data not found: {miyazawa_xlsx}")
        return {}

    try:
        import openpyxl
        wb = openpyxl.load_workbook(miyazawa_xlsx, read_only=True)
        ws = wb["A_FishPatterns_img"]

        images = {}
        for row in ws.iter_rows(min_row=2, values_only=True):
            img_file = row[0]  # Original filename (e.g., "12345AF.jpg" or "Chaur_u0.jpg")
            source = row[1]    # FishBase or FishPix
            family = row[2]
            species = row[4]

            if family == "Chaetodontidae":
                images[img_file] = {
                    'species': species,
                    'source': source,
                }

        wb.close()
        return images
    except Exception as e:
        print(f"  Error loading Miyazawa data: {e}")
        return {}


def _is_miyazawa_image(filename):
    """Check if an image appears in Miyazawa (2020) study.

    Handles various filename formats:
    - Direct match: "12345AF.jpg"
    - With prefix: "Chaetodon_auriga_FishPix_12345AF.jpg"
    - FishBase style: "Chaur_u0.jpg" or "Chaetodon_auriga_FishBase_Chaur_u0.jpg"
    """
    import re

    # Direct match
    if filename in _miyazawa_images:
        return True

    # Try extracting the original filename part

    # FishPix pattern: look for "12345AF.jpg"
    match = re.search(r'(\d+AF\.jpg)', filename, re.IGNORECASE)
    if match and match.group(1) in _miyazawa_images:
        return True

    # FishBase pattern: look for "Chaur_u0.jpg" style
    match = re.search(r'([A-Z][a-z]{3,4}_u\d+\.jpg)', filename, re.IGNORECASE)
    if match and match.group(1) in _miyazawa_images:
        return True

    # Try suffix matching (our naming adds Genus_species_Source_ prefix)
    for miyazawa_file in _miyazawa_images:
        if filename.endswith(miyazawa_file):
            return True

    return False


def _load_duplicate_mapping():
    """Compute cross-database duplicate mapping using perceptual hashing.

    Compares images across ALL source pairs within each species directory
    (Bishop, FishBase, FishWise, etc.) to identify the same photograph
    appearing in multiple databases.

    Uses direct pairwise matching only (no transitive closure) with a tight
    threshold to avoid false positives from similar-looking fish of the same
    species. Each non-canonical image is assigned to the single best-matching
    canonical image from a higher-priority source.

    Returns:
        Tuple of (canonical_to_equivalents, duplicate_pngs_set)
        - canonical_to_equivalents: dict mapping canonical_png -> list of equivalent dicts
        - duplicate_pngs_set: set of non-canonical PNGs to hide from display
    """
    # Source priority: lower = higher priority (kept as canonical)
    source_priority = {"Bishop": 0, "FishBase": 1, "FishPix": 2, "FishWise": 3, "FBUser": 4, "iNat": 5, "Other": 6}
    HASH_THRESHOLD = 15  # Very tight: ~6% of 256 bits — true duplicates only

    # Load confirmed duplicates from earlier review (these bypass hash threshold)
    confirmed_pairs = set()
    confirmed_csv = os.path.join(GMM_DIR, "randall_dup_confirmed.csv")
    if os.path.exists(confirmed_csv):
        with open(confirmed_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("is_duplicate") == "yes":
                    fb_png = os.path.splitext(row.get("fishbase_file", ""))[0] + ".png"
                    b_png = os.path.splitext(row.get("bishop_file", ""))[0] + ".png"
                    if fb_png and b_png:
                        confirmed_pairs.add((fb_png, b_png))

    # Load FishWise Randall duplicates (Bishop <-> FishWise)
    fw_randall_csv = os.path.join(DATA_DIR, "fishwise_randall_to_exclude.csv")
    if os.path.exists(fw_randall_csv):
        with open(fw_randall_csv, newline="") as f:
            for row in csv.DictReader(f):
                fw_file = row.get("fishwise_file", "")
                bishop_file = row.get("duplicate_of_bishop", "")
                if fw_file and bishop_file:
                    # Convert to PNG names (segmented images)
                    fw_png = os.path.splitext(fw_file)[0] + ".png"
                    b_png = os.path.splitext(bishop_file)[0] + ".png"
                    confirmed_pairs.add((fw_png, b_png))

    canonical_map = defaultdict(list)
    dup_set = set()

    for species in _species_list:
        sp_dir = species_to_dirname(species)
        seg_dir = os.path.join(SEGMENTED_DIR, sp_dir)

        if not os.path.isdir(seg_dir):
            continue

        files = [f for f in os.listdir(seg_dir) if f.endswith(".png")]
        if len(files) < 2:
            continue

        # Group files by source
        source_groups = defaultdict(list)
        for f in files:
            src = _classify_source(f)
            source_groups[src].append(f)

        # Skip species with only one source
        if len(source_groups) < 2:
            continue

        # Compute hashes for all files
        hashes = {}
        for f in files:
            h = _compute_phash(os.path.join(seg_dir, f))
            if h is not None:
                hashes[f] = h

        prefix = sp_dir + "_"

        # Process confirmed duplicate pairs first (these don't need hash check)
        for fb_png, b_png in confirmed_pairs:
            if fb_png in hashes and b_png in hashes:
                src_fb = _classify_source(fb_png)
                src_b = _classify_source(b_png)
                pri_fb = source_priority.get(src_fb, 99)
                pri_b = source_priority.get(src_b, 99)
                if pri_fb < pri_b:
                    canonical, duplicate = fb_png, b_png
                else:
                    canonical, duplicate = b_png, fb_png
                if duplicate not in dup_set:
                    dup_set.add(duplicate)
                    h1 = np.frombuffer(hashes[canonical], dtype=np.bool_)
                    h2 = np.frombuffer(hashes[duplicate], dtype=np.bool_)
                    dist = int(np.sum(h1 != h2))
                    eq_display = duplicate
                    if eq_display.startswith(prefix):
                        eq_display = eq_display[len(prefix):]
                    canonical_map[canonical].append({
                        'png_name': duplicate,
                        'source': _classify_source(duplicate),
                        'display_name': eq_display,
                        'hash_distance': dist,
                    })

        # Pairwise cross-source comparison via perceptual hashing
        # For each lower-priority image, find the best match in a higher-priority source
        file_list = list(hashes.keys())
        for f_low in file_list:
            if f_low in dup_set:
                continue  # Already matched via confirmed pairs
            src_low = _classify_source(f_low)
            pri_low = source_priority.get(src_low, 99)

            best_match = None
            best_dist = HASH_THRESHOLD + 1

            for f_high in file_list:
                if f_high == f_low or f_high in dup_set:
                    continue
                src_high = _classify_source(f_high)
                pri_high = source_priority.get(src_high, 99)
                if pri_high >= pri_low:
                    continue  # Only match against higher-priority sources

                h1 = np.frombuffer(hashes[f_high], dtype=np.bool_)
                h2 = np.frombuffer(hashes[f_low], dtype=np.bool_)
                dist = int(np.sum(h1 != h2))
                if dist < best_dist:
                    best_dist = dist
                    best_match = f_high

            if best_match is not None and best_dist <= HASH_THRESHOLD:
                dup_set.add(f_low)
                eq_display = f_low
                if eq_display.startswith(prefix):
                    eq_display = eq_display[len(prefix):]
                canonical_map[best_match].append({
                    'png_name': f_low,
                    'source': _classify_source(f_low),
                    'display_name': eq_display,
                    'hash_distance': best_dist,
                })

    return dict(canonical_map), dup_set


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_DIR, "data")


def _load_photographer_data():
    """Load photographer metadata from all available sources."""
    data = {}

    # Primary source: enriched image_metadata.csv (has ALL photographer data)
    meta_csv = os.path.join(DATA_DIR, "image_metadata.csv")
    if os.path.exists(meta_csv):
        with open(meta_csv, newline="") as f:
            for row in csv.DictReader(f):
                fname = row.get("filename", "")
                photographer = row.get("photographer", "").strip()
                if fname and photographer:
                    data[fname] = photographer

    # Fallback: FishWise download manifest (in case image_metadata.csv is stale)
    manifest = os.path.join(DATA_DIR, "fishwise", "fishwise_download_manifest.csv")
    if os.path.exists(manifest):
        with open(manifest, newline="") as f:
            for row in csv.DictReader(f):
                local_fname = row.get("local_filename", "")
                photographer = row.get("photographer", "")
                if local_fname and photographer and local_fname not in data:
                    data[local_fname] = photographer

    # Fallback: FishBase UserContrib metadata
    uc_meta = os.path.join(DATA_DIR, "fishbase_usercontrib_metadata.csv")
    if os.path.exists(uc_meta):
        with open(uc_meta, newline="") as f:
            for row in csv.DictReader(f):
                fname = row.get("filename", "")
                photographer = row.get("photographer", "")
                if fname and photographer and fname not in data:
                    data[fname] = photographer

    # Fallback: FishBase photographer recovery CSV
    fb_meta = os.path.join(DATA_DIR, "fishbase_photographer_metadata.csv")
    if os.path.exists(fb_meta):
        with open(fb_meta, newline="") as f:
            for row in csv.DictReader(f):
                fname = row.get("filename", "")
                photographer = row.get("photographer", "")
                if fname and photographer and fname not in data:
                    data[fname] = photographer

    # Fallback: iNaturalist photographer recovery CSV
    inat_meta = os.path.join(DATA_DIR, "inaturalist_photographer_metadata.csv")
    if os.path.exists(inat_meta):
        with open(inat_meta, newline="") as f:
            for row in csv.DictReader(f):
                fname = row.get("filename", "")
                photographer = row.get("photographer", "")
                if fname and photographer and fname not in data:
                    data[fname] = photographer

    # Fallback: FishPix photographer recovery CSV
    fp_meta = os.path.join(DATA_DIR, "fishpix_photographer_metadata.csv")
    if os.path.exists(fp_meta):
        with open(fp_meta, newline="") as f:
            for row in csv.DictReader(f):
                fname = row.get("filename", "")
                photographer = row.get("photographer", "")
                if fname and photographer and fname not in data:
                    data[fname] = photographer

    # FishWise attributions (includes Jack Randall photos)
    fw_attr = os.path.join(REPO_DIR, "fishwise_attributions", "attributions.csv")
    if os.path.exists(fw_attr):
        with open(fw_attr, newline="") as f:
            for row in csv.DictReader(f):
                species = row.get("species", "")
                photographer = row.get("photographer", "")
                fw_filename = row.get("filename", "")  # e.g., "000669F000024W000002.jpg"
                if species and photographer and fw_filename:
                    # Convert to local filename format: Species_name_FishWise_ID.jpg
                    fw_id = fw_filename.replace(".jpg", "")
                    local_fname = f"{species.replace(' ', '_')}_FishWise_{fw_id}.jpg"
                    if local_fname not in data:
                        data[local_fname] = photographer

    # Fallback: Bishop Museum — all Jack Randall
    for row in _inventory:
        if row.get("source") == "BishopMuseum" and row["filename"] not in data:
            data[row["filename"]] = "Jack Randall"

    return data


def _load_image_metadata():
    """Load enriched image metadata (type, quality indicators, etc.)."""
    metadata = {}
    csv_path = os.path.join(DATA_DIR, "image_metadata.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                metadata[row["filename"]] = row
    return metadata


def _load_underwater_flags():
    """Load underwater detection report into a dict keyed by filename."""
    flags = {}
    csv_path = os.path.join(GMM_DIR, "underwater_detection_report.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                flags[row["filename"]] = row
    return flags


def _load_image_filters():
    """Load persisted filter state from CSV."""
    filters = {}
    if os.path.exists(FILTERS_CSV):
        with open(FILTERS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                fname = row["filename"]
                filter_name = row["filter"]
                enabled = row["enabled"] == "True"
                if fname not in filters:
                    filters[fname] = {}
                filters[fname][filter_name] = enabled
    return filters


def _save_image_filters():
    """Persist filter state to CSV."""
    rows = []
    for fname, fdict in _image_filters.items():
        for filter_name, enabled in fdict.items():
            if enabled:  # only persist enabled filters
                rows.append({"filename": fname, "filter": filter_name,
                             "enabled": str(enabled)})
    with open(FILTERS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "filter", "enabled"])
        writer.writeheader()
        writer.writerows(rows)


def _apply_perspective_correction(img, dorsal_ventral_angle, head_tail_angle=0):
    """Apply perspective correction centered around image center (no translation).

    Uses the standard approach from photo editing software:
    1. Translate center to origin
    2. Apply perspective transformation
    3. Translate back

    This is done by composing transformation matrices:
    M_final = T_back @ P @ T_center

    Args:
        img: PIL Image in RGBA mode
        dorsal_ventral_angle: Tilt around horizontal axis (-45 to +45 degrees)
                              Positive = top tilts toward camera
        head_tail_angle: Tilt around vertical axis (-45 to +45 degrees)
                         Positive = left tilts toward camera

    Returns:
        Perspective-corrected PIL Image
    """
    import math

    if dorsal_ventral_angle == 0 and head_tail_angle == 0:
        return img

    arr = np.array(img)
    h, w = arr.shape[:2]
    cx, cy = w / 2.0, h / 2.0  # Center point

    # Convert angles to perspective factors
    # Using sin() gives a natural perspective effect
    # Scale factor controls intensity - 0.0015 gives reasonable effect at 45°
    scale = 0.0015
    dv_factor = math.sin(math.radians(dorsal_ventral_angle)) * scale
    ht_factor = math.sin(math.radians(head_tail_angle)) * scale

    # Build the composite transformation matrix
    # Standard approach: T_back @ P @ T_center
    #
    # T_center: translate so center is at origin
    # P: perspective transform (with perspective terms in bottom row)
    # T_back: translate back to original position

    # For a 3x3 homography matrix:
    # [a, b, c]     [x]     [ax + by + c]
    # [d, e, f]  @  [y]  =  [dx + ey + f]
    # [g, h, 1]     [1]     [gx + hy + 1]
    #
    # Output = (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1)
    #
    # The g and h terms create perspective distortion
    # g affects how x changes with y (vertical perspective)
    # h affects how y changes with x (horizontal perspective)

    # Compose the matrices manually for clarity
    # T_center @ point moves point so that (cx, cy) becomes (0, 0)
    # P applies perspective at origin
    # T_back @ point moves (0, 0) back to (cx, cy)

    # The composed matrix for center-pivoted perspective:
    # This formula comes from multiplying T_back @ P @ T_center
    M = np.array([
        [1 - ht_factor * cx,  0,                    ht_factor * cx * cx],
        [0,                   1 - dv_factor * cy,   dv_factor * cy * cy],
        [ht_factor,           dv_factor,            1 - ht_factor * cx - dv_factor * cy]
    ], dtype=np.float64)

    print(f"[Perspective] DV={dorsal_ventral_angle}° HT={head_tail_angle}°")
    print(f"[Perspective] Factors: dv={dv_factor:.6f}, ht={ht_factor:.6f}")

    # Apply the perspective transformation
    result = cv2.warpPerspective(arr, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))

    return Image.fromarray(result)


def _normalize_single_image(rgba_img, species, species_dirname, filename):
    """Normalize luminance of a single image and save to normalized/ directory.

    Uses species mean L from existing normalized images if available,
    otherwise computes from the image itself.

    Args:
        rgba_img: PIL Image in RGBA mode
        species: Species name
        species_dirname: Directory name for species
        filename: Output filename (PNG)

    Returns:
        Path to normalized image, or None on failure
    """
    from skimage.color import rgb2lab, lab2rgb

    try:
        arr = np.array(rgba_img)
        mask = arr[:, :, 3] > 0
        rgb = arr[:, :, :3]

        if np.sum(mask) < 100:
            return None

        # Convert to LAB
        rgb_float = rgb.astype(np.float64) / 255.0
        lab = rgb2lab(rgb_float)

        # Get body pixels
        body_L = lab[:, :, 0][mask]

        # Try to get species mean L from existing normalized images
        norm_dir = os.path.join(NORMALIZED_DIR, species_dirname)
        species_mean_L = None

        if os.path.isdir(norm_dir):
            existing_L_values = []
            for f in os.listdir(norm_dir)[:10]:  # Sample up to 10 images
                if f.endswith(".png"):
                    try:
                        existing_img = Image.open(os.path.join(norm_dir, f)).convert("RGBA")
                        ex_arr = np.array(existing_img)
                        ex_mask = ex_arr[:, :, 3] > 0
                        ex_rgb = ex_arr[:, :, :3].astype(np.float64) / 255.0
                        ex_lab = rgb2lab(ex_rgb)
                        ex_L = ex_lab[:, :, 0][ex_mask]
                        if len(ex_L) > 100:
                            existing_L_values.append(np.mean(ex_L))
                    except:
                        pass
            if existing_L_values:
                species_mean_L = np.mean(existing_L_values)

        # If no existing normalized images, use this image's mean (no shift)
        if species_mean_L is None:
            species_mean_L = np.mean(body_L)

        # Compute L shift
        current_mean_L = np.mean(body_L)
        L_shift = species_mean_L - current_mean_L

        # Apply shift to L channel
        lab_shifted = lab.copy()
        lab_shifted[:, :, 0] = np.clip(lab[:, :, 0] + L_shift, 0, 100)

        # Convert back to RGB
        rgb_shifted = lab2rgb(lab_shifted)
        rgb_uint8 = np.clip(rgb_shifted * 255, 0, 255).astype(np.uint8)

        # Reconstruct RGBA
        result = np.zeros_like(arr)
        result[:, :, :3] = rgb_uint8
        result[:, :, 3] = arr[:, :, 3]

        # Save
        ensure_dir(NORMALIZED_DIR)
        ensure_dir(norm_dir)
        out_path = os.path.join(norm_dir, filename)
        Image.fromarray(result).save(out_path)

        print(f"[Normalize] {filename}: L shift = {L_shift:.1f} (current={current_mean_L:.1f}, target={species_mean_L:.1f})")
        return out_path

    except Exception as e:
        print(f"[Normalize] Error normalizing {filename}: {e}")
        return None


def _sync_annotations_for_image(filename, species, action):
    """Sync a single image's review action to image_annotations.csv."""
    global _annotations

    # Map action to annotation type
    action_to_ann_type = {
        "exclude": "manual_exclude",
        "fix_orientation": "fix_orientation",
        "alt_morph": "alternate_morph",
        "resegment": "needs_resegment",
    }

    if action in action_to_ann_type:
        ann_type = action_to_ann_type[action]
        new_ann = {
            "filename": filename,
            "species": species,
            "directory": "",
            "annotation_type": ann_type,
            "annotation_detail": f"Flagged during manual review",
            "n_fish_noted": "",
            "annotated_by": "user_review",
            "date_annotated": datetime.now().strftime("%Y-%m-%d"),
        }
        # Find directory from inventory
        for row in _inventory:
            if row["filename"] == filename:
                new_ann["directory"] = row["directory"]
                break
        _annotations[filename] = new_ann
        append_annotations([new_ann])

    elif action == "keep":
        # Remove any review-created annotation
        ann = _annotations.get(filename)
        if ann and ann.get("annotation_type") in ("manual_exclude", "fix_orientation", "alternate_morph", "needs_resegment") \
                and ann.get("annotated_by") == "user_review":
            del _annotations[filename]
            all_anns = list(_annotations.values())
            save_annotations(all_anns)


# ══════════════════════════════════════════════════════════════
#  Helper: get images for a species review page
# ══════════════════════════════════════════════════════════════

def _classify_source(filename):
    """Get image source from filename."""
    if "Bishop" in filename:
        return "Bishop"
    elif "FishWise" in filename:
        return "FishWise"
    elif "FishBaseUser" in filename:
        return "FBUser"
    elif "FishBase" in filename:
        return "FishBase"
    elif "FishPix" in filename:
        return "FishPix"
    elif "iNaturalist" in filename:
        return "iNat"
    return "Other"


def _is_randall_image(filename, species=None):
    """Check if image is a Randall photograph.

    All Bishop images are Randall. Some FishBase images are also Randall
    (same photos uploaded to both databases, or FishBase-only Randall images).
    FishWise images with photographer 'Jack Randall' are also Randall.
    """
    if "Bishop" in filename:
        return True

    # FishWise Randall images - check photographer data
    if "FishWise" in filename:
        # Photographer data is keyed by .jpg, but filename may be .png
        jpg_name = os.path.splitext(filename)[0] + ".jpg"
        return _photographer_data.get(jpg_name, "") == "Jack Randall"

    # FishBase-only Randall images (from Randall analysis)
    fishbase_randall_species = {
        'Chaetodon burgessi', 'Chaetodon capistratus', 'Chaetodon interruptus',
        'Chaetodon oxycephalus', 'Chaetodon quadrimaculatus', 'Chaetodon sedentarius',
        'Chaetodon striatus', 'Chaetodon triangulum', 'Hemitaurichthys thompsoni',
        'Johnrandallia nigrirostris', 'Prognathodes aculeatus',
        'Roa excelsa', 'Roa jayakari', 'Roa modesta',
    }

    if species and "FishBase" in filename and species in fishbase_randall_species:
        return True

    return False


def _apply_auto_exclude_defaults(species, images):
    """Auto-exclude iNat and FBUser images if this species hasn't been visited before.

    Returns True if any defaults were applied.
    """
    global _initialized_species

    # Check if any images for this species already have review state
    # If so, this species was already visited — don't re-apply defaults
    species_has_reviews = any(
        rs.get("species") == species for rs in _review_state.values()
    )

    if species_has_reviews:
        return False

    # Apply defaults: exclude all iNat and FishBaseUser images
    # These are typically lower-quality or underwater shots with color casts
    applied = False
    now = datetime.now().isoformat()
    for im in images:
        if im["source"] in ("iNat", "FBUser") and im["orig_fname"] not in _review_state:
            _review_state[im["orig_fname"]] = {
                "filename": im["orig_fname"],
                "species": species,
                "action": "exclude",
                "reviewed_at": now,
            }
            _sync_annotations_for_image(im["orig_fname"], species, "exclude")
            applied = True

    if applied:
        _save_review_state()

    return applied


def _get_species_images(species, apply_defaults=True):
    """Get all images for a species with metadata for the review page.

    Returns list of dicts sorted by source priority then filename.
    If apply_defaults=True and species is unvisited, auto-excludes iNat and FBUser images.
    """
    sp_dir = species_to_dirname(species)
    seg_dir = os.path.join(SEGMENTED_DIR, sp_dir)
    norm_dir = os.path.join(NORMALIZED_DIR, sp_dir)

    # Get all segmented PNGs
    seg_files = set()
    if os.path.isdir(seg_dir):
        seg_files = {f for f in os.listdir(seg_dir) if f.endswith(".png")}

    norm_files = set()
    if os.path.isdir(norm_dir):
        norm_files = {f for f in os.listdir(norm_dir) if f.endswith(".png")}

    all_files = seg_files | norm_files

    # Filter out FishBase duplicates (images that are identical to Bishop images)
    # These will be noted on the Bishop image instead of shown separately
    all_files = {f for f in all_files if f not in _fishbase_duplicates}

    images = []
    source_order = {"Bishop": 0, "FishBase": 1, "FishPix": 2, "FishWise": 3, "iNat": 4, "FBUser": 5, "Other": 6}

    for png_name in sorted(all_files):
        base = os.path.splitext(png_name)[0]
        # Find original filename in inventory
        orig_fname = None
        for row in _inventory:
            if row["species"] == species and os.path.splitext(row["filename"])[0] == base:
                orig_fname = row["filename"]
                break
        if orig_fname is None:
            orig_fname = base + ".jpg"

        source = _classify_source(png_name)

        # Strip species prefix for compact display
        display_name = png_name
        prefix = sp_dir + "_"
        if display_name.startswith(prefix):
            display_name = display_name[len(prefix):]

        has_normalized = png_name in norm_files
        has_segmented = png_name in seg_files

        # Get current review state (before defaults)
        review = _review_state.get(orig_fname, {})
        action = review.get("action", "keep")

        # Check existing annotations
        ann = _annotations.get(orig_fname)
        ann_badge = None
        if ann:
            at = ann.get("annotation_type", "")
            if at == "potential_larva":
                ann_badge = "potential larva"
            elif at == "outlier":
                ann_badge = f"outlier: {ann.get('annotation_detail', '')[:30]}"
            elif at == "multi_fish":
                ann_badge = f"multi-fish ({ann.get('n_fish_noted', '?')})"
            elif at == "larva":
                ann_badge = "larva (confirmed)"

        orient_warning = orig_fname in _orient_warnings

        # Check if this is the exemplar
        is_exemplar = False
        if species in _exemplars:
            if _exemplars[species].get("png_name") == png_name:
                is_exemplar = True

        # Check if oriented version exists
        oriented_path = os.path.join(ORIENTED_DIR, sp_dir, png_name)
        has_oriented = os.path.exists(oriented_path)

        # Check if this is a Randall image (Bishop or FishBase-only Randall)
        is_randall = _is_randall_image(png_name, species)

        # Check if this image is from Miyazawa (2020) study
        is_miyazawa = _is_miyazawa_image(orig_fname)

        # Get FishBase equivalent names for Bishop images (duplicates that are hidden)
        fishbase_equivalents = []
        if "Bishop" in png_name and png_name in _duplicate_mapping:
            fishbase_equivalents = [dup['display_name'] for dup in _duplicate_mapping[png_name]]

        # Cross-database equivalents (generalized)
        cross_db_equivalents = []
        if png_name in _duplicate_mapping:
            cross_db_equivalents = [
                {"source": _classify_source(dup.get("png_name", "")),
                 "name": dup["display_name"]}
                for dup in _duplicate_mapping[png_name]
            ]

        # Photographer data
        photographer = _photographer_data.get(orig_fname, "")

        # Enriched metadata
        meta = _image_metadata.get(orig_fname, {})
        image_type = meta.get("image_type", "")
        is_grayscale = meta.get("is_grayscale", "") == "yes"
        is_lateral = meta.get("is_lateral", "") == "yes"
        is_single_fish = meta.get("is_single_fish", "") == "yes"
        seg_quality = meta.get("segmentation_quality", "")

        # Underwater detection flags
        uw = _underwater_flags.get(orig_fname, {})
        uw_whole = uw.get("whole_is_uw", "") == "True"
        uw_body = uw.get("body_is_uw", "") == "True"

        # Depth estimates
        depth = _depth_estimates.get(orig_fname, {})
        has_depth = bool(depth.get("whole_mean_depth", ""))
        depth_mean = depth.get("whole_mean_depth", "")
        depth_range = depth.get("depth_range_m", "")
        body_depth_mean = depth.get("body_mean_depth", "")
        # Flag Bishop museum specimens where depth estimate is camera
        # distance to a jar/tray, not underwater depth
        depth_is_specimen = has_depth and source == "Bishop"

        images.append({
            "png_name": png_name,
            "orig_fname": orig_fname,
            "display_name": display_name,
            "source": source,
            "source_order": source_order.get(source, 6),
            "has_normalized": has_normalized,
            "has_segmented": has_segmented,
            "has_oriented": has_oriented,
            "action": action,
            "ann_badge": ann_badge,
            "orient_warning": orient_warning,
            "is_exemplar": is_exemplar,
            "is_randall": is_randall,
            "is_miyazawa": is_miyazawa,
            "fishbase_equivalents": fishbase_equivalents,
            "cross_db_equivalents": cross_db_equivalents,
            "photographer": photographer,
            "image_type": image_type,
            "is_grayscale": is_grayscale,
            "is_lateral": is_lateral,
            "is_single_fish": is_single_fish,
            "seg_quality": seg_quality,
            "uw_whole": uw_whole,
            "uw_body": uw_body,
            "has_depth": has_depth,
            "depth_mean": depth_mean,
            "depth_range": depth_range,
            "body_depth_mean": body_depth_mean,
            "depth_is_specimen": depth_is_specimen,
            "filters": _image_filters.get(orig_fname, {}),
        })

    # Sort by: included first, then exemplar first, then source priority, then filename
    # Excluded actions: exclude, alt_morph, resegment, color_correct
    excluded_actions = {"exclude", "alt_morph", "resegment", "color_correct"}

    def sort_key(x):
        is_excluded = 1 if x["action"] in excluded_actions else 0
        is_exemplar = 0 if x["is_exemplar"] else 1  # Exemplar first (0 before 1)
        return (is_excluded, is_exemplar, x["source_order"], x["png_name"])

    images.sort(key=sort_key)

    # Add divider marker between included and excluded images
    found_first_excluded = False
    for im in images:
        if im["action"] in excluded_actions and not found_first_excluded:
            im["is_first_excluded"] = True
            found_first_excluded = True
        else:
            im["is_first_excluded"] = False

    # Apply auto-exclude defaults if this is first visit
    if apply_defaults:
        if _apply_auto_exclude_defaults(species, images):
            # Re-read actions after defaults applied
            for im in images:
                review = _review_state.get(im["orig_fname"], {})
                im["action"] = review.get("action", "keep")

    return images


def _get_species_summary():
    """Build summary data for the index page."""
    summaries = []
    for species in _species_list:
        sp_dir = species_to_dirname(species)
        seg_dir = os.path.join(SEGMENTED_DIR, sp_dir)
        norm_dir = os.path.join(NORMALIZED_DIR, sp_dir)

        n_seg = len([f for f in os.listdir(seg_dir) if f.endswith(".png")]) \
            if os.path.isdir(seg_dir) else 0
        n_norm = len([f for f in os.listdir(norm_dir) if f.endswith(".png")]) \
            if os.path.isdir(norm_dir) else 0

        gk = _gestalt_k.get(species, {})
        gestalt = gk.get("gestalt_k", "")
        status = gk.get("review_status", "not_reviewed")

        # Count actions for this species
        n_excluded = 0
        n_fix = 0
        n_alt_morph = 0
        for fname, rs in _review_state.items():
            if rs.get("species") == species:
                if rs["action"] == "exclude":
                    n_excluded += 1
                elif rs["action"] == "fix_orientation":
                    n_fix += 1
                elif rs["action"] == "alt_morph":
                    n_alt_morph += 1

        # Check exemplar
        has_exemplar = species in _exemplars

        summaries.append({
            "species": species,
            "dirname": sp_dir,
            "n_seg": n_seg,
            "n_norm": n_norm,
            "gestalt_k": gestalt,
            "status": status,
            "n_excluded": n_excluded,
            "n_fix": n_fix,
            "n_alt_morph": n_alt_morph,
            "has_exemplar": has_exemplar,
        })
    return summaries


# ══════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    summaries = _get_species_summary()
    n_reviewed = sum(1 for s in summaries if s["status"] == "reviewed")
    n_in_progress = sum(1 for s in summaries if s["status"] == "in_progress")
    show_all = request.args.get("show_all", "0") == "1"

    return render_template_string(INDEX_HTML,
                                  summaries=summaries,
                                  n_total=len(summaries),
                                  n_reviewed=n_reviewed,
                                  n_in_progress=n_in_progress,
                                  show_all=show_all)


@app.route("/review/<species_dirname>")
def review(species_dirname):
    species = dirname_to_species(species_dirname)
    if species not in _species_list:
        return f"Species not found: {species}", 404

    idx = _species_list.index(species)
    prev_sp = species_to_dirname(_species_list[idx - 1]) if idx > 0 else None
    next_sp = species_to_dirname(_species_list[idx + 1]) if idx < len(_species_list) - 1 else None

    # Get images with auto-defaults for iNat
    images = _get_species_images(species, apply_defaults=True)

    gk = _gestalt_k.get(species, {})
    gestalt_k = gk.get("gestalt_k", "4")
    notes = gk.get("notes", "")
    status = gk.get("review_status", "not_reviewed")

    n_excluded = sum(1 for im in images if im["action"] == "exclude")
    n_fix = sum(1 for im in images if im["action"] == "fix_orientation")
    n_alt_morph = sum(1 for im in images if im["action"] == "alt_morph")
    n_resegment = sum(1 for im in images if im["action"] == "resegment")
    n_color_correct = sum(1 for im in images if im["action"] == "color_correct")
    has_exemplar = species in _exemplars

    return render_template_string(REVIEW_HTML,
                                  species=species,
                                  species_dirname=species_dirname,
                                  idx=idx,
                                  n_total=len(_species_list),
                                  prev_sp=prev_sp,
                                  next_sp=next_sp,
                                  images=images,
                                  gestalt_k=gestalt_k,
                                  notes=notes,
                                  status=status,
                                  n_excluded=n_excluded,
                                  n_fix=n_fix,
                                  n_alt_morph=n_alt_morph,
                                  n_resegment=n_resegment,
                                  n_color_correct=n_color_correct,
                                  has_exemplar=has_exemplar)


@app.route("/image/<img_type>/<species_dirname>/<filename>")
def serve_image(img_type, species_dirname, filename):
    if img_type == "normalized":
        base_dir = NORMALIZED_DIR
    elif img_type == "segmented":
        base_dir = SEGMENTED_DIR
    elif img_type == "oriented":
        base_dir = ORIENTED_DIR
    else:
        return "Invalid type", 404

    path = os.path.join(base_dir, species_dirname, filename)
    if not os.path.exists(path):
        return "Not found", 404

    try:
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        if h > 0:
            scale = THUMB_HEIGHT / h
            new_w = max(1, int(w * scale))
            img = img.resize((new_w, THUMB_HEIGHT), Image.LANCZOS)

        # Apply Sea-thru correction if requested
        # Estimates params from the full original JPEG (better water column info),
        # then applies them to the segmented/oriented fish pixels.
        # Supports different white balance methods via wb_method parameter:
        #   - max_channel (default): original Sea-thru behavior
        #   - gray_world: assume average color should be neutral (paper Sec 4.4.2)
        #   - gray_world_10p: gray world using top 10% brightest pixels
        #   - red_boost: boost red toward green (counteract underwater absorption)
        seathru_mode = request.args.get("seathru")
        if seathru_mode and seathru_mode != "0":
            # seathru=1 means default (max_channel), otherwise it's the wb_method name
            wb_method = seathru_mode if seathru_mode in WB_METHODS else "max_channel"

            arr_rgba = np.array(img, dtype=np.uint8)

            if wb_method == "depth_seathru":
                # Depth-aware physics correction
                base_no_ext = os.path.splitext(filename)[0]
                # Find original filename to load depth map
                orig_fname = None
                for inv_row in _inventory:
                    inv_base = os.path.splitext(inv_row["filename"])[0]
                    if inv_base == base_no_ext:
                        orig_fname = inv_row["filename"]
                        break
                depth_map = load_depth_map(orig_fname) if orig_fname else None
                if depth_map is not None:
                    # Resize depth to match thumbnail
                    if depth_map.shape[:2] != arr_rgba.shape[:2]:
                        depth_pil = Image.fromarray(depth_map)
                        depth_pil = depth_pil.resize(
                            (arr_rgba.shape[1], arr_rgba.shape[0]),
                            Image.LANCZOS)
                        depth_map = np.array(depth_pil, dtype=np.float32)
                    arr_rgba = seathru_depth_correct(arr_rgba, depth_map)
                else:
                    # Fall back to gray_world if no depth map
                    arr_rgba = seathru_image(arr_rgba, wb_method="gray_world")
            else:
                base = os.path.splitext(filename)[0]
                orig_path = _orig_image_index.get(base)
                params = None
                if orig_path and os.path.exists(orig_path):
                    try:
                        orig_img = Image.open(orig_path).convert("RGB")
                        # Downsample for speed
                        ow, oh = orig_img.size
                        if max(ow, oh) > 512:
                            s = 512 / max(ow, oh)
                            orig_img = orig_img.resize(
                                (max(1, int(ow * s)), max(1, int(oh * s))),
                                Image.LANCZOS)
                        orig_pixels = np.array(orig_img).reshape(-1, 3)
                        params = estimate_seathru_params(orig_pixels, wb_method=wb_method)
                    except Exception:
                        pass  # fall back to self-estimation
                arr_rgba = seathru_image(arr_rgba, params=params, wb_method=wb_method)

            img = Image.fromarray(arr_rgba, "RGBA")

        # Composite on gray background
        arr = np.array(img, dtype=np.float32)
        rgb = arr[:, :, :3]
        alpha = (arr[:, :, 3] / 255.0)[:, :, np.newaxis]
        bg = 180.0
        result = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)

        out = Image.fromarray(result)
        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return f"Error: {e}", 500


@app.route("/image/source/<filename>")
def serve_source_image(filename):
    """Serve the original unsegmented source image (JPEG) as a thumbnail."""
    base = os.path.splitext(filename)[0]
    orig_path = _orig_image_index.get(base)
    if not orig_path or not os.path.exists(orig_path):
        return "Not found", 404
    try:
        img = Image.open(orig_path).convert("RGB")
        w, h = img.size
        if h > 0:
            scale = THUMB_HEIGHT / h
            new_w = max(1, int(w * scale))
            img = img.resize((new_w, THUMB_HEIGHT), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        return f"Error: {e}", 500


# ── API routes ─────────────────────────────────────────────

@app.route("/api/save_image_action", methods=["POST"])
def api_save_image_action():
    data = request.get_json()
    filename = data.get("filename", "")
    species = data.get("species", "")
    action = data.get("action", "keep")

    if not filename or not species:
        return jsonify({"error": "missing filename or species"}), 400

    valid_actions = ("keep", "exclude", "fix_orientation", "alt_morph", "resegment", "color_correct")
    if action not in valid_actions:
        return jsonify({"error": "invalid action"}), 400

    now = datetime.now().isoformat()

    if action == "keep":
        if filename in _review_state:
            old_action = _review_state[filename].get("action", "keep")
            del _review_state[filename]
            if old_action != "keep":
                _sync_annotations_for_image(filename, species, "keep")
    else:
        _review_state[filename] = {
            "filename": filename,
            "species": species,
            "action": action,
            "reviewed_at": now,
        }
        _sync_annotations_for_image(filename, species, action)

    _save_review_state()

    # Mark species as in_progress if not already reviewed
    if species not in _gestalt_k:
        _gestalt_k[species] = {
            "species": species,
            "gestalt_k": "4",
            "review_status": "in_progress",
            "notes": "",
            "reviewed_at": now,
        }
    elif _gestalt_k[species].get("review_status") == "not_reviewed":
        _gestalt_k[species]["review_status"] = "in_progress"
        _gestalt_k[species]["reviewed_at"] = now
    _save_gestalt_k()

    return jsonify({"ok": True})


@app.route("/api/save_gestalt_k", methods=["POST"])
def api_save_gestalt_k():
    data = request.get_json()
    species = data.get("species", "")
    k_val = data.get("gestalt_k", "4")
    notes = data.get("notes", "")

    if not species:
        return jsonify({"error": "missing species"}), 400

    now = datetime.now().isoformat()

    if species not in _gestalt_k:
        _gestalt_k[species] = {
            "species": species,
            "gestalt_k": str(k_val),
            "review_status": "in_progress",
            "notes": notes,
            "reviewed_at": now,
        }
    else:
        _gestalt_k[species]["gestalt_k"] = str(k_val)
        _gestalt_k[species]["notes"] = notes
        _gestalt_k[species]["reviewed_at"] = now
        if _gestalt_k[species].get("review_status") == "not_reviewed":
            _gestalt_k[species]["review_status"] = "in_progress"

    _save_gestalt_k()
    return jsonify({"ok": True})


@app.route("/api/save_filter", methods=["POST"])
def api_save_filter():
    """Save filter state for an image.

    For Sea-thru filter, also accepts 'method' parameter to store the
    white balance method: gray_world, gray_world_10p, red_boost, max_channel
    """
    data = request.get_json()
    filename = data.get("filename", "")
    filter_name = data.get("filter", "")
    enabled = data.get("enabled", False)
    method = data.get("method", None)  # For Sea-thru wb_method

    if not filename or not filter_name:
        return jsonify({"error": "missing filename or filter"}), 400

    if filename not in _image_filters:
        _image_filters[filename] = {}

    _image_filters[filename][filter_name] = enabled

    # Store Sea-thru method if provided
    if filter_name == "seathru" and method:
        _image_filters[filename]["seathru_method"] = method

    _save_image_filters()

    return jsonify({"ok": True, "filter": filter_name, "enabled": enabled, "method": method})


@app.route("/api/export_metadata")
def api_export_metadata():
    """Export all image metadata as a CSV download.

    Query params:
        scope: "included" (default) or "all"
            - included: only images with action "keep"
            - all: all images regardless of action
    """
    scope = request.args.get("scope", "included")
    excluded_actions = {"exclude", "alt_morph", "resegment", "color_correct"}

    fieldnames = [
        "filename", "species", "source", "photographer", "image_type",
        "is_lateral", "is_single_fish", "is_grayscale", "segmentation_quality",
        "uw_whole", "uw_body", "uw_score_whole", "uw_score_body",
        "mean_depth", "median_depth", "depth_range_m", "body_mean_depth",
        "action", "has_seathru", "seathru_method", "has_randall", "is_exemplar",
    ]

    rows = []
    for row in _inventory:
        fname = row["filename"]
        species = row["species"]

        # Action/review state
        review = _review_state.get(fname, {})
        action = review.get("action", "keep")

        # Filter by scope
        if scope == "included" and action in excluded_actions:
            continue

        # Metadata
        meta = _image_metadata.get(fname, {})
        uw = _underwater_flags.get(fname, {})
        depth = _depth_estimates.get(fname, {})
        filters = _image_filters.get(fname, {})

        # Exemplar check
        exemplar_data = _exemplars.get(species, {})
        is_exemplar = exemplar_data.get("filename", "") == fname

        rows.append({
            "filename": fname,
            "species": species,
            "source": row.get("source", ""),
            "photographer": _photographer_data.get(fname, ""),
            "image_type": meta.get("image_type", ""),
            "is_lateral": meta.get("is_lateral", ""),
            "is_single_fish": meta.get("is_single_fish", ""),
            "is_grayscale": meta.get("is_grayscale", ""),
            "segmentation_quality": meta.get("segmentation_quality", ""),
            "uw_whole": uw.get("whole_is_uw", ""),
            "uw_body": uw.get("body_is_uw", ""),
            "uw_score_whole": uw.get("whole_uw_score", ""),
            "uw_score_body": uw.get("body_uw_score", ""),
            "mean_depth": depth.get("whole_mean_depth", ""),
            "median_depth": depth.get("whole_median_depth", ""),
            "depth_range_m": depth.get("depth_range_m", ""),
            "body_mean_depth": depth.get("body_mean_depth", ""),
            "action": action,
            "has_seathru": str(bool(filters.get("seathru"))),
            "seathru_method": filters.get("seathru_method", ""),
            "has_randall": str(bool(filters.get("randall"))),
            "is_exemplar": str(is_exemplar),
        })

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    output = io.BytesIO(buf.getvalue().encode("utf-8"))
    output.seek(0)
    dl_name = f"chaetodontidae_metadata_{scope}.csv"
    return send_file(output, mimetype="text/csv",
                     as_attachment=True, download_name=dl_name)


@app.route("/api/mark_reviewed", methods=["POST"])
def api_mark_reviewed():
    data = request.get_json()
    species = data.get("species", "")
    if not species:
        return jsonify({"error": "missing species"}), 400

    now = datetime.now().isoformat()
    if species not in _gestalt_k:
        _gestalt_k[species] = {
            "species": species,
            "gestalt_k": "4",
            "review_status": "reviewed",
            "notes": "",
            "reviewed_at": now,
        }
    else:
        _gestalt_k[species]["review_status"] = "reviewed"
        _gestalt_k[species]["reviewed_at"] = now

    _save_gestalt_k()
    return jsonify({"ok": True})


@app.route("/api/set_exemplar", methods=["POST"])
def api_set_exemplar():
    """Set or clear the exemplar image for a species."""
    data = request.get_json()
    species = data.get("species", "")
    png_name = data.get("png_name", "")
    filename = data.get("filename", "")
    source = data.get("source", "")
    clear = data.get("clear", False)

    if not species:
        return jsonify({"error": "missing species"}), 400

    now = datetime.now().isoformat()

    if clear:
        # Clear the exemplar for this species
        if species in _exemplars:
            del _exemplars[species]
            _save_exemplars()
        return jsonify({"ok": True, "cleared": True})

    if not png_name:
        return jsonify({"error": "missing png_name"}), 400

    # Determine if this is a Randall image
    is_randall = _is_randall_image(png_name, species)

    _exemplars[species] = {
        "species": species,
        "filename": filename,
        "png_name": png_name,
        "source": source,
        "selected_at": now,
        "is_randall": is_randall,
    }
    _save_exemplars()
    return jsonify({"ok": True})


@app.route("/image/full/<species_dirname>/<filename>")
def serve_full_image(species_dirname, filename):
    """Serve full-size segmented image for orientation digitizing."""
    path = os.path.join(SEGMENTED_DIR, species_dirname, filename)
    if not os.path.exists(path):
        return "Not found", 404

    try:
        img = Image.open(path).convert("RGBA")

        # Composite on gray background
        arr = np.array(img, dtype=np.float32)
        rgb = arr[:, :, :3]
        alpha = (arr[:, :, 3] / 255.0)[:, :, np.newaxis]
        bg = 180.0
        result = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)

        out = Image.fromarray(result)
        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return f"Error: {e}", 500


@app.route("/api/get_orient_landmarks/<filename>")
def api_get_orient_landmarks(filename):
    """Get existing landmarks for an image."""
    if filename in _orient_landmarks:
        return jsonify(_orient_landmarks[filename])
    return jsonify({})


@app.route("/api/save_orient_landmarks", methods=["POST"])
def api_save_orient_landmarks():
    """Save mouth/tail landmarks and compute rotation angle."""
    data = request.get_json()
    filename = data.get("filename", "")
    species = data.get("species", "")
    mouth_x = data.get("mouth_x")
    mouth_y = data.get("mouth_y")
    tail_x = data.get("tail_x")
    tail_y = data.get("tail_y")
    adjusted_angle = data.get("adjusted_angle")  # Optional: manually adjusted angle
    flip_h = data.get("flip_h", False)  # Horizontal flip
    flip_v = data.get("flip_v", False)  # Vertical flip
    dv_tilt = data.get("dv_tilt", 0)    # Dorsal-ventral perspective tilt
    ht_tilt = data.get("ht_tilt", 0)    # Head-tail perspective tilt

    if not filename or mouth_x is None or tail_x is None:
        return jsonify({"error": "missing data"}), 400

    # Compute angle: we want fish horizontal, mouth on LEFT
    import math
    dx = tail_x - mouth_x
    dy = tail_y - mouth_y
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # Use adjusted angle if provided
    final_angle = float(adjusted_angle) if adjusted_angle is not None else angle_deg

    # We want the tail on the right, so we rotate by -angle to make horizontal
    # The result will have mouth-tail vector pointing right (0 degrees)
    # Then we'll flip if needed to put mouth on left

    now = datetime.now().isoformat()
    _orient_landmarks[filename] = {
        "filename": filename,
        "species": species,
        "mouth_x": str(mouth_x),
        "mouth_y": str(mouth_y),
        "tail_x": str(tail_x),
        "tail_y": str(tail_y),
        "original_angle": str(round(angle_deg, 2)),
        "angle": str(round(final_angle, 2)),
        "flip_h": "yes" if flip_h else "no",
        "flip_v": "yes" if flip_v else "no",
        "dv_tilt": str(dv_tilt),
        "ht_tilt": str(ht_tilt),
        "status": "digitized",
        "saved_at": now,
    }
    _save_orient_landmarks()

    return jsonify({"ok": True, "angle": final_angle})


@app.route("/api/apply_orientation", methods=["POST"])
def api_apply_orientation():
    """Apply rotation to image and save to oriented/ directory.

    If perspective correction was applied, saves BOTH versions:
    - Standard rotation: goes to oriented/ and normalized/ (INCLUDED in Pavo)
    - Perspective-corrected: goes to perspective_corrected/ (EXCLUDED by default)
    """
    data = request.get_json()
    filename = data.get("filename", "")
    species = data.get("species", "")
    species_dirname = data.get("species_dirname", "")

    if not filename or filename not in _orient_landmarks:
        return jsonify({"error": "no landmarks for this image"}), 400

    landmarks = _orient_landmarks[filename]
    angle = float(landmarks["angle"])
    dv_tilt = float(landmarks.get("dv_tilt", 0))
    ht_tilt = float(landmarks.get("ht_tilt", 0))

    # Load the original segmented image
    seg_path = os.path.join(SEGMENTED_DIR, species_dirname, filename)
    if not os.path.exists(seg_path):
        return jsonify({"error": "segmented image not found"}), 404

    try:
        img = Image.open(seg_path).convert("RGBA")

        # Rotate by the angle to make horizontal (PIL rotates counter-clockwise)
        # expand=True to avoid cropping
        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)

        # After rotation by atan2(dy, dx), the mouth-to-tail vector points along
        # the positive x-axis. This means tail is on RIGHT, mouth is on LEFT.
        # This is correct for left lateral view - no automatic flip needed.

        # Apply user-requested flips (for upside-down fish, etc.)
        flip_h = landmarks.get("flip_h", "no") == "yes"
        flip_v = landmarks.get("flip_v", "no") == "yes"
        if flip_h:
            rotated = rotated.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_v:
            rotated = rotated.transpose(Image.FLIP_TOP_BOTTOM)

        # Save STANDARD rotation to oriented directory (always)
        ensure_dir(ORIENTED_DIR)
        sp_orient_dir = os.path.join(ORIENTED_DIR, species_dirname)
        ensure_dir(sp_orient_dir)
        out_path = os.path.join(sp_orient_dir, filename)
        rotated.save(out_path)
        print(f"[Apply] Saved standard rotation to {out_path}")

        # Check if ANY perspective correction was applied
        has_perspective = dv_tilt != 0 or ht_tilt != 0

        # If perspective correction is used, DON'T save the standard version to normalized/
        # (both versions get excluded from Pavo by default)
        if has_perspective:
            # Save standard rotation to oriented/ but NOT to normalized/
            print(f"[Apply] Perspective correction used - standard rotation NOT added to normalized/")
            norm_path = None
        else:
            # No perspective correction - save to normalized/ for Pavo
            norm_path = _normalize_single_image(rotated, species, species_dirname, filename)

        # If perspective correction was requested, save THAT version too
        perspective_path = None
        if has_perspective:
            perspective_corrected = _apply_perspective_correction(rotated, dv_tilt, ht_tilt)

            # Save to perspective_corrected directory
            ensure_dir(PERSPECTIVE_DIR)
            sp_persp_dir = os.path.join(PERSPECTIVE_DIR, species_dirname)
            ensure_dir(sp_persp_dir)
            perspective_path = os.path.join(sp_persp_dir, filename)
            perspective_corrected.save(perspective_path)
            print(f"[Apply] Saved perspective-corrected version to {perspective_path}")
            print(f"[Apply] DV tilt: {dv_tilt}°, HT tilt: {ht_tilt}°")

            # Also normalize the perspective-corrected version (for comparison, but still excluded)
            persp_norm_dir = os.path.join(PERSPECTIVE_DIR, "normalized", species_dirname)
            ensure_dir(os.path.join(PERSPECTIVE_DIR, "normalized"))
            ensure_dir(persp_norm_dir)
            _normalize_single_image(perspective_corrected, species,
                                    os.path.join("perspective_corrected", "normalized", species_dirname),
                                    filename)

        # Update status
        _orient_landmarks[filename]["status"] = "applied"
        _orient_landmarks[filename]["saved_at"] = datetime.now().isoformat()
        _save_orient_landmarks()

        # Get the original filename to update review state
        orig_fname = filename.replace(".png", ".jpg")
        if not any(orig_fname in row.get("filename", "") for row in _inventory):
            # Try other extensions
            for ext in [".jpeg", ".png", ".gif"]:
                test_fname = filename.replace(".png", ext)
                if any(test_fname in row.get("filename", "") for row in _inventory):
                    orig_fname = test_fname
                    break

        return jsonify({
            "ok": True,
            "path": out_path,
            "normalized": norm_path is not None,
            "perspective_corrected": perspective_path is not None,
            "perspective_path": perspective_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/image/oriented/<species_dirname>/<filename>")
def serve_oriented_image(species_dirname, filename):
    """Serve oriented image (after rotation applied)."""
    path = os.path.join(ORIENTED_DIR, species_dirname, filename)
    if not os.path.exists(path):
        return "Not found", 404

    try:
        img = Image.open(path).convert("RGBA")

        # Resize for display
        w, h = img.size
        if h > 0:
            scale = THUMB_HEIGHT / h
            new_w = max(1, int(w * scale))
            img = img.resize((new_w, THUMB_HEIGHT), Image.LANCZOS)

        # Composite on gray background
        arr = np.array(img, dtype=np.float32)
        rgb = arr[:, :, :3]
        alpha = (arr[:, :, 3] / 255.0)[:, :, np.newaxis]
        bg = 180.0
        result = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)

        out = Image.fromarray(result)
        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return f"Error: {e}", 500


@app.route("/api/preview_rotation", methods=["POST"])
def api_preview_rotation():
    """Generate a preview of the rotated image without saving."""
    import base64
    data = request.get_json()
    filename = data.get("filename", "")
    species_dirname = data.get("species_dirname", "")
    mouth_x = data.get("mouth_x")
    mouth_y = data.get("mouth_y")
    tail_x = data.get("tail_x")
    tail_y = data.get("tail_y")
    angle_override = data.get("angle_override")  # Optional manual angle adjustment
    flip_h = data.get("flip_h", False)  # Horizontal flip
    flip_v = data.get("flip_v", False)  # Vertical flip
    dv_tilt = data.get("dv_tilt", 0)  # Dorsal-ventral perspective tilt
    ht_tilt = data.get("ht_tilt", 0)  # Head-tail perspective tilt
    offset_x = data.get("offset_x", 0)  # X offset for centering
    offset_y = data.get("offset_y", 0)  # Y offset for centering
    scale = data.get("scale", 100)  # Scale percentage
    auto_crop = data.get("auto_crop", False)  # Auto-crop margins

    if not filename or mouth_x is None:
        return jsonify({"error": "missing data"}), 400

    seg_path = os.path.join(SEGMENTED_DIR, species_dirname, filename)
    if not os.path.exists(seg_path):
        return jsonify({"error": "image not found"}), 404

    try:
        import math
        dx = tail_x - mouth_x
        dy = tail_y - mouth_y
        # atan2 gives angle from positive x-axis (right)
        # In image coords, y increases downward, so:
        # - Fish pointing right: angle ≈ 0
        # - Fish tilting down-right: positive angle
        # - Fish tilting up-right: negative angle
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Use manual angle override if provided
        if angle_override is not None:
            angle_deg = float(angle_override)

        print(f"[Preview] mouth=({mouth_x},{mouth_y}) tail=({tail_x},{tail_y}) dx={dx} dy={dy} angle={angle_deg:.1f}°")

        img = Image.open(seg_path).convert("RGBA")
        print(f"[Preview] Original image size: {img.size}")

        # PIL rotate() rotates counter-clockwise by the given angle
        # atan2 gives angle from horizontal: positive = tilting down toward tail, negative = tilting up
        # To make the line horizontal, we need to rotate BY that angle (not negative)
        # because PIL rotates counter-clockwise, and we want to "undo" the tilt
        rotated = img.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
        print(f"[Preview] After rotation by {angle_deg:.1f}°: {rotated.size}")

        # After rotation by atan2(dy, dx), the mouth-to-tail vector points along
        # the positive x-axis. This means tail is to the RIGHT of mouth.
        # For left lateral view: mouth on LEFT, tail on RIGHT - already correct!
        #
        # NO automatic flip needed - the rotation handles orientation correctly.
        # The manual Flip H/V buttons remain available for fine-tuning (e.g., upside down fish).
        print(f"[Preview] Rotation positions mouth on left, tail on right (left lateral view)")

        # Apply user-requested flips (for upside-down fish, etc.)
        if flip_h:
            rotated = rotated.transpose(Image.FLIP_LEFT_RIGHT)
            print(f"[Preview] Applied user horizontal flip")
        if flip_v:
            rotated = rotated.transpose(Image.FLIP_TOP_BOTTOM)
            print(f"[Preview] Applied user vertical flip")

        # Create a fixed-size canvas and place the fish on it with zoom and pan.
        # THEN apply perspective - this ensures warp is around canvas center
        # (which aligns with the red grid lines in the preview).

        # Fixed canvas size for preview
        canvas_size = 800

        # Apply zoom (scale the fish, not the canvas)
        fish_img = rotated
        if scale != 100:
            w, h = fish_img.size
            scale_factor = scale / 100.0
            new_w = max(1, int(w * scale_factor))
            new_h = max(1, int(h * scale_factor))
            fish_img = fish_img.resize((new_w, new_h), Image.LANCZOS)
            print(f"[Preview] Applied zoom: {scale}%")

        # Create a fixed-size RGBA canvas
        canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))

        # Calculate where to place the fish on the canvas
        # Center by default, then apply offset
        fish_w, fish_h = fish_img.size
        base_x = (canvas_size - fish_w) // 2
        base_y = (canvas_size - fish_h) // 2

        # Apply user offset (pan)
        paste_x = base_x + int(offset_x)
        paste_y = base_y + int(offset_y)

        print(f"[Preview] Canvas {canvas_size}x{canvas_size}, fish {fish_w}x{fish_h}, "
              f"paste at ({paste_x}, {paste_y})")

        # Paste the fish onto the canvas (handling out-of-bounds)
        fish_arr = np.array(fish_img)
        canvas_arr = np.array(canvas)

        # Calculate the valid regions for both source and destination
        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        src_x2 = min(fish_w, canvas_size - paste_x)
        src_y2 = min(fish_h, canvas_size - paste_y)

        dst_x1 = max(0, paste_x)
        dst_y1 = max(0, paste_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            fish_region = fish_arr[src_y1:src_y2, src_x1:src_x2]
            canvas_arr[dst_y1:dst_y2, dst_x1:dst_x2] = fish_region

        canvas_with_fish = Image.fromarray(canvas_arr)

        # NOW apply perspective correction to the entire canvas
        # This warps around the canvas center (400, 400) which matches the grid lines
        if dv_tilt != 0 or ht_tilt != 0:
            canvas_with_fish = _apply_perspective_correction(canvas_with_fish, dv_tilt, ht_tilt)
            print(f"[Preview] Applied perspective correction: DV={dv_tilt}°, HT={ht_tilt}°")

        rotated = canvas_with_fish

        # Composite on gray
        arr = np.array(rotated, dtype=np.float32)
        rgb = arr[:, :, :3]
        alpha = (arr[:, :, 3] / 255.0)[:, :, np.newaxis]
        bg = 180.0
        result = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)

        out = Image.fromarray(result)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        buf.seek(0)

        # Return as base64
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return jsonify({"ok": True, "image": f"data:image/png;base64,{b64}", "angle": angle_deg})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/image/exemplar_silhouette/<species_dirname>")
def serve_exemplar_silhouette(species_dirname):
    """Serve the exemplar image as a silhouette (black shape) for overlay reference."""
    species = dirname_to_species(species_dirname)

    # Check if we have an exemplar for this species
    if species not in _exemplars:
        return "No exemplar set", 404

    exemplar = _exemplars[species]
    png_name = exemplar.get("png_name", "")
    source = exemplar.get("source", "segmented")

    # Determine the image path based on source
    if source == "oriented":
        img_path = os.path.join(ORIENTED_DIR, species_dirname, png_name)
    else:
        img_path = os.path.join(SEGMENTED_DIR, species_dirname, png_name)

    if not os.path.exists(img_path):
        return "Exemplar image not found", 404

    try:
        img = Image.open(img_path).convert("RGBA")

        # Create silhouette: make all non-transparent pixels black
        arr = np.array(img)
        # Keep alpha channel, set RGB to 0 (black)
        arr[:, :, 0] = 0  # R
        arr[:, :, 1] = 0  # G
        arr[:, :, 2] = 0  # B
        # Alpha stays the same

        silhouette = Image.fromarray(arr)

        # Resize for display (similar to preview size)
        w, h = silhouette.size
        max_dim = 600
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            silhouette = silhouette.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        silhouette.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    except Exception as e:
        return f"Error: {e}", 500


# ══════════════════════════════════════════════════════════════
#  HTML Templates
# ══════════════════════════════════════════════════════════════

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fish Image Review</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f0f0f0; color: #222; padding: 20px; }
  h1 { margin-bottom: 8px; }
  .progress-bar { background: #ddd; border-radius: 8px; height: 24px; width: 400px;
                  margin: 12px 0 20px 0; overflow: hidden; position: relative; }
  .progress-fill { background: #4caf50; height: 100%; transition: width 0.3s; }
  .progress-fill-partial { background: #ff9800; height: 100%; position: absolute; top: 0; }
  .progress-text { position: absolute; top: 0; left: 0; right: 0; text-align: center;
                   line-height: 24px; font-size: 13px; font-weight: 600; color: #333; }
  table { border-collapse: collapse; width: 100%; background: white; border-radius: 8px;
          overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
  th { background: #333; color: white; text-align: left; padding: 10px 12px; font-size: 13px; }
  td { padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 14px; }
  tr:hover td { background: #f5f5f5; }
  tr.clickable { cursor: pointer; }
  a { color: #1a73e8; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .species-name { font-style: italic; font-weight: 500; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px;
           font-weight: 600; }
  .badge-reviewed { background: #c8e6c9; color: #2e7d32; }
  .badge-progress { background: #fff3e0; color: #e65100; }
  .badge-none { background: #eee; color: #666; }
  .filter-bar { margin-bottom: 16px; display: flex; gap: 20px; align-items: center; }
  .filter-bar label { font-size: 14px; cursor: pointer; }
  .jump-to { display: flex; gap: 8px; align-items: center; }
  .jump-to select { padding: 4px 8px; font-size: 14px; }
  .counts { color: #888; font-size: 13px; }
  .summary { margin: 12px 0; font-size: 14px; color: #555; }
  .export-dropdown { position: relative; display: inline-block; }
  .export-btn { padding: 5px 12px; font-size: 13px; cursor: pointer;
                background: #e8f5e9; border: 1px solid #81c784; border-radius: 4px; }
  .export-btn:hover { background: #c8e6c9; }
  .export-menu { display: none; position: absolute; top: 100%; left: 0;
                 background: white; border: 1px solid #ddd; border-radius: 4px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.15); z-index: 10; min-width: 180px; }
  .export-menu.open { display: block; }
  .export-menu a { display: block; padding: 8px 12px; color: #333;
                   text-decoration: none; font-size: 13px; }
  .export-menu a:hover { background: #f5f5f5; }
</style>
</head>
<body>
<h1>Chaetodontidae Image Review</h1>
<div class="summary">
  {{ n_total }} species &middot; {{ n_reviewed }} reviewed &middot; {{ n_in_progress }} in progress
</div>
<div class="progress-bar">
  <div class="progress-fill" style="width: {{ (n_reviewed / n_total * 100)|round(1) }}%"></div>
  <div class="progress-fill-partial" style="left: {{ (n_reviewed / n_total * 100)|round(1) }}%;
       width: {{ (n_in_progress / n_total * 100)|round(1) }}%"></div>
  <div class="progress-text">{{ n_reviewed }} / {{ n_total }} reviewed</div>
</div>

<div class="filter-bar">
  <label><input type="checkbox" id="filterUnreviewed"
    onchange="filterTable()"> Show unreviewed only</label>
  <div class="jump-to">
    <label>Jump to:</label>
    <select id="jumpSelect" onchange="jumpToSpecies()">
      <option value="">-- Select species --</option>
      {% for s in summaries %}
      <option value="{{ s.dirname }}">{{ s.species }}</option>
      {% endfor %}
    </select>
  </div>
  <div class="export-dropdown">
    <button class="export-btn" onclick="this.nextElementSibling.classList.toggle('open')">&#x1F4E5; Export CSV</button>
    <div class="export-menu">
      <a href="/api/export_metadata?scope=included">Export Included</a>
      <a href="/api/export_metadata?scope=all">Export All</a>
    </div>
  </div>
</div>

<table id="speciesTable">
<thead>
<tr>
  <th>#</th>
  <th>Species</th>
  <th>Seg</th>
  <th>Norm</th>
  <th>k</th>
  <th>Excl</th>
  <th>Fix</th>
  <th>Alt</th>
  <th>Exemp</th>
  <th>Status</th>
</tr>
</thead>
<tbody>
{% for s in summaries %}
<tr data-status="{{ s.status }}" class="clickable" onclick="window.location='/review/{{ s.dirname }}'">
  <td>{{ loop.index }}</td>
  <td><span class="species-name">{{ s.species }}</span></td>
  <td>{{ s.n_seg }}</td>
  <td>{{ s.n_norm }}</td>
  <td>{{ s.gestalt_k or '—' }}</td>
  <td>{{ s.n_excluded if s.n_excluded else '' }}</td>
  <td>{{ s.n_fix if s.n_fix else '' }}</td>
  <td>{{ s.n_alt_morph if s.n_alt_morph else '' }}</td>
  <td>{% if s.has_exemplar %}<span style="color: #4caf50;">&#9733;</span>{% endif %}</td>
  <td>
    {% if s.status == 'reviewed' %}
      <span class="badge badge-reviewed">Done</span>
    {% elif s.status == 'in_progress' %}
      <span class="badge badge-progress">WIP</span>
    {% else %}
      <span class="badge badge-none">—</span>
    {% endif %}
  </td>
</tr>
{% endfor %}
</tbody>
</table>

<script>
function filterTable() {
  const hide = document.getElementById('filterUnreviewed').checked;
  document.querySelectorAll('#speciesTable tbody tr').forEach(tr => {
    if (hide && tr.dataset.status === 'reviewed') {
      tr.style.display = 'none';
    } else {
      tr.style.display = '';
    }
  });
}

function jumpToSpecies() {
  const sel = document.getElementById('jumpSelect');
  if (sel.value) {
    window.location = '/review/' + sel.value;
  }
}

document.addEventListener('click', function(e) {
  if (!e.target.closest('.export-dropdown')) {
    document.querySelectorAll('.export-menu').forEach(m => m.classList.remove('open'));
  }
});
</script>
</body>
</html>"""


REVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Review: {{ species }}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #e8e8e8; color: #222; }
  .header { background: #333; color: white; padding: 12px 20px; display: flex;
            align-items: center; justify-content: space-between; position: sticky;
            top: 0; z-index: 100; }
  .header a { color: #adf; text-decoration: none; font-size: 14px; }
  .header a:hover { text-decoration: underline; }
  .header .title { font-size: 20px; font-style: italic; font-weight: 600; }
  .header .pos { font-size: 13px; color: #bbb; margin-left: 12px; }
  .nav-links { display: flex; gap: 20px; align-items: center; }
  .nav-btn { background: #555; color: white; padding: 6px 14px; border-radius: 4px;
             text-decoration: none; font-size: 13px; }
  .nav-btn:hover { background: #666; text-decoration: none; }
  .nav-btn.disabled { background: #444; color: #777; pointer-events: none; }

  .controls { background: white; padding: 16px 20px; border-bottom: 1px solid #ccc;
              display: flex; gap: 24px; align-items: center; flex-wrap: wrap; }
  .controls label { font-size: 14px; font-weight: 500; }
  .controls input[type="number"] { width: 60px; padding: 4px 8px; font-size: 16px;
    border: 2px solid #ccc; border-radius: 4px; text-align: center; }
  .controls input[type="text"] { width: 300px; padding: 4px 8px; font-size: 14px;
    border: 1px solid #ccc; border-radius: 4px; }
  .controls .status-badge { margin-left: auto; }

  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
          gap: 16px; padding: 20px; }
  .card { background: white; border-radius: 8px; overflow: hidden;
          box-shadow: 0 1px 4px rgba(0,0,0,0.12); transition: box-shadow 0.2s; }
  .card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
  .card.excluded { opacity: 0.5; }
  .card.fix-orient { border: 3px solid #ff9800; }
  .excluded-divider { grid-column: 1 / -1; text-align: center; padding: 15px 0;
                      border-top: 2px dashed #999; margin-top: 10px; }
  .excluded-divider span { background: #f5f5f5; padding: 5px 20px; color: #666;
                           font-weight: 500; font-size: 14px; border-radius: 4px; }
  .card.has-oriented { border: 3px solid #2196f3; background: #e3f2fd; }
  .card.alt-morph { border: 3px solid #9c27b0; }
  .card.resegment { border: 3px solid #f44336; }
  .card.color-correct { border: 3px solid #00bcd4; }
  .card .imgs { background: #b4b4b4; padding: 4px; text-align: center; }
  .card .imgs img { display: block; margin: 0 auto 4px auto; max-width: 100%;
                    height: auto; border-radius: 2px; }
  .card .imgs .label { font-size: 10px; color: #666; margin-bottom: 2px; }
  .card .imgs .oriented-label { color: #2e7d32; font-weight: 600; }
  .card .info { padding: 8px 10px; }
  .card .fname { font-size: 12px; font-weight: 500; word-break: break-all; }
  .card .source { font-size: 11px; margin-top: 2px; }
  .source-badge { display: inline-block; padding: 1px 6px; border-radius: 8px;
                  font-size: 10px; font-weight: 600; }
  .source-Bishop { background: #c8e6c9; color: #2e7d32; }
  .source-FishBase { background: #bbdefb; color: #1565c0; }
  .source-FishPix { background: #d1c4e9; color: #4527a0; }
  .source-FishWise { background: #ffe0b2; color: #e65100; }
  .source-FBUser { background: #f0f4c3; color: #827717; }
  .source-iNat { background: #fff9c4; color: #f57f17; }
  .photographer-badge { display: inline-block; padding: 1px 5px; border-radius: 8px;
                         font-size: 9px; background: #f3e5f5; color: #6a1b9a; margin-left: 2px; }
  .type-badge { display: inline-block; padding: 1px 5px; border-radius: 8px; font-size: 9px; margin-left: 2px; }
  .type-underwater { background: #e0f7fa; color: #006064; }
  .type-specimen_photo { background: #efebe9; color: #4e342e; }
  .type-illustration { background: #fce4ec; color: #880e4f; }
  .type-photo { background: #f5f5f5; color: #616161; }
  .quality-badges { font-size: 9px; margin-top: 2px; }
  .quality-ok { color: #2e7d32; margin-right: 4px; }
  .quality-warn { color: #e65100; margin-right: 4px; }
  .uw-badges { font-size: 9px; margin-top: 2px; }
  .uw-badge { display: inline-block; padding: 1px 5px; border-radius: 8px;
              font-size: 9px; font-weight: 600; margin-right: 3px; }
  .uw-whole { background: #e3f2fd; color: #1565c0; }
  .uw-body { background: #e0f7fa; color: #00838f; }
  .depth-badges { font-size: 9px; margin-top: 2px; }
  .depth-badge { display: inline-block; padding: 1px 5px; border-radius: 8px;
                 font-size: 9px; font-weight: 600; background: #e0f2f1; color: #00695c; }
  .depth-badge.depth-warn { background: #fff3e0; color: #e65100; }

  /* Show Source button */
  .show-source-btn { background: none; border: 1px dashed #999; color: #666;
    padding: 1px 6px; border-radius: 3px; font-size: 9px; cursor: pointer;
    margin-top: 3px; }
  .show-source-btn:hover { background: #fff3e0; border-color: #ff9800; color: #e65100; }

  /* Original source image popup overlay */
  .source-popup-overlay { display: none; position: fixed; top: 0; left: 0;
    width: 100%; height: 100%; background: rgba(0,0,0,0.6); z-index: 2000;
    justify-content: center; align-items: center; }
  .source-popup-overlay.visible { display: flex; }
  .source-popup { position: relative; max-width: 80vw; max-height: 85vh;
    background: #222; border-radius: 8px; overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
  .source-popup img { display: block; max-width: 80vw; max-height: 80vh;
    object-fit: contain; }
  .source-popup .popup-header { display: flex; justify-content: space-between;
    align-items: center; padding: 6px 12px; background: #333; }
  .source-popup .popup-title { color: #ff9800; font-size: 12px; font-weight: 600; }
  .source-popup .popup-close { background: none; border: none; color: #aaa;
    font-size: 18px; cursor: pointer; padding: 0 4px; }
  .source-popup .popup-close:hover { color: white; }
  .cross-db-equiv { font-size: 9px; color: #666; margin-top: 2px; }
  .equiv-label { font-weight: 500; color: #333; font-size: 9px; }
  .source-Other { background: #eee; color: #666; }
  .randall-badge { display: inline-block; padding: 1px 6px; border-radius: 8px;
                   font-size: 10px; font-weight: 600; background: #fff3e0; color: #e65100;
                   border: 1px solid #ffb74d; margin-left: 4px; }
  .miyazawa-badge { display: inline-block; padding: 1px 6px; border-radius: 8px;
                    font-size: 10px; font-weight: 600; background: #e3f2fd; color: #1565c0;
                    border: 1px solid #64b5f6; margin-left: 4px; }
  .fishbase-equiv { font-size: 9px; color: #666; margin-top: 2px; font-style: italic; }
  .fishbase-equiv-label { color: #1565c0; font-weight: 500; }
  .warn-badge { display: inline-block; padding: 1px 6px; border-radius: 8px;
                font-size: 10px; background: #ffcdd2; color: #c62828; margin-top: 2px; }
  .ann-badge { display: inline-block; padding: 1px 6px; border-radius: 8px;
               font-size: 10px; background: #ffe0b2; color: #e65100; margin-top: 2px; }
  .card .actions { padding: 6px 10px 10px; display: flex; gap: 8px; }
  .action-col { display: flex; flex-direction: column; gap: 3px; flex: 1; }
  .action-col-header { font-size: 10px; font-weight: 600; color: #666;
                       text-transform: uppercase; margin-bottom: 2px;
                       padding-bottom: 2px; border-bottom: 1px solid #ddd; }
  .action-col.exclude-col .action-col-header { color: #c62828; }
  .action-col.include-col .action-col-header { color: #2e7d32; }
  .card .actions label { font-size: 11px; cursor: pointer; display: flex;
                         align-items: center; gap: 5px; position: relative; }
  .card .actions input[type="checkbox"] { width: 14px; height: 14px; cursor: pointer; }
  .card .actions input[type="radio"] { width: 14px; height: 14px; cursor: pointer; }
  .exemplar-label { color: #4caf50; font-weight: 600; }

  /* Tooltips */
  .has-tooltip { position: relative; }
  .has-tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    left: 100%;
    top: 50%;
    transform: translateY(-50%);
    margin-left: 8px;
    background: #333;
    color: white;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: normal;
    white-space: nowrap;
    z-index: 1000;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.2s, visibility 0.2s;
    pointer-events: none;
    max-width: 250px;
    white-space: normal;
  }
  .has-tooltip:hover::after { opacity: 1; visibility: visible; }

  /* Filter section */
  .filter-section { margin-top: 4px; padding-top: 4px; border-top: 1px dashed #c8e6c9; }
  .filter-header { font-size: 9px; color: #666; text-transform: uppercase; margin-bottom: 3px; }
  .filter-btn { background: #f1f8e9; border: 1px solid #aed581; color: #558b2f;
                padding: 4px 8px; border-radius: 4px; font-size: 10px; cursor: pointer;
                display: flex; align-items: center; gap: 4px; width: 100%;
                position: relative; transition: all 0.2s; }
  .filter-btn:hover { background: #dcedc8; border-color: #8bc34a; }
  .filter-btn.active { background: #7cb342; color: white; border-color: #689f38; }
  .filter-btn .filter-icon { font-size: 12px; }
  .filter-btn .filter-tooltip {
    position: absolute;
    left: 100%;
    top: 50%;
    transform: translateY(-50%);
    margin-left: 8px;
    background: #333;
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 11px;
    white-space: normal;
    width: 200px;
    z-index: 1000;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.2s;
    pointer-events: none;
  }
  .filter-btn:hover .filter-tooltip { opacity: 1; visibility: visible; }
  /* Sea-thru inline buttons */
  .seathru-buttons { display: flex; flex-wrap: wrap; gap: 2px; margin-top: 3px; }
  .seathru-btn-inline { background: #e3f2fd; border: 1px solid #90caf9; color: #1565c0;
                        padding: 2px 6px; border-radius: 3px; font-size: 9px; cursor: pointer;
                        white-space: nowrap; transition: all 0.15s; }
  .seathru-btn-inline:hover { background: #bbdefb; border-color: #64b5f6; }
  .seathru-btn-inline.active { background: #1976d2; color: white; border-color: #1565c0; }
  .seathru-btn-inline.recommended { box-shadow: 0 0 0 1px #42a5f5; }
  .seathru-btn-inline.depth-btn { background: #e0f2f1; border-color: #80cbc4; color: #00695c; }
  .seathru-btn-inline.depth-btn:hover { background: #b2dfdb; border-color: #4db6ac; }
  .seathru-btn-inline.depth-btn.active { background: #00897b; color: white; border-color: #00695c; }

  .exemplar-badge { background: #4caf50; color: white; padding: 2px 6px; border-radius: 4px;
                    font-size: 10px; font-weight: bold; display: inline-block; margin-left: 4px; }
  .card.is-exemplar { border: 3px solid #4caf50; }
  .card.is-exemplar .info { background: #e8f5e9; }
  .not-norm { font-size: 10px; color: #999; font-style: italic; text-align: center;
              padding: 4px; }

  /* View toggle buttons */
  .view-toggle-group { display: flex; gap: 4px; align-items: center; }
  .view-toggle-group label { font-size: 14px; font-weight: 500; margin-right: 8px; }
  .view-btn { padding: 5px 12px; font-size: 12px; border: 1px solid #ccc;
              background: #f5f5f5; border-radius: 4px; cursor: pointer;
              transition: all 0.15s; }
  .view-btn:hover { background: #e0e0e0; }
  .view-btn.active { background: #1a73e8; color: white; border-color: #1a73e8; }

  /* Image display modes */
  .card .imgs .img-panel { display: block; }
  .card .imgs .img-panel.hidden { display: none; }
  body.view-original .card .imgs .img-processed { display: none; }
  body.view-processed .card .imgs .img-original { display: none; }
  body.view-both .card .imgs .img-panel { display: block; }

  .footer { background: white; padding: 16px 20px; border-top: 1px solid #ccc;
            display: flex; gap: 16px; align-items: center; position: sticky;
            bottom: 0; z-index: 100; flex-wrap: wrap; }
  .footer .stats { font-size: 13px; color: #555; margin-right: auto; }
  .btn { padding: 8px 20px; border: none; border-radius: 6px; font-size: 14px;
         font-weight: 600; cursor: pointer; transition: background 0.2s; }
  .btn-primary { background: #4caf50; color: white; }
  .btn-primary:hover { background: #43a047; }
  .btn-secondary { background: #eee; color: #333; }
  .btn-secondary:hover { background: #ddd; }
  .btn-next { background: #1a73e8; color: white; text-decoration: none; display: inline-block; }
  .btn-next:hover { background: #1565c0; }
  .btn-prev { background: #666; color: white; text-decoration: none; display: inline-block; }
  .btn-prev:hover { background: #555; }

  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px;
           font-weight: 600; }
  .badge-reviewed { background: #c8e6c9; color: #2e7d32; }
  .badge-progress { background: #fff3e0; color: #e65100; }
  .badge-none { background: #555; color: #ccc; }

  /* Keyboard hint */
  .kbd-hint { font-size: 11px; color: #888; margin-top: 8px; }
  kbd { background: #eee; padding: 2px 6px; border-radius: 3px; font-size: 11px;
        border: 1px solid #ccc; }
</style>
</head>
<body>

<div class="header">
  <div class="nav-links">
    {% if prev_sp %}
      <a href="/review/{{ prev_sp }}" class="nav-btn">&larr; Prev</a>
    {% else %}
      <span class="nav-btn disabled">&larr; Prev</span>
    {% endif %}
    <a href="/">Index</a>
  </div>
  <div>
    <span class="title">{{ species }}</span>
    <span class="pos">({{ idx + 1 }} of {{ n_total }})</span>
  </div>
  <div class="nav-links">
    {% if next_sp %}
      <a href="/review/{{ next_sp }}" class="nav-btn">Next &rarr;</a>
    {% else %}
      <span class="nav-btn disabled">Next &rarr;</span>
    {% endif %}
  </div>
</div>

<div class="controls">
  <label>Gestalt k:
    <input type="number" id="gestaltK" value="{{ gestalt_k }}" min="1" max="20"
           onchange="saveGestaltK()">
  </label>
  <label>Notes:
    <input type="text" id="notes" value="{{ notes }}" placeholder="Optional notes..."
           onchange="saveGestaltK()">
  </label>
  <div class="view-toggle-group">
    <label>View:</label>
    <button class="view-btn" data-view="original" onclick="setViewMode('original')">Original</button>
    <button class="view-btn active" data-view="both" onclick="setViewMode('both')">Both</button>
    <button class="view-btn" data-view="processed" onclick="setViewMode('processed')">Processed</button>
  </div>
  <div class="status-badge">
    {% if status == 'reviewed' %}
      <span class="badge badge-reviewed">Reviewed</span>
    {% elif status == 'in_progress' %}
      <span class="badge badge-progress">In Progress</span>
    {% else %}
      <span class="badge badge-none">Not reviewed</span>
    {% endif %}
  </div>
</div>

<div class="grid">
{% for im in images %}
  {% if im.is_first_excluded %}
  <div class="excluded-divider">
    <span>Excluded Images</span>
  </div>
  {% endif %}
  <div class="card {{ 'excluded' if im.action == 'exclude' else '' }}
              {{ 'fix-orient' if im.action == 'fix_orientation' else '' }}
              {{ 'alt-morph' if im.action == 'alt_morph' else '' }}
              {{ 'resegment' if im.action == 'resegment' else '' }}
              {{ 'color-correct' if im.action == 'color_correct' else '' }}
              {{ 'has-oriented' if im.has_oriented else '' }}
              {{ 'is-exemplar' if im.is_exemplar else '' }}"
       id="card-{{ loop.index0 }}"
       data-source="{{ im.source }}"
       data-fname="{{ im.orig_fname }}">
    <div class="imgs">
      {% if im.has_segmented %}
        <div class="img-panel img-original">
          <div class="label">Original</div>
          <img src="/image/segmented/{{ species_dirname }}/{{ im.png_name }}"
               alt="original" loading="lazy">
        </div>
      {% endif %}
      {% if im.has_oriented %}
        <div class="img-panel img-processed">
          <div class="label oriented-label">Processed &#10003;</div>
          <img src="/image/oriented/{{ species_dirname }}/{{ im.png_name }}"
               alt="processed" loading="lazy"
               data-has-randall="{{ 'true' if im.filters.get('randall') else 'false' }}"
               data-has-seathru="{{ 'true' if im.filters.get('seathru') else 'false' }}"
               data-seathru-method="{{ im.filters.get('seathru_method', 'gray_world') }}">
        </div>
      {% elif im.has_normalized %}
        <div class="img-panel img-processed">
          <div class="label">Processed</div>
          <img src="/image/normalized/{{ species_dirname }}/{{ im.png_name }}"
               alt="processed" loading="lazy"
               data-has-randall="{{ 'true' if im.filters.get('randall') else 'false' }}"
               data-has-seathru="{{ 'true' if im.filters.get('seathru') else 'false' }}"
               data-seathru-method="{{ im.filters.get('seathru_method', 'gray_world') }}">
        </div>
      {% endif %}
      {% if not im.has_segmented and not im.has_normalized %}
        <div class="not-norm">Not processed (iNat excluded)</div>
      {% endif %}
      <button class="show-source-btn" onclick="showSourcePopup('{{ im.orig_fname }}', '{{ im.display_name }}')">Show source image</button>
    </div>
    <div class="info">
      <div class="fname">{{ im.display_name }}</div>
      <div class="source">
        <span class="source-badge source-{{ im.source }}">{{ im.source }}</span>
        {% if im.is_randall %}
          <span class="randall-badge">Randall</span>
        {% endif %}
        {% if im.is_miyazawa %}
          <span class="miyazawa-badge">Miyazawa 2020</span>
        {% endif %}
        {% if im.photographer and im.photographer != 'Jack Randall' %}
          <span class="photographer-badge">{{ im.photographer }}</span>
        {% endif %}
        {% if im.image_type %}
          <span class="type-badge type-{{ im.image_type }}">{{ im.image_type | replace('_', ' ') }}</span>
        {% endif %}
        {% if im.orient_warning %}
          <span class="warn-badge">Orient?</span>
        {% endif %}
        {% if im.ann_badge %}
          <span class="ann-badge">{{ im.ann_badge }}</span>
        {% endif %}
      </div>
      {% if im.is_lateral or im.is_single_fish or im.is_grayscale %}
        <div class="quality-badges">
          {% if im.is_lateral %}<span class="quality-ok">Lateral &#10003;</span>{% endif %}
          {% if im.is_single_fish %}<span class="quality-ok">Single &#10003;</span>{% endif %}
          {% if im.is_grayscale %}<span class="quality-warn">B&amp;W</span>{% endif %}
        </div>
      {% endif %}
      {% if im.uw_whole or im.uw_body %}
        <div class="uw-badges">
          {% if im.uw_whole %}<span class="uw-badge uw-whole">UW image</span>{% endif %}
          {% if im.uw_body %}<span class="uw-badge uw-body">UW body</span>{% endif %}
        </div>
      {% endif %}
      {% if im.has_depth %}
        <div class="depth-badges">
          {% if im.depth_is_specimen %}
          <span class="depth-badge depth-warn has-tooltip" data-tooltip="Specimen photo - depth estimate is camera distance, not underwater depth. Sea-thru correction not recommended.">Not underwater</span>
          {% else %}
          <span class="depth-badge has-tooltip" data-tooltip="Est. depth: {{ im.depth_mean }}m | Range: {{ im.depth_range }}m{% if im.body_depth_mean %} | Body: {{ im.body_depth_mean }}m{% endif %}">&#x1F4CF; {{ im.depth_mean }}m</span>
          {% endif %}
        </div>
      {% endif %}
      {% if im.cross_db_equivalents %}
        <div class="cross-db-equiv">
          <span class="equiv-label">Also in:</span>
          {% for eq in im.cross_db_equivalents %}
            <span class="source-badge source-{{ eq.source }}">{{ eq.source }}</span>
          {% endfor %}
        </div>
      {% elif im.fishbase_equivalents %}
        <div class="fishbase-equiv">
          <span class="fishbase-equiv-label">FishBase:</span>
          {{ im.fishbase_equivalents | join(', ') }}
        </div>
      {% endif %}
    </div>
    <div class="actions">
      <div class="action-col exclude-col">
        <div class="action-col-header">Exclude</div>
        <label class="has-tooltip" data-tooltip="Remove this image from the analysis entirely">
          <input type="checkbox" data-fname="{{ im.orig_fname }}" data-action="exclude"
                 {{ 'checked' if im.action == 'exclude' else '' }}
                 onchange="toggleAction(this)">
          Exclude
        </label>
        <label class="has-tooltip" data-tooltip="Exclude: different color morph (juvenile, male/female, regional variant)">
          <input type="checkbox" data-fname="{{ im.orig_fname }}" data-action="alt_morph"
                 {{ 'checked' if im.action == 'alt_morph' else '' }}
                 onchange="toggleAction(this)">
          Alt Morph
        </label>
        <label class="has-tooltip" data-tooltip="Exclude: segmentation mask needs to be regenerated">
          <input type="checkbox" data-fname="{{ im.orig_fname }}" data-action="resegment"
                 data-png="{{ im.png_name }}"
                 {{ 'checked' if im.action == 'resegment' else '' }}
                 onchange="toggleAction(this)">
          Resegment
        </label>
        <label class="has-tooltip" data-tooltip="Exclude: color correction needed (white balance, exposure)">
          <input type="checkbox" data-fname="{{ im.orig_fname }}" data-action="color_correct"
                 {{ 'checked' if im.action == 'color_correct' else '' }}
                 onchange="toggleAction(this)">
          Color Fix
        </label>
      </div>
      <div class="action-col include-col">
        <div class="action-col-header">Include</div>
        <label class="exemplar-label has-tooltip" data-tooltip="Set as the reference image for this species">
          <input type="radio" name="exemplar" data-fname="{{ im.orig_fname }}"
                 data-png="{{ im.png_name }}" data-source="{{ im.source }}"
                 {{ 'checked' if im.is_exemplar else '' }}
                 onchange="setExemplar(this)">
          ★ Exemplar
        </label>
        <label class="has-tooltip" data-tooltip="Manually adjust orientation (rotate, flip) to face left">
          <input type="checkbox" data-fname="{{ im.orig_fname }}" data-action="fix_orientation"
                 data-png="{{ im.png_name }}"
                 {{ 'checked' if im.action == 'fix_orientation' else '' }}
                 onchange="handleFixOrientation(this)">
          Reorient
        </label>
        <div class="filter-section">
          <div class="filter-header has-tooltip" data-tooltip="Sea-thru color correction for underwater images">Sea-thru</div>
          <div class="seathru-buttons">
            {% if im.has_depth and not im.depth_is_specimen %}
            <button class="seathru-btn-inline depth-btn {{ 'active' if im.filters.get('seathru') and im.filters.get('seathru_method') == 'depth_seathru' else '' }} {{ 'recommended' if (im.uw_whole or im.uw_body) else '' }}"
                    data-fname="{{ im.orig_fname }}" data-png="{{ im.png_name }}"
                    data-method="depth_seathru"
                    title="Depth Sea-thru: Full physics model using estimated depth map. Corrects backscatter and attenuation per-pixel. Best for true underwater photos."
                    onclick="clickSeathruBtn(this)">Depth</button>
            {% endif %}
            <button class="seathru-btn-inline {{ 'active' if im.filters.get('seathru') and im.filters.get('seathru_method') == 'gray_world' else '' }} {{ 'recommended' if (im.uw_whole or im.uw_body) and (not im.has_depth or im.depth_is_specimen) else '' }}"
                    data-fname="{{ im.orig_fname }}" data-png="{{ im.png_name }}"
                    data-method="gray_world"
                    title="Gray World: Assumes the average scene color should be neutral gray. Good general-purpose white balance for mild-moderate color casts."
                    onclick="clickSeathruBtn(this)">GrayW</button>
            <button class="seathru-btn-inline {{ 'active' if im.filters.get('seathru') and im.filters.get('seathru_method') == 'gray_world_10p' else '' }}"
                    data-fname="{{ im.orig_fname }}" data-png="{{ im.png_name }}"
                    data-method="gray_world_10p"
                    title="Gray World 10%: Uses only the brightest 10% of pixels for estimation. Better when the fish fills most of the frame and the scene is unevenly lit."
                    onclick="clickSeathruBtn(this)">GW10%</button>
            <button class="seathru-btn-inline {{ 'active' if im.filters.get('seathru') and im.filters.get('seathru_method') == 'red_boost' else '' }}"
                    data-fname="{{ im.orig_fname }}" data-png="{{ im.png_name }}"
                    data-method="red_boost"
                    title="Red Boost: Boosts the red channel toward the green channel mean. Specifically targets the red absorption typical of deeper underwater photos."
                    onclick="clickSeathruBtn(this)">RedB</button>
            <button class="seathru-btn-inline {{ 'active' if im.filters.get('seathru') and im.filters.get('seathru_method') == 'max_channel' else '' }}"
                    data-fname="{{ im.orig_fname }}" data-png="{{ im.png_name }}"
                    data-method="max_channel"
                    title="Max Channel: Normalizes each channel by its 90th percentile (original Sea-thru approach). Most aggressive correction, can oversaturate."
                    onclick="clickSeathruBtn(this)">MaxCh</button>
          </div>
        </div>
      </div>
    </div>
  </div>
{% endfor %}
</div>

<!-- Source image popup (shared, one for all cards) -->
<div class="source-popup-overlay" id="sourcePopup" onclick="closeSourcePopup(event)">
  <div class="source-popup">
    <div class="popup-header">
      <span class="popup-title" id="sourcePopupTitle"></span>
      <button class="popup-close" onclick="closeSourcePopup()">&times;</button>
    </div>
    <img id="sourcePopupImg" alt="source">
  </div>
</div>

<div class="footer">
  <div class="stats">
    {{ images|length }} images &middot;
    <span id="nExcluded">{{ n_excluded }}</span> excl &middot;
    <span id="nFix">{{ n_fix }}</span> fix &middot;
    <span id="nAltMorph">{{ n_alt_morph }}</span> alt &middot;
    <span id="nResegment">{{ n_resegment }}</span> reseg &middot;
    <span id="nColorCorrect">{{ n_color_correct }}</span> color &middot;
    <span id="exemplarStatus">{{ '&#9733; Exemplar set' if has_exemplar else 'No exemplar' | safe }}</span>
  </div>
  {% if prev_sp %}
    <a href="/review/{{ prev_sp }}" class="btn btn-prev">&larr; Previous</a>
  {% endif %}
  <button class="btn btn-primary" onclick="markReviewed()">Mark Reviewed</button>
  {% if next_sp %}
    <a href="/review/{{ next_sp }}" class="btn btn-next">Next &rarr;</a>
  {% endif %}
  <div class="kbd-hint">
    <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> navigate &nbsp;
    <kbd>Enter</kbd> mark reviewed + next
  </div>
</div>

<!-- First Visit Welcome Modal -->
<div id="firstVisitModal" class="modal" style="display:none;">
  <div class="modal-content first-visit-modal">
    <div class="modal-header first-visit-header">
      <h3>👋 Welcome to <em>{{ species }}</em></h3>
      <button class="modal-close" onclick="closeFirstVisitModal()">&times;</button>
    </div>
    <div class="modal-body">
      <p class="first-visit-intro">
        This is your first time reviewing this species. Let's start by setting the
        <strong>gestalt k</strong> value and selecting an <strong>exemplar image</strong>.
      </p>

      <div class="first-visit-section">
        <h4>1. Set Gestalt K</h4>
        <p class="hint">How many distinct color pattern groups are there?</p>
        <p class="hint" style="color: #e65100; font-size: 12px;">Note: Current analysis uses k=3-4 for valid Jt metric (k=2 creates artifact)</p>
        <div class="k-selector">
          {% for k in range(2, 11) %}
          <button class="k-btn {{ 'active' if gestalt_k|int == k else '' }}"
                  onclick="selectK({{ k }})">{{ k }}</button>
          {% endfor %}
        </div>
        <p class="k-selected">Selected: <strong id="selectedK">{{ gestalt_k }}</strong></p>
      </div>

      <div class="first-visit-section">
        <h4>2. Select Exemplar Image</h4>
        <p class="hint">Click on a fish image below to set it as the exemplar for this species.
           You can scroll through and browse before deciding.</p>
        <div class="exemplar-gallery" id="exemplarGallery">
          {% for im in images %}
          {% if im.has_segmented or im.has_oriented %}
          <div class="exemplar-thumb {{ 'selected' if im.is_exemplar else '' }}"
               data-fname="{{ im.orig_fname }}"
               data-png="{{ im.png_name }}"
               data-source="{{ 'oriented' if im.has_oriented else 'segmented' }}"
               onclick="selectExemplarFromModal(this)">
            <img src="/image/{{ 'oriented' if im.has_oriented else 'segmented' }}/{{ species_dirname }}/{{ im.png_name }}"
                 alt="{{ im.display_name }}" loading="lazy">
            {% if im.is_exemplar %}
            <span class="exemplar-check">★</span>
            {% endif %}
          </div>
          {% endif %}
          {% endfor %}
        </div>
        <p class="exemplar-selected" id="exemplarSelected">
          {{ '★ Exemplar selected' if has_exemplar else 'No exemplar selected yet' }}
        </p>
      </div>
    </div>
    <div class="modal-footer first-visit-footer">
      <button class="btn btn-secondary" onclick="closeFirstVisitModal()">Skip for now</button>
      <button class="btn btn-primary" onclick="confirmFirstVisit()">
        Continue to Review →
      </button>
    </div>
  </div>
</div>

<!-- Orientation Digitizer Modal -->
<div id="orientModal" class="modal" style="display:none;">
  <div class="modal-content">
    <div class="modal-header">
      <h3>Digitize Orientation: <span id="modalFilename"></span></h3>
      <button class="modal-close" onclick="closeOrientModal()">&times;</button>
    </div>
    <div class="modal-body">
      <p class="modal-instructions">
        Click <strong>1) MOUTH</strong> first, then <strong>2) BASE OF TAIL</strong>.
        Fish will be rotated to face left.
      </p>
      <div class="modal-image-container">
        <img id="modalImage" src="" alt="Fish image" onclick="handleImageClick(event)">
        <div id="mouthMarker" class="marker marker-mouth" style="display:none;">M</div>
        <div id="tailMarker" class="marker marker-tail" style="display:none;">T</div>
      </div>
      <div class="modal-coords">
        <span>Mouth: (<span id="mouthCoords">—</span>)</span>
        <span>Tail: (<span id="tailCoords">—</span>)</span>
        <span>Angle: <span id="angleDisplay">—</span>°</span>
      </div>
      <div id="previewContainer" style="display:none;">
        <h4>Preview:
          <label style="font-weight:normal; margin-left:20px;">
            <input type="checkbox" id="gridToggle" onchange="toggleGrid()" checked> Show grid
          </label>
        </h4>
        <div class="preview-wrapper" id="previewWrapper">
          <img id="exemplarSilhouette" class="exemplar-silhouette" src="" alt="" style="display:none;">
          <img id="previewImage" src="" alt="Preview">
          <svg id="previewGrid" class="preview-grid"></svg>
        </div>
        <div class="angle-adjust">
          <label>
            Fine-tune angle:
            <input type="range" id="angleSlider" min="-45" max="45" step="0.5" value="0"
                   oninput="adjustAngle(this.value)">
            <span id="angleAdjustValue">0</span>&deg;
          </label>
          <button class="btn btn-secondary btn-sm" onclick="resetAngleAdjust()">Reset to auto</button>
          <span class="flip-separator">|</span>
          <button class="btn btn-secondary btn-sm" id="flipHBtn" onclick="toggleFlipH()">Flip H</button>
          <button class="btn btn-secondary btn-sm" id="flipVBtn" onclick="toggleFlipV()">Flip V</button>
        </div>
        <div class="perspective-adjust">
          <div class="perspective-row">
            <label>
              Dorsal/Ventral tilt (rotate around long axis):
              <input type="range" id="dvTiltSlider" min="-30" max="30" step="1" value="0"
                     oninput="adjustDVTilt(this.value)">
              <span id="dvTiltValue">0</span>&deg;
            </label>
            <span class="tilt-hint">(+ = dorsal toward camera)</span>
          </div>
          <div class="perspective-row">
            <label>
              Head/Tail tilt (rotate around vertical axis):
              <input type="range" id="htTiltSlider" min="-30" max="30" step="1" value="0"
                     oninput="adjustHTTilt(this.value)">
              <span id="htTiltValue">0</span>&deg;
            </label>
            <span class="tilt-hint">(+ = head toward camera)</span>
          </div>
          <button class="btn btn-secondary btn-sm" onclick="resetPerspective()">Reset Tilt</button>
          <span class="perspective-warning" id="perspectiveWarning" style="display:none;">
            ⚠️ Perspective-corrected images are EXCLUDED from Pavo by default. Consider excluding images needing &gt;15° correction.
          </span>
        </div>
        <div class="offset-adjust">
          <div class="offset-row">
            <label>
              X offset:
              <input type="range" id="offsetXSlider" min="-200" max="200" step="5" value="0"
                     oninput="adjustOffsetX(this.value)">
              <span id="offsetXValue">0</span>px
            </label>
            <label>
              Y offset:
              <input type="range" id="offsetYSlider" min="-200" max="200" step="5" value="0"
                     oninput="adjustOffsetY(this.value)">
              <span id="offsetYValue">0</span>px
            </label>
            <label>
              <strong>Zoom:</strong>
              <input type="range" id="scaleSlider" min="25" max="200" step="5" value="100"
                     oninput="adjustScale(this.value)">
              <span id="scaleValue">100</span>%
            </label>
          </div>
          <div class="offset-row">
            <label>
              <input type="checkbox" id="exemplarToggle" onchange="toggleExemplar()"> Show exemplar silhouette
            </label>
            <button class="btn btn-secondary btn-sm" onclick="resetOffset()">Reset Position/Zoom</button>
            <button class="btn btn-warning btn-sm" onclick="resetAllAdjustments()">⟲ Reset All</button>
          </div>
          <div class="drag-hint">💡 Click and drag on the preview to pan the fish. Use Zoom to resize.</div>
        </div>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn btn-secondary" onclick="resetDigitize()">Reset Points</button>
      <button class="btn btn-secondary" id="previewBtn" onclick="previewRotation()" disabled>Preview</button>
      <button class="btn btn-primary" id="applyBtn" onclick="applyRotation()" disabled>Apply & Save</button>
      <button class="btn btn-secondary" onclick="closeOrientModal()">Cancel</button>
    </div>
  </div>
</div>

<style>
  .modal { position: fixed; top: 0; left: 0; width: 100%; height: 100%;
           background: rgba(0,0,0,0.7); z-index: 1000; display: flex;
           align-items: center; justify-content: center; }
  .modal-content { background: white; border-radius: 12px; max-width: 90vw;
                   max-height: 90vh; overflow: auto; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .modal-header { display: flex; justify-content: space-between; align-items: center;
                  padding: 16px 20px; border-bottom: 1px solid #ddd; background: #f5f5f5;
                  border-radius: 12px 12px 0 0; }
  .modal-header h3 { margin: 0; font-size: 16px; }
  .modal-close { background: none; border: none; font-size: 24px; cursor: pointer;
                 color: #666; padding: 0 8px; }
  .modal-close:hover { color: #333; }
  .modal-body { padding: 20px; }
  .modal-instructions { margin: 0 0 12px 0; font-size: 14px; color: #555;
                        background: #fff3e0; padding: 10px 14px; border-radius: 6px; }
  .modal-image-container { position: relative; display: inline-block; cursor: crosshair;
                           border: 2px solid #ccc; border-radius: 4px; background: #b4b4b4; }
  .modal-image-container img { display: block; max-width: 100%; max-height: 60vh; }
  .marker { position: absolute; width: 24px; height: 24px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 12px; font-weight: bold; color: white;
            transform: translate(-50%, -50%); pointer-events: none;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
  .marker-mouth { background: #e53935; }
  .marker-tail { background: #1e88e5; }
  .modal-coords { margin-top: 12px; font-size: 13px; color: #666;
                  display: flex; gap: 20px; }
  .modal-footer { padding: 16px 20px; border-top: 1px solid #ddd;
                  display: flex; gap: 12px; justify-content: flex-end; }

  /* First Visit Modal Styles */
  .first-visit-modal { max-width: 800px; width: 90vw; }
  .first-visit-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
  .first-visit-header h3 { color: white; font-size: 20px; }
  .first-visit-header .modal-close { color: rgba(255,255,255,0.8); }
  .first-visit-header .modal-close:hover { color: white; }
  .first-visit-intro { font-size: 15px; color: #555; margin-bottom: 20px;
                       padding: 12px 16px; background: #e8f4fd; border-radius: 8px;
                       border-left: 4px solid #2196f3; }
  .first-visit-section { margin-bottom: 24px; }
  .first-visit-section h4 { margin: 0 0 8px 0; font-size: 16px; color: #333; }
  .first-visit-section .hint { font-size: 13px; color: #777; margin: 0 0 12px 0; }
  .k-selector { display: flex; gap: 8px; flex-wrap: wrap; }
  .k-btn { width: 44px; height: 44px; border-radius: 8px; border: 2px solid #ddd;
           background: white; font-size: 18px; font-weight: 600; cursor: pointer;
           transition: all 0.15s ease; }
  .k-btn:hover { border-color: #667eea; background: #f0f4ff; }
  .k-btn.active { background: #667eea; color: white; border-color: #667eea; }
  .k-selected { margin-top: 10px; font-size: 14px; color: #555; }
  .exemplar-gallery { display: flex; gap: 10px; overflow-x: auto; padding: 10px 0;
                      max-height: 180px; }
  .exemplar-thumb { flex-shrink: 0; width: 120px; height: 120px; border-radius: 8px;
                    border: 3px solid #ddd; overflow: hidden; cursor: pointer;
                    position: relative; background: #b4b4b4;
                    transition: all 0.15s ease; display: flex;
                    align-items: center; justify-content: center; }
  .exemplar-thumb img { max-width: 100%; max-height: 100%; object-fit: contain; }
  .exemplar-thumb:hover { border-color: #667eea; transform: scale(1.02); }
  .exemplar-thumb.selected { border-color: #4caf50; border-width: 4px; }
  .exemplar-check { position: absolute; top: 4px; right: 4px; background: #4caf50;
                    color: white; width: 24px; height: 24px; border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 14px; }
  .exemplar-selected { margin-top: 10px; font-size: 14px; color: #4caf50; font-weight: 600; }
  .first-visit-footer { justify-content: space-between; }
  #previewContainer { margin-top: 16px; padding: 12px; background: #f0f0f0;
                      border-radius: 8px; }
  #previewContainer h4 { margin: 0 0 8px 0; font-size: 14px; }
  .preview-wrapper { position: relative; display: inline-block; cursor: grab; }
  .preview-wrapper.dragging { cursor: grabbing; }
  #previewImage { max-width: 100%; max-height: 400px; border-radius: 4px; display: block;
                  user-select: none; -webkit-user-drag: none; }
  .exemplar-silhouette { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                         max-width: 100%; max-height: 400px; opacity: 0.3;
                         pointer-events: none; filter: brightness(0); }
  .preview-grid { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                  pointer-events: none; }
  .preview-grid line { stroke: rgba(0, 255, 255, 0.5); stroke-width: 1; }
  .preview-grid .center-line { stroke: rgba(255, 0, 0, 0.7); stroke-width: 2; }
  .angle-adjust { margin-top: 12px; padding: 10px; background: #e8e8e8;
                  border-radius: 6px; display: flex; align-items: center; gap: 12px; }
  .angle-adjust label { display: flex; align-items: center; gap: 8px; font-size: 13px; }
  .angle-adjust input[type="range"] { width: 200px; cursor: pointer; }
  .angle-adjust #angleAdjustValue { display: inline-block; width: 40px; text-align: right;
                                    font-weight: 600; font-family: monospace; }
  .btn-sm { padding: 4px 10px; font-size: 11px; }
  .flip-separator { color: #ccc; margin: 0 4px; }
  .btn-flip-active { background: #2196f3 !important; color: white !important; }
  .btn-warning { background: #ff9800; color: white; border-color: #f57c00; }
  .btn-warning:hover { background: #f57c00; border-color: #ef6c00; }
  .perspective-adjust { margin-top: 10px; padding: 10px; background: #fff3e0;
                        border-radius: 6px; border: 1px solid #ffcc80; }
  .perspective-adjust .perspective-row { display: flex; align-items: center; gap: 8px;
                                         margin-bottom: 8px; }
  .perspective-adjust label { display: flex; align-items: center; gap: 8px; font-size: 13px; }
  .perspective-adjust input[type="range"] { width: 180px; cursor: pointer; }
  .perspective-adjust .tilt-hint { font-size: 11px; color: #888; font-style: italic; }
  .perspective-adjust #dvTiltValue, .perspective-adjust #htTiltValue {
    display: inline-block; width: 40px; text-align: right;
    font-weight: 600; font-family: monospace; }
  .perspective-warning { display: block; margin-top: 8px; font-size: 11px; color: #e65100;
                         background: #fff8e1; padding: 4px 8px; border-radius: 4px; }
  .offset-adjust { margin-top: 10px; padding: 10px; background: #e3f2fd;
                   border-radius: 6px; border: 1px solid #90caf9; }
  .offset-adjust .offset-row { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  .offset-adjust label { display: flex; align-items: center; gap: 8px; font-size: 13px; }
  .offset-adjust input[type="range"] { width: 100px; cursor: pointer; }
  .offset-adjust input[type="checkbox"] { width: 16px; height: 16px; }
  .offset-adjust #offsetXValue, .offset-adjust #offsetYValue, .offset-adjust #scaleValue {
    display: inline-block; width: 45px; text-align: right;
    font-weight: 600; font-family: monospace; }
  .drag-hint { font-size: 11px; color: #666; font-style: italic; margin-top: 6px; }
</style>

<script>
const SPECIES = "{{ species }}";
const SPECIES_DIR = "{{ species_dirname }}";
const PREV_SP = "{{ prev_sp or '' }}";
const NEXT_SP = "{{ next_sp or '' }}";
const IS_FIRST_VISIT = {{ 'true' if status == 'not_reviewed' and not has_exemplar else 'false' }};

// ══════════════════════════════════════════════════════════════
// First Visit Modal Functions
// ══════════════════════════════════════════════════════════════

function showFirstVisitModal() {
  document.getElementById('firstVisitModal').style.display = 'flex';
  document.body.style.overflow = 'hidden';
}

function closeFirstVisitModal() {
  document.getElementById('firstVisitModal').style.display = 'none';
  document.body.style.overflow = '';
}

function selectK(k) {
  // Update UI
  document.querySelectorAll('.k-btn').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('selectedK').textContent = k;

  // Also update the main page k input
  document.getElementById('gestaltK').value = k;

  // Save to server
  const notes = document.getElementById('notes').value;
  fetch('/api/save_gestalt_k', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({species: SPECIES, gestalt_k: k, notes: notes})
  });
}

function selectExemplarFromModal(thumb) {
  const fname = thumb.dataset.fname;
  const pngName = thumb.dataset.png;
  const source = thumb.dataset.source;

  // Update modal UI
  document.querySelectorAll('.exemplar-thumb').forEach(t => {
    t.classList.remove('selected');
    t.querySelector('.exemplar-check')?.remove();
  });
  thumb.classList.add('selected');

  // Add checkmark
  const check = document.createElement('span');
  check.className = 'exemplar-check';
  check.textContent = '★';
  thumb.appendChild(check);

  document.getElementById('exemplarSelected').textContent = '★ Exemplar selected';

  // Update main page exemplar radio
  const mainRadio = document.querySelector(`input[name="exemplar"][data-fname="${fname}"]`);
  if (mainRadio) {
    mainRadio.checked = true;
    // Update card styling
    document.querySelectorAll('.card').forEach(c => c.classList.remove('is-exemplar'));
    mainRadio.closest('.card').classList.add('is-exemplar');
  }

  // Update footer status
  document.getElementById('exemplarStatus').innerHTML = '&#9733; Exemplar set';

  // Save to server
  fetch('/api/set_exemplar', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      species: SPECIES,
      filename: fname,
      png_name: pngName,
      source: source
    })
  });
}

function confirmFirstVisit() {
  closeFirstVisitModal();
  // Scroll to top of image grid for review
  document.querySelector('.grid').scrollIntoView({ behavior: 'smooth' });
}

// Show first visit modal on page load if this is a new species
document.addEventListener('DOMContentLoaded', function() {
  if (IS_FIRST_VISIT) {
    // Small delay to let page render first
    setTimeout(showFirstVisitModal, 300);
  }
});

// ══════════════════════════════════════════════════════════════
// Image Action Functions
// ══════════════════════════════════════════════════════════════

function toggleAction(cb) {
  const fname = cb.dataset.fname;
  const action = cb.dataset.action;

  const card = cb.closest('.card');
  const excludeCb = card.querySelector('[data-action="exclude"]');
  const fixCb = card.querySelector('[data-action="fix_orientation"]');
  const altCb = card.querySelector('[data-action="alt_morph"]');
  const resegCb = card.querySelector('[data-action="resegment"]');
  const colorCb = card.querySelector('[data-action="color_correct"]');

  // Determine what action to send (only one can be active)
  let sendAction = "keep";
  if (cb.checked) {
    sendAction = action;
    // Uncheck others and auto-exclude for alt_morph, resegment, color_correct
    if (action === "exclude") { fixCb.checked = false; altCb.checked = false; resegCb.checked = false; colorCb.checked = false; }
    if (action === "fix_orientation") { excludeCb.checked = false; altCb.checked = false; resegCb.checked = false; colorCb.checked = false; }
    if (action === "alt_morph") { fixCb.checked = false; resegCb.checked = false; colorCb.checked = false; excludeCb.checked = true; }
    if (action === "resegment") { fixCb.checked = false; altCb.checked = false; colorCb.checked = false; excludeCb.checked = true; }
    if (action === "color_correct") { fixCb.checked = false; altCb.checked = false; resegCb.checked = false; excludeCb.checked = true; }
  }

  // Update card styling
  card.classList.toggle('excluded', excludeCb.checked);
  card.classList.toggle('fix-orient', fixCb.checked);
  card.classList.toggle('alt-morph', altCb.checked);
  card.classList.toggle('resegment', resegCb.checked);
  card.classList.toggle('color-correct', colorCb.checked);

  // Save
  fetch('/api/save_image_action', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: fname, species: SPECIES, action: sendAction})
  }).then(r => r.json()).then(() => {
    updateCounts();
    reorderCards();
  });
}

function updateCounts() {
  const cards = document.querySelectorAll('.card');
  let nExcl = 0, nFix = 0, nAlt = 0, nReseg = 0, nColor = 0;
  cards.forEach(c => {
    if (c.querySelector('[data-action="exclude"]').checked) nExcl++;
    if (c.querySelector('[data-action="fix_orientation"]').checked) nFix++;
    if (c.querySelector('[data-action="alt_morph"]').checked) nAlt++;
    if (c.querySelector('[data-action="resegment"]').checked) nReseg++;
    if (c.querySelector('[data-action="color_correct"]').checked) nColor++;
  });
  document.getElementById('nExcluded').textContent = nExcl;
  document.getElementById('nFix').textContent = nFix;
  document.getElementById('nAltMorph').textContent = nAlt;
  document.getElementById('nResegment').textContent = nReseg;
  document.getElementById('nColorCorrect').textContent = nColor;
}

function reorderCards() {
  const grid = document.querySelector('.grid');
  const cards = Array.from(grid.querySelectorAll('.card'));
  const divider = grid.querySelector('.excluded-divider');

  // Source priority order
  const sourceOrder = {Bishop: 0, FishBase: 1, FishPix: 2, FishWise: 3, iNat: 4, FBUser: 5, Other: 6};

  // Sort cards
  cards.sort((a, b) => {
    const aExcluded = a.classList.contains('excluded') || a.classList.contains('alt-morph') ||
                      a.classList.contains('resegment') || a.classList.contains('color-correct');
    const bExcluded = b.classList.contains('excluded') || b.classList.contains('alt-morph') ||
                      b.classList.contains('resegment') || b.classList.contains('color-correct');
    const aExemplar = a.classList.contains('is-exemplar');
    const bExemplar = b.classList.contains('is-exemplar');
    const aSource = a.dataset.source || 'Other';
    const bSource = b.dataset.source || 'Other';

    // Excluded status (included first)
    if (aExcluded !== bExcluded) return aExcluded ? 1 : -1;
    // Exemplar first
    if (aExemplar !== bExemplar) return aExemplar ? -1 : 1;
    // Source priority
    const aOrder = sourceOrder[aSource] ?? 6;
    const bOrder = sourceOrder[bSource] ?? 6;
    if (aOrder !== bOrder) return aOrder - bOrder;
    // Filename
    return (a.dataset.fname || '').localeCompare(b.dataset.fname || '');
  });

  // Remove divider if exists
  if (divider) divider.remove();

  // Re-append cards in sorted order, adding divider before first excluded
  let addedDivider = false;
  cards.forEach(card => {
    const isExcluded = card.classList.contains('excluded') || card.classList.contains('alt-morph') ||
                       card.classList.contains('resegment') || card.classList.contains('color-correct');
    if (isExcluded && !addedDivider) {
      const newDivider = document.createElement('div');
      newDivider.className = 'excluded-divider';
      newDivider.innerHTML = '<span>Excluded Images</span>';
      grid.appendChild(newDivider);
      addedDivider = true;
    }
    grid.appendChild(card);
  });
}

function setExemplar(radio) {
  const fname = radio.dataset.fname;
  const pngName = radio.dataset.png;
  const source = radio.dataset.source;
  const card = radio.closest('.card');

  // Remove is-exemplar from all cards
  document.querySelectorAll('.card').forEach(c => c.classList.remove('is-exemplar'));

  // Add to this card
  card.classList.add('is-exemplar');

  // Save to server
  fetch('/api/set_exemplar', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      species: SPECIES,
      filename: fname,
      png_name: pngName,
      source: source
    })
  }).then(r => r.json()).then(() => {
    document.getElementById('exemplarStatus').innerHTML = '&#9733; Exemplar set';
  });
}

// ══════════════════════════════════════════════════════════════
// View Mode Toggle
// ══════════════════════════════════════════════════════════════

function setViewMode(mode) {
  // Remove all view classes
  document.body.classList.remove('view-original', 'view-processed', 'view-both');
  // Add selected view class
  document.body.classList.add('view-' + mode);

  // Update button active states
  document.querySelectorAll('.view-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.view === mode);
  });

  // Save preference (optional - persists across page loads)
  localStorage.setItem('chaetview-mode', mode);
}

// Initialize view mode from localStorage or default to 'both'
document.addEventListener('DOMContentLoaded', function() {
  const savedMode = localStorage.getItem('chaetview-mode') || 'both';
  setViewMode(savedMode);

  // Apply Sea-thru filter to processed images that have it enabled
  document.querySelectorAll('.img-processed img[data-has-seathru="true"]').forEach(img => {
    const method = img.dataset.seathruMethod || 'gray_world';
    const apply = () => applySeathruFilter(img, method);
    if (img.complete && img.naturalWidth > 0) {
      apply();
    } else {
      img.addEventListener('load', apply, { once: true });
    }
  });
});

// ══════════════════════════════════════════════════════════════
// Sea-thru Inline Buttons
// ══════════════════════════════════════════════════════════════

function applySeathruFilter(img, method) {
  if (!img.dataset.originalSrc) {
    img.dataset.originalSrc = img.src;
  }
  const baseSrc = img.dataset.originalSrc;
  const sep = baseSrc.includes('?') ? '&' : '?';
  img.src = baseSrc + sep + 'seathru=' + method;
  img.classList.add('filter-seathru');
  img.dataset.seathruMethod = method;
}

function removeSeathruFilter(img) {
  if (img.dataset.originalSrc) {
    img.src = img.dataset.originalSrc;
    img.classList.remove('filter-seathru');
    delete img.dataset.seathruMethod;
  }
}

function clickSeathruBtn(btn) {
  const card = btn.closest('.card');
  const fname = btn.dataset.fname;
  const method = btn.dataset.method;
  const wasActive = btn.classList.contains('active');
  const processedImg = card.querySelector('.img-processed img');

  // Deactivate all sibling seathru buttons in this card
  card.querySelectorAll('.seathru-btn-inline').forEach(b => b.classList.remove('active'));

  if (wasActive) {
    // Toggle off
    if (processedImg) removeSeathruFilter(processedImg);
    fetch('/api/save_filter', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: fname, filter: 'seathru', enabled: false})
    });
  } else {
    // Activate this method
    btn.classList.add('active');
    if (processedImg) applySeathruFilter(processedImg, method);
    fetch('/api/save_filter', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({filename: fname, filter: 'seathru', enabled: true, method: method})
    });
  }
}

// Source image popup
function showSourcePopup(fname, displayName) {
  const popup = document.getElementById('sourcePopup');
  const img = document.getElementById('sourcePopupImg');
  const title = document.getElementById('sourcePopupTitle');
  title.textContent = 'Source: ' + displayName;
  img.src = '/image/source/' + fname;
  popup.classList.add('visible');
}

function closeSourcePopup(event) {
  // Close if clicking the overlay background or the close button
  if (!event || event.target.id === 'sourcePopup' || event.target.classList.contains('popup-close')) {
    document.getElementById('sourcePopup').classList.remove('visible');
  }
}

// Close popup on Escape
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    document.getElementById('sourcePopup').classList.remove('visible');
  }
});

let gestaltTimer = null;
function saveGestaltK() {
  clearTimeout(gestaltTimer);
  gestaltTimer = setTimeout(() => {
    const k = document.getElementById('gestaltK').value;
    const notes = document.getElementById('notes').value;
    fetch('/api/save_gestalt_k', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({species: SPECIES, gestalt_k: k, notes: notes})
    });
  }, 500);
}

function markReviewed() {
  fetch('/api/mark_reviewed', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({species: SPECIES})
  }).then(r => r.json()).then(() => {
    const badge = document.querySelector('.status-badge');
    badge.innerHTML = '<span class="badge badge-reviewed">Reviewed</span>';
    if (NEXT_SP) {
      window.location.href = '/review/' + NEXT_SP;
    } else {
      window.location.href = '/';
    }
  });
}

// Keyboard navigation
document.addEventListener('keydown', function(e) {
  // Don't trigger if typing in an input or modal is open
  if (e.target.tagName === 'INPUT') return;
  if (document.getElementById('orientModal').style.display !== 'none') {
    if (e.key === 'Escape') closeOrientModal();
    return;
  }

  if (e.key === 'ArrowLeft' && PREV_SP) {
    window.location.href = '/review/' + PREV_SP;
  } else if (e.key === 'ArrowRight' && NEXT_SP) {
    window.location.href = '/review/' + NEXT_SP;
  } else if (e.key === 'Enter') {
    markReviewed();
  }
});

// ======== Orientation Digitizer ========
let orientState = {
  filename: '',
  pngName: '',
  checkbox: null,
  mouthX: null, mouthY: null,
  tailX: null, tailY: null,
  clickCount: 0,
  autoAngle: null,      // Calculated angle from landmarks
  adjustedAngle: null,  // User-adjusted angle (null means use autoAngle)
  flipH: false,         // Horizontal flip
  flipV: false,         // Vertical flip
  dvTilt: 0,            // Dorsal-ventral tilt (rotate around long axis)
  htTilt: 0,            // Head-tail tilt (rotate around vertical axis)
  offsetX: 0,           // X offset for centering (pan)
  offsetY: 0,           // Y offset for centering (pan)
  scale: 100,           // Zoom percentage
  showExemplar: false,  // Show exemplar silhouette
  isDragging: false,    // For drag-to-pan
  dragStartX: 0,
  dragStartY: 0
};

function handleFixOrientation(cb) {
  if (cb.checked) {
    // Open the digitizer modal
    orientState.filename = cb.dataset.fname;
    orientState.pngName = cb.dataset.png;
    orientState.checkbox = cb;
    openOrientModal();
  } else {
    // Unchecking - just toggle the action
    toggleAction(cb);
  }
}

function openOrientModal() {
  const modal = document.getElementById('orientModal');
  const img = document.getElementById('modalImage');
  document.getElementById('modalFilename').textContent = orientState.pngName;

  // Load full-size image
  img.src = '/image/full/' + SPECIES_DIR + '/' + orientState.pngName;

  // Reset state
  resetDigitize();

  modal.style.display = 'flex';
}

function closeOrientModal() {
  document.getElementById('orientModal').style.display = 'none';

  // If we didn't apply, uncheck the checkbox
  if (orientState.checkbox && !orientState.applied) {
    orientState.checkbox.checked = false;
  }
  orientState.applied = false;
}

function resetDigitize() {
  orientState.mouthX = null;
  orientState.mouthY = null;
  orientState.tailX = null;
  orientState.tailY = null;
  orientState.clickCount = 0;
  orientState.autoAngle = null;
  orientState.adjustedAngle = null;
  orientState.flipH = false;
  orientState.flipV = false;
  orientState.dvTilt = 0;
  orientState.htTilt = 0;
  orientState.offsetX = 0;
  orientState.offsetY = 0;

  document.getElementById('mouthMarker').style.display = 'none';
  document.getElementById('tailMarker').style.display = 'none';
  document.getElementById('mouthCoords').textContent = '—';
  document.getElementById('tailCoords').textContent = '—';
  document.getElementById('angleDisplay').textContent = '—';
  document.getElementById('previewContainer').style.display = 'none';
  document.getElementById('previewBtn').disabled = true;
  document.getElementById('applyBtn').disabled = true;

  // Reset angle slider, flip buttons, and perspective
  document.getElementById('angleSlider').value = 0;
  document.getElementById('angleAdjustValue').textContent = '0';
  document.getElementById('flipHBtn').classList.remove('btn-flip-active');
  document.getElementById('flipVBtn').classList.remove('btn-flip-active');
  document.getElementById('dvTiltSlider').value = 0;
  document.getElementById('dvTiltValue').textContent = '0';
  document.getElementById('htTiltSlider').value = 0;
  document.getElementById('htTiltValue').textContent = '0';
  document.getElementById('offsetXSlider').value = 0;
  document.getElementById('offsetXValue').textContent = '0';
  document.getElementById('offsetYSlider').value = 0;
  document.getElementById('offsetYValue').textContent = '0';
  document.getElementById('scaleSlider').value = 100;
  document.getElementById('scaleValue').textContent = '100';
  document.getElementById('exemplarToggle').checked = false;
  document.getElementById('exemplarSilhouette').style.display = 'none';
  document.getElementById('perspectiveWarning').style.display = 'none';
}

function handleImageClick(event) {
  const img = event.target;
  const rect = img.getBoundingClientRect();

  // Get click position relative to image
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  // Scale to actual image coordinates
  // Use naturalWidth/Height if available, otherwise fall back to displayed size
  const scaleX = img.naturalWidth > 0 ? img.naturalWidth / img.width : 1;
  const scaleY = img.naturalHeight > 0 ? img.naturalHeight / img.height : 1;
  const imgX = Math.round(x * scaleX);
  const imgY = Math.round(y * scaleY);

  console.log('Click:', {x, y, imgX, imgY, scaleX, scaleY, naturalW: img.naturalWidth, naturalH: img.naturalHeight});

  if (orientState.clickCount === 0) {
    // First click: mouth
    orientState.mouthX = imgX;
    orientState.mouthY = imgY;
    orientState.clickCount = 1;

    const marker = document.getElementById('mouthMarker');
    marker.style.left = x + 'px';
    marker.style.top = y + 'px';
    marker.style.display = 'flex';
    document.getElementById('mouthCoords').textContent = imgX + ', ' + imgY;

  } else if (orientState.clickCount === 1) {
    // Second click: tail
    orientState.tailX = imgX;
    orientState.tailY = imgY;
    orientState.clickCount = 2;

    const marker = document.getElementById('tailMarker');
    marker.style.left = x + 'px';
    marker.style.top = y + 'px';
    marker.style.display = 'flex';
    document.getElementById('tailCoords').textContent = imgX + ', ' + imgY;

    // Calculate angle
    const dx = orientState.tailX - orientState.mouthX;
    const dy = orientState.tailY - orientState.mouthY;
    const angle = Math.atan2(dy, dx) * 180 / Math.PI;
    document.getElementById('angleDisplay').textContent = angle.toFixed(1);

    // Enable preview button
    document.getElementById('previewBtn').disabled = false;
  }
}

function previewRotation() {
  if (orientState.clickCount < 2) return;

  const payload = {
    filename: orientState.pngName,
    species_dirname: SPECIES_DIR,
    mouth_x: orientState.mouthX,
    mouth_y: orientState.mouthY,
    tail_x: orientState.tailX,
    tail_y: orientState.tailY,
    flip_h: orientState.flipH,
    flip_v: orientState.flipV,
    dv_tilt: orientState.dvTilt,
    ht_tilt: orientState.htTilt,
    offset_x: orientState.offsetX,
    offset_y: orientState.offsetY,
    scale: orientState.scale
  };

  // Include angle_override if user has adjusted it
  if (orientState.adjustedAngle !== null) {
    payload.angle_override = orientState.adjustedAngle;
  }

  console.log('Preview request:', payload);

  fetch('/api/preview_rotation', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  }).then(r => r.json()).then(data => {
    console.log('Preview response:', data.ok ? 'OK, angle=' + data.angle : 'Error: ' + data.error);
    if (data.ok) {
      const img = document.getElementById('previewImage');
      img.src = data.image;
      document.getElementById('previewContainer').style.display = 'block';
      document.getElementById('applyBtn').disabled = false;

      // Update grid when image loads
      img.onload = function() {
        updatePreviewGrid();
      };

      // Store the auto-calculated angle (only on first preview, not when slider adjusts)
      if (orientState.autoAngle === null) {
        orientState.autoAngle = data.angle;
        // Reset slider to 0 (center) since this is the auto-detected angle
        document.getElementById('angleSlider').value = 0;
        document.getElementById('angleAdjustValue').textContent = '0';
      }
    } else {
      alert('Preview error: ' + data.error);
    }
  }).catch(err => {
    console.error('Preview fetch error:', err);
    alert('Preview network error: ' + err);
  });
}

// Draw grid overlay on preview image
function updatePreviewGrid() {
  const img = document.getElementById('previewImage');
  const svg = document.getElementById('previewGrid');
  const showGrid = document.getElementById('gridToggle').checked;

  if (!showGrid || !img.complete) {
    svg.innerHTML = '';
    return;
  }

  const w = img.offsetWidth;
  const h = img.offsetHeight;

  // Create grid lines
  let lines = '';
  const gridSpacing = 40; // pixels between grid lines

  // Vertical lines
  for (let x = gridSpacing; x < w; x += gridSpacing) {
    const isCenter = Math.abs(x - w/2) < gridSpacing/2;
    lines += '<line x1="' + x + '" y1="0" x2="' + x + '" y2="' + h + '"' +
             (isCenter ? ' class="center-line"' : '') + '/>';
  }

  // Horizontal lines
  for (let y = gridSpacing; y < h; y += gridSpacing) {
    const isCenter = Math.abs(y - h/2) < gridSpacing/2;
    lines += '<line x1="0" y1="' + y + '" x2="' + w + '" y2="' + y + '"' +
             (isCenter ? ' class="center-line"' : '') + '/>';
  }

  svg.innerHTML = lines;
}

// Toggle grid visibility
function toggleGrid() {
  updatePreviewGrid();
}

// Adjust angle with slider (real-time preview)
let angleDebounce = null;
function adjustAngle(adjustment) {
  const adjust = parseFloat(adjustment);
  document.getElementById('angleAdjustValue').textContent = adjust >= 0 ? '+' + adjust : adjust;

  // Calculate new angle = autoAngle + adjustment
  if (orientState.autoAngle !== null) {
    orientState.adjustedAngle = orientState.autoAngle + adjust;
    document.getElementById('angleDisplay').textContent = orientState.adjustedAngle.toFixed(1);

    // Debounce the preview request
    clearTimeout(angleDebounce);
    angleDebounce = setTimeout(() => {
      previewRotation();
    }, 150);
  }
}

// Reset angle adjustment to auto-detected
function resetAngleAdjust() {
  orientState.adjustedAngle = null;
  document.getElementById('angleSlider').value = 0;
  document.getElementById('angleAdjustValue').textContent = '0';

  if (orientState.autoAngle !== null) {
    document.getElementById('angleDisplay').textContent = orientState.autoAngle.toFixed(1);
    previewRotation();
  }
}

// Toggle horizontal flip
function toggleFlipH() {
  orientState.flipH = !orientState.flipH;
  document.getElementById('flipHBtn').classList.toggle('btn-flip-active', orientState.flipH);
  previewRotation();
}

// Toggle vertical flip
function toggleFlipV() {
  orientState.flipV = !orientState.flipV;
  document.getElementById('flipVBtn').classList.toggle('btn-flip-active', orientState.flipV);
  previewRotation();
}

// Adjust dorsal-ventral tilt (rotate around long axis)
let dvTiltDebounce = null;
function adjustDVTilt(angle) {
  const tiltAngle = parseFloat(angle);
  orientState.dvTilt = tiltAngle;
  document.getElementById('dvTiltValue').textContent = tiltAngle >= 0 ? '+' + tiltAngle : tiltAngle;
  updatePerspectiveWarning();

  // Debounce the preview request
  clearTimeout(dvTiltDebounce);
  dvTiltDebounce = setTimeout(() => {
    previewRotation();
  }, 150);
}

// Adjust head-tail tilt (rotate around vertical axis)
let htTiltDebounce = null;
function adjustHTTilt(angle) {
  const tiltAngle = parseFloat(angle);
  orientState.htTilt = tiltAngle;
  document.getElementById('htTiltValue').textContent = tiltAngle >= 0 ? '+' + tiltAngle : tiltAngle;
  updatePerspectiveWarning();

  // Debounce the preview request
  clearTimeout(htTiltDebounce);
  htTiltDebounce = setTimeout(() => {
    previewRotation();
  }, 150);
}

// Show/hide warning based on whether any perspective correction is applied
function updatePerspectiveWarning() {
  const hasPerspective = orientState.dvTilt !== 0 || orientState.htTilt !== 0;
  document.getElementById('perspectiveWarning').style.display = hasPerspective ? 'block' : 'none';
}

// Reset both perspective tilts to 0
function resetPerspective() {
  orientState.dvTilt = 0;
  orientState.htTilt = 0;
  document.getElementById('dvTiltSlider').value = 0;
  document.getElementById('dvTiltValue').textContent = '0';
  document.getElementById('htTiltSlider').value = 0;
  document.getElementById('htTiltValue').textContent = '0';
  document.getElementById('perspectiveWarning').style.display = 'none';
  previewRotation();
}

// Adjust X offset (pan left/right)
let offsetXDebounce = null;
function adjustOffsetX(val) {
  const offset = parseInt(val);
  orientState.offsetX = offset;
  document.getElementById('offsetXValue').textContent = offset >= 0 ? '+' + offset : offset;

  clearTimeout(offsetXDebounce);
  offsetXDebounce = setTimeout(() => {
    previewRotation();
  }, 100);
}

// Adjust Y offset (pan up/down)
let offsetYDebounce = null;
function adjustOffsetY(val) {
  const offset = parseInt(val);
  orientState.offsetY = offset;
  document.getElementById('offsetYValue').textContent = offset >= 0 ? '+' + offset : offset;

  clearTimeout(offsetYDebounce);
  offsetYDebounce = setTimeout(() => {
    previewRotation();
  }, 100);
}

// Reset offset and scale
function resetOffset() {
  orientState.offsetX = 0;
  orientState.offsetY = 0;
  orientState.scale = 100;
  document.getElementById('offsetXSlider').value = 0;
  document.getElementById('offsetXValue').textContent = '0';
  document.getElementById('offsetYSlider').value = 0;
  document.getElementById('offsetYValue').textContent = '0';
  document.getElementById('scaleSlider').value = 100;
  document.getElementById('scaleValue').textContent = '100';
  previewRotation();
}

// Reset ALL adjustments back to original image state
function resetAllAdjustments() {
  // Reset rotation to auto-calculated angle
  orientState.adjustedAngle = null;
  document.getElementById('angleSlider').value = 0;
  document.getElementById('angleAdjustValue').textContent = '0';

  // Reset flips
  orientState.flipH = false;
  orientState.flipV = false;
  document.getElementById('flipHBtn').classList.remove('btn-flip-active');
  document.getElementById('flipVBtn').classList.remove('btn-flip-active');

  // Reset perspective tilts
  orientState.dvTilt = 0;
  orientState.htTilt = 0;
  document.getElementById('dvTiltSlider').value = 0;
  document.getElementById('dvTiltValue').textContent = '0';
  document.getElementById('htTiltSlider').value = 0;
  document.getElementById('htTiltValue').textContent = '0';
  updatePerspectiveWarning();

  // Reset position and zoom
  orientState.offsetX = 0;
  orientState.offsetY = 0;
  orientState.scale = 100;
  document.getElementById('offsetXSlider').value = 0;
  document.getElementById('offsetXValue').textContent = '0';
  document.getElementById('offsetYSlider').value = 0;
  document.getElementById('offsetYValue').textContent = '0';
  document.getElementById('scaleSlider').value = 100;
  document.getElementById('scaleValue').textContent = '100';

  // Regenerate preview with original settings
  previewRotation();
}

// Adjust scale
let scaleDebounce = null;
function adjustScale(val) {
  const scale = parseInt(val);
  orientState.scale = scale;
  document.getElementById('scaleValue').textContent = scale;

  clearTimeout(scaleDebounce);
  scaleDebounce = setTimeout(() => {
    previewRotation();
  }, 100);
}

// Toggle exemplar silhouette
function toggleExemplar() {
  orientState.showExemplar = document.getElementById('exemplarToggle').checked;
  const silhouette = document.getElementById('exemplarSilhouette');
  if (orientState.showExemplar) {
    // Load exemplar image for this species
    silhouette.src = '/image/exemplar_silhouette/' + SPECIES_DIR;
    silhouette.style.display = 'block';
  } else {
    silhouette.style.display = 'none';
  }
}

// Drag to pan functionality
function initDragToPan() {
  const wrapper = document.getElementById('previewWrapper');
  const preview = document.getElementById('previewImage');

  wrapper.addEventListener('mousedown', function(e) {
    if (e.target === preview) {
      orientState.isDragging = true;
      orientState.dragStartX = e.clientX;
      orientState.dragStartY = e.clientY;
      wrapper.classList.add('dragging');
      e.preventDefault();
    }
  });

  document.addEventListener('mousemove', function(e) {
    if (orientState.isDragging) {
      const dx = e.clientX - orientState.dragStartX;
      const dy = e.clientY - orientState.dragStartY;

      // Update offset (scale the movement to image coordinates)
      orientState.offsetX += dx;
      orientState.offsetY += dy;

      // Clamp to slider range
      orientState.offsetX = Math.max(-200, Math.min(200, orientState.offsetX));
      orientState.offsetY = Math.max(-200, Math.min(200, orientState.offsetY));

      // Update sliders
      document.getElementById('offsetXSlider').value = orientState.offsetX;
      document.getElementById('offsetXValue').textContent = orientState.offsetX >= 0 ? '+' + orientState.offsetX : orientState.offsetX;
      document.getElementById('offsetYSlider').value = orientState.offsetY;
      document.getElementById('offsetYValue').textContent = orientState.offsetY >= 0 ? '+' + orientState.offsetY : orientState.offsetY;

      orientState.dragStartX = e.clientX;
      orientState.dragStartY = e.clientY;

      // Debounced preview update
      clearTimeout(offsetXDebounce);
      offsetXDebounce = setTimeout(() => {
        previewRotation();
      }, 50);
    }
  });

  document.addEventListener('mouseup', function(e) {
    if (orientState.isDragging) {
      orientState.isDragging = false;
      document.getElementById('previewWrapper').classList.remove('dragging');
    }
  });
}

// Initialize drag-to-pan when page loads
document.addEventListener('DOMContentLoaded', initDragToPan);

function applyRotation() {
  // Build landmarks payload
  const landmarksPayload = {
    filename: orientState.pngName,
    species: SPECIES,
    mouth_x: orientState.mouthX,
    mouth_y: orientState.mouthY,
    tail_x: orientState.tailX,
    tail_y: orientState.tailY,
    flip_h: orientState.flipH,
    flip_v: orientState.flipV,
    dv_tilt: orientState.dvTilt,
    ht_tilt: orientState.htTilt
  };

  // Include adjusted angle if user modified it
  if (orientState.adjustedAngle !== null) {
    landmarksPayload.adjusted_angle = orientState.adjustedAngle;
  }

  // First save the landmarks
  fetch('/api/save_orient_landmarks', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(landmarksPayload)
  }).then(r => r.json()).then(data => {
    if (!data.ok) {
      throw new Error('Save landmarks failed: ' + (data.error || 'unknown'));
    }
    // Then apply the rotation
    return fetch('/api/apply_orientation', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        filename: orientState.pngName,
        species: SPECIES,
        species_dirname: SPECIES_DIR
      })
    });
  }).then(r => r.json()).then(data => {
    if (data.ok) {
      // Uncheck the fix_orientation checkbox since we've now fixed it
      orientState.checkbox.checked = false;

      const card = orientState.checkbox.closest('.card');
      card.classList.remove('fix-orient');
      card.classList.add('has-oriented');

      // Determine the action based on whether perspective correction was applied
      // If ANY perspective correction is used, BOTH versions are excluded by default
      const hasPerspective = orientState.dvTilt !== 0 || orientState.htTilt !== 0;
      const action = hasPerspective ? 'perspective_corrected' : 'keep';

      // Save action - perspective-corrected images are EXCLUDED by default (both versions!)
      fetch('/api/save_image_action', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          filename: orientState.checkbox.dataset.fname,
          species: SPECIES,
          action: action
        })
      });

      orientState.applied = true;
      closeOrientModal();

      // Different message based on whether perspective correction was applied
      if (hasPerspective) {
        alert('Orientation fixed with perspective correction!\\n\\nBOTH versions saved but BOTH EXCLUDED from Pavo by default:\\n- Standard rotation only\\n- With perspective correction\\n\\nYou can compare these against good laterals to validate the correction approach.');
      } else {
        alert('Orientation fixed and normalized! The image will now be INCLUDED in analysis.');
      }
      window.location.reload();
    } else {
      alert('Apply error: ' + data.error);
    }
  }).catch(err => {
    console.error('Apply rotation error:', err);
    alert('Error: ' + err.message);
  });
}
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════
#  Startup
# ══════════════════════════════════════════════════════════════

def init_app():
    """Load all data into memory."""
    global _inventory, _species_list, _annotations
    global _review_state, _gestalt_k, _exemplars, _orient_warnings, _orient_landmarks
    global _duplicate_mapping, _fishbase_duplicates
    global _photographer_data, _image_metadata, _miyazawa_images
    global _underwater_flags, _orig_image_index

    print("Loading inventory...")
    _inventory = load_inventory()
    _species_list = all_species(_inventory)
    print(f"  {len(_inventory)} images across {len(_species_list)} species")

    print("Loading annotations...")
    _annotations = load_annotations()
    print(f"  {len(_annotations)} annotations")

    print("Loading review state...")
    _review_state = _load_review_state()
    print(f"  {len(_review_state)} image reviews")

    print("Loading gestalt k values...")
    _gestalt_k = _load_gestalt_k()
    n_reviewed = sum(1 for v in _gestalt_k.values()
                     if v.get("review_status") == "reviewed")
    print(f"  {n_reviewed} species reviewed")

    print("Loading exemplar selections...")
    _exemplars = _load_exemplars()
    print(f"  {len(_exemplars)} species with exemplars")

    print("Loading orientation warnings...")
    _orient_warnings = _load_orient_warnings()
    print(f"  {len(_orient_warnings)} images with orientation uncertainty")

    print("Loading orientation landmarks...")
    _orient_landmarks = _load_orient_landmarks()
    print(f"  {len(_orient_landmarks)} images with digitized landmarks")

    print("Loading photographer metadata...")
    _photographer_data = _load_photographer_data()
    print(f"  {len(_photographer_data)} images with photographer data")

    print("Loading image metadata (type, quality)...")
    _image_metadata = _load_image_metadata()
    print(f"  {len(_image_metadata)} images with enriched metadata")

    print("Loading perceptual hash cache...")
    global _phash_cache
    _phash_cache = _load_phash_cache()
    cached_count = len(_phash_cache)
    print(f"  {cached_count} cached hashes loaded")

    print("Computing cross-database duplicate mapping...")
    _duplicate_mapping, _fishbase_duplicates = _load_duplicate_mapping()
    n_with_dups = sum(1 for v in _duplicate_mapping.values() if v)
    print(f"  {n_with_dups} canonical images with cross-database duplicates")

    # Save updated cache if new hashes were computed
    if len(_phash_cache) > cached_count:
        print(f"  Saving {len(_phash_cache) - cached_count} new hashes to cache...")
        _save_phash_cache()

    print("Loading Miyazawa (2020) study images...")
    _miyazawa_images = _load_miyazawa_images()
    print(f"  {len(_miyazawa_images)} images from Miyazawa (2020) study")
    print(f"  {len(_fishbase_duplicates)} duplicate images to hide")

    print("Loading underwater detection flags...")
    _underwater_flags = _load_underwater_flags()
    if _underwater_flags:
        n_whole_uw = sum(1 for v in _underwater_flags.values()
                         if v.get("whole_is_uw") == "True")
        n_body_uw = sum(1 for v in _underwater_flags.values()
                        if v.get("body_is_uw") == "True")
        print(f"  {len(_underwater_flags)} images analyzed, "
              f"{n_whole_uw} UW (whole), {n_body_uw} UW (body)")
    else:
        print("  No underwater detection report found (run detect_underwater.py)")

    print("Loading depth estimates...")
    global _depth_estimates
    _depth_estimates = load_depth_estimates()
    if _depth_estimates:
        print(f"  {len(_depth_estimates)} images with depth estimates")
    else:
        print("  No depth estimates found (run estimate_depths.py)")

    print("Loading saved filter states...")
    _image_filters.update(_load_image_filters())
    n_filters = sum(1 for f in _image_filters.values()
                    if any(f.values()))
    print(f"  {n_filters} images with active filters")

    print("Building original image index...")
    _orig_image_index = {}
    for row in _inventory:
        base = os.path.splitext(row["filename"])[0]
        orig_path = os.path.join(REPO_DIR, row["directory"], row["filename"])
        _orig_image_index[base] = orig_path
    print(f"  {len(_orig_image_index)} images indexed")


def main():
    parser = argparse.ArgumentParser(description="Fish image review app")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_app()

    print(f"\nStarting review app at http://localhost:{args.port}")
    print("Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
