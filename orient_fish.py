#!/usr/bin/env python3
"""
Standardize fish orientation: all fish face LEFT on a horizontal axis.

Strategy for Chaetodontidae (disc-shaped butterflyfish):
  1. TRUST museum sources: Bishop, FishBase, and FishPix images are assumed
     to be correctly oriented (head facing left). Only apply minor tilt
     correction to these.
  2. For iNaturalist images: use the species reference image (Bishop preferred)
     as a template. Compare depth profiles to decide if the image needs flipping.
  3. Generate a review list for low-confidence decisions.

Why we trust museum images:
  - Bishop Museum, FishBase, and FishPix photos follow standard ichthyological
    convention (head left, lateral view)
  - These disc-shaped fish are too symmetric for reliable automated flip detection
  - The few exceptions (maybe 5-10% of museum images face right) are better
    caught by manual review than by error-prone auto-flipping

Works on RGBA segmented PNGs — rotates and flips the full image+mask.

Usage:
    python3 orient_fish.py                                     # all segmented
    python3 orient_fish.py --species "Chaetodon baronessa"     # one species
    python3 orient_fish.py --image path/to/seg.png --show      # test one image
    python3 orient_fish.py --dry-run                           # report only

Reads:  analysis/approach_1_gmm/segmented/
Writes: analysis/approach_1_gmm/segmented/  (overwrites in place)
        analysis/approach_1_gmm/orientation_report.csv
        analysis/approach_1_gmm/orientation_review.csv  (low-confidence cases)
"""

import os
import csv
import argparse
import numpy as np
from PIL import Image

from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR,
    load_inventory, load_annotations, species_to_dirname,
    ensure_dir,
)

REPORT_CSV = os.path.join(GMM_DIR, "orientation_report.csv")
REVIEW_CSV = os.path.join(GMM_DIR, "orientation_review.csv")
REPORT_FIELDS = [
    "filename", "species", "source", "rotation_deg", "flipped_horizontal",
    "method", "confidence", "reference_image", "needs_review",
]


# ──────────────────────────────────────────────────────────────
# Source classification
# ──────────────────────────────────────────────────────────────

def classify_source(filename):
    """Classify the image source from filename."""
    if "Bishop" in filename:
        return "bishop"
    elif "FishBase" in filename:
        return "fishbase"
    elif "FishPix" in filename:
        return "fishpix"
    elif "iNaturalist" in filename:
        return "inaturalist"
    return "other"


TRUSTED_SOURCES = {"bishop", "fishbase", "fishpix"}


# ──────────────────────────────────────────────────────────────
# Reference image selection
# ──────────────────────────────────────────────────────────────

def select_reference_image(species_rows, seg_dir):
    """Pick the best reference image for a species.

    Priority: Bishop > FishBase > FishPix > iNaturalist.
    Within each tier, prefer larger mask area.
    """
    tiers = {"bishop": [], "fishbase": [], "fishpix": [],
             "inaturalist": [], "other": []}

    for row in species_rows:
        fname = row["filename"]
        png_name = os.path.splitext(fname)[0] + ".png"
        seg_path = os.path.join(seg_dir, png_name)
        if not os.path.exists(seg_path):
            continue
        tier = classify_source(fname)
        tiers[tier].append((fname, seg_path))

    for tier_name in ["bishop", "fishbase", "fishpix", "inaturalist", "other"]:
        candidates = tiers[tier_name]
        if not candidates:
            continue

        best = None
        best_area = 0
        for fname, seg_path in candidates:
            try:
                img = Image.open(seg_path).convert("RGBA")
                arr = np.array(img)
                area = (arr[:, :, 3] > 128).sum()
                if area > best_area:
                    best_area = area
                    best = (fname, seg_path, tier_name)
            except Exception:
                continue

        if best:
            return best

    return None


# ──────────────────────────────────────────────────────────────
# Depth profile for template matching
# ──────────────────────────────────────────────────────────────

def compute_depth_profile(mask, n_bins=50):
    """Compute the vertical-depth profile of a fish mask.

    Divides the mask horizontally into n_bins slices. For each slice,
    measures the vertical depth (pixel rows with foreground).

    Returns a normalized profile array of length n_bins.
    """
    ys, xs = np.where(mask)
    if len(ys) < 50:
        return np.zeros(n_bins)

    x_min, x_max = xs.min(), xs.max()
    width = x_max - x_min + 1
    if width < n_bins:
        return np.zeros(n_bins)

    bin_edges = np.linspace(x_min, x_max + 1, n_bins + 1, dtype=int)
    profile = np.zeros(n_bins)

    for i in range(n_bins):
        col_start = bin_edges[i]
        col_end = max(col_start + 1, bin_edges[i + 1])
        strip = mask[:, col_start:col_end]
        profile[i] = strip.any(axis=1).sum()

    peak = profile.max()
    if peak > 0:
        profile = profile / peak

    return profile


def profile_flip_score(target_profile, ref_profile):
    """Score whether the target needs flipping relative to the reference.

    Returns (should_flip, confidence):
      - confidence: how much better the flipped version matches
    """
    if len(target_profile) == 0 or len(ref_profile) == 0:
        return False, 0.0
    if target_profile.sum() < 0.01 or ref_profile.sum() < 0.01:
        return False, 0.0

    corr_normal = np.corrcoef(ref_profile, target_profile)[0, 1]
    corr_flipped = np.corrcoef(ref_profile, target_profile[::-1])[0, 1]

    if np.isnan(corr_normal):
        corr_normal = 0.0
    if np.isnan(corr_flipped):
        corr_flipped = 0.0

    should_flip = corr_flipped > corr_normal + 0.05  # need clear advantage
    confidence = abs(corr_normal - corr_flipped)

    return should_flip, round(confidence, 4)


# ──────────────────────────────────────────────────────────────
# Tilt correction
# ──────────────────────────────────────────────────────────────

def get_horizontal_tilt(mask):
    """Estimate tilt using the leftmost and rightmost mask points."""
    ys, xs = np.where(mask)
    if len(ys) < 50:
        return 0.0

    x_min, x_max = xs.min(), xs.max()
    dx = x_max - x_min
    if dx < 20:
        return 0.0

    strip = max(3, int(dx * 0.02))
    left_ys = ys[xs <= x_min + strip]
    right_ys = ys[xs >= x_max - strip]

    if len(left_ys) == 0 or len(right_ys) == 0:
        return 0.0

    left_y = np.median(left_ys)
    right_y = np.median(right_ys)
    dy = right_y - left_y

    return np.degrees(np.arctan2(dy, dx))


def apply_tilt_correction(rgba_arr, max_tilt=20):
    """Apply small tilt correction. Returns (corrected_arr, rotation_deg)."""
    mask = rgba_arr[:, :, 3] > 128
    tilt = get_horizontal_tilt(mask)

    if abs(tilt) < 2 or abs(tilt) > max_tilt:
        return rgba_arr, 0.0

    rotation = round(-tilt, 1)
    img = Image.fromarray(rgba_arr, "RGBA")
    rotated = img.rotate(rotation, resample=Image.BICUBIC,
                          expand=True, fillcolor=(0, 0, 0, 0))
    return np.array(rotated), rotation


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standardize fish orientation (face left, horizontal)"
    )
    parser.add_argument("--species", type=str, nargs="*", default=None)
    parser.add_argument("--image", type=str, default=None,
                        help="Test on a single segmented PNG")
    parser.add_argument("--show", action="store_true",
                        help="Save before/after to /tmp for inspection")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report without modifying files")
    args = parser.parse_args()

    # Single image test mode
    if args.image:
        print(f"Testing: {args.image}")
        rgba = np.array(Image.open(args.image).convert("RGBA"))
        corrected, rot = apply_tilt_correction(rgba)
        print(f"  Tilt correction: {rot}°")
        if args.show:
            Image.fromarray(rgba, "RGBA").save("/tmp/oriented_before.png")
            Image.fromarray(corrected, "RGBA").save("/tmp/oriented_after.png")
            print(f"  Saved: /tmp/oriented_before.png, /tmp/oriented_after.png")
        return

    print("[1] Loading inventory...")
    inventory = load_inventory()
    annotations = load_annotations()

    species_images = {}
    for row in inventory:
        sp = row["species"]
        if sp not in species_images:
            species_images[sp] = []
        species_images[sp].append(row)

    if args.species:
        requested = set(args.species)
        species_images = {sp: rows for sp, rows in species_images.items()
                         if sp in requested}

    print(f"    {len(species_images)} species")

    # Select reference images and compute profiles
    print("[2] Selecting reference images...")
    species_ref = {}

    for species, rows in sorted(species_images.items()):
        sp_dir = species_to_dirname(species)
        seg_dir = os.path.join(SEGMENTED_DIR, sp_dir)
        ref = select_reference_image(rows, seg_dir)

        if ref:
            fname, seg_path, tier = ref
            ref_rgba = np.array(Image.open(seg_path).convert("RGBA"))
            # Tilt-correct the reference for profile computation
            ref_corrected, _ = apply_tilt_correction(ref_rgba)
            ref_mask = ref_corrected[:, :, 3] > 128
            ref_profile = compute_depth_profile(ref_mask)
            species_ref[species] = {
                "fname": fname, "tier": tier, "profile": ref_profile
            }
            print(f"    {species}: ref={fname} ({tier})")
        else:
            species_ref[species] = None
            print(f"    {species}: NO REFERENCE")

    n_with_ref = sum(1 for v in species_ref.values() if v is not None)
    print(f"    {n_with_ref}/{len(species_images)} species have references")

    # Process all images
    print("[3] Orienting fish...")
    report_rows = []
    review_rows = []
    n_rotated = 0
    n_flipped = 0
    n_unchanged = 0
    n_review = 0

    total_images = sum(len(rows) for rows in species_images.values())
    processed = 0

    for species, rows in sorted(species_images.items()):
        sp_dir = species_to_dirname(species)
        ref_info = species_ref.get(species)
        ref_profile = ref_info["profile"] if ref_info else None
        ref_fname = ref_info["fname"] if ref_info else "none"

        for row in rows:
            fname = row["filename"]
            png_name = os.path.splitext(fname)[0] + ".png"
            seg_path = os.path.join(SEGMENTED_DIR, sp_dir, png_name)

            if not os.path.exists(seg_path):
                continue

            processed += 1
            source = classify_source(fname)

            try:
                rgba = np.array(Image.open(seg_path).convert("RGBA"))

                # Step 1: Tilt correction (all sources)
                corrected, rot = apply_tilt_correction(rgba)

                # Step 2: Flip decision
                flipped = False
                confidence = 1.0
                method = "trusted_source"
                needs_review = "no"

                if source in TRUSTED_SOURCES:
                    # TRUST museum sources — don't flip
                    # But flag as needing review if it's the reference itself
                    # or if we detect it might face right
                    if ref_profile is not None and fname != ref_fname:
                        corr_mask = corrected[:, :, 3] > 128
                        tgt_profile = compute_depth_profile(corr_mask)
                        should_flip, conf = profile_flip_score(
                            tgt_profile, ref_profile)
                        confidence = conf

                        if should_flip and conf > 0.20:
                            # High-confidence mismatch with reference —
                            # flag for review but DON'T auto-flip
                            needs_review = "yes"
                            method = "trusted_flagged"
                            n_review += 1
                    else:
                        confidence = 1.0

                else:
                    # iNaturalist / other: use template matching
                    if ref_profile is not None:
                        corr_mask = corrected[:, :, 3] > 128
                        tgt_profile = compute_depth_profile(corr_mask)
                        should_flip, conf = profile_flip_score(
                            tgt_profile, ref_profile)
                        confidence = conf

                        if should_flip and conf > 0.15:
                            flipped = True
                            method = "template_flip"
                        elif should_flip:
                            # Low confidence — flag for review
                            needs_review = "yes"
                            method = "template_uncertain"
                            n_review += 1
                        else:
                            method = "template_ok"
                    else:
                        method = "no_reference"

                if flipped:
                    corrected = np.fliplr(corrected).copy()

                # Save
                did_something = abs(rot) > 0 or flipped
                if did_something and not args.dry_run:
                    Image.fromarray(corrected, "RGBA").save(seg_path)

                if abs(rot) > 0:
                    n_rotated += 1
                if flipped:
                    n_flipped += 1
                if not did_something:
                    n_unchanged += 1

                result = {
                    "filename": fname,
                    "species": species,
                    "source": source,
                    "rotation_deg": rot,
                    "flipped_horizontal": "yes" if flipped else "no",
                    "method": method,
                    "confidence": confidence,
                    "reference_image": ref_fname,
                    "needs_review": needs_review,
                }
                report_rows.append(result)
                if needs_review == "yes":
                    review_rows.append(result)

            except Exception as e:
                print(f"    Error: {fname}: {e}")

            if processed % 200 == 0:
                print(f"    ... {processed}/{total_images}")

    # Write reports
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    with open(REVIEW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(review_rows)

    action = "Would orient" if args.dry_run else "Oriented"
    print(f"\n[4] {action} {len(report_rows)} images")
    print(f"    Tilt-corrected: {n_rotated}")
    print(f"    Flipped (iNat only): {n_flipped}")
    print(f"    Unchanged: {n_unchanged}")
    print(f"    Flagged for review: {n_review}")
    print(f"    Report: {REPORT_CSV}")
    if review_rows:
        print(f"    Review list: {REVIEW_CSV}")
        print(f"\n    Review needed (trusted source may face right):")
        for r in review_rows[:20]:
            print(f"      {r['filename']:55s} source={r['source']:10s} "
                  f"conf={r['confidence']}")


if __name__ == "__main__":
    main()
