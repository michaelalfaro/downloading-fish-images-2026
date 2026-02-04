#!/usr/bin/env python3
"""
Compare Apple Vision and DINO+SAM segmentation masks, pick the best per image,
and flag images that need manual review.

For each image with both masks available, computes quality metrics for both
and selects the better one. Also flags images where NEITHER method produces
a good mask (e.g., dark background matching stripe colors).

Strategy:
  - Apple Vision is the default (generally better)
  - DINO+SAM is used as fallback when Apple Vision fails or has issues
  - Final selection goes to analysis/approach_1_gmm/segmented_final/

Reads:
  analysis/approach_1_gmm/segmented/          (Apple Vision masks)
  analysis/approach_1_gmm/segmented_dinosam/  (DINO+SAM masks)
  analysis/approach_1_gmm/mask_quality_report.csv
  analysis/approach_1_gmm/dinosam_report.csv
  analysis/approach_1_gmm/segmentation_report.csv

Writes:
  analysis/approach_1_gmm/comparison_report.csv
  analysis/approach_1_gmm/manual_review_list.csv

Usage:
    python3 compare_methods.py
"""

import os
import csv
import numpy as np
from collections import defaultdict

from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR,
    load_inventory, species_to_dirname, ensure_dir,
    get_image_path,
)

DINOSAM_DIR = os.path.join(GMM_DIR, "segmented_dinosam")
COMPARISON_CSV = os.path.join(GMM_DIR, "comparison_report.csv")
REVIEW_CSV = os.path.join(GMM_DIR, "manual_review_list.csv")
QUALITY_CSV = os.path.join(GMM_DIR, "mask_quality_report.csv")
DINOSAM_REPORT = os.path.join(GMM_DIR, "dinosam_report.csv")


def compute_mask_quality(seg_path):
    """Quick quality metrics for a single mask (shared between methods)."""
    from PIL import Image
    from scipy.ndimage import label as scipy_label, binary_erosion
    from scipy.spatial import ConvexHull

    img = Image.open(seg_path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]
    h, w = alpha.shape

    binary = (alpha > 128).astype(np.uint8)
    mask_area = int(binary.sum())
    total = h * w
    mask_frac = mask_area / total if total > 0 else 0

    # Connected components
    labeled, n_comp = scipy_label(binary)
    comp_sizes = sorted(
        [(labeled == i).sum() for i in range(1, n_comp + 1)], reverse=True
    )
    largest_frac = comp_sizes[0] / mask_area if mask_area > 0 else 0

    # Solidity
    ys, xs = np.where(binary > 0)
    solidity = 1.0
    if len(ys) >= 10:
        try:
            pts = np.column_stack([xs, ys])
            if len(pts) > 3000:
                idx = np.random.default_rng(42).choice(len(pts), 3000,
                                                        replace=False)
                pts = pts[idx]
            hull = ConvexHull(pts)
            hull_area = hull.volume
            if hull_area > 0:
                solidity = mask_area / hull_area
        except Exception:
            pass

    # Margin bleed
    margin = max(3, int(min(h, w) * 0.02))
    margin_px = (binary[:margin, :].sum() + binary[-margin:, :].sum() +
                 binary[:, :margin].sum() + binary[:, -margin:].sum())
    margin_total = 2 * margin * w + 2 * margin * h
    margin_bleed = margin_px / margin_total if margin_total > 0 else 0

    # Compactness
    largest_idx = 1
    best_size = 0
    for i in range(1, n_comp + 1):
        s = (labeled == i).sum()
        if s > best_size:
            best_size = s
            largest_idx = i
    largest_mask = (labeled == largest_idx).astype(np.uint8)
    eroded = binary_erosion(largest_mask)
    if eroded is None:
        eroded = np.zeros_like(largest_mask)
    perim = int((largest_mask.astype(bool) & ~eroded.astype(bool)).sum())
    compactness = 4 * np.pi * best_size / (perim ** 2) if perim > 0 else 0

    return {
        "mask_frac": mask_frac,
        "n_components": n_comp,
        "largest_frac": largest_frac,
        "solidity": solidity,
        "compactness": compactness,
        "margin_bleed": margin_bleed,
    }


def detect_dark_bg_stripe_conflict(image_path):
    """Detect if an image has a dark background that matches stripe colors.

    This is the unsolvable case: segmentation cannot distinguish the fish
    body from background when they share the same color.

    Returns (is_problematic: bool, confidence: float).
    """
    from PIL import Image

    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    gray = np.mean(img, axis=2)

    # Check overall image darkness
    mean_brightness = gray.mean()

    # Sample margin pixels (definitely background)
    margin = max(5, int(min(h, w) * 0.05))
    bg_samples = np.concatenate([
        gray[:margin, :].ravel(),
        gray[-margin:, :].ravel(),
        gray[:, :margin].ravel(),
        gray[:, -margin:].ravel(),
    ])
    bg_mean = bg_samples.mean()
    bg_dark = bg_mean < 60  # dark background

    if not bg_dark:
        return False, 0.0

    # Check if there are dark stripes in the center region
    center = gray[h//4:3*h//4, w//4:3*w//4]
    # Bimodal test: do we have both dark (<60) and bright (>120) pixels?
    dark_frac = (center < 60).mean()
    bright_frac = (center > 120).mean()

    # Problematic if background is dark AND image has substantial
    # dark regions mixed with bright (= dark stripes on dark bg)
    if dark_frac > 0.2 and bright_frac > 0.15:
        confidence = min(1.0, dark_frac * 2)
        return True, round(confidence, 3)

    return False, 0.0


def main():
    print("[1] Loading reports...")
    inventory = load_inventory()
    inv_by_fname = {r["filename"]: r for r in inventory}

    # Load Apple Vision quality report
    apple_quality = {}
    if os.path.exists(QUALITY_CSV):
        with open(QUALITY_CSV, newline="") as f:
            for row in csv.DictReader(f):
                apple_quality[row["filename"]] = row

    # Load Apple Vision segmentation report
    apple_seg = {}
    seg_report = os.path.join(GMM_DIR, "segmentation_report.csv")
    if os.path.exists(seg_report):
        with open(seg_report, newline="") as f:
            for row in csv.DictReader(f):
                apple_seg[row["filename"]] = row

    # Load DINO+SAM report
    dino_seg = {}
    if os.path.exists(DINOSAM_REPORT):
        with open(DINOSAM_REPORT, newline="") as f:
            for row in csv.DictReader(f):
                dino_seg[row["filename"]] = row

    print(f"    Apple Vision masks: {len(apple_quality)}")
    print(f"    DINO+SAM masks: {len(dino_seg)}")

    # Images that were flagged or failed Apple Vision
    flagged = set()
    for fname, row in apple_quality.items():
        if row["recommended_action"] in ("resegment_dinosam", "review"):
            flagged.add(fname)
    for fname, row in apple_seg.items():
        if row["status"] == "no_detection":
            flagged.add(fname)

    print(f"    Flagged/no-detection: {len(flagged)}")

    print("[2] Comparing methods for flagged images...")
    comparison = []
    review_list = []

    for fname in sorted(flagged):
        inv_row = inv_by_fname.get(fname)
        if not inv_row:
            continue

        species = inv_row["species"]
        sp_dir = species_to_dirname(species)
        png_name = os.path.splitext(fname)[0] + ".png"

        apple_path = os.path.join(SEGMENTED_DIR, sp_dir, png_name)
        dino_path = os.path.join(DINOSAM_DIR, sp_dir, png_name)

        has_apple = os.path.exists(apple_path)
        has_dino = os.path.exists(dino_path)
        dino_status = dino_seg.get(fname, {}).get("status", "not_run")

        # Score both if available
        apple_metrics = None
        dino_metrics = None
        if has_apple:
            try:
                apple_metrics = compute_mask_quality(apple_path)
            except Exception:
                pass
        if has_dino and dino_status == "success":
            try:
                dino_metrics = compute_mask_quality(dino_path)
            except Exception:
                pass

        # Check for dark-bg stripe conflict
        image_path = get_image_path(inv_row)
        is_dark_stripe, dark_conf = detect_dark_bg_stripe_conflict(image_path)

        # --- Selection logic ---
        selected = "apple_vision"  # default
        reason = "default"

        if not has_apple and has_dino and dino_status == "success":
            selected = "dinosam"
            reason = "apple_failed"
        elif has_apple and apple_metrics and dino_metrics:
            # Compare: Apple wins unless it has clear problems AND DINO is better
            apple_ok = (apple_metrics["solidity"] > 0.7 and
                       apple_metrics["largest_frac"] > 0.85)
            dino_ok = (dino_metrics["solidity"] > 0.7 and
                      dino_metrics["largest_frac"] > 0.85)

            if apple_ok:
                selected = "apple_vision"
                reason = "apple_acceptable"
            elif dino_ok and not apple_ok:
                # DINO better only if it's actually good AND Apple has problems
                if (dino_metrics["solidity"] > apple_metrics["solidity"] + 0.1 and
                        dino_metrics["margin_bleed"] < apple_metrics["margin_bleed"]):
                    selected = "dinosam"
                    reason = "dino_better_solidity"
                else:
                    selected = "apple_vision"
                    reason = "apple_preferred"
            else:
                selected = "apple_vision"
                reason = "neither_great"
        elif has_apple:
            selected = "apple_vision"
            reason = "only_apple"
        elif not has_apple and not has_dino:
            selected = "none"
            reason = "both_failed"

        # Flag for manual review
        needs_review = False
        review_reason = []

        if selected == "none":
            needs_review = True
            review_reason.append("no_mask_available")
        if is_dark_stripe:
            needs_review = True
            review_reason.append(f"dark_bg_stripe_conflict({dark_conf})")
        if apple_metrics and apple_metrics["solidity"] < 0.65:
            needs_review = True
            review_reason.append(f"low_solidity({apple_metrics['solidity']:.2f})")
        if apple_metrics and apple_metrics["largest_frac"] < 0.8:
            needs_review = True
            review_reason.append("fragmented_mask")

        comp_row = {
            "filename": fname,
            "species": species,
            "directory": inv_row["directory"],
            "apple_available": "yes" if has_apple else "no",
            "dino_available": "yes" if (has_dino and dino_status == "success") else "no",
            "apple_solidity": round(apple_metrics["solidity"], 4) if apple_metrics else "",
            "apple_mask_frac": round(apple_metrics["mask_frac"], 4) if apple_metrics else "",
            "dino_solidity": round(dino_metrics["solidity"], 4) if dino_metrics else "",
            "dino_mask_frac": round(dino_metrics["mask_frac"], 4) if dino_metrics else "",
            "dark_bg_stripe": "yes" if is_dark_stripe else "no",
            "selected_method": selected,
            "selection_reason": reason,
            "needs_review": "yes" if needs_review else "no",
            "review_reasons": "; ".join(review_reason) if review_reason else "",
        }
        comparison.append(comp_row)

        if needs_review:
            review_list.append({
                "filename": fname,
                "species": species,
                "directory": inv_row["directory"],
                "reasons": "; ".join(review_reason),
                "apple_path": apple_path if has_apple else "",
                "dino_path": dino_path if (has_dino and dino_status == "success") else "",
                "original_path": image_path,
            })

    # Write comparison report
    comp_fields = [
        "filename", "species", "directory",
        "apple_available", "dino_available",
        "apple_solidity", "apple_mask_frac",
        "dino_solidity", "dino_mask_frac",
        "dark_bg_stripe", "selected_method", "selection_reason",
        "needs_review", "review_reasons",
    ]
    with open(COMPARISON_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comp_fields)
        writer.writeheader()
        writer.writerows(comparison)

    # Write manual review list
    review_fields = [
        "filename", "species", "directory", "reasons",
        "apple_path", "dino_path", "original_path",
    ]
    with open(REVIEW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows(review_list)

    # Summary
    from collections import Counter
    methods = Counter(r["selected_method"] for r in comparison)
    n_review = sum(1 for r in comparison if r["needs_review"] == "yes")
    n_dark = sum(1 for r in comparison if r["dark_bg_stripe"] == "yes")

    print(f"\n[3] Comparison complete")
    print(f"    Images compared: {len(comparison)}")
    print(f"    Selected method:")
    for method, ct in sorted(methods.items()):
        print(f"      {method}: {ct}")
    print(f"    Dark bg + stripe conflicts: {n_dark}")
    print(f"    Need manual review: {n_review}")
    print(f"    Reports: {COMPARISON_CSV}")
    print(f"             {REVIEW_CSV}")


if __name__ == "__main__":
    main()
