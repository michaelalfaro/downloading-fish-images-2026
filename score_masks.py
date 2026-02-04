#!/usr/bin/env python3
"""
Score Apple Vision segmentation masks for quality and flag problematic ones.

Detects common failure modes:
  1. FRAGMENTED MASK — black stripes / dark patches treated as background,
     splitting the fish into disconnected pieces (e.g., Amphichaetodon).
  2. BACKGROUND INCLUSION — coral, shells, substrate included as foreground
     because they are salient or adjacent to the fish.
  3. MASK TOO SMALL — Vision missed most of the fish body.
  4. MARGIN BLEED — foreground mask extends to the image edges, indicating
     likely background contamination (fish are typically centered).

Key metrics:
  - fragmentation_score: fraction of mask NOT in the largest connected component
  - solidity: mask_area / convex_hull_area — low solidity = concavities from
    bg inclusions. Fish (even long-snouted) are fairly convex (>0.6)
  - compactness: 4*pi*area/perimeter^2 — low for elongated shapes
  - margin_bleed: fraction of edge pixels that are foreground
  - mask_frac_ratio: image mask fraction vs species median

Usage:
    python3 score_masks.py                    # score all masks
    python3 score_masks.py --threshold 0.5    # custom flag threshold

Reads:  analysis/approach_1_gmm/segmented/
        analysis/approach_1_gmm/segmentation_report.csv
Writes: analysis/approach_1_gmm/mask_quality_report.csv
"""

import os
import csv
import argparse
import numpy as np
from collections import defaultdict

from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR,
    load_inventory, species_to_dirname, ensure_dir,
)

QUALITY_CSV = os.path.join(GMM_DIR, "mask_quality_report.csv")
QUALITY_FIELDS = [
    "filename", "species", "directory",
    "n_components", "largest_component_frac", "fragmentation_score",
    "compactness", "solidity", "margin_bleed_frac", "mask_fraction",
    "species_median_mask_frac", "mask_frac_ratio",
    "center_offset", "convex_hull_area",
    "quality_score", "flag_fragmented", "flag_bg_inclusion",
    "flag_small_mask", "flag_margin_bleed", "n_flags",
    "recommended_action",
]


def score_mask(seg_path, orig_shape_hw):
    """Score a segmented RGBA PNG for quality.

    Returns dict of quality metrics.
    """
    from PIL import Image
    from scipy.ndimage import label as scipy_label, binary_erosion
    from scipy.spatial import ConvexHull

    img = Image.open(seg_path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]
    h, w = alpha.shape

    binary = (alpha > 128).astype(np.uint8)
    mask_area = int(binary.sum())
    total_area = h * w
    mask_fraction = mask_area / total_area if total_area > 0 else 0

    # --- Connected components ---
    labeled, n_components = scipy_label(binary)
    component_sizes = []
    for i in range(1, n_components + 1):
        component_sizes.append(int((labeled == i).sum()))
    component_sizes.sort(reverse=True)

    largest_frac = component_sizes[0] / mask_area if mask_area > 0 else 0
    fragmentation_score = 1.0 - largest_frac

    # Use the largest component for shape analysis
    if component_sizes:
        largest_idx = 1
        best_size = 0
        for i in range(1, n_components + 1):
            s = (labeled == i).sum()
            if s > best_size:
                best_size = s
                largest_idx = i
        largest_mask = (labeled == largest_idx).astype(np.uint8)
    else:
        largest_mask = binary

    area_largest = int(largest_mask.sum())

    # --- Compactness (circularity) on largest component ---
    eroded = binary_erosion(largest_mask)
    if eroded is None:
        eroded = np.zeros_like(largest_mask)
    perimeter = int((largest_mask.astype(bool) & ~eroded.astype(bool)).sum())
    if perimeter > 0:
        compactness = 4 * np.pi * area_largest / (perimeter ** 2)
    else:
        compactness = 0

    # --- Solidity (convexity) on largest component ---
    # solidity = area / convex_hull_area
    # This distinguishes bg inclusions (low solidity, big concavities)
    # from legitimate elongated fish shapes (still high solidity).
    ys_lc, xs_lc = np.where(largest_mask > 0)
    convex_hull_area = 0
    solidity = 1.0
    if len(ys_lc) >= 10:
        try:
            points = np.column_stack([xs_lc, ys_lc])
            # Subsample for speed if many points
            if len(points) > 5000:
                idx = np.random.default_rng(42).choice(len(points), 5000,
                                                        replace=False)
                points = points[idx]
            hull = ConvexHull(points)
            convex_hull_area = hull.volume  # 2D: volume = area
            if convex_hull_area > 0:
                solidity = area_largest / convex_hull_area
        except Exception:
            solidity = 1.0

    # --- Solidity on FULL mask (all components) ---
    # If bg inclusion creates separate component, this catches it too
    ys_all, xs_all = np.where(binary > 0)
    full_solidity = 1.0
    if len(ys_all) >= 10:
        try:
            pts_all = np.column_stack([xs_all, ys_all])
            if len(pts_all) > 5000:
                idx = np.random.default_rng(42).choice(len(pts_all), 5000,
                                                        replace=False)
                pts_all = pts_all[idx]
            hull_all = ConvexHull(pts_all)
            full_hull_area = hull_all.volume
            if full_hull_area > 0:
                full_solidity = mask_area / full_hull_area
        except Exception:
            full_solidity = 1.0

    # Use the LOWER of the two solidities as the indicator
    solidity = min(solidity, full_solidity)

    # --- Margin bleed ---
    margin = max(3, int(min(h, w) * 0.02))
    top_bleed = binary[:margin, :].sum()
    bot_bleed = binary[-margin:, :].sum()
    left_bleed = binary[:, :margin].sum()
    right_bleed = binary[:, -margin:].sum()
    margin_pixels = top_bleed + bot_bleed + left_bleed + right_bleed
    margin_total = 2 * margin * w + 2 * margin * h
    margin_bleed_frac = margin_pixels / margin_total if margin_total > 0 else 0

    # --- Center offset ---
    if len(ys_all) > 0:
        cy, cx = ys_all.mean(), xs_all.mean()
        center_offset = np.sqrt(((cy - h/2) / h)**2 + ((cx - w/2) / w)**2)
    else:
        center_offset = 1.0

    return {
        "n_components": n_components,
        "largest_component_frac": round(largest_frac, 4),
        "fragmentation_score": round(fragmentation_score, 4),
        "compactness": round(compactness, 4),
        "solidity": round(solidity, 4),
        "margin_bleed_frac": round(margin_bleed_frac, 4),
        "mask_fraction": round(mask_fraction, 4),
        "center_offset": round(center_offset, 4),
        "convex_hull_area": int(convex_hull_area),
    }


def main():
    parser = argparse.ArgumentParser(description="Score segmentation masks")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Quality score threshold for flagging (default 0.5)")
    args = parser.parse_args()

    print("[1] Loading segmentation report...")
    seg_report_path = os.path.join(GMM_DIR, "segmentation_report.csv")
    seg_rows = {}
    with open(seg_report_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "success":
                seg_rows[row["filename"]] = row
    print(f"    {len(seg_rows)} successfully segmented images")

    # Group by species to compute per-species median mask fraction
    species_masks = defaultdict(list)
    for fname, row in seg_rows.items():
        species_masks[row["species"]].append(float(row["mask_fraction"]))
    species_median = {sp: np.median(fracs) for sp, fracs in species_masks.items()}

    print("[2] Scoring masks...")
    results = []

    for i, (fname, row) in enumerate(sorted(seg_rows.items())):
        species = row["species"]
        sp_dir = species_to_dirname(species)
        png_name = os.path.splitext(fname)[0] + ".png"
        seg_path = os.path.join(SEGMENTED_DIR, sp_dir, png_name)

        if not os.path.exists(seg_path):
            continue

        try:
            metrics = score_mask(seg_path, (int(row["image_height"]),
                                            int(row["image_width"])))
        except Exception as e:
            print(f"    Error scoring {fname}: {e}")
            continue

        # Per-species mask fraction ratio
        med_frac = species_median.get(species, 0.2)
        mf = metrics["mask_fraction"]
        mask_frac_ratio = mf / med_frac if med_frac > 0 else 1.0

        # --- Flags ---
        # 1. Fragmented: multiple components making up significant fraction
        flag_fragmented = (metrics["fragmentation_score"] > 0.12 and
                          metrics["n_components"] > 2)

        # 2. Background inclusion: low solidity (big concavities), OR
        #    mask is much bigger than species median + low solidity
        #    Solidity < 0.7 is suspicious for a fish-only mask.
        #    Legit fish shapes (even Forcipiger) have solidity > 0.75.
        flag_bg_inclusion = (
            metrics["solidity"] < 0.65 or
            (metrics["solidity"] < 0.75 and mask_frac_ratio > 1.4) or
            (metrics["margin_bleed_frac"] > 0.2 and mask_frac_ratio > 1.5)
        )

        # 3. Small mask: much smaller than species typical
        flag_small = mask_frac_ratio < 0.3

        # 4. Margin bleed: extensive edge contact
        flag_margin = metrics["margin_bleed_frac"] > 0.3

        n_flags = sum([flag_fragmented, flag_bg_inclusion,
                       flag_small, flag_margin])

        # --- Composite quality score (0–1, higher = better) ---
        qs = 1.0
        # Penalize fragmentation (fish should be one connected piece)
        qs -= 0.35 * metrics["fragmentation_score"]
        # Penalize low solidity (concavities from bg inclusion)
        if metrics["solidity"] < 0.85:
            qs -= 0.3 * (0.85 - metrics["solidity"])
        # Penalize margin bleed
        qs -= 0.2 * metrics["margin_bleed_frac"]
        # Penalize deviation from species median size
        size_dev = abs(np.log(max(mask_frac_ratio, 0.01)))
        qs -= 0.08 * min(size_dev, 3.0)
        # Penalize very low compactness (but less weight — legitimate elongation)
        if metrics["compactness"] < 0.2:
            qs -= 0.1 * (0.2 - metrics["compactness"])
        qs = max(0, min(1, qs))

        # Decision
        action = "ok"
        if qs < args.threshold or n_flags >= 2:
            action = "resegment_dinosam"
        elif n_flags == 1:
            action = "review"

        result_row = {
            "filename": fname,
            "species": species,
            "directory": row["directory"],
            "species_median_mask_frac": round(med_frac, 4),
            "mask_frac_ratio": round(mask_frac_ratio, 4),
            "quality_score": round(qs, 4),
            "flag_fragmented": "yes" if flag_fragmented else "no",
            "flag_bg_inclusion": "yes" if flag_bg_inclusion else "no",
            "flag_small_mask": "yes" if flag_small else "no",
            "flag_margin_bleed": "yes" if flag_margin else "no",
            "n_flags": n_flags,
            "recommended_action": action,
        }
        result_row.update(metrics)
        results.append(result_row)

        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}/{len(seg_rows)} scored")

    # Write report
    with open(QUALITY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUALITY_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Summary
    from collections import Counter
    actions = Counter(r["recommended_action"] for r in results)
    flag_counts = Counter()
    for r in results:
        for flag in ["flag_fragmented", "flag_bg_inclusion",
                     "flag_small_mask", "flag_margin_bleed"]:
            if r[flag] == "yes":
                flag_counts[flag] += 1

    qs_vals = [r['quality_score'] for r in results]
    print(f"\n[3] Quality scoring complete")
    print(f"    Images scored: {len(results)}")
    print(f"    Quality score: mean={np.mean(qs_vals):.3f}, "
          f"median={np.median(qs_vals):.3f}, min={min(qs_vals):.3f}")
    print(f"    Recommended actions:")
    for action, ct in sorted(actions.items()):
        print(f"      {action}: {ct}")
    print(f"    Flags raised:")
    for flag, ct in sorted(flag_counts.items()):
        print(f"      {flag}: {ct}")
    print(f"    Report: {QUALITY_CSV}")


if __name__ == "__main__":
    main()
