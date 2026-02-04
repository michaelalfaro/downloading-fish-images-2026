#!/usr/bin/env python3
"""
Detect potential larval/juvenile fish in segmented images.

Larvae differ from adults in several measurable ways:
  1. SIZE: Much smaller body area relative to species median
  2. SHAPE: More elongated (lower depth:length ratio); adult Chaetodontidae
     are disc-shaped with aspect ratios near 1.0
  3. COLOR: Silvery/translucent — higher L, lower a/b saturation in CIELAB
  4. TRANSPARENCY: Higher fraction of semi-transparent pixels (partial alpha)

Algorithm:
  - For each species with 3+ images, compute species-level stats
  - Flag images where multiple larval cues fire simultaneously
  - Score each image on a 0–1 "larva likelihood" scale

Usage:
    python3 flag_larvae.py                    # all species
    python3 flag_larvae.py --threshold 0.5    # custom flag threshold
    python3 flag_larvae.py --dry-run          # report only, don't add annotations

Reads:  analysis/approach_1_gmm/segmented/
Writes: analysis/approach_1_gmm/larva_flag_report.csv
        analysis/image_annotations.csv (appends, unless --dry-run)
"""

import os
import csv
import argparse
import numpy as np
from collections import defaultdict

from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR,
    load_inventory, load_annotations, species_to_dirname,
    load_segmented_image, body_pixels_to_lab, append_annotations,
    ensure_dir,
)

REPORT_CSV = os.path.join(GMM_DIR, "larva_flag_report.csv")
REPORT_FIELDS = [
    "filename", "species", "directory",
    "mask_area", "species_median_area", "area_ratio",
    "aspect_ratio", "species_median_aspect", "aspect_z",
    "depth_length_ratio", "species_median_dl_ratio", "dl_ratio_z",
    "mean_L", "species_mean_L", "L_z",
    "mean_chroma", "species_mean_chroma", "chroma_z",
    "larva_score", "flagged",
]


def measure_fish_shape(mask):
    """Measure shape properties of a fish mask.

    Returns dict with:
        mask_area: total foreground pixels
        bbox_aspect: bounding box width/height
        depth_length_ratio: max body depth / total length
    """
    ys, xs = np.where(mask)
    if len(ys) < 50:
        return None

    mask_area = int(mask.sum())
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    # Bounding box aspect ratio (width / height)
    bbox_aspect = width / max(1, height)

    # Depth-to-length ratio: how "disc-like" the fish is
    # For each column, measure the vertical extent of the mask
    # Take the maximum depth and divide by total length
    col_depths = []
    # Sample columns for speed
    n_samples = min(50, width)
    sample_cols = np.linspace(x_min, x_max, n_samples, dtype=int)
    for col in sample_cols:
        col_mask = mask[:, col]
        rows_on = np.where(col_mask)[0]
        if len(rows_on) > 0:
            col_depths.append(rows_on[-1] - rows_on[0] + 1)

    max_depth = max(col_depths) if col_depths else height
    depth_length_ratio = max_depth / max(1, width)

    return {
        "mask_area": mask_area,
        "bbox_aspect": round(bbox_aspect, 3),
        "depth_length_ratio": round(depth_length_ratio, 3),
    }


def measure_fish_color(rgb, mask):
    """Measure color properties of fish body pixels.

    Returns dict with:
        mean_L: mean luminance in CIELAB
        mean_chroma: mean chroma (sqrt(a^2 + b^2)) — low for silvery larvae
        std_ab: std of a,b channels — low for uniform silvery coloring
    """
    body_pixels = rgb[mask]
    if len(body_pixels) < 100:
        return None

    lab = body_pixels_to_lab(body_pixels)
    L_vals = lab[:, 0]
    a_vals = lab[:, 1]
    b_vals = lab[:, 2]

    # Chroma = saturation in a,b plane
    chroma = np.sqrt(a_vals ** 2 + b_vals ** 2)

    return {
        "mean_L": round(float(np.mean(L_vals)), 2),
        "mean_chroma": round(float(np.mean(chroma)), 2),
        "std_ab": round(float(np.std(np.column_stack([a_vals, b_vals]))), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Flag potential larvae")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Larva score threshold for flagging (default 0.5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only, don't modify annotations")
    args = parser.parse_args()

    print("[1] Loading inventory...")
    inventory = load_inventory()
    annotations = load_annotations()

    # Group by species
    species_images = defaultdict(list)
    for row in inventory:
        fname = row["filename"]
        # Skip already-annotated larvae (don't re-flag)
        ann = annotations.get(fname)
        if ann and ann["annotation_type"] == "larva":
            continue
        species_images[row["species"]].append(row)

    print(f"    {len(species_images)} species")

    print("[2] Measuring fish properties...")
    # First pass: collect measurements for all images
    all_measures = {}  # filename -> measurements dict

    for species, rows in sorted(species_images.items()):
        sp_dir = species_to_dirname(species)

        for row in rows:
            fname = row["filename"]
            png_name = os.path.splitext(fname)[0] + ".png"
            seg_path = os.path.join(SEGMENTED_DIR, sp_dir, png_name)

            if not os.path.exists(seg_path):
                continue

            try:
                rgb, mask = load_segmented_image(seg_path)
                shape_m = measure_fish_shape(mask)
                color_m = measure_fish_color(rgb, mask)

                if shape_m and color_m:
                    m = {**shape_m, **color_m}
                    m["species"] = species
                    m["directory"] = row["directory"]
                    all_measures[fname] = m
            except Exception as e:
                print(f"    Error measuring {fname}: {e}")

    print(f"    Measured {len(all_measures)} images")

    # Second pass: compute per-species stats and score
    print("[3] Scoring for larval characteristics...")
    species_stats = defaultdict(lambda: {
        "areas": [], "aspects": [], "dl_ratios": [],
        "L_vals": [], "chromas": [],
    })

    for fname, m in all_measures.items():
        sp = m["species"]
        species_stats[sp]["areas"].append(m["mask_area"])
        species_stats[sp]["aspects"].append(m["bbox_aspect"])
        species_stats[sp]["dl_ratios"].append(m["depth_length_ratio"])
        species_stats[sp]["L_vals"].append(m["mean_L"])
        species_stats[sp]["chromas"].append(m["mean_chroma"])

    # Compute medians and stds
    sp_summary = {}
    for sp, st in species_stats.items():
        n = len(st["areas"])
        sp_summary[sp] = {
            "n": n,
            "med_area": float(np.median(st["areas"])),
            "std_area": float(np.std(st["areas"])) if n > 2 else 0,
            "med_aspect": float(np.median(st["aspects"])),
            "std_aspect": float(np.std(st["aspects"])) if n > 2 else 0,
            "med_dl_ratio": float(np.median(st["dl_ratios"])),
            "std_dl_ratio": float(np.std(st["dl_ratios"])) if n > 2 else 0,
            "mean_L": float(np.mean(st["L_vals"])),
            "std_L": float(np.std(st["L_vals"])) if n > 2 else 0,
            "mean_chroma": float(np.mean(st["chromas"])),
            "std_chroma": float(np.std(st["chromas"])) if n > 2 else 0,
        }

    # Score each image
    results = []
    n_flagged = 0

    for fname, m in sorted(all_measures.items()):
        sp = m["species"]
        ss = sp_summary[sp]

        if ss["n"] < 3:
            # Can't reliably detect larvae with fewer than 3 images
            continue

        # --- Cue 1: Size anomaly (smaller than species typical) ---
        area_ratio = m["mask_area"] / max(1, ss["med_area"])
        # Score: 1.0 if very small (< 20% of median), 0 if normal
        size_score = max(0, 1.0 - area_ratio / 0.5) if area_ratio < 0.5 else 0.0

        # --- Cue 2: Shape anomaly (more elongated, less disc-shaped) ---
        # Larvae have lower depth_length_ratio (thinner body)
        dl_z = 0.0
        if ss["std_dl_ratio"] > 0.01:
            dl_z = (ss["med_dl_ratio"] - m["depth_length_ratio"]) / ss["std_dl_ratio"]
        # Also check aspect ratio deviation
        aspect_z = 0.0
        if ss["std_aspect"] > 0.01:
            aspect_z = abs(m["bbox_aspect"] - ss["med_aspect"]) / ss["std_aspect"]

        # Score: higher if much thinner than species norm
        shape_score = min(1.0, max(0, dl_z / 2.0))  # Positive dl_z = thinner than median

        # --- Cue 3: Color anomaly (silvery = high L, low chroma) ---
        L_z = 0.0
        if ss["std_L"] > 0.5:
            L_z = (m["mean_L"] - ss["mean_L"]) / ss["std_L"]
        chroma_z = 0.0
        if ss["std_chroma"] > 0.5:
            chroma_z = (ss["mean_chroma"] - m["mean_chroma"]) / ss["std_chroma"]

        # Score: higher if brighter AND less colorful than species norm
        color_score = 0.0
        if L_z > 1.0 and chroma_z > 0.5:
            color_score = min(1.0, (L_z - 1.0) * 0.3 + chroma_z * 0.3)
        elif chroma_z > 2.0:
            color_score = min(1.0, (chroma_z - 1.5) * 0.3)

        # --- Combined score ---
        # Multiple cues needed — any single cue could be a legitimate variant
        # Weight: size (0.4) + shape (0.35) + color (0.25)
        larva_score = size_score * 0.4 + shape_score * 0.35 + color_score * 0.25

        # Boost if multiple cues fire (more confident)
        n_cues = sum([size_score > 0.3, shape_score > 0.3, color_score > 0.3])
        if n_cues >= 2:
            larva_score = min(1.0, larva_score * 1.3)

        flagged = larva_score >= args.threshold

        result = {
            "filename": fname,
            "species": sp,
            "directory": m["directory"],
            "mask_area": m["mask_area"],
            "species_median_area": int(ss["med_area"]),
            "area_ratio": round(area_ratio, 3),
            "aspect_ratio": m["bbox_aspect"],
            "species_median_aspect": round(ss["med_aspect"], 3),
            "aspect_z": round(aspect_z, 2),
            "depth_length_ratio": m["depth_length_ratio"],
            "species_median_dl_ratio": round(ss["med_dl_ratio"], 3),
            "dl_ratio_z": round(dl_z, 2),
            "mean_L": m["mean_L"],
            "species_mean_L": round(ss["mean_L"], 2),
            "L_z": round(L_z, 2),
            "mean_chroma": m["mean_chroma"],
            "species_mean_chroma": round(ss["mean_chroma"], 2),
            "chroma_z": round(chroma_z, 2),
            "larva_score": round(larva_score, 3),
            "flagged": "yes" if flagged else "no",
        }
        results.append(result)

        if flagged:
            n_flagged += 1

    # Write report
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # Add flagged images to annotations
    if not args.dry_run and n_flagged > 0:
        new_annotations = []
        for r in results:
            if r["flagged"] == "yes":
                detail_parts = []
                if float(r["area_ratio"]) < 0.5:
                    detail_parts.append(
                        f"small_body ({r['area_ratio']}x species median)")
                if float(r["dl_ratio_z"]) > 1.5:
                    detail_parts.append(
                        f"elongated (dl_z={r['dl_ratio_z']})")
                if float(r["chroma_z"]) > 1.0:
                    detail_parts.append(
                        f"low_chroma (z={r['chroma_z']})")
                if float(r["L_z"]) > 1.5:
                    detail_parts.append(
                        f"high_luminance (L_z={r['L_z']})")

                new_annotations.append({
                    "filename": r["filename"],
                    "species": r["species"],
                    "directory": r["directory"],
                    "annotation_type": "potential_larva",
                    "annotation_detail": "; ".join(detail_parts) if detail_parts
                        else f"larva_score={r['larva_score']}",
                    "n_fish_noted": "",
                    "annotated_by": "pipeline_larva_detector",
                    "date_annotated": __import__("datetime").date.today().isoformat(),
                })
        append_annotations(new_annotations)
        print(f"    Added {len(new_annotations)} potential_larva annotations")

    # Summary
    flagged_results = [r for r in results if r["flagged"] == "yes"]
    print(f"\n[4] Larva detection complete")
    print(f"    Images scored: {len(results)}")
    print(f"    Flagged as potential larvae: {n_flagged}")
    if flagged_results:
        print(f"\n    Flagged images:")
        for r in flagged_results:
            print(f"      {r['filename']:55s}  score={r['larva_score']:.3f}  "
                  f"area_ratio={r['area_ratio']:.2f}  "
                  f"dl_z={r['dl_ratio_z']:.1f}  chroma_z={r['chroma_z']:.1f}")
    print(f"\n    Report: {REPORT_CSV}")
    if args.dry_run:
        print("    (dry-run mode — annotations not modified)")


if __name__ == "__main__":
    main()
