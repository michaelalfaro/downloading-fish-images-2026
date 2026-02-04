#!/usr/bin/env python3
"""
Detect outlier images within each species pool.

For each species (3+ images), computes a 6D color feature vector per image
(mean and std of L, a, b in CIELAB space) and flags images that are
statistically distant from the species consensus. Catches:
  - Missed larvae or juveniles
  - Sexual dimorphism
  - Wrong species contamination
  - Extreme lighting anomalies

Reads: segmented PNGs from analysis/approach_1_gmm/segmented/
Writes: analysis/approach_1_gmm/outlier_report.csv
Updates: analysis/image_annotations.csv (appends outlier flags)

Usage:
    python3 detect_outliers.py
"""

import os
import csv
import numpy as np
from collections import defaultdict

from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR, ANNOTATIONS_CSV,
    load_inventory, load_annotations, append_annotations,
    species_to_dirname, load_segmented_image, body_pixels_to_lab,
    all_species,
)

OUTLIER_CSV = os.path.join(GMM_DIR, "outlier_report.csv")
REPORT_FIELDS = [
    "filename", "species", "directory",
    "mean_L", "mean_a", "mean_b", "std_L", "std_a", "std_b",
    "distance", "distance_method", "species_n_images",
    "distance_zscore",
    "flag_color_outlier", "flag_small_mask", "flag_large_mask",
    "flag_luminance", "n_flags", "suggested_action",
]

ZSCORE_THRESHOLD = 2.5  # flag images > 2.5 SD from species mean distance
MASK_SMALL_FRAC = 0.3   # mask area < 30% of species median
MASK_LARGE_FRAC = 2.5   # mask area > 250% of species median
LUM_ZSCORE = 2.0         # mean L > 2 SD from species mean L


def compute_image_features(png_path):
    """Compute 6D color feature vector for a segmented image.

    Returns dict with mean_L, mean_a, mean_b, std_L, std_a, std_b, n_pixels.
    Returns None if image can't be loaded or has no body pixels.
    """
    try:
        rgb, mask = load_segmented_image(png_path)
    except Exception:
        return None

    body_pixels = rgb[mask]
    if len(body_pixels) < 100:
        return None

    lab = body_pixels_to_lab(body_pixels)

    return {
        "mean_L": float(np.mean(lab[:, 0])),
        "mean_a": float(np.mean(lab[:, 1])),
        "mean_b": float(np.mean(lab[:, 2])),
        "std_L": float(np.std(lab[:, 0])),
        "std_a": float(np.std(lab[:, 1])),
        "std_b": float(np.std(lab[:, 2])),
        "n_pixels": len(body_pixels),
    }


def mahalanobis_distances(features_matrix):
    """Compute Mahalanobis distance of each row from the centroid.

    Falls back to Euclidean if covariance is singular.
    Returns (distances, method_str).
    """
    from scipy.spatial.distance import mahalanobis

    centroid = np.mean(features_matrix, axis=0)
    n = features_matrix.shape[0]

    if n < features_matrix.shape[1] + 2:
        # Not enough samples for reliable covariance — use Euclidean
        dists = np.linalg.norm(features_matrix - centroid, axis=1)
        return dists, "euclidean"

    cov = np.cov(features_matrix, rowvar=False)
    try:
        cov_inv = np.linalg.inv(cov)
        dists = np.array([mahalanobis(row, centroid, cov_inv)
                          for row in features_matrix])
        return dists, "mahalanobis"
    except np.linalg.LinAlgError:
        dists = np.linalg.norm(features_matrix - centroid, axis=1)
        return dists, "euclidean"


def main():
    print("[1] Loading segmentation report...")
    seg_report_path = os.path.join(GMM_DIR, "segmentation_report.csv")
    seg_rows = {}
    with open(seg_report_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "success":
                seg_rows[row["filename"]] = row

    print(f"    {len(seg_rows)} successfully segmented images")

    # Group by species
    species_images = defaultdict(list)
    for fname, row in seg_rows.items():
        species_images[row["species"]].append(row)

    print(f"    {len(species_images)} species")

    print("[2] Computing per-image color features...")
    # Compute features for all images
    image_features = {}  # filename -> features dict
    for i, (species, rows) in enumerate(sorted(species_images.items())):
        sp_dir = species_to_dirname(species)
        for row in rows:
            png_name = os.path.splitext(row["filename"])[0] + ".png"
            png_path = os.path.join(SEGMENTED_DIR, sp_dir, png_name)
            if not os.path.exists(png_path):
                continue
            feats = compute_image_features(png_path)
            if feats:
                feats["mask_area"] = int(row["selected_mask_area"])
                image_features[row["filename"]] = feats

        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(species_images)} species")

    print(f"    Computed features for {len(image_features)} images")

    print("[3] Detecting outliers...")
    outlier_rows = []
    n_color_outliers = 0
    n_small_mask = 0
    n_large_mask = 0
    n_luminance = 0
    n_too_few = 0

    for species, rows in sorted(species_images.items()):
        fnames = [r["filename"] for r in rows if r["filename"] in image_features]
        n_images = len(fnames)

        if n_images < 3:
            # Can't do meaningful outlier detection
            n_too_few += 1
            for fname in fnames:
                feats = image_features.get(fname, {})
                row_data = seg_rows[fname]
                outlier_rows.append({
                    "filename": fname,
                    "species": species,
                    "directory": row_data["directory"],
                    "mean_L": round(feats.get("mean_L", 0), 2),
                    "mean_a": round(feats.get("mean_a", 0), 2),
                    "mean_b": round(feats.get("mean_b", 0), 2),
                    "std_L": round(feats.get("std_L", 0), 2),
                    "std_a": round(feats.get("std_a", 0), 2),
                    "std_b": round(feats.get("std_b", 0), 2),
                    "distance": 0,
                    "distance_method": "too_few",
                    "species_n_images": n_images,
                    "distance_zscore": 0,
                    "flag_color_outlier": "no",
                    "flag_small_mask": "no",
                    "flag_large_mask": "no",
                    "flag_luminance": "no",
                    "n_flags": 0,
                    "suggested_action": "manual_review",
                })
            continue

        # Build feature matrix (6D: mean_L, mean_a, mean_b, std_L, std_a, std_b)
        feat_matrix = np.array([
            [image_features[f]["mean_L"], image_features[f]["mean_a"],
             image_features[f]["mean_b"], image_features[f]["std_L"],
             image_features[f]["std_a"], image_features[f]["std_b"]]
            for f in fnames
        ])

        # Compute distances
        distances, method = mahalanobis_distances(feat_matrix)

        # Compute z-scores of distances
        dist_mean = np.mean(distances)
        dist_std = np.std(distances)
        if dist_std > 0:
            zscores = (distances - dist_mean) / dist_std
        else:
            zscores = np.zeros_like(distances)

        # Mask area statistics for heuristic flags
        mask_areas = np.array([image_features[f]["mask_area"] for f in fnames])
        median_area = np.median(mask_areas)

        # Mean L statistics
        mean_Ls = np.array([image_features[f]["mean_L"] for f in fnames])
        L_mean = np.mean(mean_Ls)
        L_std = np.std(mean_Ls)

        for j, fname in enumerate(fnames):
            feats = image_features[fname]
            row_data = seg_rows[fname]

            flag_color = "yes" if zscores[j] > ZSCORE_THRESHOLD else "no"
            flag_small = "yes" if (median_area > 0 and
                                   feats["mask_area"] < median_area * MASK_SMALL_FRAC) else "no"
            flag_large = "yes" if (median_area > 0 and
                                   feats["mask_area"] > median_area * MASK_LARGE_FRAC) else "no"
            flag_lum = "no"
            if L_std > 0:
                l_zscore = abs(feats["mean_L"] - L_mean) / L_std
                flag_lum = "yes" if l_zscore > LUM_ZSCORE else "no"

            flags = sum(1 for f in [flag_color, flag_small, flag_large, flag_lum]
                        if f == "yes")

            if flag_color == "yes":
                n_color_outliers += 1
            if flag_small == "yes":
                n_small_mask += 1
            if flag_large == "yes":
                n_large_mask += 1
            if flag_lum == "yes":
                n_luminance += 1

            action = "review" if flags > 0 else "ok"

            outlier_rows.append({
                "filename": fname,
                "species": species,
                "directory": row_data["directory"],
                "mean_L": round(feats["mean_L"], 2),
                "mean_a": round(feats["mean_a"], 2),
                "mean_b": round(feats["mean_b"], 2),
                "std_L": round(feats["std_L"], 2),
                "std_a": round(feats["std_a"], 2),
                "std_b": round(feats["std_b"], 2),
                "distance": round(distances[j], 3),
                "distance_method": method,
                "species_n_images": n_images,
                "distance_zscore": round(zscores[j], 3),
                "flag_color_outlier": flag_color,
                "flag_small_mask": flag_small,
                "flag_large_mask": flag_large,
                "flag_luminance": flag_lum,
                "n_flags": flags,
                "suggested_action": action,
            })

    # Write report
    with open(OUTLIER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(outlier_rows)

    # Append outlier flags to annotations
    from datetime import date
    new_annotations = []
    for row in outlier_rows:
        if row["n_flags"] > 0:
            flags_detail = []
            if row["flag_color_outlier"] == "yes":
                flags_detail.append("color_outlier")
            if row["flag_small_mask"] == "yes":
                flags_detail.append("small_mask")
            if row["flag_large_mask"] == "yes":
                flags_detail.append("large_mask")
            if row["flag_luminance"] == "yes":
                flags_detail.append("luminance_anomaly")
            new_annotations.append({
                "filename": row["filename"],
                "species": row["species"],
                "directory": row["directory"],
                "annotation_type": "outlier",
                "annotation_detail": "; ".join(flags_detail),
                "n_fish_noted": "",
                "annotated_by": "pipeline",
                "date_annotated": date.today().isoformat(),
            })

    if new_annotations:
        append_annotations(new_annotations)

    # Summary
    n_flagged = sum(1 for r in outlier_rows if r["n_flags"] > 0)
    print(f"\n[4] Outlier detection complete")
    print(f"    Images analyzed: {len(outlier_rows)}")
    print(f"    Species with < 3 images (manual review): {n_too_few}")
    print(f"    Flags raised:")
    print(f"      Color outliers:    {n_color_outliers}")
    print(f"      Small mask:        {n_small_mask}")
    print(f"      Large mask:        {n_large_mask}")
    print(f"      Luminance anomaly: {n_luminance}")
    print(f"    Total images flagged: {n_flagged}")
    print(f"    Report: {OUTLIER_CSV}")
    print(f"    Annotations updated: {ANNOTATIONS_CSV}")


if __name__ == "__main__":
    main()
