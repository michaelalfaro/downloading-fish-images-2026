#!/usr/bin/env python3
"""
Fit Gaussian Mixture Models per species to determine optimal k (number of
color classes) using BIC model selection.

For each species:
1. Pool body pixels from all normalized segmented images
2. Convert to CIELAB, sample up to 50,000 pixels
3. Fit GMM with k=2..8, select k with lowest BIC
4. Save fitted models for downstream zone map generation

Reads: analysis/approach_1_gmm/normalized/
Writes: analysis/approach_1_gmm/k_selection/species_k_selection.csv
        analysis/approach_1_gmm/k_selection/bic_curves/
        analysis/approach_1_gmm/k_selection/models/ (joblib)

Usage:
    python3 fit_gmm_species.py
    python3 fit_gmm_species.py --species "Chaetodon auriga"
    python3 fit_gmm_species.py --k-max 6   # try k=2..6 instead of 2..8
"""

import os
import csv
import json
import argparse
import numpy as np
from collections import defaultdict

from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, NORMALIZED_DIR, K_SELECTION_DIR,
    load_inventory, load_annotations, species_to_dirname,
    load_segmented_image, body_pixels_to_lab, all_species, ensure_dir,
)

K_SELECTION_CSV = os.path.join(K_SELECTION_DIR, "species_k_selection.csv")
MODELS_DIR = os.path.join(K_SELECTION_DIR, "models")
BIC_CURVES_DIR = os.path.join(K_SELECTION_DIR, "bic_curves")

MAX_PIXELS = 50000
K_MIN = 2
K_MAX_DEFAULT = 8
N_INIT = 5
MAX_ITER = 200
RANDOM_STATE = 42


def collect_species_pixels(species, rows, annotations):
    """Collect body pixels from all normalized images for a species.

    Returns (pixels_lab Nx3, image_counts list).
    """
    sp_dir = species_to_dirname(species)
    norm_dir = os.path.join(NORMALIZED_DIR, sp_dir)

    all_pixels = []
    image_counts = []

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

        lab = body_pixels_to_lab(body_pixels)
        all_pixels.append(lab)
        image_counts.append(len(lab))

    if not all_pixels:
        return None, []

    return np.vstack(all_pixels), image_counts


def sample_pixels(pixels, max_n, rng):
    """Subsample pixels if there are more than max_n."""
    if len(pixels) <= max_n:
        return pixels
    idx = rng.choice(len(pixels), size=max_n, replace=False)
    return pixels[idx]


def fit_gmm_sweep(pixels, k_min, k_max, n_init, max_iter, random_state):
    """Fit GMMs for k=k_min..k_max. Returns dict of k -> (model, bic)."""
    from sklearn.mixture import GaussianMixture

    results = {}
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        gmm.fit(pixels)
        bic = gmm.bic(pixels)
        results[k] = (gmm, bic)

    return results


def plot_bic_curve(species, bic_values, k_selected, output_path):
    """Save a BIC vs k plot for a species."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks = sorted(bic_values.keys())
        bics = [bic_values[k] for k in ks]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(ks, bics, "o-", color="#2c3e50", linewidth=1.5, markersize=6)
        ax.axvline(k_selected, color="#e74c3c", linestyle="--", alpha=0.7,
                   label=f"selected k={k_selected}")
        ax.set_xlabel("Number of components (k)")
        ax.set_ylabel("BIC")
        ax.set_title(species.replace("_", " "), style="italic", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xticks(ks)
        fig.tight_layout()
        fig.savefig(output_path, dpi=100)
        plt.close(fig)
    except Exception:
        pass  # plotting is optional


def main():
    parser = argparse.ArgumentParser(description="Fit GMM per species")
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--k-max", type=int, default=K_MAX_DEFAULT)
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip BIC curve plots")
    args = parser.parse_args()

    ensure_dir(MODELS_DIR)
    ensure_dir(BIC_CURVES_DIR)

    print("[1] Loading inventory and annotations...")
    inventory = load_inventory()
    annotations = load_annotations()

    species_images = defaultdict(list)
    for row in inventory:
        species_images[row["species"]].append(row)

    if args.species:
        species_images = {args.species: species_images.get(args.species, [])}

    print(f"    {len(species_images)} species to process")

    print("[2] Fitting GMMs per species...")
    report_rows = []
    rng = np.random.RandomState(RANDOM_STATE)

    for i, (species, rows) in enumerate(sorted(species_images.items())):
        pixels, image_counts = collect_species_pixels(
            species, rows, annotations)

        if pixels is None or len(pixels) < 200:
            print(f"  [{i+1:3d}] {species} -- SKIPPED (too few pixels)")
            continue

        n_total = len(pixels)
        sampled = sample_pixels(pixels, MAX_PIXELS, rng)
        n_sampled = len(sampled)

        # Fit GMM sweep
        results = fit_gmm_sweep(
            sampled, K_MIN, args.k_max, N_INIT, MAX_ITER, RANDOM_STATE)

        # Select best k by BIC
        bic_values = {k: bic for k, (model, bic) in results.items()}
        best_k = min(bic_values, key=bic_values.get)
        best_model = results[best_k][0]

        # BIC delta (confidence measure)
        sorted_bics = sorted(bic_values.values())
        bic_delta = sorted_bics[1] - sorted_bics[0] if len(sorted_bics) > 1 else 0

        # Cluster means as JSON
        cluster_means = best_model.means_.tolist()
        cluster_means_json = json.dumps(
            [{"L": round(m[0], 1), "a": round(m[1], 1), "b": round(m[2], 1)}
             for m in cluster_means])

        # Save model
        import joblib
        sp_safe = species_to_dirname(species)
        model_path = os.path.join(MODELS_DIR, f"{sp_safe}_gmm.joblib")
        joblib.dump(best_model, model_path)

        # BIC curve plot
        if not args.no_plots:
            plot_path = os.path.join(BIC_CURVES_DIR, f"{sp_safe}_bic.png")
            plot_bic_curve(species, bic_values, best_k, plot_path)

        # Report row
        row_data = {
            "species": species,
            "n_images": len(image_counts),
            "n_pixels_total": n_total,
            "n_pixels_sampled": n_sampled,
            "k_selected": best_k,
            "bic_delta": round(bic_delta, 1),
            "cluster_means": cluster_means_json,
        }
        for k in range(K_MIN, args.k_max + 1):
            row_data[f"bic_k{k}"] = round(bic_values.get(k, 0), 1)

        report_rows.append(row_data)
        print(f"  [{i+1:3d}] {species} -- k={best_k}, "
              f"n_img={len(image_counts)}, pixels={n_sampled}")

    # Write report
    fieldnames = ["species", "n_images", "n_pixels_total", "n_pixels_sampled",
                  "k_selected", "bic_delta", "cluster_means"]
    for k in range(K_MIN, args.k_max + 1):
        fieldnames.append(f"bic_k{k}")

    with open(K_SELECTION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    # Summary
    if report_rows:
        ks = [r["k_selected"] for r in report_rows]
        print(f"\n[3] GMM fitting complete")
        print(f"    Species processed: {len(report_rows)}")
        print(f"    k distribution: min={min(ks)}, max={max(ks)}, "
              f"median={sorted(ks)[len(ks)//2]}")
        print(f"    Report: {K_SELECTION_CSV}")
        print(f"    Models: {MODELS_DIR}")
    else:
        print("\n[3] No species processed.")


if __name__ == "__main__":
    main()
