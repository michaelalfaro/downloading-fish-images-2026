#!/usr/bin/env python3
"""
Color cast correction for FishPix images using Bishop/Randall as reference.

Strategy:
1. Estimate global color shift by comparing mean a,b values between sources
2. Apply correction to a,b channels in LAB space (preserving L)
3. Can also do per-species correction if both sources exist for that species

Usage:
    python3 correct_color_cast.py --analyze          # Show color statistics
    python3 correct_color_cast.py --correct          # Apply correction to segmented images
    python3 correct_color_cast.py --preview SPECIES  # Preview correction for one species
"""

import os
import csv
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils")); from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR, NORMALIZED_DIR,
    load_inventory, species_to_dirname, dirname_to_species, ensure_dir,
)

# Output directory for color-corrected images
CORRECTED_DIR = os.path.join(GMM_DIR, "color_corrected")
CORRECTION_REPORT = os.path.join(GMM_DIR, "color_correction_report.csv")


def load_segmented_lab(path, max_pixels=50000):
    """Load segmented image and return LAB pixels + mask."""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    mask = arr[:, :, 3] > 0
    rgb = arr[:, :, :3]

    # Convert entire image to LAB
    rgb_float = rgb.astype(np.float64) / 255.0
    lab = rgb2lab(rgb_float)

    # Get body pixel values for stats
    body_pixels = lab[mask]
    if len(body_pixels) > max_pixels:
        idx = np.random.choice(len(body_pixels), max_pixels, replace=False)
        body_pixels = body_pixels[idx]

    return lab, mask, body_pixels, arr[:, :, 3]  # return alpha too


def get_mean_ab(body_pixels):
    """Get mean a,b values from LAB body pixels."""
    return np.mean(body_pixels[:, 1]), np.mean(body_pixels[:, 2])


def apply_ab_correction(lab_img, mask, delta_a, delta_b):
    """Apply a,b shift to LAB image, only on masked pixels."""
    corrected = lab_img.copy()

    # Shift a,b channels
    corrected[mask, 1] += delta_a
    corrected[mask, 2] += delta_b

    # Clamp to valid LAB ranges
    corrected[:, :, 1] = np.clip(corrected[:, :, 1], -128, 127)
    corrected[:, :, 2] = np.clip(corrected[:, :, 2], -128, 127)

    return corrected


def lab_to_rgba(lab_img, alpha):
    """Convert LAB image back to RGBA."""
    rgb_float = lab2rgb(lab_img)
    rgb_uint8 = np.clip(rgb_float * 255, 0, 255).astype(np.uint8)

    # Add alpha channel back
    rgba = np.zeros((rgb_uint8.shape[0], rgb_uint8.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb_uint8
    rgba[:, :, 3] = alpha

    return rgba


def analyze_sources(inventory):
    """Analyze color distributions by source."""
    source_stats = defaultdict(list)

    for row in inventory:
        source = row["source"]
        if source not in ("FishPix", "BishopMuseum", "FishBase", "FishBaseUser", "iNaturalist"):
            continue

        sp_dir = species_to_dirname(row["species"])
        seg_path = os.path.join(SEGMENTED_DIR, sp_dir,
                               os.path.splitext(row["filename"])[0] + ".png")

        if not os.path.exists(seg_path):
            continue

        try:
            _, _, body_pixels, _ = load_segmented_lab(seg_path, max_pixels=10000)
            mean_a, mean_b = get_mean_ab(body_pixels)
            source_stats[source].append({
                'filename': row["filename"],
                'species': row["species"],
                'mean_a': mean_a,
                'mean_b': mean_b,
            })
        except Exception as e:
            pass

    return source_stats


def compute_correction_params(source_stats, reference="BishopMuseum"):
    """Compute correction parameters using Bishop/Randall as reference."""
    if reference not in source_stats or not source_stats[reference]:
        print(f"Warning: No {reference} images found for reference")
        return {}

    ref_stats = source_stats[reference]
    ref_mean_a = np.mean([s['mean_a'] for s in ref_stats])
    ref_mean_b = np.mean([s['mean_b'] for s in ref_stats])

    corrections = {}

    for source, stats in source_stats.items():
        if source == reference:
            corrections[source] = {'delta_a': 0, 'delta_b': 0}
            continue

        src_mean_a = np.mean([s['mean_a'] for s in stats])
        src_mean_b = np.mean([s['mean_b'] for s in stats])

        # Correction = reference - source (to shift source toward reference)
        delta_a = ref_mean_a - src_mean_a
        delta_b = ref_mean_b - src_mean_b

        corrections[source] = {
            'delta_a': delta_a,
            'delta_b': delta_b,
            'n_images': len(stats),
        }

    return corrections, {'mean_a': ref_mean_a, 'mean_b': ref_mean_b}


def correct_image(seg_path, output_path, delta_a, delta_b):
    """Apply color correction to a segmented image."""
    lab_img, mask, _, alpha = load_segmented_lab(seg_path)

    corrected_lab = apply_ab_correction(lab_img, mask, delta_a, delta_b)
    corrected_rgba = lab_to_rgba(corrected_lab, alpha)

    ensure_dir(os.path.dirname(output_path))
    Image.fromarray(corrected_rgba).save(output_path)


def run_analysis():
    """Run color analysis and show statistics."""
    print("Loading inventory...")
    inventory = load_inventory()

    print("Analyzing color distributions by source...\n")
    source_stats = analyze_sources(inventory)

    print("=" * 70)
    print(f"{'Source':<15} {'N':>6} {'mean a':>10} {'std a':>10} {'mean b':>10} {'std b':>10}")
    print("=" * 70)

    for source in ['BishopMuseum', 'FishBase', 'FishPix', 'iNaturalist']:
        if source not in source_stats:
            continue
        stats = source_stats[source]
        a_vals = [s['mean_a'] for s in stats]
        b_vals = [s['mean_b'] for s in stats]
        print(f"{source:<15} {len(stats):>6} {np.mean(a_vals):>10.2f} {np.std(a_vals):>10.2f} "
              f"{np.mean(b_vals):>10.2f} {np.std(b_vals):>10.2f}")

    print("=" * 70)

    # Compute corrections
    corrections, ref = compute_correction_params(source_stats)

    print(f"\nReference (Bishop/Randall): a={ref['mean_a']:.2f}, b={ref['mean_b']:.2f}")
    print("\nRecommended corrections (to match Bishop/Randall):")
    print("-" * 50)

    for source, corr in corrections.items():
        if source == "BishopMuseum":
            continue
        print(f"  {source}: Δa = {corr['delta_a']:+.2f}, Δb = {corr['delta_b']:+.2f}")

    print("\nNote: FishPix shows a blue cast (lower b values).")
    print("Correction would add ~7-8 to the b channel (toward yellow).")

    return source_stats, corrections


def run_correction(source_to_correct="FishPix", dry_run=False):
    """Apply color correction to all images from a source."""
    print("Loading inventory...")
    inventory = load_inventory()

    print("Analyzing color distributions...")
    source_stats = analyze_sources(inventory)
    corrections, ref = compute_correction_params(source_stats)

    if source_to_correct not in corrections:
        print(f"No correction computed for {source_to_correct}")
        return

    corr = corrections[source_to_correct]
    delta_a = corr['delta_a']
    delta_b = corr['delta_b']

    print(f"\nApplying correction to {source_to_correct}:")
    print(f"  Δa = {delta_a:+.2f}")
    print(f"  Δb = {delta_b:+.2f}")

    if dry_run:
        print("\n[DRY RUN - no files modified]")
        return

    # Find all images from this source
    to_correct = [r for r in inventory if r["source"] == source_to_correct]
    print(f"\nCorrecting {len(to_correct)} images...")

    report = []
    n_success = 0
    n_skip = 0

    for row in to_correct:
        sp_dir = species_to_dirname(row["species"])
        seg_path = os.path.join(SEGMENTED_DIR, sp_dir,
                               os.path.splitext(row["filename"])[0] + ".png")

        if not os.path.exists(seg_path):
            n_skip += 1
            continue

        output_path = os.path.join(CORRECTED_DIR, sp_dir,
                                   os.path.splitext(row["filename"])[0] + ".png")

        try:
            # Get original stats
            _, _, body_pixels, _ = load_segmented_lab(seg_path, max_pixels=10000)
            orig_a, orig_b = get_mean_ab(body_pixels)

            correct_image(seg_path, output_path, delta_a, delta_b)

            # Get corrected stats
            _, _, body_pixels, _ = load_segmented_lab(output_path, max_pixels=10000)
            new_a, new_b = get_mean_ab(body_pixels)

            report.append({
                'filename': row["filename"],
                'species': row["species"],
                'source': row["source"],
                'orig_a': orig_a,
                'orig_b': orig_b,
                'new_a': new_a,
                'new_b': new_b,
                'delta_a': delta_a,
                'delta_b': delta_b,
            })
            n_success += 1

            if n_success % 20 == 0:
                print(f"  Processed {n_success}...")

        except Exception as e:
            print(f"  Error on {row['filename']}: {e}")

    print(f"\nDone: {n_success} corrected, {n_skip} skipped (no segmented file)")

    # Save report
    if report:
        fields = ['filename', 'species', 'source', 'orig_a', 'orig_b',
                  'new_a', 'new_b', 'delta_a', 'delta_b']
        with open(CORRECTION_REPORT, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(report)
        print(f"Report saved to {CORRECTION_REPORT}")


def preview_species(species_name):
    """Generate a side-by-side preview for one species."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    inventory = load_inventory()
    source_stats = analyze_sources(inventory)
    corrections, _ = compute_correction_params(source_stats)

    sp_dir = species_to_dirname(species_name)
    seg_dir = os.path.join(SEGMENTED_DIR, sp_dir)

    if not os.path.isdir(seg_dir):
        print(f"No segmented images for {species_name}")
        return

    # Find FishPix images for this species
    fishpix_files = []
    for row in inventory:
        if row["species"] == species_name and row["source"] == "FishPix":
            seg_path = os.path.join(seg_dir, os.path.splitext(row["filename"])[0] + ".png")
            if os.path.exists(seg_path):
                fishpix_files.append((row["filename"], seg_path))

    if not fishpix_files:
        print(f"No FishPix images found for {species_name}")
        return

    corr = corrections.get("FishPix", {'delta_a': 0, 'delta_b': 0})
    delta_a = corr['delta_a']
    delta_b = corr['delta_b']

    print(f"Generating preview for {species_name}")
    print(f"Correction: Δa={delta_a:+.2f}, Δb={delta_b:+.2f}")

    # Create comparison figure
    n_images = min(len(fishpix_files), 4)
    fig, axes = plt.subplots(n_images, 2, figsize=(10, 4*n_images))
    if n_images == 1:
        axes = [axes]

    for i, (fname, seg_path) in enumerate(fishpix_files[:n_images]):
        lab_img, mask, _, alpha = load_segmented_lab(seg_path)

        # Original
        orig_rgba = lab_to_rgba(lab_img, alpha)

        # Corrected
        corrected_lab = apply_ab_correction(lab_img, mask, delta_a, delta_b)
        corr_rgba = lab_to_rgba(corrected_lab, alpha)

        # Composite on gray
        def composite(rgba, bg=180):
            rgb = rgba[:, :, :3].astype(float)
            a = (rgba[:, :, 3] / 255.0)[:, :, np.newaxis]
            return (rgb * a + bg * (1 - a)).astype(np.uint8)

        axes[i][0].imshow(composite(orig_rgba))
        axes[i][0].set_title(f"Original: {fname[:40]}")
        axes[i][0].axis('off')

        axes[i][1].imshow(composite(corr_rgba))
        axes[i][1].set_title(f"Corrected (Δa={delta_a:+.1f}, Δb={delta_b:+.1f})")
        axes[i][1].axis('off')

    plt.suptitle(f"{species_name} - FishPix Color Correction Preview", fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(GMM_DIR, f"color_correction_preview_{sp_dir}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Preview saved to {output_path}")
    os.system(f'open "{output_path}"')


def main():
    parser = argparse.ArgumentParser(description="Color cast correction")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze color distributions by source")
    parser.add_argument("--correct", action="store_true",
                       help="Apply correction to FishPix images")
    parser.add_argument("--correct-source", default="FishPix",
                       help="Source to correct (default: FishPix)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without modifying files")
    parser.add_argument("--preview", metavar="SPECIES",
                       help="Generate preview for a species")
    args = parser.parse_args()

    if args.analyze:
        run_analysis()
    elif args.correct:
        run_correction(args.correct_source, args.dry_run)
    elif args.preview:
        preview_species(args.preview)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
