#!/usr/bin/env python3
"""
Analyze reference colors (white, yellow, black) from Randall/Bishop photos.

Most butterflyfish have white and/or yellow regions that can serve as
reference points for color correction. This script:
1. Extracts the whitest and yellowest pixels from Bishop images
2. Computes the LAB distribution of these "known" colors
3. Provides correction targets for underwater photo restoration

White in fish photos should be close to L*=100, a*=0, b*=0
Yellow (typical for butterflyfish) should be L*~85, a*~0, b*~70
"""

import os
import sys
import json
from collections import defaultdict

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from analysis_utils import (
    REPO_DIR, GMM_DIR, SEGMENTED_DIR,
    load_inventory, species_to_dirname,
)

OUTPUT_DIR = os.path.join(GMM_DIR, "color_reference")


def extract_color_extremes(seg_path, n_pixels=100):
    """Extract the whitest, yellowest, and darkest pixels from segmented image.

    Returns:
        dict with 'white', 'yellow', 'dark' containing LAB statistics
    """
    try:
        img = Image.open(seg_path).convert("RGBA")
        arr = np.array(img)
        mask = arr[:, :, 3] > 0
        rgb = arr[:, :, :3]

        # Convert to LAB
        rgb_float = rgb.astype(np.float64) / 255.0
        lab = rgb2lab(rgb_float)

        # Get body pixels
        body_mask = mask.flatten()
        lab_flat = lab.reshape(-1, 3)
        body_lab = lab_flat[body_mask]

        if len(body_lab) < 1000:
            return None

        L, a, b = body_lab[:, 0], body_lab[:, 1], body_lab[:, 2]

        results = {}

        # White pixels: highest L, low saturation (low |a| + |b|)
        # Compute "whiteness" score: high L, low chroma
        chroma = np.sqrt(a**2 + b**2)
        whiteness = L - chroma  # Higher is more white
        white_idx = np.argsort(whiteness)[-n_pixels:]
        results['white'] = {
            'L': float(np.mean(L[white_idx])),
            'a': float(np.mean(a[white_idx])),
            'b': float(np.mean(b[white_idx])),
            'L_std': float(np.std(L[white_idx])),
            'a_std': float(np.std(a[white_idx])),
            'b_std': float(np.std(b[white_idx])),
        }

        # Yellow pixels: high L, high b*, near-zero a*
        # Yellowness score: high b, moderate-high L, low |a|
        yellowness = b + 0.5 * L - 2 * np.abs(a)
        yellow_idx = np.argsort(yellowness)[-n_pixels:]
        results['yellow'] = {
            'L': float(np.mean(L[yellow_idx])),
            'a': float(np.mean(a[yellow_idx])),
            'b': float(np.mean(b[yellow_idx])),
            'L_std': float(np.std(L[yellow_idx])),
            'a_std': float(np.std(a[yellow_idx])),
            'b_std': float(np.std(b[yellow_idx])),
        }

        # Dark pixels: lowest L (often black stripes/markings)
        dark_idx = np.argsort(L)[:n_pixels]
        results['dark'] = {
            'L': float(np.mean(L[dark_idx])),
            'a': float(np.mean(a[dark_idx])),
            'b': float(np.mean(b[dark_idx])),
            'L_std': float(np.std(L[dark_idx])),
            'a_std': float(np.std(a[dark_idx])),
            'b_std': float(np.std(b[dark_idx])),
        }

        # Overall mean
        results['overall'] = {
            'L': float(np.mean(L)),
            'a': float(np.mean(a)),
            'b': float(np.mean(b)),
        }

        return results

    except Exception as e:
        print(f"  Error: {e}")
        return None


def analyze_bishop_reference_colors(inventory):
    """Analyze white and yellow reference colors from all Bishop images."""

    bishop_images = [r for r in inventory if r["source"] == "BishopMuseum"]
    print(f"Analyzing {len(bishop_images)} Bishop/Randall images...")

    all_white = []
    all_yellow = []
    all_dark = []

    species_colors = defaultdict(list)

    for i, row in enumerate(bishop_images):
        species = row["species"]
        filename = row["filename"]
        sp_dir = species_to_dirname(species)

        seg_path = os.path.join(SEGMENTED_DIR, sp_dir,
                               os.path.splitext(filename)[0] + ".png")

        if not os.path.exists(seg_path):
            continue

        colors = extract_color_extremes(seg_path)
        if colors is None:
            continue

        all_white.append(colors['white'])
        all_yellow.append(colors['yellow'])
        all_dark.append(colors['dark'])

        species_colors[species].append({
            'filename': filename,
            'colors': colors,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(bishop_images)} images...")

    return all_white, all_yellow, all_dark, species_colors


def compute_reference_statistics(all_white, all_yellow, all_dark):
    """Compute aggregate reference color statistics."""

    print("\n" + "=" * 70)
    print("BISHOP/RANDALL REFERENCE COLOR STATISTICS")
    print("=" * 70)

    # White reference
    white_L = [w['L'] for w in all_white]
    white_a = [w['a'] for w in all_white]
    white_b = [w['b'] for w in all_white]

    print("\n[WHITE REFERENCE] (fish body white regions)")
    print(f"  N images: {len(all_white)}")
    print(f"  L*: {np.mean(white_L):.1f} ± {np.std(white_L):.1f} (range: {np.min(white_L):.1f} - {np.max(white_L):.1f})")
    print(f"  a*: {np.mean(white_a):.1f} ± {np.std(white_a):.1f} (range: {np.min(white_a):.1f} - {np.max(white_a):.1f})")
    print(f"  b*: {np.mean(white_b):.1f} ± {np.std(white_b):.1f} (range: {np.min(white_b):.1f} - {np.max(white_b):.1f})")
    print(f"  → Ideal white: L*=100, a*=0, b*=0")
    print(f"  → Bishop white offset from ideal: L*={np.mean(white_L)-100:.1f}, a*={np.mean(white_a):.1f}, b*={np.mean(white_b):.1f}")

    # Yellow reference
    yellow_L = [y['L'] for y in all_yellow]
    yellow_a = [y['a'] for y in all_yellow]
    yellow_b = [y['b'] for y in all_yellow]

    print("\n[YELLOW REFERENCE] (fish body yellow regions)")
    print(f"  N images: {len(all_yellow)}")
    print(f"  L*: {np.mean(yellow_L):.1f} ± {np.std(yellow_L):.1f} (range: {np.min(yellow_L):.1f} - {np.max(yellow_L):.1f})")
    print(f"  a*: {np.mean(yellow_a):.1f} ± {np.std(yellow_a):.1f} (range: {np.min(yellow_a):.1f} - {np.max(yellow_a):.1f})")
    print(f"  b*: {np.mean(yellow_b):.1f} ± {np.std(yellow_b):.1f} (range: {np.min(yellow_b):.1f} - {np.max(yellow_b):.1f})")

    # Dark reference
    dark_L = [d['L'] for d in all_dark]
    dark_a = [d['a'] for d in all_dark]
    dark_b = [d['b'] for d in all_dark]

    print("\n[DARK REFERENCE] (fish body dark/black regions)")
    print(f"  N images: {len(all_dark)}")
    print(f"  L*: {np.mean(dark_L):.1f} ± {np.std(dark_L):.1f} (range: {np.min(dark_L):.1f} - {np.max(dark_L):.1f})")
    print(f"  a*: {np.mean(dark_a):.1f} ± {np.std(dark_a):.1f} (range: {np.min(dark_a):.1f} - {np.max(dark_a):.1f})")
    print(f"  b*: {np.mean(dark_b):.1f} ± {np.std(dark_b):.1f} (range: {np.min(dark_b):.1f} - {np.max(dark_b):.1f})")

    # Return reference values
    reference = {
        'white': {
            'L': float(np.mean(white_L)),
            'a': float(np.mean(white_a)),
            'b': float(np.mean(white_b)),
            'L_std': float(np.std(white_L)),
            'a_std': float(np.std(white_a)),
            'b_std': float(np.std(white_b)),
        },
        'yellow': {
            'L': float(np.mean(yellow_L)),
            'a': float(np.mean(yellow_a)),
            'b': float(np.mean(yellow_b)),
            'L_std': float(np.std(yellow_L)),
            'a_std': float(np.std(yellow_a)),
            'b_std': float(np.std(yellow_b)),
        },
        'dark': {
            'L': float(np.mean(dark_L)),
            'a': float(np.mean(dark_a)),
            'b': float(np.mean(dark_b)),
            'L_std': float(np.std(dark_L)),
            'a_std': float(np.std(dark_a)),
            'b_std': float(np.std(dark_b)),
        },
    }

    return reference


def suggest_correction_parameters(reference):
    """Suggest color correction parameters based on reference colors."""

    print("\n" + "=" * 70)
    print("SUGGESTED COLOR CORRECTION APPROACH")
    print("=" * 70)

    white_ref = reference['white']
    yellow_ref = reference['yellow']

    print("\n[1] WHITE POINT CORRECTION")
    print("    If underwater image has green/blue cast, the white regions will have:")
    print("    - Negative a* (greenish) → need to add positive a*")
    print("    - Negative b* (bluish) → need to add positive b*")
    print(f"    TARGET white: L*={white_ref['L']:.0f}, a*={white_ref['a']:.1f}, b*={white_ref['b']:.1f}")
    print("\n    Simple correction: shift all pixels toward white reference")
    print("    For image with detected white at (L', a', b'):")
    print(f"    Δa = {white_ref['a']:.1f} - a'_white")
    print(f"    Δb = {white_ref['b']:.1f} - b'_white")
    print("    Apply Δa, Δb to all pixels in image")

    print("\n[2] YELLOW VERIFICATION")
    print(f"    After white correction, yellow should be near:")
    print(f"    L*={yellow_ref['L']:.0f}, a*={yellow_ref['a']:.1f}, b*={yellow_ref['b']:.1f}")

    print("\n[3] RECOLORIZE APPROACH (Hannah Weller's package)")
    print("    Could use Bishop colors as the color palette constraint:")
    print("    - Extract color histogram from Bishop/Randall photos")
    print("    - Apply to underwater photos as target distribution")
    print("    - This would preserve patterns while shifting colors")

    print("\n[4] AGGRESSIVE UNDERWATER CORRECTION")
    print("    For images with strong blue/green cast (a* < -10, b* < -10):")
    print("    1. Detect white regions (highest L, lowest chroma)")
    print("    2. Compute shift needed to reach Bishop white reference")
    print("    3. Apply shift uniformly OR with gradient based on depth")
    print("    4. Optionally enhance red channel (compensate for absorption)")


def create_color_distribution_plot(all_white, all_yellow, all_dark, output_path):
    """Create visualization of reference color distributions."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # White distribution
    ax = axes[0]
    white_a = [w['a'] for w in all_white]
    white_b = [w['b'] for w in all_white]
    ax.scatter(white_a, white_b, c='gray', alpha=0.5, s=30)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    ax.scatter([np.mean(white_a)], [np.mean(white_b)], c='black', s=200, marker='X',
               edgecolors='white', linewidths=2, label='Mean')
    ax.set_xlabel("a* (green ← → red)")
    ax.set_ylabel("b* (blue ← → yellow)")
    ax.set_title(f"White Reference\n(n={len(all_white)})")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 30)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Yellow distribution
    ax = axes[1]
    yellow_a = [y['a'] for y in all_yellow]
    yellow_b = [y['b'] for y in all_yellow]
    ax.scatter(yellow_a, yellow_b, c='gold', alpha=0.5, s=30, edgecolors='orange')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    ax.scatter([np.mean(yellow_a)], [np.mean(yellow_b)], c='darkorange', s=200, marker='X',
               edgecolors='white', linewidths=2, label='Mean')
    ax.set_xlabel("a* (green ← → red)")
    ax.set_ylabel("b* (blue ← → yellow)")
    ax.set_title(f"Yellow Reference\n(n={len(all_yellow)})")
    ax.set_xlim(-15, 30)
    ax.set_ylim(20, 80)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dark distribution
    ax = axes[2]
    dark_a = [d['a'] for d in all_dark]
    dark_b = [d['b'] for d in all_dark]
    ax.scatter(dark_a, dark_b, c='dimgray', alpha=0.5, s=30)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.5)
    ax.scatter([np.mean(dark_a)], [np.mean(dark_b)], c='black', s=200, marker='X',
               edgecolors='white', linewidths=2, label='Mean')
    ax.set_xlabel("a* (green ← → red)")
    ax.set_ylabel("b* (blue ← → yellow)")
    ax.set_title(f"Dark Reference\n(n={len(all_dark)})")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Bishop/Randall Reference Color Distributions\n(a*-b* plane)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to: {output_path}")


def main():
    print("=" * 70)
    print("Reference Color Analysis from Bishop/Randall Photos")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load inventory
    print("\n[1] Loading inventory...")
    inventory = load_inventory()
    print(f"    {len(inventory)} images")

    # Analyze Bishop reference colors
    print("\n[2] Extracting reference colors from Bishop images...")
    all_white, all_yellow, all_dark, species_colors = analyze_bishop_reference_colors(inventory)

    # Compute statistics
    print("\n[3] Computing reference statistics...")
    reference = compute_reference_statistics(all_white, all_yellow, all_dark)

    # Suggest correction approach
    suggest_correction_parameters(reference)

    # Save reference data
    json_path = os.path.join(OUTPUT_DIR, "bishop_reference_colors.json")
    print(f"\n[4] Saving reference data to {json_path}")
    with open(json_path, 'w') as f:
        json.dump(reference, f, indent=2)

    # Create visualization
    plot_path = os.path.join(OUTPUT_DIR, "bishop_reference_colors.png")
    print(f"\n[5] Creating visualization...")
    create_color_distribution_plot(all_white, all_yellow, all_dark, plot_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
