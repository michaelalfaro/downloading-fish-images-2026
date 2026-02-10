#!/usr/bin/env python3
"""
Visualize color distribution in LAB a*b* morphospace: whole image vs segmented fish.

Creates a two-panel plot:
- Left: Color from the WHOLE original image (includes background/water)
- Right: Color from SEGMENTED fish only (background removed)

This comparison can reveal:
- How much the background (water, tank, etc.) affects overall color
- Whether underwater lighting conditions are captured in the background
- Species where the fish is well-centered vs off-center
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from collections import defaultdict

from PIL import Image
from skimage.color import rgb2lab

import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils")); from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR,
    load_inventory, species_to_dirname, get_image_path,
)

OUTPUT_PNG = os.path.join(GMM_DIR, "color_morphospace_dual_panel.png")


def load_whole_image_lab_stats(path, max_pixels=10000):
    """Load whole image and return mean a, b values (center region)."""
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size

        # Sample from center region (center 50% of image)
        # This is where the fish typically is
        left = w // 4
        right = 3 * w // 4
        top = h // 4
        bottom = 3 * h // 4

        img_cropped = img.crop((left, top, right, bottom))
        arr = np.array(img_cropped)

        # Convert to LAB
        rgb_float = arr.astype(np.float64) / 255.0
        lab = rgb2lab(rgb_float)

        # Flatten to pixels
        pixels = lab.reshape(-1, 3)
        if len(pixels) > max_pixels:
            idx = np.random.choice(len(pixels), max_pixels, replace=False)
            pixels = pixels[idx]

        mean_a = np.mean(pixels[:, 1])
        mean_b = np.mean(pixels[:, 2])
        mean_L = np.mean(pixels[:, 0])
        return mean_a, mean_b, mean_L
    except Exception:
        return None, None, None


def load_segmented_lab_stats(path, max_pixels=10000):
    """Load segmented image and return mean a, b values (fish only)."""
    try:
        img = Image.open(path).convert("RGBA")
        arr = np.array(img)
        mask = arr[:, :, 3] > 0
        rgb = arr[:, :, :3]

        # Convert to LAB
        rgb_float = rgb.astype(np.float64) / 255.0
        lab = rgb2lab(rgb_float)

        # Get body pixel values
        body_pixels = lab[mask]
        if len(body_pixels) < 100:
            return None, None, None

        if len(body_pixels) > max_pixels:
            idx = np.random.choice(len(body_pixels), max_pixels, replace=False)
            body_pixels = body_pixels[idx]

        mean_a = np.mean(body_pixels[:, 1])
        mean_b = np.mean(body_pixels[:, 2])
        mean_L = np.mean(body_pixels[:, 0])
        return mean_a, mean_b, mean_L
    except Exception:
        return None, None, None


def collect_color_data(inventory):
    """Collect a*b* data for all images by source, for both whole and segmented."""
    whole_data = defaultdict(list)
    seg_data = defaultdict(list)

    sources_to_include = ["BishopMuseum", "FishBase", "FishBaseUser", "FishPix", "iNaturalist"]

    print("Collecting color data...")
    for i, row in enumerate(inventory):
        source = row["source"]
        if source not in sources_to_include:
            continue

        sp_dir = species_to_dirname(row["species"])

        # Get original image path
        orig_path = get_image_path(row)
        # Get segmented image path
        seg_path = os.path.join(SEGMENTED_DIR, sp_dir,
                               os.path.splitext(row["filename"])[0] + ".png")

        if not os.path.exists(orig_path) or not os.path.exists(seg_path):
            continue

        # Whole image stats
        whole_a, whole_b, whole_L = load_whole_image_lab_stats(orig_path)
        # Segmented stats
        seg_a, seg_b, seg_L = load_segmented_lab_stats(seg_path)

        if whole_a is not None and seg_a is not None:
            whole_data[source].append({
                'filename': row["filename"],
                'species': row["species"],
                'mean_a': whole_a,
                'mean_b': whole_b,
                'mean_L': whole_L,
            })
            seg_data[source].append({
                'filename': row["filename"],
                'species': row["species"],
                'mean_a': seg_a,
                'mean_b': seg_b,
                'mean_L': seg_L,
            })

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(inventory)} images...")

    return whole_data, seg_data


def create_panel(ax, source_data, title):
    """Create a single morphospace panel."""

    source_styles = {
        'BishopMuseum': {'color': '#2ca02c', 'marker': 's', 'label': 'Bishop/Randall'},
        'FishBase': {'color': '#1f77b4', 'marker': 'o', 'label': 'FishBase'},
        'FishBaseUser': {'color': '#9467bd', 'marker': 'D', 'label': 'FishBase User'},
        'FishPix': {'color': '#ff7f0e', 'marker': '^', 'label': 'FishPix'},
        'iNaturalist': {'color': '#d62728', 'marker': 'v', 'label': 'iNaturalist'},
    }

    # Plot convex hulls
    for source, data in source_data.items():
        if len(data) < 4:
            continue

        points = np.array([[d['mean_a'], d['mean_b']] for d in data])

        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])

            style = source_styles.get(source, {'color': 'gray'})
            polygon = Polygon(hull_points,
                            facecolor=style['color'],
                            edgecolor=style['color'],
                            alpha=0.15,
                            linewidth=2,
                            linestyle='--',
                            zorder=1)
            ax.add_patch(polygon)
        except Exception:
            pass

    # Plot points
    for source, data in source_data.items():
        if not data:
            continue

        a_vals = [d['mean_a'] for d in data]
        b_vals = [d['mean_b'] for d in data]

        style = source_styles.get(source, {'color': 'gray', 'marker': '.', 'label': source})

        ax.scatter(a_vals, b_vals,
                  c=style['color'],
                  marker=style['marker'],
                  s=25,
                  alpha=0.6,
                  label=f"{style['label']} (n={len(data)})",
                  edgecolors='white',
                  linewidths=0.3,
                  zorder=3)

    # Add centroids
    for source, data in source_data.items():
        if not data:
            continue
        mean_a = np.mean([d['mean_a'] for d in data])
        mean_b = np.mean([d['mean_b'] for d in data])
        style = source_styles.get(source, {'color': 'gray'})

        ax.scatter([mean_a], [mean_b],
                  c=style['color'],
                  marker='X',
                  s=150,
                  edgecolors='black',
                  linewidths=2,
                  zorder=10)

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, zorder=0)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, zorder=0)

    ax.set_xlabel("a* (green ← → red)", fontsize=10)
    ax.set_ylabel("b* (blue ← → yellow)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Set consistent axis limits
    all_a = [d['mean_a'] for data in source_data.values() for d in data]
    all_b = [d['mean_b'] for data in source_data.values() for d in data]

    margin = 5
    ax.set_xlim(min(all_a) - margin, max(all_a) + margin)
    ax.set_ylim(min(all_b) - margin, max(all_b) + margin)


def create_dual_panel_plot(whole_data, seg_data):
    """Create the two-panel comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Calculate global limits
    all_a_whole = [d['mean_a'] for data in whole_data.values() for d in data]
    all_b_whole = [d['mean_b'] for data in whole_data.values() for d in data]
    all_a_seg = [d['mean_a'] for data in seg_data.values() for d in data]
    all_b_seg = [d['mean_b'] for data in seg_data.values() for d in data]

    all_a = all_a_whole + all_a_seg
    all_b = all_b_whole + all_b_seg

    margin = 5
    xlim = (min(all_a) - margin, max(all_a) + margin)
    ylim = (min(all_b) - margin, max(all_b) + margin)

    create_panel(ax1, whole_data, "Whole Image (Center 50%)")
    create_panel(ax2, seg_data, "Segmented Fish Only")

    # Apply consistent limits
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    fig.suptitle("Color Distribution Comparison: Whole Image vs Segmented Fish\n"
                 "Large X markers = centroids; dashed polygons = convex hulls",
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("Dual-Panel Color Morphospace Visualization")
    print("=" * 70)

    # Load inventory
    print("\n[1] Loading inventory...")
    inventory = load_inventory()
    print(f"    {len(inventory)} images")

    # Collect color data
    print("\n[2] Analyzing image colors (whole + segmented)...")
    whole_data, seg_data = collect_color_data(inventory)

    print("\n    Whole image stats:")
    for source, data in sorted(whole_data.items()):
        print(f"      {source}: {len(data)} images")

    print("\n    Segmented stats:")
    for source, data in sorted(seg_data.items()):
        print(f"      {source}: {len(data)} images")

    # Create plot
    print("\n[3] Creating dual-panel plot...")
    fig = create_dual_panel_plot(whole_data, seg_data)

    # Save
    print(f"\n[4] Saving to {OUTPUT_PNG}")
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Print comparison statistics
    print("\n" + "=" * 70)
    print("Color Shift: Whole Image vs Segmented")
    print("=" * 70)
    print(f"{'Source':<20} {'Whole a':>10} {'Seg a':>10} {'Δa':>8} {'Whole b':>10} {'Seg b':>10} {'Δb':>8}")
    print("-" * 80)

    for source in ['BishopMuseum', 'FishBase', 'FishBaseUser', 'FishPix', 'iNaturalist']:
        wd = whole_data.get(source, [])
        sd = seg_data.get(source, [])
        if wd and sd:
            wa = np.mean([d['mean_a'] for d in wd])
            wb = np.mean([d['mean_b'] for d in wd])
            sa = np.mean([d['mean_a'] for d in sd])
            sb = np.mean([d['mean_b'] for d in sd])
            print(f"{source:<20} {wa:>10.2f} {sa:>10.2f} {sa-wa:>+8.2f} "
                  f"{wb:>10.2f} {sb:>10.2f} {sb-wb:>+8.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
