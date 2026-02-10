#!/usr/bin/env python3
"""
Visualize color distribution in LAB a*b* morphospace by source.

Creates a scatter plot with:
- Points for each image colored/shaped by source
- Convex hull polygons around each source cluster
- Bishop/Randall as reference (high-quality curated images)
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
    load_inventory, species_to_dirname,
)

OUTPUT_PNG = os.path.join(GMM_DIR, "color_morphospace_by_source.png")


def load_segmented_lab_stats(path, max_pixels=10000):
    """Load segmented image and return mean a, b values."""
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
            return None, None

        if len(body_pixels) > max_pixels:
            idx = np.random.choice(len(body_pixels), max_pixels, replace=False)
            body_pixels = body_pixels[idx]

        mean_a = np.mean(body_pixels[:, 1])
        mean_b = np.mean(body_pixels[:, 2])
        return mean_a, mean_b
    except Exception:
        return None, None


def collect_color_data(inventory):
    """Collect a*b* data for all images by source."""
    source_data = defaultdict(list)

    sources_to_include = ["BishopMuseum", "FishBase", "FishBaseUser", "FishPix", "iNaturalist"]

    print("Collecting color data...")
    for i, row in enumerate(inventory):
        source = row["source"]
        if source not in sources_to_include:
            continue

        sp_dir = species_to_dirname(row["species"])
        seg_path = os.path.join(SEGMENTED_DIR, sp_dir,
                               os.path.splitext(row["filename"])[0] + ".png")

        if not os.path.exists(seg_path):
            continue

        mean_a, mean_b = load_segmented_lab_stats(seg_path)
        if mean_a is not None:
            source_data[source].append({
                'filename': row["filename"],
                'species': row["species"],
                'mean_a': mean_a,
                'mean_b': mean_b,
            })

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(inventory)} images...")

    return source_data


def create_morphospace_plot(source_data):
    """Create the a*b* morphospace visualization."""

    # Source styling
    source_styles = {
        'BishopMuseum': {'color': '#2ca02c', 'marker': 's', 'label': 'Bishop/Randall', 'zorder': 5},
        'FishBase': {'color': '#1f77b4', 'marker': 'o', 'label': 'FishBase (curated)', 'zorder': 3},
        'FishBaseUser': {'color': '#9467bd', 'marker': 'D', 'label': 'FishBase User', 'zorder': 3},
        'FishPix': {'color': '#ff7f0e', 'marker': '^', 'label': 'FishPix', 'zorder': 4},
        'iNaturalist': {'color': '#d62728', 'marker': 'v', 'label': 'iNaturalist', 'zorder': 2},
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot convex hulls first (background)
    hull_colors = {
        'BishopMuseum': '#2ca02c',
        'FishBase': '#1f77b4',
        'FishBaseUser': '#9467bd',
        'FishPix': '#ff7f0e',
        'iNaturalist': '#d62728',
    }

    for source, data in source_data.items():
        if len(data) < 4:
            continue

        points = np.array([[d['mean_a'], d['mean_b']] for d in data])

        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

            # Close the polygon
            hull_points = np.vstack([hull_points, hull_points[0]])

            polygon = Polygon(hull_points,
                            facecolor=hull_colors[source],
                            edgecolor=hull_colors[source],
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

        style = source_styles.get(source, {'color': 'gray', 'marker': '.', 'label': source, 'zorder': 1})

        ax.scatter(a_vals, b_vals,
                  c=style['color'],
                  marker=style['marker'],
                  s=40 if source == 'BishopMuseum' else 25,
                  alpha=0.7,
                  label=f"{style['label']} (n={len(data)})",
                  edgecolors='white',
                  linewidths=0.5,
                  zorder=style['zorder'])

    # Add source centroids with labels
    for source, data in source_data.items():
        if not data:
            continue
        mean_a = np.mean([d['mean_a'] for d in data])
        mean_b = np.mean([d['mean_b'] for d in data])

        style = source_styles.get(source, {'color': 'gray'})

        # Mark centroid with a larger marker
        ax.scatter([mean_a], [mean_b],
                  c=style['color'],
                  marker='X',
                  s=200,
                  edgecolors='black',
                  linewidths=2,
                  zorder=10)

    # Add reference lines at a=0, b=0
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, zorder=0)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5, zorder=0)

    # Labels and formatting
    ax.set_xlabel("a* (green ← → red)", fontsize=12)
    ax.set_ylabel("b* (blue ← → yellow)", fontsize=12)
    ax.set_title("Chaetodontidae Image Color Distribution in LAB a*b* Space\n"
                "Large X markers = source centroids; dashed polygons = convex hulls",
                fontsize=14)

    # Add annotation for color interpretation
    ax.annotate("Greener", xy=(-25, 0), fontsize=10, color='green', ha='center')
    ax.annotate("Redder", xy=(15, 0), fontsize=10, color='red', ha='center')
    ax.annotate("Bluer", xy=(0, -15), fontsize=10, color='blue', ha='center')
    ax.annotate("Yellower", xy=(0, 45), fontsize=10, color='orange', ha='center')

    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Set axis limits based on data
    all_a = [d['mean_a'] for data in source_data.values() for d in data]
    all_b = [d['mean_b'] for data in source_data.values() for d in data]

    margin = 5
    ax.set_xlim(min(all_a) - margin, max(all_a) + margin)
    ax.set_ylim(min(all_b) - margin, max(all_b) + margin)

    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("Color Morphospace Visualization")
    print("=" * 70)

    # Load inventory
    print("\n[1] Loading inventory...")
    inventory = load_inventory()
    print(f"    {len(inventory)} images")

    # Collect color data
    print("\n[2] Analyzing image colors...")
    source_data = collect_color_data(inventory)

    for source, data in sorted(source_data.items()):
        print(f"    {source}: {len(data)} images")

    # Create plot
    print("\n[3] Creating morphospace plot...")
    fig = create_morphospace_plot(source_data)

    # Save
    print(f"\n[4] Saving to {OUTPUT_PNG}")
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics (mean ± std)")
    print("=" * 70)
    print(f"{'Source':<20} {'N':>6} {'mean a':>10} {'std a':>10} {'mean b':>10} {'std b':>10}")
    print("-" * 70)

    for source in ['BishopMuseum', 'FishBase', 'FishBaseUser', 'FishPix', 'iNaturalist']:
        data = source_data.get(source, [])
        if data:
            a_vals = [d['mean_a'] for d in data]
            b_vals = [d['mean_b'] for d in data]
            print(f"{source:<20} {len(data):>6} {np.mean(a_vals):>10.2f} {np.std(a_vals):>10.2f} "
                  f"{np.mean(b_vals):>10.2f} {np.std(b_vals):>10.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
