#!/usr/bin/env python3
"""
L-channel normalization in CIELAB space for segmented fish images.

For each species, computes the mean luminance (L) across all body pixels
in all images, then shifts each image's L channel so its mean matches
the species mean. Preserves hue (a, b channels unchanged).

Optionally generates per-species comparison panels showing:
  Row 1: original segmented images (on gray background)
  Row 2: luminance-normalized images (on gray background)
  Labels: filename + relative path under each thumbnail
  Stats bar: within-species color variability before and after normalization

Options:
  --exclude-inat     Exclude iNaturalist images from analysis
  --visualize        Generate comparison panels

Reads: analysis/approach_1_gmm/segmented/
Writes: analysis/approach_1_gmm/normalized/
        analysis/approach_1_gmm/normalization_report.csv
        analysis/approach_1_gmm/comparison_panels/  (if --visualize)

Usage:
    python3 normalize_luminance.py                                  # all species
    python3 normalize_luminance.py --species "Chaetodon meyeri"     # one species
    python3 normalize_luminance.py --visualize                      # + panels
    python3 normalize_luminance.py --visualize --exclude-inat       # no iNat
"""

import os
import csv
import argparse
import numpy as np
from collections import defaultdict
from PIL import Image

from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR, NORMALIZED_DIR,
    load_inventory, load_annotations, species_to_dirname,
    load_segmented_image, body_pixels_to_lab, lab_to_rgb_pixels,
    all_species, ensure_dir,
)

REPORT_CSV = os.path.join(GMM_DIR, "normalization_report.csv")
PANELS_DIR = os.path.join(GMM_DIR, "comparison_panels")
REPORT_FIELDS = [
    "filename", "species", "source", "original_mean_L", "species_mean_L",
    "L_shift", "n_body_pixels", "n_pixels_clamped",
]


def make_thumbnail(rgba_arr, target_height=300):
    """Resize RGBA array to target height, preserving aspect ratio."""
    h, w = rgba_arr.shape[:2]
    if h <= 0:
        return rgba_arr
    scale = target_height / h
    new_w = max(1, int(w * scale))
    img = Image.fromarray(rgba_arr, "RGBA")
    img = img.resize((new_w, target_height), Image.LANCZOS)
    return np.array(img)


def composite_on_gray(rgba_arr, bg_value=180):
    """Composite RGBA onto neutral gray background, return RGB.

    Gray background makes luminance differences much more visible
    than white background.
    """
    rgb = rgba_arr[:, :, :3].astype(float)
    alpha = (rgba_arr[:, :, 3] / 255.0)[:, :, np.newaxis]
    bg = float(bg_value)
    result = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
    return result


def compute_color_variation(image_data_list):
    """Compute within-species color variation statistics.

    For each image, computes mean L, a, b and chroma.
    Returns dict of variation statistics useful for detecting
    species with unusually high color variation.
    """
    if not image_data_list:
        return {}

    per_image_stats = []
    for fname, png_path, rgb, mask in image_data_list:
        body_px = rgb[mask]
        if len(body_px) < 100:
            continue
        lab = body_pixels_to_lab(body_px)
        L_mean = float(np.mean(lab[:, 0]))
        a_mean = float(np.mean(lab[:, 1]))
        b_mean = float(np.mean(lab[:, 2]))
        chroma = float(np.mean(np.sqrt(lab[:, 1]**2 + lab[:, 2]**2)))
        L_std = float(np.std(lab[:, 0]))
        a_std = float(np.std(lab[:, 1]))
        b_std = float(np.std(lab[:, 2]))
        per_image_stats.append({
            "fname": fname,
            "L_mean": L_mean, "a_mean": a_mean, "b_mean": b_mean,
            "chroma": chroma,
            "L_std": L_std, "a_std": a_std, "b_std": b_std,
        })

    if len(per_image_stats) < 2:
        return {"n_images": len(per_image_stats)}

    # Across-image variation (how different are images from each other)
    L_means = [s["L_mean"] for s in per_image_stats]
    a_means = [s["a_mean"] for s in per_image_stats]
    b_means = [s["b_mean"] for s in per_image_stats]
    chromas = [s["chroma"] for s in per_image_stats]

    # Within-image variation (average complexity of coloring per image)
    L_stds = [s["L_std"] for s in per_image_stats]
    a_stds = [s["a_std"] for s in per_image_stats]
    b_stds = [s["b_std"] for s in per_image_stats]

    return {
        "n_images": len(per_image_stats),
        # Across-image variation (how consistent is the species)
        "L_cv_across": float(np.std(L_means)),    # L variation across images
        "a_cv_across": float(np.std(a_means)),     # a* variation
        "b_cv_across": float(np.std(b_means)),     # b* variation
        "chroma_cv_across": float(np.std(chromas)),  # chroma variation
        # Mean within-image variation (pattern complexity)
        "L_complexity": float(np.mean(L_stds)),
        "a_complexity": float(np.mean(a_stds)),
        "b_complexity": float(np.mean(b_stds)),
        # Total color distance spread (max pairwise Lab distance between images)
        "max_Lab_dist": _max_pairwise_lab_dist(L_means, a_means, b_means),
    }


def _max_pairwise_lab_dist(L_means, a_means, b_means):
    """Max pairwise Euclidean distance in Lab space between image means."""
    n = len(L_means)
    if n < 2:
        return 0.0
    max_dist = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((L_means[i] - L_means[j])**2 +
                        (a_means[i] - a_means[j])**2 +
                        (b_means[i] - b_means[j])**2)
            max_dist = max(max_dist, d)
    return float(max_dist)


def truncate_path(filepath, max_chars=40):
    """Truncate a file path for display, keeping the informative end."""
    if len(filepath) <= max_chars:
        return filepath
    return "..." + filepath[-(max_chars - 3):]


def generate_species_panel(species, seg_images, norm_images, stats,
                           color_stats, output_path, thumb_height=300):
    """Generate a comparison panel for one species.

    Improved layout:
      Title bar: species name, n images, color variation summary
      Row 1 (labeled "Original"): thumbnails on gray background
      Row 2 (labeled "Normalized"): thumbnails on gray background
      Filename labels below each column
      Stats bar: L-channel + color variability before/after

    seg_images: list of (filename, directory, rgba_array)
    norm_images: list of (filename, directory, rgba_array)
    stats: dict with L_std_before, L_std_after, mean_L, n_images, etc.
    color_stats: dict with color variation metrics
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(seg_images)
    if n == 0:
        return

    # Make thumbnails — composite on GRAY for better visibility
    seg_thumbs = []
    norm_thumbs = []
    fnames = []
    rel_paths = []
    for (fname, directory, rgba) in seg_images:
        t = make_thumbnail(rgba, thumb_height)
        seg_thumbs.append(composite_on_gray(t, bg_value=180))
        # Build relative path for label
        sp_dir = species_to_dirname(species)
        rel_path = f"{directory}/{fname}"
        fnames.append(fname)
        rel_paths.append(rel_path)
    for (fname, directory, rgba) in norm_images:
        t = make_thumbnail(rgba, thumb_height)
        norm_thumbs.append(composite_on_gray(t, bg_value=180))

    # Layout: show all images, wrapping to multiple rows if > 10
    max_per_row = 10
    n_show = min(n, max_per_row)
    thumb_widths = [t.shape[1] for t in seg_thumbs[:n_show]]
    gap = 8
    total_thumb_width = sum(thumb_widths) + (n_show - 1) * gap
    label_margin = 80  # left margin for "Original" / "Normalized" labels
    panel_width = max(total_thumb_width + label_margin + 20, 800)

    # Height: title + 2 image rows + label rows + stats bar
    label_row_height = 50  # space for filename labels below thumbnails
    fig_w = panel_width / 80
    fig_h = (thumb_height * 2 + label_row_height * 2 + 280) / 80

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#b8b8b8")

    # Title
    title_text = f"{species} (n={stats['n_images']})"
    fig.text(0.5, 0.96, title_text,
             ha="center", va="top", fontsize=16, fontweight="bold",
             fontstyle="italic", color="black")

    # Layout with gridspec: orig_row, orig_labels, norm_row, norm_labels, stats
    gs = GridSpec(5, 1, figure=fig,
                  height_ratios=[thumb_height, label_row_height,
                                 thumb_height, label_row_height, 60],
                  top=0.92, bottom=0.02, left=0.01, right=0.99,
                  hspace=0.02)

    # --- Row 1: Original segmented ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, panel_width)
    ax1.set_ylim(0, thumb_height)
    ax1.set_facecolor("#b8b8b8")
    ax1.axis("off")
    ax1.text(5, thumb_height / 2, "Original", va="center", ha="left",
             fontsize=11, fontweight="bold", rotation=90, color="black")
    x_positions = []  # track x center for each thumbnail
    x_offset = label_margin
    for i in range(n_show):
        t = seg_thumbs[i]
        tw = t.shape[1]
        ax1.imshow(t, extent=[x_offset, x_offset + tw, 0, thumb_height],
                   aspect="auto")
        x_positions.append(x_offset + tw / 2)
        x_offset += tw + gap

    # --- Label row 1: filenames for original ---
    ax1_labels = fig.add_subplot(gs[1])
    ax1_labels.set_xlim(0, panel_width)
    ax1_labels.set_ylim(0, label_row_height)
    ax1_labels.set_facecolor("#b8b8b8")
    ax1_labels.axis("off")
    for i in range(n_show):
        # Short filename (just the base, no species prefix)
        short_name = fnames[i]
        # Remove common species prefix for readability
        parts = short_name.split("_")
        if len(parts) > 2:
            short_name = "_".join(parts[2:])  # drop Genus_species prefix
        short_name = os.path.splitext(short_name)[0]

        # Two-line label: short name + relative path
        ax1_labels.text(x_positions[i], label_row_height * 0.7,
                        short_name,
                        ha="center", va="top", fontsize=5,
                        fontweight="bold", color="#333333")
        ax1_labels.text(x_positions[i], label_row_height * 0.3,
                        truncate_path(rel_paths[i], 45),
                        ha="center", va="top", fontsize=4,
                        color="#666666")

    # --- Row 2: Normalized ---
    ax2 = fig.add_subplot(gs[2])
    ax2.set_xlim(0, panel_width)
    ax2.set_ylim(0, thumb_height)
    ax2.set_facecolor("#b8b8b8")
    ax2.axis("off")
    ax2.text(5, thumb_height / 2, "Normalized", va="center", ha="left",
             fontsize=11, fontweight="bold", rotation=90, color="black")
    x_offset = label_margin
    for i in range(n_show):
        t = norm_thumbs[i]
        tw = t.shape[1]
        ax2.imshow(t, extent=[x_offset, x_offset + tw, 0, thumb_height],
                   aspect="auto")
        x_offset += tw + gap

    # --- Label row 2: L values before/after for each image ---
    ax2_labels = fig.add_subplot(gs[3])
    ax2_labels.set_xlim(0, panel_width)
    ax2_labels.set_ylim(0, label_row_height)
    ax2_labels.set_facecolor("#b8b8b8")
    ax2_labels.axis("off")
    for i in range(n_show):
        if i < len(stats.get("per_image_L", [])):
            orig_L, norm_L = stats["per_image_L"][i]
            shift = norm_L - orig_L
            sign = "+" if shift >= 0 else ""
            ax2_labels.text(x_positions[i], label_row_height * 0.7,
                            f"L: {orig_L:.0f} → {norm_L:.0f} ({sign}{shift:.1f})",
                            ha="center", va="top", fontsize=5,
                            color="#333333")

    # --- Stats bar ---
    ax3 = fig.add_subplot(gs[4])
    ax3.set_facecolor("#b8b8b8")
    ax3.axis("off")

    # Build multi-line stats text
    line1 = (f"L-normalization:  Mean L={stats['mean_L']:.1f}   |   "
             f"L std: {stats['L_std_before']:.2f} → {stats['L_std_after']:.2f}  "
             f"({stats['L_std_reduction']:.0%} reduction)   |   "
             f"Max |shift|: {stats['max_abs_shift']:.1f}")

    # Color variation stats
    line2_parts = []
    if color_stats.get("n_images", 0) >= 2:
        line2_parts.append(
            f"Color variation across images:  "
            f"a* σ={color_stats.get('a_cv_across', 0):.2f}   "
            f"b* σ={color_stats.get('b_cv_across', 0):.2f}   "
            f"Chroma σ={color_stats.get('chroma_cv_across', 0):.2f}   |   "
            f"Max ΔE(Lab)={color_stats.get('max_Lab_dist', 0):.1f}"
        )
    line2 = line2_parts[0] if line2_parts else ""

    ax3.text(0.5, 0.7, line1, ha="center", va="center",
             fontsize=8, transform=ax3.transAxes, color="black",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0e0e0",
                       edgecolor="#aaaaaa"))
    if line2:
        ax3.text(0.5, 0.2, line2, ha="center", va="center",
                 fontsize=7, transform=ax3.transAxes, color="#333333",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8e8e0",
                           edgecolor="#aaaaaa"))

    ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=120, bbox_inches="tight",
                facecolor="#b8b8b8")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="L-channel normalization + optional visualization"
    )
    parser.add_argument("--species", type=str, nargs="*", default=None,
                        help="Process specific species (space-separated)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate comparison panels")
    parser.add_argument("--exclude-inat", action="store_true",
                        help="Exclude iNaturalist images from analysis")
    args = parser.parse_args()
    print("[1] Loading inventory and annotations...")
    inventory = load_inventory()
    annotations = load_annotations()

    # Group images by species, with exclusions
    species_images = defaultdict(list)
    n_inat_excluded = 0
    for row in inventory:
        fname = row["filename"]
        ann = annotations.get(fname)
        # Exclude larvae, potential larvae, and manual exclusions
        if ann and ann["annotation_type"] in (
            "larva", "potential_larva", "manual_exclude"
        ):
            continue
        # Optionally exclude iNaturalist images
        if args.exclude_inat and row.get("directory") == "images_inaturalist":
            n_inat_excluded += 1
            continue
        species_images[row["species"]].append(row)

    # Filter to requested species if specified
    if args.species:
        requested = set(args.species)
        species_images = {sp: rows for sp, rows in species_images.items()
                         if sp in requested}

    print(f"    {len(species_images)} species")
    total_images = sum(len(rows) for rows in species_images.values())
    print(f"    {total_images} images to process")
    if args.exclude_inat:
        print(f"    {n_inat_excluded} iNaturalist images excluded")

    print("[2] Computing species-level mean L and normalizing...")
    report_rows = []
    n_processed = 0
    n_skipped = 0
    n_panels = 0

    # Track cross-species color variation for summary
    all_species_color_stats = []

    for i, (species, rows) in enumerate(sorted(species_images.items())):
        sp_dir = species_to_dirname(species)
        seg_dir = os.path.join(SEGMENTED_DIR, sp_dir)
        norm_dir = os.path.join(NORMALIZED_DIR, sp_dir)
        ensure_dir(norm_dir)

        # First pass: compute mean L for the species
        all_L_values = []
        image_data = []  # (filename, directory, png_path, rgb, mask)

        for row in rows:
            png_name = os.path.splitext(row["filename"])[0] + ".png"
            png_path = os.path.join(seg_dir, png_name)
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
            all_L_values.append(np.mean(lab[:, 0]))
            image_data.append((row["filename"], row["directory"],
                               png_path, rgb, mask))

        if not image_data:
            n_skipped += 1
            continue

        species_mean_L = float(np.mean(all_L_values))
        L_std_before = float(np.std(all_L_values))

        # Compute color variation stats
        color_data = [(f, p, r, m) for f, d, p, r, m in image_data]
        color_stats = compute_color_variation(color_data)
        color_stats["species"] = species
        all_species_color_stats.append(color_stats)

        # Collect for visualization
        seg_rgba_list = []   # original segmented (fname, directory, RGBA)
        norm_rgba_list = []  # normalized (fname, directory, RGBA)
        shifts = []
        per_image_L = []  # (orig_L, norm_L) for labels

        # Second pass: normalize each image
        for fname, directory, png_path, rgb, mask in image_data:
            body_pixels = rgb[mask]
            lab = body_pixels_to_lab(body_pixels)

            image_mean_L = float(np.mean(lab[:, 0]))
            L_shift = species_mean_L - image_mean_L
            shifts.append(L_shift)

            # Shift L channel
            lab[:, 0] += L_shift

            # Clamp to valid range
            n_clamped = int(np.sum(lab[:, 0] < 0) + np.sum(lab[:, 0] > 100))
            lab[:, 0] = np.clip(lab[:, 0], 0, 100)

            norm_mean_L = float(np.mean(lab[:, 0]))
            per_image_L.append((image_mean_L, norm_mean_L))

            # Convert back to RGB
            new_rgb = lab_to_rgb_pixels(lab)

            # Rebuild RGBA image
            h, w = rgb.shape[:2]
            rgba_norm = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_norm[mask, :3] = new_rgb
            rgba_norm[:, :, 3] = (mask * 255).astype(np.uint8)

            # Save
            out_name = os.path.splitext(fname)[0] + ".png"
            out_path = os.path.join(norm_dir, out_name)
            Image.fromarray(rgba_norm, "RGBA").save(out_path)

            # Collect for panel
            if args.visualize:
                rgba_orig = np.zeros((h, w, 4), dtype=np.uint8)
                rgba_orig[:, :, :3] = rgb
                rgba_orig[:, :, 3] = (mask * 255).astype(np.uint8)
                seg_rgba_list.append((fname, directory, rgba_orig))
                norm_rgba_list.append((fname, directory, rgba_norm))

            # Determine source for report
            source = "unknown"
            if "FishPix" in fname:
                source = "fishpix"
            elif "Bishop" in fname:
                source = "bishop"
            elif "FishBase" in fname:
                source = "fishbase"
            elif "iNaturalist" in fname:
                source = "inaturalist"

            report_rows.append({
                "filename": fname,
                "species": species,
                "source": source,
                "original_mean_L": round(image_mean_L, 2),
                "species_mean_L": round(species_mean_L, 2),
                "L_shift": round(L_shift, 2),
                "n_body_pixels": len(body_pixels),
                "n_pixels_clamped": n_clamped,
            })
            n_processed += 1

        # Compute post-normalization L std
        if shifts:
            post_means = [all_L_values[j] + shifts[j]
                         for j in range(len(shifts))]
            L_std_after = float(np.std(post_means))
        else:
            L_std_after = 0.0

        # Generate comparison panel
        if args.visualize and seg_rgba_list:
            stats = {
                "n_images": len(image_data),
                "mean_L": species_mean_L,
                "L_std_before": L_std_before,
                "L_std_after": L_std_after,
                "L_std_reduction": (1 - L_std_after / L_std_before
                                    if L_std_before > 0 else 1.0),
                "max_abs_shift": max(abs(s) for s in shifts) if shifts else 0,
                "per_image_L": per_image_L,
            }
            panel_path = os.path.join(PANELS_DIR,
                                      f"{species_to_dirname(species)}.png")
            try:
                generate_species_panel(species, seg_rgba_list, norm_rgba_list,
                                       stats, color_stats, panel_path)
                n_panels += 1
                print(f"    Panel: {panel_path}")
            except Exception as e:
                print(f"    Panel error for {species}: {e}")

        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{len(species_images)} species")

    # Write report
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    # Write color variation summary across all species
    if all_species_color_stats:
        color_var_csv = os.path.join(GMM_DIR, "species_color_variation.csv")
        color_fields = [
            "species", "n_images", "L_cv_across", "a_cv_across",
            "b_cv_across", "chroma_cv_across", "L_complexity",
            "a_complexity", "b_complexity", "max_Lab_dist",
        ]
        with open(color_var_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=color_fields,
                                    extrasaction="ignore")
            writer.writeheader()
            for cs in sorted(all_species_color_stats,
                             key=lambda x: -x.get("max_Lab_dist", 0)):
                # Round numeric values
                for k in cs:
                    if isinstance(cs[k], float):
                        cs[k] = round(cs[k], 3)
                writer.writerow(cs)
        print(f"    Color variation report: {color_var_csv}")

        # Report species with highest variation
        sorted_by_var = sorted(all_species_color_stats,
                               key=lambda x: -x.get("max_Lab_dist", 0))
        print(f"\n    Top 10 species by color variation (max ΔE(Lab)):")
        for cs in sorted_by_var[:10]:
            print(f"      {cs['species']:40s}  ΔE={cs.get('max_Lab_dist', 0):.1f}  "
                  f"n={cs.get('n_images', 0)}")

    print(f"\n[3] Normalization complete")
    print(f"    Images normalized: {n_processed}")
    print(f"    Species skipped (no segmented images): {n_skipped}")
    if args.visualize:
        print(f"    Comparison panels: {n_panels} in {PANELS_DIR}")
    print(f"    Report: {REPORT_CSV}")


if __name__ == "__main__":
    main()
