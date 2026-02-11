#!/usr/bin/env python3
"""
Interactive Color Morphospace Visualization.

Creates an interactive HTML visualization with:
- Two panels: Original images vs Segmented images
- Hover tooltips showing image thumbnails
- Click to open image in browser/file manager
- Sources: Bishop/Randall, FishBase, FishPix, FishWise, iNaturalist, FishBaseUser

This helps explore color variation across sources to identify correction needs.
"""

import os
import sys
import json
import base64
from io import BytesIO
from collections import defaultdict

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from analysis_utils import (
    REPO_DIR, GMM_DIR, SEGMENTED_DIR,
    load_inventory, species_to_dirname,
)

# Output paths
OUTPUT_HTML = os.path.join(GMM_DIR, "color_morphospace_interactive.html")
THUMBNAIL_SIZE = (100, 100)

# Image directories
IMAGE_DIRS = {
    'BishopMuseum': os.path.join(REPO_DIR, "images_bishop"),
    'FishBase': os.path.join(REPO_DIR, "images"),
    'FishBaseExtra': os.path.join(REPO_DIR, "images_fishbase_extra"),
    'FishBaseUser': os.path.join(REPO_DIR, "images_fishbase_usercontrib"),
    'FishPix': os.path.join(REPO_DIR, "images"),
    'FishWise': os.path.join(REPO_DIR, "images_fishwise"),
    'iNaturalist': os.path.join(REPO_DIR, "images_inaturalist"),
}


def get_original_image_path(row):
    """Get path to original image file."""
    directory = row.get("directory", "images")

    # Map source to actual directory
    source = row["source"]
    if source == "BishopMuseum":
        base_dir = IMAGE_DIRS['BishopMuseum']
    elif source == "FishBaseUser":
        base_dir = IMAGE_DIRS['FishBaseUser']
    elif source == "FishWise":
        base_dir = IMAGE_DIRS['FishWise']
    elif source == "iNaturalist":
        base_dir = IMAGE_DIRS['iNaturalist']
    elif source == "FishBase" and directory == "images_fishbase_extra":
        base_dir = IMAGE_DIRS['FishBaseExtra']
    else:
        base_dir = os.path.join(REPO_DIR, directory)

    return os.path.join(base_dir, row["filename"])


def create_thumbnail_base64(image_path, size=THUMBNAIL_SIZE):
    """Create a base64-encoded thumbnail for embedding in HTML."""
    try:
        img = Image.open(image_path)
        img.thumbnail(size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=70)
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        return None


def compute_lab_stats(image_path, is_segmented=False, max_pixels=10000):
    """Compute mean a*, b* values from an image.

    For segmented images, uses alpha channel mask.
    For original images, uses all pixels.
    """
    try:
        img = Image.open(image_path)

        if is_segmented:
            img = img.convert("RGBA")
            arr = np.array(img)
            mask = arr[:, :, 3] > 0
            rgb = arr[:, :, :3]
        else:
            img = img.convert("RGB")
            arr = np.array(img)
            rgb = arr
            mask = np.ones(arr.shape[:2], dtype=bool)

        # Convert to LAB
        rgb_float = rgb.astype(np.float64) / 255.0
        lab = rgb2lab(rgb_float)

        # Get pixel values
        body_pixels = lab[mask]
        if len(body_pixels) < 100:
            return None, None, None

        # Sample if too many pixels
        if len(body_pixels) > max_pixels:
            idx = np.random.choice(len(body_pixels), max_pixels, replace=False)
            body_pixels = body_pixels[idx]

        mean_L = np.mean(body_pixels[:, 0])
        mean_a = np.mean(body_pixels[:, 1])
        mean_b = np.mean(body_pixels[:, 2])

        return mean_L, mean_a, mean_b
    except Exception as e:
        return None, None, None


def collect_color_data(inventory, include_originals=True, include_segmented=True):
    """Collect LAB color data for all images."""

    sources_to_include = ["BishopMuseum", "FishBase", "FishBaseUser", "FishPix",
                          "FishWise", "iNaturalist"]

    # Separate non-Randall FishBase from FishBase
    # We'll categorize FishBase images that are duplicates of Bishop as "Randall"

    data = []

    print("Collecting color data...")
    for i, row in enumerate(inventory):
        source = row["source"]
        if source not in sources_to_include:
            continue

        species = row["species"]
        filename = row["filename"]
        sp_dir = species_to_dirname(species)

        # Get paths
        orig_path = get_original_image_path(row)
        seg_path = os.path.join(SEGMENTED_DIR, sp_dir,
                               os.path.splitext(filename)[0] + ".png")

        # Determine display source (separate Bishop/Randall as its own category)
        display_source = source
        if source == "BishopMuseum":
            display_source = "Bishop (Randall)"

        record = {
            'filename': filename,
            'species': species,
            'source': source,
            'display_source': display_source,
            'orig_path': orig_path,
            'seg_path': seg_path,
            'has_original': os.path.exists(orig_path),
            'has_segmented': os.path.exists(seg_path),
        }

        # Compute original image stats
        if include_originals and record['has_original']:
            L, a, b = compute_lab_stats(orig_path, is_segmented=False)
            if a is not None:
                record['orig_L'] = L
                record['orig_a'] = a
                record['orig_b'] = b

        # Compute segmented image stats
        if include_segmented and record['has_segmented']:
            L, a, b = compute_lab_stats(seg_path, is_segmented=True)
            if a is not None:
                record['seg_L'] = L
                record['seg_a'] = a
                record['seg_b'] = b

        # Only include if we have at least one set of stats
        if 'orig_a' in record or 'seg_a' in record:
            data.append(record)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(inventory)} images...")

    print(f"  Collected data for {len(data)} images")
    return data


def create_interactive_plot(data):
    """Create interactive Plotly visualization with two panels."""

    # Source styling
    source_colors = {
        'Bishop (Randall)': '#2ca02c',  # Green
        'FishBase': '#1f77b4',           # Blue
        'FishBaseUser': '#9467bd',       # Purple
        'FishPix': '#ff7f0e',            # Orange
        'FishWise': '#17becf',           # Cyan
        'iNaturalist': '#d62728',        # Red
    }

    source_symbols = {
        'Bishop (Randall)': 'square',
        'FishBase': 'circle',
        'FishBaseUser': 'diamond',
        'FishPix': 'triangle-up',
        'FishWise': 'pentagon',
        'iNaturalist': 'triangle-down',
    }

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Original Images', 'Segmented Images'),
        horizontal_spacing=0.08,
    )

    # Group data by source
    for display_source in source_colors.keys():
        source_data = [d for d in data if d['display_source'] == display_source]
        if not source_data:
            continue

        color = source_colors[display_source]
        symbol = source_symbols[display_source]

        # Original images panel (left)
        orig_data = [d for d in source_data if 'orig_a' in d]
        if orig_data:
            # Create hover text with thumbnails
            hover_texts = []
            custom_data = []
            for d in orig_data:
                hover_text = (
                    f"<b>{d['species']}</b><br>"
                    f"File: {d['filename']}<br>"
                    f"Source: {d['display_source']}<br>"
                    f"L*: {d['orig_L']:.1f}, a*: {d['orig_a']:.1f}, b*: {d['orig_b']:.1f}"
                )
                hover_texts.append(hover_text)
                custom_data.append(d['orig_path'])

            fig.add_trace(
                go.Scatter(
                    x=[d['orig_a'] for d in orig_data],
                    y=[d['orig_b'] for d in orig_data],
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=8,
                        symbol=symbol,
                        line=dict(width=0.5, color='white'),
                        opacity=0.7,
                    ),
                    name=f"{display_source} (n={len(orig_data)})",
                    text=hover_texts,
                    hoverinfo='text',
                    customdata=custom_data,
                    legendgroup=display_source,
                ),
                row=1, col=1
            )

        # Segmented images panel (right)
        seg_data = [d for d in source_data if 'seg_a' in d]
        if seg_data:
            hover_texts = []
            custom_data = []
            for d in seg_data:
                hover_text = (
                    f"<b>{d['species']}</b><br>"
                    f"File: {d['filename']}<br>"
                    f"Source: {d['display_source']}<br>"
                    f"L*: {d['seg_L']:.1f}, a*: {d['seg_a']:.1f}, b*: {d['seg_b']:.1f}"
                )
                hover_texts.append(hover_text)
                custom_data.append(d['seg_path'])

            fig.add_trace(
                go.Scatter(
                    x=[d['seg_a'] for d in seg_data],
                    y=[d['seg_b'] for d in seg_data],
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=8,
                        symbol=symbol,
                        line=dict(width=0.5, color='white'),
                        opacity=0.7,
                    ),
                    name=f"{display_source}",
                    text=hover_texts,
                    hoverinfo='text',
                    customdata=custom_data,
                    legendgroup=display_source,
                    showlegend=False,  # Only show in legend once
                ),
                row=1, col=2
            )

    # Add reference lines
    for col in [1, 2]:
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=col)
        fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=col)

    # Update layout
    fig.update_layout(
        title=dict(
            text="Chaetodontidae Image Color Distribution in LAB a*b* Space<br>"
                 "<sub>Click points to open images • Hover for details</sub>",
            font=dict(size=16),
        ),
        height=700,
        width=1400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        hovermode='closest',
    )

    # Update axes
    fig.update_xaxes(title_text="a* (green ← → red)", row=1, col=1)
    fig.update_xaxes(title_text="a* (green ← → red)", row=1, col=2)
    fig.update_yaxes(title_text="b* (blue ← → yellow)", row=1, col=1)
    fig.update_yaxes(title_text="b* (blue ← → yellow)", row=1, col=2)

    # Make axes equal scale
    all_a_orig = [d['orig_a'] for d in data if 'orig_a' in d]
    all_b_orig = [d['orig_b'] for d in data if 'orig_b' in d]
    all_a_seg = [d['seg_a'] for d in data if 'seg_a' in d]
    all_b_seg = [d['seg_b'] for d in data if 'seg_b' in d]

    all_a = all_a_orig + all_a_seg
    all_b = all_b_orig + all_b_seg

    if all_a and all_b:
        margin = 5
        a_range = [min(all_a) - margin, max(all_a) + margin]
        b_range = [min(all_b) - margin, max(all_b) + margin]

        fig.update_xaxes(range=a_range, row=1, col=1)
        fig.update_xaxes(range=a_range, row=1, col=2)
        fig.update_yaxes(range=b_range, row=1, col=1)
        fig.update_yaxes(range=b_range, row=1, col=2)

    return fig


def add_thumbnail_hover_js(html_content, data):
    """Add JavaScript for thumbnail hover and click-to-open functionality."""

    # Create a mapping of coordinates to image paths for thumbnails
    # We'll use a simpler approach: embed thumbnails as base64 and create
    # a custom hover handler

    js_code = """
<script>
// Click handler to open images
document.addEventListener('DOMContentLoaded', function() {
    var plot = document.querySelector('.plotly-graph-div');
    if (plot) {
        plot.on('plotly_click', function(data) {
            var point = data.points[0];
            if (point && point.customdata) {
                // Open the file path (works on local machines)
                var path = point.customdata;
                // Try to open in new tab/window
                window.open('file://' + path, '_blank');
            }
        });
    }
});
</script>

<style>
/* Style for image preview tooltip */
.custom-tooltip {
    position: fixed;
    background: white;
    border: 2px solid #333;
    border-radius: 8px;
    padding: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 10000;
    max-width: 250px;
    pointer-events: none;
}
.custom-tooltip img {
    max-width: 200px;
    max-height: 200px;
    display: block;
    margin-bottom: 8px;
}
.custom-tooltip .info {
    font-size: 11px;
    color: #333;
}
</style>
"""

    # Insert before closing </body> tag
    html_content = html_content.replace('</body>', js_code + '</body>')

    return html_content


def generate_summary_stats(data):
    """Generate summary statistics by source."""

    stats = defaultdict(lambda: {'orig': [], 'seg': []})

    for d in data:
        source = d['display_source']
        if 'orig_a' in d:
            stats[source]['orig'].append((d['orig_L'], d['orig_a'], d['orig_b']))
        if 'seg_a' in d:
            stats[source]['seg'].append((d['seg_L'], d['seg_a'], d['seg_b']))

    print("\n" + "=" * 90)
    print("Summary Statistics by Source")
    print("=" * 90)

    print("\nOriginal Images:")
    print(f"{'Source':<20} {'N':>6} {'mean L':>10} {'mean a':>10} {'mean b':>10} {'std a':>10} {'std b':>10}")
    print("-" * 90)

    for source in ['Bishop (Randall)', 'FishBase', 'FishPix', 'FishWise', 'iNaturalist', 'FishBaseUser']:
        orig_data = stats[source]['orig']
        if orig_data:
            L_vals = [x[0] for x in orig_data]
            a_vals = [x[1] for x in orig_data]
            b_vals = [x[2] for x in orig_data]
            print(f"{source:<20} {len(orig_data):>6} {np.mean(L_vals):>10.2f} {np.mean(a_vals):>10.2f} "
                  f"{np.mean(b_vals):>10.2f} {np.std(a_vals):>10.2f} {np.std(b_vals):>10.2f}")

    print("\nSegmented Images:")
    print(f"{'Source':<20} {'N':>6} {'mean L':>10} {'mean a':>10} {'mean b':>10} {'std a':>10} {'std b':>10}")
    print("-" * 90)

    for source in ['Bishop (Randall)', 'FishBase', 'FishPix', 'FishWise', 'iNaturalist', 'FishBaseUser']:
        seg_data = stats[source]['seg']
        if seg_data:
            L_vals = [x[0] for x in seg_data]
            a_vals = [x[1] for x in seg_data]
            b_vals = [x[2] for x in seg_data]
            print(f"{source:<20} {len(seg_data):>6} {np.mean(L_vals):>10.2f} {np.mean(a_vals):>10.2f} "
                  f"{np.mean(b_vals):>10.2f} {np.std(a_vals):>10.2f} {np.std(b_vals):>10.2f}")

    # Calculate color shift from Bishop reference
    bishop_orig = stats['Bishop (Randall)']['orig']
    if bishop_orig:
        ref_a = np.mean([x[1] for x in bishop_orig])
        ref_b = np.mean([x[2] for x in bishop_orig])

        print("\n" + "=" * 90)
        print(f"Color Shift from Bishop Reference (orig a*={ref_a:.2f}, b*={ref_b:.2f})")
        print("=" * 90)
        print(f"{'Source':<20} {'Δa* (green shift)':>20} {'Δb* (blue shift)':>20}")
        print("-" * 60)

        for source in ['FishBase', 'FishPix', 'FishWise', 'iNaturalist', 'FishBaseUser']:
            orig_data = stats[source]['orig']
            if orig_data:
                mean_a = np.mean([x[1] for x in orig_data])
                mean_b = np.mean([x[2] for x in orig_data])
                delta_a = mean_a - ref_a
                delta_b = mean_b - ref_b
                print(f"{source:<20} {delta_a:>+20.2f} {delta_b:>+20.2f}")

    return stats


def main():
    print("=" * 70)
    print("Interactive Color Morphospace Visualization")
    print("=" * 70)

    # Load inventory
    print("\n[1] Loading inventory...")
    inventory = load_inventory()
    print(f"    {len(inventory)} images")

    # Count by source
    source_counts = defaultdict(int)
    for row in inventory:
        source_counts[row["source"]] += 1
    print("    Source distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"      {source}: {count}")

    # Collect color data
    print("\n[2] Analyzing image colors...")
    data = collect_color_data(inventory)

    # Generate summary stats
    stats = generate_summary_stats(data)

    # Create interactive plot
    print("\n[3] Creating interactive visualization...")
    fig = create_interactive_plot(data)

    # Save as HTML
    print(f"\n[4] Saving to {OUTPUT_HTML}")
    html_content = fig.to_html(include_plotlyjs=True, full_html=True)
    html_content = add_thumbnail_hover_js(html_content, data)

    with open(OUTPUT_HTML, 'w') as f:
        f.write(html_content)

    print("\nDone! Open the HTML file in a browser to explore.")
    print(f"  {OUTPUT_HTML}")

    # Also save data as JSON for further analysis
    json_path = OUTPUT_HTML.replace('.html', '_data.json')
    print(f"\n[5] Saving data to {json_path}")

    # Prepare JSON-serializable data
    json_data = []
    for d in data:
        record = {
            'filename': d['filename'],
            'species': d['species'],
            'source': d['source'],
            'display_source': d['display_source'],
        }
        if 'orig_a' in d:
            record['orig'] = {'L': d['orig_L'], 'a': d['orig_a'], 'b': d['orig_b']}
        if 'seg_a' in d:
            record['seg'] = {'L': d['seg_L'], 'a': d['seg_a'], 'b': d['seg_b']}
        json_data.append(record)

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
