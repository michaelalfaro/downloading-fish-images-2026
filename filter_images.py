#!/usr/bin/env python3
"""
Filter images to remove problematic files from the Chaetodontidae image database.

Filters applied:
1. Remove GIF files (typically black-and-white line drawings from old FishBase)
2. Remove known wrong-species images (C. pictus Chvag/Chpic contamination)
3. Detect cross-species duplicates (same file hash assigned to different species)
4. Flag very small images (<10KB, likely thumbnails or stamps)
5. Flag potential larvae/juveniles based on image dimensions (very small images
   that FishBase may have misclassified as adult)

This script:
- Moves filtered images to a quarantine directory (images_quarantine/)
- Updates all_images_inventory.csv with filter status
- Rebuilds species_image_summary.csv
- Reports all actions taken
"""

import os
import csv
import hashlib
import shutil
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# All image directories
IMAGE_DIRS = {
    "images": os.path.join(SCRIPT_DIR, "images"),
    "images_bishop": os.path.join(SCRIPT_DIR, "images_bishop"),
    "images_fishbase_extra": os.path.join(SCRIPT_DIR, "images_fishbase_extra"),
    "images_inaturalist": os.path.join(SCRIPT_DIR, "images_inaturalist"),
}
QUARANTINE_DIR = os.path.join(SCRIPT_DIR, "images_quarantine")

# Output files
INVENTORY_CSV = os.path.join(SCRIPT_DIR, "all_images_inventory.csv")
SUMMARY_CSV = os.path.join(SCRIPT_DIR, "species_image_summary.csv")
DEDUP_CSV = os.path.join(SCRIPT_DIR, "dedup_report.csv")
FILTER_LOG = os.path.join(SCRIPT_DIR, "filter_log.csv")

# --- Known bad images to remove ---
# These are images that FishBase incorrectly associates with a Chaetodontidae species
# but actually depict a different species entirely
KNOWN_BAD_IMAGES = {
    # C. pictus: FishBase links Chvag_u0.jpg and Chvag_u2.jpg (both C. vagabundus)
    # and Chpic_u2.jpg (Chaunax pictus, a sea toad)
    "Chaetodon_pictus_FishBase_Chpic_u2.jpg": "Wrong species: Chaunax pictus (sea toad), not Chaetodon pictus",
    "Chaetodon_pictus_FishBase_Chvag_u2.jpg": "Wrong species: duplicate of C. vagabundus image Chvag_u2.jpg mislabeled as C. pictus",
    "Chaetodon_pictus_FishBase_Chvag_u0.jpg": "Wrong species: C. vagabundus image Chvag_u0.jpg mislabeled as C. pictus by FishBase",
}


def compute_md5(filepath):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_image_dimensions(filepath):
    """Get image width and height by reading JPEG/PNG header."""
    try:
        with open(filepath, "rb") as f:
            data = f.read(50000)

        # JPEG
        if data[:2] == b'\xff\xd8':
            i = 2
            while i < len(data) - 4:
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i+1]
                if marker in (0xC0, 0xC2):  # SOF0 or SOF2
                    height = (data[i+5] << 8) + data[i+6]
                    width = (data[i+7] << 8) + data[i+8]
                    return width, height
                elif marker == 0xD9:  # EOI
                    break
                elif marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0x01):
                    i += 2
                else:
                    length = (data[i+2] << 8) + data[i+3]
                    i += 2 + length

        # PNG
        elif data[:4] == b'\x89PNG':
            width = int.from_bytes(data[16:20], 'big')
            height = int.from_bytes(data[20:24], 'big')
            return width, height

    except Exception:
        pass

    return None, None


def scan_all_images():
    """Scan all image directories and collect metadata."""
    images = []

    for dirname, dirpath in IMAGE_DIRS.items():
        if not os.path.exists(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            filepath = os.path.join(dirpath, fname)
            if not os.path.isfile(filepath):
                continue
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                continue

            fsize = os.path.getsize(filepath)
            md5 = compute_md5(filepath)
            w, h = get_image_dimensions(filepath)

            # Extract species
            parts = fname.split("_")
            species = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else fname

            # Determine source
            if "_FishPix_" in fname:
                source = "FishPix"
            elif "_FishBase_" in fname:
                source = "FishBase"
            elif "_Bishop_" in fname:
                source = "BishopMuseum"
            elif "_iNaturalist_" in fname:
                source = "iNaturalist"
            else:
                source = "unknown"

            images.append({
                "filename": fname,
                "filepath": filepath,
                "directory": dirname,
                "species": species,
                "source": source,
                "md5": md5,
                "size": fsize,
                "width": w,
                "height": h,
                "is_gif": fname.lower().endswith(".gif"),
            })

    return images


def apply_filters(images):
    """Apply all filters and return list of (image, reason) to quarantine."""
    to_quarantine = []
    reasons = {}

    # --- Filter 1: GIF files (line drawings) ---
    for img in images:
        if img["is_gif"]:
            reasons[img["filepath"]] = "GIF format: likely black-and-white line drawing"

    # --- Filter 2: Known bad images ---
    for img in images:
        if img["filename"] in KNOWN_BAD_IMAGES:
            reasons[img["filepath"]] = KNOWN_BAD_IMAGES[img["filename"]]

    # --- Filter 3: Cross-species duplicates (same hash, different species) ---
    by_hash = defaultdict(list)
    for img in images:
        by_hash[img["md5"]].append(img)

    for h, group in by_hash.items():
        if len(group) <= 1:
            continue
        species_set = set(g["species"] for g in group)
        if len(species_set) > 1:
            # This is a cross-species duplicate - flag all but keep the one
            # in the highest-priority directory
            dir_priority = {"images": 0, "images_bishop": 1, "images_fishbase_extra": 2, "images_inaturalist": 3}
            group.sort(key=lambda x: dir_priority.get(x["directory"], 99))
            kept = group[0]
            for dup in group[1:]:
                if dup["filepath"] not in reasons:
                    reasons[dup["filepath"]] = (
                        f"Cross-species duplicate: same image as "
                        f"{kept['species']} ({kept['directory']}/{kept['filename']})"
                    )

    # --- Filter 3b: Within-species duplicates (same hash) ---
    for h, group in by_hash.items():
        if len(group) <= 1:
            continue
        species_set = set(g["species"] for g in group)
        if len(species_set) == 1:
            dir_priority = {"images": 0, "images_bishop": 1, "images_fishbase_extra": 2, "images_inaturalist": 3}
            group.sort(key=lambda x: dir_priority.get(x["directory"], 99))
            kept = group[0]
            for dup in group[1:]:
                if dup["filepath"] not in reasons:
                    reasons[dup["filepath"]] = (
                        f"Within-species duplicate: same image as "
                        f"{kept['directory']}/{kept['filename']}"
                    )

    # --- Filter 4: Very small files (<8KB) likely stamps or thumbnails ---
    for img in images:
        if img["size"] < 8000 and img["filepath"] not in reasons:
            reasons[img["filepath"]] = f"Very small file ({img['size']} bytes): likely stamp or thumbnail"

    # --- Filter 5: Potential larvae/juveniles ---
    # Flag images where both dimensions are very small (likely tholichthys larvae)
    # FishBase larvae images tend to be <200px on the short side
    for img in images:
        if img["filepath"] in reasons:
            continue
        w, h = img["width"], img["height"]
        if w and h:
            short_side = min(w, h)
            if short_side < 150 and img["source"] in ("FishBase", "FishPix"):
                reasons[img["filepath"]] = (
                    f"Potential larva/juvenile: very small image ({w}x{h}px)"
                )

    return reasons


def quarantine_images(images, reasons):
    """Move filtered images to quarantine directory."""
    os.makedirs(QUARANTINE_DIR, exist_ok=True)

    quarantined = []
    for img in images:
        if img["filepath"] in reasons:
            reason = reasons[img["filepath"]]
            dest = os.path.join(QUARANTINE_DIR, img["filename"])

            # Handle name conflicts in quarantine
            if os.path.exists(dest):
                base, ext = os.path.splitext(img["filename"])
                dest = os.path.join(QUARANTINE_DIR, f"{base}_from_{img['directory']}{ext}")

            shutil.move(img["filepath"], dest)
            quarantined.append({
                "filename": img["filename"],
                "species": img["species"],
                "source": img["source"],
                "directory": img["directory"],
                "reason": reason,
                "quarantine_path": dest,
            })

    return quarantined


def rebuild_inventory_and_summary(tree_species_file=None):
    """Rebuild inventory and summary CSVs after filtering."""
    from download_additional_images import (
        parse_tree_species, build_full_inventory,
        deduplicate, build_species_summary,
    )

    # Parse tree
    import re
    tree_path = os.path.join(SCRIPT_DIR, "tree", "Butterflyfish_concat_final.tre")
    tree_species = parse_tree_species(tree_path)

    # Rebuild inventory
    inventory = build_full_inventory()
    dedup_log, n_dups = deduplicate(inventory)

    # Write inventory
    with open(INVENTORY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "species", "filename", "source", "directory", "md5_hash", "is_duplicate",
        ])
        writer.writeheader()
        for item in inventory:
            writer.writerow({
                "species": item["species"],
                "filename": item["filename"],
                "source": item["source"],
                "directory": item["directory"],
                "md5_hash": item["md5_hash"],
                "is_duplicate": item["is_duplicate"],
            })

    # Write dedup
    with open(DEDUP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "md5_hash", "kept_file", "kept_source",
            "duplicate_file", "duplicate_source", "species",
        ])
        writer.writeheader()
        if dedup_log:
            writer.writerows(dedup_log)

    # Build summary
    summary_rows = build_species_summary(inventory, tree_species)
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "species", "in_tree", "n_fishpix", "n_fishbase", "n_bishop",
            "n_fishbase_extra", "n_inaturalist", "n_total", "sources",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    return inventory, summary_rows


def main():
    print("=" * 70)
    print("Image Quality Filter")
    print("=" * 70)

    # Scan all images
    print("\n[1] Scanning all image directories...")
    images = scan_all_images()
    print(f"    Found {len(images)} total images")

    # Count by type
    n_gif = sum(1 for i in images if i["is_gif"])
    print(f"    GIF files: {n_gif}")

    # Apply filters
    print("\n[2] Applying filters...")
    reasons = apply_filters(images)
    print(f"    Images to quarantine: {len(reasons)}")

    # Categorize
    categories = defaultdict(int)
    for fp, reason in reasons.items():
        if "GIF format" in reason:
            categories["GIF/line drawing"] += 1
        elif "Wrong species" in reason or "Cross-species" in reason:
            categories["Wrong species"] += 1
        elif "Within-species duplicate" in reason:
            categories["Within-species duplicate"] += 1
        elif "Very small" in reason:
            categories["Very small/stamp"] += 1
        elif "larva" in reason.lower():
            categories["Potential larva"] += 1
        else:
            categories["Other"] += 1

    for cat, count in sorted(categories.items()):
        print(f"      {cat}: {count}")

    # Show details
    print("\n[3] Filter details:")
    for img in images:
        if img["filepath"] in reasons:
            print(f"    {img['directory']}/{img['filename']}")
            print(f"      -> {reasons[img['filepath']]}")

    # Quarantine
    print(f"\n[4] Moving {len(reasons)} images to quarantine...")
    quarantined = quarantine_images(images, reasons)

    # Write filter log
    with open(FILTER_LOG, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "species", "source", "directory", "reason",
        ])
        writer.writeheader()
        for q in quarantined:
            writer.writerow({
                "filename": q["filename"],
                "species": q["species"],
                "source": q["source"],
                "directory": q["directory"],
                "reason": q["reason"],
            })
    print(f"    Filter log: {FILTER_LOG}")

    # Rebuild inventory and summary
    print("\n[5] Rebuilding inventory and summary...")
    inventory, summary_rows = rebuild_inventory_and_summary()

    total_remaining = len(inventory)
    unique_remaining = sum(1 for i in inventory if i["is_duplicate"] == "no")
    print(f"    Total images remaining: {total_remaining}")
    print(f"    Unique images remaining: {unique_remaining}")

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Quarantined: {len(quarantined)} images")
    print(f"  Remaining:   {total_remaining} images")
    print(f"  Quarantine dir: {QUARANTINE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
