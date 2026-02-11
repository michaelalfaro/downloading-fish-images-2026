#!/usr/bin/env python3
"""
Unified Image Metadata System for Fish Image Downloads.

This module provides a centralized way to track:
- Image source (Bishop, FishBase, FishPix, FishWise, iNaturalist)
- Study provenance (Miyazawa 2020, etc.)
- Quality attributes (is_randall, is_underwater, etc.)
- Processing status (segmented, normalized, oriented)

The metadata is stored in data/image_metadata.csv and can be loaded
by the visualizer and analysis scripts.

Usage:
    from image_metadata import ImageMetadataManager

    mgr = ImageMetadataManager()
    mgr.add_image("Chaetodon_auriga_FishPix_12345AF.jpg", {
        'species': 'Chaetodon auriga',
        'source': 'FishPix',
        'in_miyazawa': True,
    })
    mgr.save()
"""

import os
import csv
import json
import hashlib
from datetime import datetime
from collections import defaultdict

import openpyxl


# ---------- paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
METADATA_CSV = os.path.join(DATA_DIR, "image_metadata.csv")
MIYAZAWA_XLSX = os.path.join(PROJECT_DIR, "papers", "abb9107_data_file_s1.xlsx")


# Metadata fields
METADATA_FIELDS = [
    "filename",           # Local filename
    "species",            # Canonical species name
    "source",             # BishopMuseum, FishBase, FishPix, FishWise, iNaturalist
    "directory",          # Subdirectory where stored
    "original_id",        # Original ID from source database
    "url",                # Source URL
    "photographer",       # Photographer name if known
    "is_randall",         # True if Randall photograph
    "in_miyazawa_2020",   # True if used in Miyazawa (2020) study
    "miyazawa_source",    # FishBase or FishPix in Miyazawa
    "is_underwater",      # True if detected as underwater photo
    "has_segmented",      # True if segmentation exists
    "has_normalized",     # True if normalized version exists
    "has_oriented",       # True if orientation fixed
    "md5_hash",           # File hash for duplicate detection
    "file_size",          # File size in bytes
    "added_date",         # Date added to collection
    "notes",              # Free-form notes
]


class MiyazawaData:
    """Loader for Miyazawa (2020) supplementary data."""

    def __init__(self, xlsx_path=MIYAZAWA_XLSX):
        self.xlsx_path = xlsx_path
        self._images = None
        self._species = None

    def _load(self):
        """Lazy load the data."""
        if self._images is not None:
            return

        self._images = {}  # img_file -> {species, source, patterns...}
        self._species = {}  # species -> {source, patterns...}

        if not os.path.exists(self.xlsx_path):
            print(f"WARNING: Miyazawa data not found: {self.xlsx_path}")
            return

        try:
            wb = openpyxl.load_workbook(self.xlsx_path, read_only=True)

            # Load image-level data
            ws_img = wb["A_FishPatterns_img"]
            headers = None
            for row in ws_img.iter_rows(values_only=True):
                if headers is None:
                    headers = row
                    continue

                img_file = row[0]
                self._images[img_file] = {
                    'img_file': row[0],
                    'source': row[1],
                    'family': row[2],
                    'genus': row[3],
                    'species': row[4],
                    # Pattern annotations
                    'Mono': row[5],
                    'Bltc': row[6],
                    'Sp_D': row[7],
                    'Sp_L': row[8],
                    'Maze': row[9],
                    'St_H': row[10],
                    'St_D': row[11],
                    'St_V': row[12],
                    'Sddl': row[13],
                    'Eyes': row[14],
                    'Area': row[15],
                }

            # Load species-level data
            ws_sp = wb["B_FishPatterns_sp"]
            headers = None
            for row in ws_sp.iter_rows(values_only=True):
                if headers is None:
                    headers = row
                    continue

                species = row[3]  # species column
                if species:
                    self._species[species] = dict(zip(headers, row))

            wb.close()

        except Exception as e:
            print(f"WARNING: Could not load Miyazawa data: {e}")

    @property
    def images(self):
        self._load()
        return self._images

    @property
    def species(self):
        self._load()
        return self._species

    def is_miyazawa_image(self, filename):
        """Check if an image filename appears in Miyazawa (2020)."""
        self._load()

        # Try exact match first
        if filename in self._images:
            return True

        # Try stripping prefix (our naming convention adds species prefix)
        # e.g., "Chaetodon_auriga_FishPix_12345AF.jpg" -> "12345AF.jpg"
        parts = filename.split('_')
        for i in range(len(parts)):
            suffix = '_'.join(parts[i:])
            if suffix in self._images:
                return True

        # Try just the ID part for FishPix (12345AF.jpg)
        import re
        match = re.search(r'(\d+AF\.jpg)', filename, re.IGNORECASE)
        if match and match.group(1) in self._images:
            return True

        # Try FishBase style (e.g., "Chaur_u0.jpg")
        match = re.search(r'([A-Z][a-z]{3,4}_u\d+\.jpg)', filename, re.IGNORECASE)
        if match and match.group(1) in self._images:
            return True

        return False

    def get_miyazawa_info(self, filename):
        """Get Miyazawa data for an image if it exists."""
        self._load()

        # Try various matching strategies
        for img_file, data in self._images.items():
            if img_file in filename or filename.endswith(img_file):
                return data

        return None

    def get_chaetodontidae_images(self):
        """Get all Chaetodontidae images from Miyazawa data."""
        self._load()
        return {
            k: v for k, v in self._images.items()
            if v.get('family') == 'Chaetodontidae'
        }


class ImageMetadataManager:
    """Manager for unified image metadata."""

    def __init__(self, csv_path=METADATA_CSV):
        self.csv_path = csv_path
        self.metadata = {}  # filename -> dict
        self.miyazawa = MiyazawaData()
        self._load()

    def _load(self):
        """Load existing metadata from CSV."""
        if not os.path.exists(self.csv_path):
            return

        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.metadata[row["filename"]] = row

    def save(self):
        """Save metadata to CSV."""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS,
                                   extrasaction='ignore')
            writer.writeheader()
            for filename in sorted(self.metadata.keys()):
                writer.writerow(self.metadata[filename])

    def add_image(self, filename, data):
        """Add or update image metadata."""
        existing = self.metadata.get(filename, {})

        # Merge with existing
        record = {field: "" for field in METADATA_FIELDS}
        record.update(existing)
        record.update(data)
        record["filename"] = filename

        # Auto-detect Miyazawa status
        if not record.get("in_miyazawa_2020"):
            if self.miyazawa.is_miyazawa_image(filename):
                record["in_miyazawa_2020"] = "yes"
                minfo = self.miyazawa.get_miyazawa_info(filename)
                if minfo:
                    record["miyazawa_source"] = minfo.get("source", "")

        # Set added date if new
        if not record.get("added_date"):
            record["added_date"] = datetime.now().strftime("%Y-%m-%d")

        self.metadata[filename] = record
        return record

    def get(self, filename):
        """Get metadata for a single image."""
        return self.metadata.get(filename)

    def get_by_source(self, source):
        """Get all images from a specific source."""
        return {
            k: v for k, v in self.metadata.items()
            if v.get("source") == source
        }

    def get_miyazawa_images(self):
        """Get all images that appear in Miyazawa (2020)."""
        return {
            k: v for k, v in self.metadata.items()
            if v.get("in_miyazawa_2020") == "yes"
        }

    def compute_file_hash(self, filepath):
        """Compute MD5 hash of a file."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def update_from_inventory(self, inventory_csv):
        """Import data from existing all_images_inventory.csv."""
        with open(inventory_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "")
                if not filename:
                    continue

                self.add_image(filename, {
                    "species": row.get("species", ""),
                    "source": row.get("source", ""),
                    "directory": row.get("directory", ""),
                    "md5_hash": row.get("md5_hash", ""),
                })

    def summary(self):
        """Print summary statistics."""
        sources = defaultdict(int)
        miyazawa_count = 0
        randall_count = 0

        for data in self.metadata.values():
            sources[data.get("source", "Unknown")] += 1
            if data.get("in_miyazawa_2020") == "yes":
                miyazawa_count += 1
            if data.get("is_randall") == "yes":
                randall_count += 1

        print(f"Total images: {len(self.metadata)}")
        print(f"In Miyazawa (2020): {miyazawa_count}")
        print(f"Randall photos: {randall_count}")
        print("\nBy source:")
        for source, count in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"  {source}: {count}")


def main():
    """Build unified metadata from existing inventory and manifests."""
    print("=" * 60)
    print("Building Unified Image Metadata")
    print("=" * 60)

    mgr = ImageMetadataManager()

    # Import from existing inventory
    inventory_csv = os.path.join(DATA_DIR, "all_images_inventory.csv")
    if os.path.exists(inventory_csv):
        print(f"\n[1] Importing from {inventory_csv}...")
        mgr.update_from_inventory(inventory_csv)
        print(f"    {len(mgr.metadata)} images imported")

    # Import FishWise manifest
    fishwise_manifest = os.path.join(DATA_DIR, "fishwise", "fishwise_download_manifest.csv")
    if os.path.exists(fishwise_manifest):
        print(f"\n[2] Importing FishWise manifest...")
        with open(fishwise_manifest, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mgr.add_image(row.get("local_filename", ""), {
                    "species": row.get("species", ""),
                    "source": "FishWise",
                    "photographer": row.get("photographer", ""),
                    "url": row.get("full_url", ""),
                    "original_id": row.get("sid", ""),
                })

    # Import FishPix manifest if exists
    fishpix_manifest = os.path.join(DATA_DIR, "fishpix", "fishpix_download_manifest.csv")
    if os.path.exists(fishpix_manifest):
        print(f"\n[3] Importing FishPix manifest...")
        with open(fishpix_manifest, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mgr.add_image(row.get("local_filename", ""), {
                    "species": row.get("species", ""),
                    "source": "FishPix",
                    "url": row.get("url", ""),
                    "original_id": row.get("pic_id", ""),
                    "in_miyazawa_2020": row.get("in_miyazawa", ""),
                })

    # Auto-detect Miyazawa images from existing filenames
    print(f"\n[4] Detecting Miyazawa (2020) images...")
    miyazawa_count = 0
    for filename, data in mgr.metadata.items():
        if mgr.miyazawa.is_miyazawa_image(filename):
            if data.get("in_miyazawa_2020") != "yes":
                data["in_miyazawa_2020"] = "yes"
                minfo = mgr.miyazawa.get_miyazawa_info(filename)
                if minfo:
                    data["miyazawa_source"] = minfo.get("source", "")
                miyazawa_count += 1

    print(f"    Detected {miyazawa_count} additional Miyazawa images")

    # Save
    print(f"\n[5] Saving to {mgr.csv_path}...")
    mgr.save()

    # Summary
    print("\n" + "=" * 60)
    mgr.summary()


if __name__ == "__main__":
    main()
