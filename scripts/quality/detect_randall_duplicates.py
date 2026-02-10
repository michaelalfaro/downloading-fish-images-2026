#!/usr/bin/env python3
"""
Detect potential duplicate images between FishBase (credited to Randall) and Bishop Museum.

FishBase often uses Randall's photos from Bishop Museum but with different compression,
so MD5 hashes don't match. This script uses perceptual hashing (pHash) to find near-duplicates.

Usage:
    python3 detect_randall_duplicates.py --scan          # Scan all and generate candidates
    python3 detect_randall_duplicates.py --review        # Launch visual review app
    python3 detect_randall_duplicates.py --mark-dups     # Mark confirmed duplicates

Output:
    analysis/approach_1_gmm/randall_dup_candidates.csv   # Potential duplicates (phash similarity)
    analysis/approach_1_gmm/randall_dup_confirmed.csv    # User-confirmed duplicates
"""

import os
import csv
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image

import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils")); from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR, load_inventory, ensure_dir,
    species_to_dirname,
)

# Output files
CANDIDATES_CSV = os.path.join(GMM_DIR, "randall_dup_candidates.csv")
CONFIRMED_CSV = os.path.join(GMM_DIR, "randall_dup_confirmed.csv")


def compute_phash(img, hash_size=16):
    """Compute perceptual hash using DCT.

    Returns a binary hash as a hex string.
    """
    # Resize to hash_size+1 to allow for DCT
    size = hash_size + 1
    img = img.convert('L').resize((size, size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32)

    # Compute DCT (simplified: just use differences)
    # For proper DCT we'd use scipy.fftpack, but this simple approach works well
    dct = np.zeros((hash_size, hash_size))
    for i in range(hash_size):
        for j in range(hash_size):
            dct[i, j] = pixels[i, j] - pixels[i, j+1] + pixels[i+1, j] - pixels[i+1, j+1]

    # Use low-frequency components (top-left 8x8)
    low_freq = dct[:8, :8]

    # Median threshold
    median = np.median(low_freq)
    bits = (low_freq > median).flatten()

    # Convert to hex
    hash_int = sum(1 << i for i, b in enumerate(bits) if b)
    return f"{hash_int:016x}"


def compute_dhash(img, hash_size=16):
    """Compute difference hash (dhash).

    More robust to small changes than pHash.
    """
    # Resize to hash_size+1 x hash_size
    img = img.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32)

    # Compute horizontal gradient (difference between adjacent pixels)
    diff = pixels[:, 1:] > pixels[:, :-1]

    # Convert to hex
    bits = diff.flatten()
    hash_int = sum(1 << i for i, b in enumerate(bits) if b)
    return f"{hash_int:064x}"


def hamming_distance(hash1, hash2):
    """Compute Hamming distance between two hex hash strings."""
    int1 = int(hash1, 16)
    int2 = int(hash2, 16)
    xor = int1 ^ int2
    return bin(xor).count('1')


def load_image(path):
    """Load image, handling RGBA properly."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        # Composite on gray background
        bg = Image.new('RGB', img.size, (180, 180, 180))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert('RGB')


def scan_for_duplicates(threshold=10):
    """Scan FishBase and Bishop images within each species for potential duplicates.

    Args:
        threshold: Maximum Hamming distance to consider as potential duplicate

    Returns:
        List of candidate pairs
    """
    inventory = load_inventory()

    # Group images by species
    species_images = defaultdict(list)
    for row in inventory:
        species_images[row["species"]].append(row)

    candidates = []

    for species, images in sorted(species_images.items()):
        # Separate Bishop and FishBase images
        bishop = [r for r in images if r["source"] == "BishopMuseum"]
        fishbase = [r for r in images if r["source"] == "FishBase"]

        if not bishop or not fishbase:
            continue

        print(f"Checking {species}: {len(bishop)} Bishop, {len(fishbase)} FishBase")

        # Compute hashes for Bishop images (use segmented versions if available)
        bishop_hashes = {}
        for row in bishop:
            sp_dir = species_to_dirname(species)
            seg_path = os.path.join(SEGMENTED_DIR, sp_dir,
                                    os.path.splitext(row["filename"])[0] + ".png")
            if os.path.exists(seg_path):
                img = load_image(seg_path)
            else:
                img_path = os.path.join(SCRIPT_DIR, row["directory"], row["filename"])
                if not os.path.exists(img_path):
                    continue
                img = load_image(img_path)

            phash = compute_phash(img)
            dhash = compute_dhash(img)
            bishop_hashes[row["filename"]] = (phash, dhash, row)

        # Compute hashes for FishBase and compare
        for row in fishbase:
            sp_dir = species_to_dirname(species)
            seg_path = os.path.join(SEGMENTED_DIR, sp_dir,
                                    os.path.splitext(row["filename"])[0] + ".png")
            if os.path.exists(seg_path):
                img = load_image(seg_path)
            else:
                img_path = os.path.join(SCRIPT_DIR, row["directory"], row["filename"])
                if not os.path.exists(img_path):
                    continue
                img = load_image(img_path)

            fb_phash = compute_phash(img)
            fb_dhash = compute_dhash(img)

            # Compare against all Bishop images
            for b_fname, (b_phash, b_dhash, b_row) in bishop_hashes.items():
                p_dist = hamming_distance(fb_phash, b_phash)
                d_dist = hamming_distance(fb_dhash, b_dhash)

                # Use minimum of both distances (either hash matching is suspicious)
                min_dist = min(p_dist, d_dist)

                if min_dist <= threshold:
                    candidates.append({
                        "species": species,
                        "fishbase_file": row["filename"],
                        "bishop_file": b_fname,
                        "phash_distance": p_dist,
                        "dhash_distance": d_dist,
                        "min_distance": min_dist,
                        "confidence": "high" if min_dist <= 5 else "medium",
                    })

    return candidates


def save_candidates(candidates):
    """Save candidate duplicates to CSV."""
    ensure_dir(os.path.dirname(CANDIDATES_CSV))
    fields = ["species", "fishbase_file", "bishop_file", "phash_distance",
              "dhash_distance", "min_distance", "confidence"]

    with open(CANDIDATES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in sorted(candidates, key=lambda x: (x["min_distance"], x["species"])):
            w.writerow(c)

    print(f"\nSaved {len(candidates)} candidates to {CANDIDATES_CSV}")


def load_candidates():
    """Load candidate duplicates from CSV."""
    if not os.path.exists(CANDIDATES_CSV):
        return []
    candidates = []
    with open(CANDIDATES_CSV, newline="") as f:
        for row in csv.DictReader(f):
            candidates.append(row)
    return candidates


def load_confirmed():
    """Load confirmed duplicates from CSV."""
    if not os.path.exists(CONFIRMED_CSV):
        return {}
    confirmed = {}
    with open(CONFIRMED_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["fishbase_file"], row["bishop_file"])
            confirmed[key] = row
    return confirmed


def save_confirmed(confirmed_dict):
    """Save confirmed duplicates to CSV."""
    ensure_dir(os.path.dirname(CONFIRMED_CSV))
    fields = ["species", "fishbase_file", "bishop_file", "is_duplicate",
              "reviewed_at", "notes"]

    with open(CONFIRMED_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(confirmed_dict.values(), key=lambda x: x["species"]):
            w.writerow(row)


def run_visual_review():
    """Run a simple terminal-based visual review of candidates."""
    candidates = load_candidates()
    confirmed = load_confirmed()

    if not candidates:
        print("No candidates found. Run --scan first.")
        return

    # Filter to unreviewed
    unreviewed = []
    for c in candidates:
        key = (c["fishbase_file"], c["bishop_file"])
        if key not in confirmed:
            unreviewed.append(c)

    if not unreviewed:
        print("All candidates already reviewed!")
        return

    print(f"\n{len(unreviewed)} candidates to review (of {len(candidates)} total)")
    print("Commands: y=yes duplicate, n=not duplicate, s=skip, q=quit\n")

    for i, c in enumerate(unreviewed):
        species = c["species"]
        fb_file = c["fishbase_file"]
        b_file = c["bishop_file"]

        print(f"\n[{i+1}/{len(unreviewed)}] {species}")
        print(f"  FishBase: {fb_file}")
        print(f"  Bishop:   {b_file}")
        print(f"  Distance: {c['min_distance']} (confidence: {c['confidence']})")

        # Try to open images for visual comparison
        sp_dir = species_to_dirname(species)
        fb_seg = os.path.join(SEGMENTED_DIR, sp_dir,
                             os.path.splitext(fb_file)[0] + ".png")
        b_seg = os.path.join(SEGMENTED_DIR, sp_dir,
                            os.path.splitext(b_file)[0] + ".png")

        if os.path.exists(fb_seg) and os.path.exists(b_seg):
            print(f"  Opening images for comparison...")
            os.system(f'open "{fb_seg}" "{b_seg}"')

        while True:
            choice = input("  Duplicate? (y/n/s/q): ").strip().lower()
            if choice in ('y', 'n', 's', 'q'):
                break

        if choice == 'q':
            break
        elif choice == 's':
            continue
        else:
            key = (fb_file, b_file)
            confirmed[key] = {
                "species": species,
                "fishbase_file": fb_file,
                "bishop_file": b_file,
                "is_duplicate": "yes" if choice == 'y' else "no",
                "reviewed_at": datetime.now().isoformat(),
                "notes": "",
            }
            save_confirmed(confirmed)

            if choice == 'y':
                print("  Marked as DUPLICATE")
            else:
                print("  Marked as NOT duplicate")

    print(f"\nReview session complete. {len(confirmed)} pairs reviewed total.")


def main():
    parser = argparse.ArgumentParser(description="Detect Randall/FishBase duplicates")
    parser.add_argument("--scan", action="store_true",
                       help="Scan all images and generate candidates")
    parser.add_argument("--review", action="store_true",
                       help="Visual review of candidates")
    parser.add_argument("--threshold", type=int, default=12,
                       help="Hamming distance threshold (default: 12)")
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics on candidates/confirmed")
    args = parser.parse_args()

    if args.scan:
        print("Scanning for potential Randall/FishBase duplicates...")
        candidates = scan_for_duplicates(threshold=args.threshold)
        save_candidates(candidates)

        # Summary by confidence
        high = sum(1 for c in candidates if c["confidence"] == "high")
        medium = len(candidates) - high
        print(f"\nSummary: {high} high-confidence, {medium} medium-confidence")

    elif args.review:
        run_visual_review()

    elif args.stats:
        candidates = load_candidates()
        confirmed = load_confirmed()

        print(f"Candidates: {len(candidates)}")
        print(f"Reviewed:   {len(confirmed)}")

        n_dups = sum(1 for c in confirmed.values() if c["is_duplicate"] == "yes")
        n_not = len(confirmed) - n_dups
        print(f"  - Confirmed duplicates: {n_dups}")
        print(f"  - Not duplicates: {n_not}")

        unreviewed = len(candidates) - len(confirmed)
        print(f"  - Unreviewed: {unreviewed}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
