#!/usr/bin/env python3
"""
Check if Bishop Museum Randall images match FishBase Randall-credited images.
Uses web scraping since the API doesn't work reliably.

Hypothesis: All Bishop/Randall images are also on FishBase credited to Randall,
and those are the ONLY Randall-credited chaet images on FishBase.
"""

import os
import re
import time
import requests
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Configuration
BISHOP_DIR = Path("/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/images_bishop")
FISHBASE_IMAGES_DIR = Path("/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/images")
FISHBASE_EXTRA_DIR = Path("/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/images_fishbase_extra")

print("=" * 70)
print("Checking Randall Image Overlap: Bishop Museum vs FishBase (Web Scraping)")
print("=" * 70)

# Get Bishop species
bishop_files = list(BISHOP_DIR.glob("*.jpg")) + list(BISHOP_DIR.glob("*.png"))
bishop_species = set()
for f in bishop_files:
    parts = f.stem.split("_Bishop")[0]
    bishop_species.add(parts.replace("_", " "))

print(f"\nBishop Museum collection: {len(bishop_files)} images, {len(bishop_species)} species")

def get_speccode_from_summary(genus, species):
    """Get FishBase SpecCode by scraping the summary page."""
    url = f"https://www.fishbase.se/summary/{genus}-{species}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            # Look for speccode in the HTML
            match = re.search(r'speccode=(\d+)', resp.text)
            if match:
                return match.group(1)
    except Exception as e:
        print(f"    Error fetching summary: {e}")
    return None

def get_randall_images_from_photos_page(speccode):
    """Get Randall-credited images from FishBase photos page."""
    url = f"https://www.fishbase.se/photos/thumbnailssummary.php?ID={speccode}"
    randall_images = []
    all_images = []

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            # Parse all image entries with their credits
            # Pattern: Photo by Randall, J.E.</span></a><br>Location, by <a href='...'>Randall, J.E.</a> (filename.jpg)

            # Find all image entries
            pattern = r"<a class='tooltip' href='PicturesSummary\.php\?ID=\d+&what=species&pic=([^']+)'[^>]*>.*?by <a href='[^']*'>([^<]+)</a> \(([^)]+)\)"
            matches = re.findall(pattern, resp.text, re.DOTALL)

            for pic_link, photographer, filename in matches:
                all_images.append({
                    'filename': filename,
                    'photographer': photographer.strip()
                })
                if 'randall' in photographer.lower():
                    randall_images.append({
                        'filename': filename,
                        'photographer': photographer.strip()
                    })

    except Exception as e:
        print(f"    Error fetching photos: {e}")

    return randall_images, all_images

# Check all Bishop species
print("\n" + "=" * 70)
print("Querying FishBase for Randall-credited images...")
print("=" * 70)

results = []
bishop_species_list = sorted(list(bishop_species))

for i, species in enumerate(bishop_species_list):
    print(f"\n[{i+1}/{len(bishop_species_list)}] {species}")

    parts = species.split(" ")
    if len(parts) != 2:
        print(f"  WARNING: Invalid species name format")
        results.append({
            'species': species,
            'in_bishop': True,
            'speccode': None,
            'fishbase_randall_count': 0,
            'fishbase_total_images': 0,
            'randall_filenames': '',
            'note': 'Invalid name format'
        })
        continue

    genus, sp = parts

    # Get speccode
    speccode = get_speccode_from_summary(genus, sp)
    if not speccode:
        print(f"  WARNING: Could not find SpecCode")
        results.append({
            'species': species,
            'in_bishop': True,
            'speccode': None,
            'fishbase_randall_count': 0,
            'fishbase_total_images': 0,
            'randall_filenames': '',
            'note': 'SpecCode not found'
        })
        time.sleep(0.5)
        continue

    print(f"  SpecCode: {speccode}")

    # Get Randall images
    randall_images, all_images = get_randall_images_from_photos_page(speccode)

    print(f"  Total FishBase images: {len(all_images)}")
    print(f"  Randall-credited: {len(randall_images)}")

    if randall_images:
        for ri in randall_images[:3]:
            print(f"    - {ri['filename']}: {ri['photographer']}")
        if len(randall_images) > 3:
            print(f"    ... and {len(randall_images) - 3} more")

    results.append({
        'species': species,
        'in_bishop': True,
        'speccode': speccode,
        'fishbase_randall_count': len(randall_images),
        'fishbase_total_images': len(all_images),
        'randall_filenames': '; '.join([r['filename'] for r in randall_images]),
        'note': ''
    })

    time.sleep(0.5)  # Rate limiting

# Now check species NOT in Bishop
print("\n" + "=" * 70)
print("Checking species NOT in Bishop for Randall credits...")
print("=" * 70)

# Get species from our other image directories that aren't in Bishop
other_species = set()
for img_dir in [FISHBASE_IMAGES_DIR, FISHBASE_EXTRA_DIR]:
    for f in img_dir.glob("*.jpg"):
        # Extract species name
        name = f.stem
        # Remove source suffix
        for suffix in ['_FishBase', '_FishPix']:
            if suffix in name:
                name = name.split(suffix)[0]
                break
        sp = name.replace("_", " ")
        if sp not in bishop_species and len(sp.split()) == 2:
            other_species.add(sp)

print(f"\nSpecies in FishBase dirs but NOT in Bishop: {len(other_species)}")

# Check all non-Bishop species
non_bishop_list = sorted(list(other_species))
non_bishop_with_randall = []

for i, species in enumerate(non_bishop_list):
    print(f"\n[{i+1}/{len(non_bishop_list)}] {species}")

    parts = species.split(" ")
    if len(parts) != 2:
        continue

    genus, sp = parts

    speccode = get_speccode_from_summary(genus, sp)
    if not speccode:
        print(f"  WARNING: Could not find SpecCode")
        results.append({
            'species': species,
            'in_bishop': False,
            'speccode': None,
            'fishbase_randall_count': 0,
            'fishbase_total_images': 0,
            'randall_filenames': '',
            'note': 'SpecCode not found'
        })
        time.sleep(0.5)
        continue

    print(f"  SpecCode: {speccode}")

    randall_images, all_images = get_randall_images_from_photos_page(speccode)

    print(f"  Total FishBase images: {len(all_images)}")
    print(f"  Randall-credited: {len(randall_images)}")

    if randall_images:
        non_bishop_with_randall.append({
            'species': species,
            'count': len(randall_images),
            'files': [r['filename'] for r in randall_images]
        })
        print(f"  *** FOUND RANDALL IMAGES NOT IN BISHOP ***")
        for ri in randall_images:
            print(f"    - {ri['filename']}: {ri['photographer']}")

    results.append({
        'species': species,
        'in_bishop': False,
        'speccode': speccode,
        'fishbase_randall_count': len(randall_images),
        'fishbase_total_images': len(all_images),
        'randall_filenames': '; '.join([r['filename'] for r in randall_images]),
        'note': 'RANDALL NOT IN BISHOP' if randall_images else ''
    })

    time.sleep(0.5)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

df = pd.DataFrame(results)

# Bishop species with Randall on FishBase
bishop_with_randall = df[(df['in_bishop'] == True) & (df['fishbase_randall_count'] > 0)]
bishop_no_randall = df[(df['in_bishop'] == True) & (df['fishbase_randall_count'] == 0) & (df['speccode'].notna())]
bishop_not_found = df[(df['in_bishop'] == True) & (df['speccode'].isna())]

print(f"\n=== Bishop Museum Species ({len(bishop_species)} total) ===")
print(f"  With Randall images on FishBase: {len(bishop_with_randall)}")
print(f"  No Randall found on FishBase: {len(bishop_no_randall)}")
print(f"  Species not found on FishBase: {len(bishop_not_found)}")

if len(bishop_with_randall) > 0:
    total_randall = bishop_with_randall['fishbase_randall_count'].sum()
    print(f"  Total Randall images on FishBase: {total_randall}")

print(f"\n=== Non-Bishop Species ({len(non_bishop_list)} total) ===")
non_bishop_df = df[df['in_bishop'] == False]
non_bishop_with_randall_df = non_bishop_df[non_bishop_df['fishbase_randall_count'] > 0]
print(f"  With Randall images on FishBase: {len(non_bishop_with_randall_df)}")

if len(non_bishop_with_randall_df) > 0:
    print("\n  *** Species with Randall images NOT in Bishop collection: ***")
    for _, row in non_bishop_with_randall_df.iterrows():
        print(f"    - {row['species']}: {row['fishbase_randall_count']} images")
        if row['randall_filenames']:
            print(f"      Files: {row['randall_filenames']}")

# Species in Bishop but no Randall found on FishBase
if len(bishop_no_randall) > 0:
    print(f"\n  *** Bishop species with NO Randall credit found on FishBase ({len(bishop_no_randall)}): ***")
    for _, row in bishop_no_randall.iterrows():
        print(f"    - {row['species']} (SpecCode: {row['speccode']}, Total images: {row['fishbase_total_images']})")

# Hypothesis evaluation
print("\n" + "=" * 70)
print("HYPOTHESIS EVALUATION")
print("=" * 70)

print("\nHypothesis 1: All Bishop/Randall images are also on FishBase credited to Randall")
if len(bishop_with_randall) == len(bishop_species):
    print("  SUPPORTED - All Bishop species have Randall images on FishBase")
else:
    pct = len(bishop_with_randall) / len(bishop_species) * 100
    print(f"  PARTIAL - {len(bishop_with_randall)}/{len(bishop_species)} ({pct:.1f}%) Bishop species have Randall on FishBase")

print("\nHypothesis 2: FishBase Randall images are ONLY for species we have in Bishop")
if len(non_bishop_with_randall_df) == 0:
    print("  SUPPORTED - No Randall images found for species not in Bishop collection")
else:
    print(f"  NOT SUPPORTED - Found {len(non_bishop_with_randall_df)} species with Randall images not in Bishop")

# Save results
output_file = Path("/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/data/randall_fishbase_comparison.csv")
df.to_csv(output_file, index=False)
print(f"\nSaved results to: {output_file}")
