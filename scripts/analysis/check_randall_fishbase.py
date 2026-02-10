#!/usr/bin/env python3
"""
Check if Bishop Museum Randall images match FishBase Randall-credited images.

Hypothesis: All Bishop/Randall images are also on FishBase credited to Randall,
and those are the ONLY Randall-credited chaet images on FishBase.
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path

# Configuration
BISHOP_DIR = Path("/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/images_bishop")
FISHBASE_IMAGES_DIR = Path("/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/images")
FISHBASE_EXTRA_DIR = Path("/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/images_fishbase_extra")

print("=" * 70)
print("Checking Randall Image Overlap: Bishop Museum vs FishBase")
print("=" * 70)

# Get Bishop species
bishop_files = list(BISHOP_DIR.glob("*.jpg")) + list(BISHOP_DIR.glob("*.png"))
bishop_species = set()
for f in bishop_files:
    # Extract species: Genus_species from Genus_species_Bishop_ID.ext
    parts = f.stem.split("_Bishop")[0]
    bishop_species.add(parts.replace("_", " "))

print(f"\nBishop Museum collection: {len(bishop_files)} images, {len(bishop_species)} species")

# FishBase API for image metadata
FISHBASE_API = "https://fishbase.ropensci.org"

def get_species_code(species_name):
    """Get FishBase species code."""
    genus, species = species_name.split(" ", 1)
    url = f"{FISHBASE_API}/species?Genus={genus}&Species={species}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                return data["data"][0].get("SpecCode")
    except Exception as e:
        pass
    return None

def get_fishbase_images(spec_code):
    """Get image metadata from FishBase for a species."""
    url = f"{FISHBASE_API}/picturesmain?SpecCode={spec_code}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [])
    except Exception as e:
        pass
    return []

def check_randall_credits(images):
    """Check which images are credited to Randall."""
    randall_images = []
    for img in images:
        # Check various credit fields
        author = img.get("AuthName", "") or ""
        credit = img.get("Credit", "") or ""

        if "randall" in author.lower() or "randall" in credit.lower():
            randall_images.append({
                'pic_name': img.get("PicName", ""),
                'author': author,
                'credit': credit,
            })
    return randall_images

# Sample a subset of Bishop species to check
print("\n" + "=" * 70)
print("Querying FishBase for Randall-credited images...")
print("=" * 70)

results = []
bishop_species_list = sorted(list(bishop_species))

# Check all Bishop species
for i, species in enumerate(bishop_species_list):
    print(f"\n[{i+1}/{len(bishop_species_list)}] {species}")

    # Get species code
    spec_code = get_species_code(species)
    if not spec_code:
        print(f"  WARNING: Species not found in FishBase API")
        results.append({
            'species': species,
            'in_bishop': True,
            'fishbase_found': False,
            'fishbase_randall_count': 0,
            'fishbase_total_images': 0,
            'note': 'API lookup failed'
        })
        time.sleep(0.3)
        continue

    # Get images
    images = get_fishbase_images(spec_code)
    randall_images = check_randall_credits(images)

    print(f"  SpecCode: {spec_code}")
    print(f"  Total FishBase images: {len(images)}")
    print(f"  Randall-credited: {len(randall_images)}")

    if randall_images:
        for ri in randall_images[:3]:
            print(f"    - {ri['pic_name']}: {ri['author']}")

    results.append({
        'species': species,
        'in_bishop': True,
        'fishbase_found': True,
        'fishbase_randall_count': len(randall_images),
        'fishbase_total_images': len(images),
        'note': ''
    })

    time.sleep(0.3)  # Rate limiting

# Now check some species NOT in Bishop to see if they have Randall images
print("\n" + "=" * 70)
print("Checking species NOT in Bishop for Randall credits...")
print("=" * 70)

# Get species from our other image directories that aren't in Bishop
other_species = set()
for img_dir in [FISHBASE_IMAGES_DIR, FISHBASE_EXTRA_DIR]:
    for f in img_dir.glob("*.jpg"):
        parts = f.stem.split("_FishBase")[0].split("_FishPix")[0]
        sp = parts.replace("_", " ")
        if sp not in bishop_species:
            other_species.add(sp)

print(f"\nSpecies in FishBase dirs but NOT in Bishop: {len(other_species)}")

# Check a sample
non_bishop_sample = sorted(list(other_species))[:20]
for i, species in enumerate(non_bishop_sample):
    print(f"\n[{i+1}/{len(non_bishop_sample)}] {species}")

    spec_code = get_species_code(species)
    if not spec_code:
        print(f"  WARNING: Species not found in FishBase API")
        results.append({
            'species': species,
            'in_bishop': False,
            'fishbase_found': False,
            'fishbase_randall_count': 0,
            'fishbase_total_images': 0,
            'note': 'API lookup failed'
        })
        time.sleep(0.3)
        continue

    images = get_fishbase_images(spec_code)
    randall_images = check_randall_credits(images)

    print(f"  Total FishBase images: {len(images)}")
    print(f"  Randall-credited: {len(randall_images)}")

    if randall_images:
        print(f"  *** FOUND RANDALL IMAGE NOT IN BISHOP ***")
        for ri in randall_images:
            print(f"    - {ri['pic_name']}: {ri['author']}")

    results.append({
        'species': species,
        'in_bishop': False,
        'fishbase_found': True,
        'fishbase_randall_count': len(randall_images),
        'fishbase_total_images': len(images),
        'note': 'RANDALL NOT IN BISHOP' if randall_images else ''
    })

    time.sleep(0.3)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

df = pd.DataFrame(results)

# Bishop species with Randall on FishBase
bishop_with_randall = df[(df['in_bishop'] == True) & (df['fishbase_randall_count'] > 0)]
bishop_no_randall = df[(df['in_bishop'] == True) & (df['fishbase_randall_count'] == 0)]
bishop_api_failed = df[(df['in_bishop'] == True) & (df['fishbase_found'] == False)]

print(f"\nBishop species ({len(bishop_species)} total):")
print(f"  - With Randall on FishBase: {len(bishop_with_randall)}")
print(f"  - No Randall found on FishBase: {len(bishop_no_randall)}")
print(f"  - API lookup failed: {len(bishop_api_failed)}")

# Non-Bishop species with Randall on FishBase
non_bishop_with_randall = df[(df['in_bishop'] == False) & (df['fishbase_randall_count'] > 0)]
print(f"\nNon-Bishop species checked ({len(non_bishop_sample)}):")
print(f"  - With Randall on FishBase: {len(non_bishop_with_randall)}")

if len(non_bishop_with_randall) > 0:
    print("\n  *** Species with Randall images NOT in Bishop collection: ***")
    for _, row in non_bishop_with_randall.iterrows():
        print(f"    - {row['species']}: {row['fishbase_randall_count']} Randall images")

# Species where Bishop has images but FishBase shows no Randall
if len(bishop_no_randall) > 0:
    print(f"\n  *** Bishop species where FishBase shows NO Randall credit: ***")
    for _, row in bishop_no_randall.iterrows():
        print(f"    - {row['species']}")

# Save results
output_file = Path("/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/data/randall_fishbase_comparison.csv")
df.to_csv(output_file, index=False)
print(f"\nSaved results to: {output_file}")
