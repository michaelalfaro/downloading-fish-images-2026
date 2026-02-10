#!/usr/bin/env python3
"""
Reconcile taxonomy and source attribution between Bishop Museum and FishBase.

This script:
1. Identifies Bishop/Randall images and their FishBase equivalents
2. Resolves taxonomic discrepancies (Roa vs Chaetodon, zanzibar spelling)
3. Creates mapping for image source priority (Bishop > FishBase)
4. Detects duplicate images via hash comparison
5. Updates the inventory and exemplar files

Key findings from Randall analysis:
- All 97 Bishop species have Randall images on FishBase (same images)
- 15 additional species have Randall-credited FishBase images not in Bishop
- Roa species in Bishop use old genus "Chaetodon" (excelsa, modesta, jayakari)
- Chaetodon zanzibariensis (Bishop) = Chaetodon zanzibarensis (FishBase)
"""

import os
import sys
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import cv2
import numpy as np

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Configuration
REPO_DIR = Path("/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026")
DATA_DIR = REPO_DIR / "data"
ANALYSIS_DIR = REPO_DIR / "analysis" / "approach_1_gmm"

# Image directories
IMAGE_DIRS = {
    'bishop': REPO_DIR / "images_bishop",
    'fishbase': REPO_DIR / "images",
    'fishbase_extra': REPO_DIR / "images_fishbase_extra",
    'fishbase_usercontrib': REPO_DIR / "images_fishbase_usercontrib",
    'inaturalist': REPO_DIR / "images_inaturalist",
}

# Taxonomy mapping: Bishop name -> Current accepted name
TAXONOMY_MAP = {
    # Roa species (Bishop uses old Chaetodon names)
    'Chaetodon excelsa': 'Roa excelsa',
    'Chaetodon jayakari': 'Roa jayakari',
    'Chaetodon modesta': 'Roa modesta',
    # Spelling variants (same species)
    'Chaetodon zanzibariensis': 'Chaetodon zanzibarensis',
}

# Species that have Randall images on FishBase but NOT in Bishop collection
# (from Randall analysis report)
FISHBASE_ONLY_RANDALL = [
    'Chaetodon burgessi',     # Chbur_u0.jpg
    'Chaetodon capistratus',  # Chcap_u0.jpg
    'Chaetodon interruptus',  # Chuni_u2.jpg, Chuni_u3.jpg
    'Chaetodon oxycephalus',  # Choxy_u0.jpg
    'Chaetodon quadrimaculatus',  # Chqua_u2.jpg
    'Chaetodon sedentarius',  # Chsed_u1.jpg (currently exemplar!)
    'Chaetodon striatus',     # Chstr_u0.jpg
    'Chaetodon triangulum',   # Chtri_ui.jpg
    'Hemitaurichthys thompsoni',  # Hetho_u0.jpg (currently exemplar!)
    'Johnrandallia nigrirostris', # Jonig_u0.jpg
    'Prognathodes aculeatus', # Chacu_u1.jpg
    'Roa excelsa',            # Chexc_u0.jpg (currently exemplar!)
    'Roa jayakari',           # Chjay_u1.jpg
    'Roa modesta',            # Chmod_u1.jpg (currently exemplar!)
]

def compute_image_hash(image_path, hash_size=16):
    """Compute perceptual hash for image comparison."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        # Resize to small square
        img = cv2.resize(img, (hash_size, hash_size))
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Compute average
        avg = gray.mean()
        # Create binary hash
        hash_bits = gray > avg
        return hash_bits.flatten().tobytes()
    except Exception as e:
        print(f"  Error hashing {image_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    """Compute Hamming distance between two hashes."""
    if hash1 is None or hash2 is None:
        return float('inf')
    h1 = np.frombuffer(hash1, dtype=np.bool_)
    h2 = np.frombuffer(hash2, dtype=np.bool_)
    return np.sum(h1 != h2)

def get_species_from_filename(filename):
    """Extract species name from filename."""
    stem = Path(filename).stem
    # Remove source suffixes
    for suffix in ['_Bishop', '_FishBase', '_FishPix', '_FishBaseUser', '_iNaturalist']:
        if suffix in stem:
            stem = stem.split(suffix)[0]
            break
    # Also handle "Background Removed" suffix
    if ' Background Removed' in stem:
        stem = stem.split(' Background Removed')[0]
    return stem.replace('_', ' ')

def get_canonical_species(species):
    """Get canonical (accepted) species name."""
    return TAXONOMY_MAP.get(species, species)

def build_image_inventory():
    """Build comprehensive inventory of all images with source attribution."""
    print("=" * 70)
    print("Building Image Inventory with Source Attribution")
    print("=" * 70)

    records = []

    for source_name, source_dir in IMAGE_DIRS.items():
        if not source_dir.exists():
            print(f"  WARNING: {source_dir} does not exist")
            continue

        files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
        print(f"\n{source_name}: {len(files)} images")

        for f in files:
            species = get_species_from_filename(f.name)
            canonical = get_canonical_species(species)

            # Determine if this is a Randall image
            is_randall = False
            if source_name == 'bishop':
                is_randall = True  # All Bishop images are Randall
            elif source_name in ['fishbase', 'fishbase_extra']:
                # Check if species is in the FishBase-only Randall list
                if canonical in FISHBASE_ONLY_RANDALL:
                    is_randall = True
                # Or if Bishop has this species (then FishBase version is same Randall image)
                bishop_species = set()
                for bf in IMAGE_DIRS['bishop'].glob("*.jpg"):
                    bp = get_species_from_filename(bf.name)
                    bishop_species.add(get_canonical_species(bp))
                if canonical in bishop_species:
                    is_randall = True

            records.append({
                'filename': f.name,
                'filepath': str(f),
                'source': source_name,
                'species_raw': species,
                'species_canonical': canonical,
                'is_randall': is_randall,
                'taxonomy_updated': species != canonical,
            })

    df = pd.DataFrame(records)
    print(f"\nTotal images: {len(df)}")
    print(f"Randall images: {df['is_randall'].sum()}")
    print(f"Taxonomy updates needed: {df['taxonomy_updated'].sum()}")

    return df

def find_duplicate_images(inventory_df):
    """Find duplicate images between Bishop and FishBase collections."""
    print("\n" + "=" * 70)
    print("Finding Duplicate Images (Hash Comparison)")
    print("=" * 70)

    # Group by canonical species
    species_groups = inventory_df.groupby('species_canonical')

    duplicates = []

    for species, group in species_groups:
        # Get Bishop images for this species
        bishop_imgs = group[group['source'] == 'bishop']
        fishbase_imgs = group[group['source'].isin(['fishbase', 'fishbase_extra'])]

        if len(bishop_imgs) == 0 or len(fishbase_imgs) == 0:
            continue

        print(f"\n{species}: {len(bishop_imgs)} Bishop, {len(fishbase_imgs)} FishBase")

        # Compute hashes for all images
        bishop_hashes = {}
        for _, row in bishop_imgs.iterrows():
            h = compute_image_hash(row['filepath'])
            if h:
                bishop_hashes[row['filename']] = h

        # Compare FishBase to Bishop
        for _, fb_row in fishbase_imgs.iterrows():
            fb_hash = compute_image_hash(fb_row['filepath'])
            if fb_hash is None:
                continue

            for b_filename, b_hash in bishop_hashes.items():
                dist = hamming_distance(fb_hash, b_hash)
                # Threshold: images with <20% different bits are likely duplicates
                if dist < 52:  # 16*16*0.2 = 51.2
                    print(f"  DUPLICATE: {fb_row['filename']} ~ {b_filename} (dist={dist})")
                    duplicates.append({
                        'species': species,
                        'bishop_file': b_filename,
                        'fishbase_file': fb_row['filename'],
                        'hash_distance': dist,
                    })

    print(f"\nTotal duplicates found: {len(duplicates)}")
    return pd.DataFrame(duplicates)

def update_exemplar_file(inventory_df, duplicates_df):
    """Update exemplar file with Bishop priority and taxonomy fixes."""
    print("\n" + "=" * 70)
    print("Updating Exemplar File")
    print("=" * 70)

    exemplar_file = ANALYSIS_DIR / "species_exemplar.csv"
    exemplars = pd.read_csv(exemplar_file)

    print(f"Current exemplars: {len(exemplars)}")

    changes = []

    # 1. Fix taxonomy issues
    for old_name, new_name in TAXONOMY_MAP.items():
        mask = exemplars['species'] == old_name
        if mask.any():
            print(f"\nTaxonomy fix: {old_name} -> {new_name}")

            # Check if new_name already exists
            if (exemplars['species'] == new_name).any():
                # Duplicate! Merge records (keep Bishop if available)
                old_record = exemplars[exemplars['species'] == old_name].iloc[0]
                new_record = exemplars[exemplars['species'] == new_name].iloc[0]

                # Prefer Bishop source
                if 'Bishop' in old_record['filename']:
                    print(f"  Keeping Bishop version: {old_record['filename']}")
                    exemplars.loc[exemplars['species'] == new_name, 'filename'] = old_record['filename']
                    exemplars.loc[exemplars['species'] == new_name, 'png_name'] = old_record['png_name']
                    exemplars.loc[exemplars['species'] == new_name, 'source'] = 'Bishop'
                    # Remove old record
                    exemplars = exemplars[exemplars['species'] != old_name]
                    changes.append(f"Merged {old_name} into {new_name} (kept Bishop)")
                else:
                    print(f"  Removing duplicate: {old_name}")
                    exemplars = exemplars[exemplars['species'] != old_name]
                    changes.append(f"Removed duplicate {old_name} (same as {new_name})")
            else:
                # Just rename
                exemplars.loc[mask, 'species'] = new_name
                changes.append(f"Renamed {old_name} -> {new_name}")

    # 2. Add notes column for Bishop/Randall attribution
    if 'is_randall' not in exemplars.columns:
        exemplars['is_randall'] = False

    for idx, row in exemplars.iterrows():
        if 'Bishop' in row['filename']:
            exemplars.loc[idx, 'is_randall'] = True
            exemplars.loc[idx, 'source'] = 'Bishop'
        elif row['species'] in FISHBASE_ONLY_RANDALL:
            exemplars.loc[idx, 'is_randall'] = True

    # 3. For species with duplicates, prefer Bishop version
    for _, dup_row in duplicates_df.iterrows():
        species = dup_row['species']
        bishop_file = dup_row['bishop_file']

        mask = exemplars['species'] == species
        if mask.any():
            current = exemplars[mask].iloc[0]
            if 'FishBase' in current['filename']:
                print(f"\nUpdating {species}: {current['filename']} -> {bishop_file}")
                png_name = bishop_file.replace('.jpg', '.png').replace('.JPG', '.png')
                exemplars.loc[mask, 'filename'] = bishop_file
                exemplars.loc[mask, 'png_name'] = png_name
                exemplars.loc[mask, 'source'] = 'Bishop'
                exemplars.loc[mask, 'is_randall'] = True
                changes.append(f"Switched to Bishop: {species}")

    # Save updated file
    backup_file = exemplar_file.with_suffix('.csv.bak')
    pd.read_csv(exemplar_file).to_csv(backup_file, index=False)
    print(f"\nBackup saved to: {backup_file}")

    exemplars.to_csv(exemplar_file, index=False)
    print(f"Updated exemplar file: {exemplar_file}")
    print(f"Total changes: {len(changes)}")

    return changes

def update_gestalt_k_file():
    """Update gestalt_k file with taxonomy fixes."""
    print("\n" + "=" * 70)
    print("Updating Gestalt K File")
    print("=" * 70)

    gestalt_file = ANALYSIS_DIR / "species_gestalt_k.csv"
    gestalt = pd.read_csv(gestalt_file)

    changes = []

    for old_name, new_name in TAXONOMY_MAP.items():
        mask = gestalt['species'] == old_name
        if mask.any():
            # Check if new_name already exists
            if (gestalt['species'] == new_name).any():
                # Merge - keep the one that's reviewed
                old_record = gestalt[gestalt['species'] == old_name].iloc[0]
                new_record = gestalt[gestalt['species'] == new_name].iloc[0]

                # Use 'review_status' column (not 'status')
                if old_record['review_status'] == 'reviewed' and new_record['review_status'] != 'reviewed':
                    gestalt.loc[gestalt['species'] == new_name, 'gestalt_k'] = old_record['gestalt_k']
                    gestalt.loc[gestalt['species'] == new_name, 'review_status'] = 'reviewed'

                gestalt = gestalt[gestalt['species'] != old_name]
                changes.append(f"Merged {old_name} into {new_name}")
            else:
                gestalt.loc[mask, 'species'] = new_name
                changes.append(f"Renamed {old_name} -> {new_name}")

    # Save
    backup_file = gestalt_file.with_suffix('.csv.bak')
    pd.read_csv(gestalt_file).to_csv(backup_file, index=False)
    gestalt.to_csv(gestalt_file, index=False)

    print(f"Changes: {len(changes)}")
    return changes

def generate_report(inventory_df, duplicates_df, exemplar_changes, gestalt_changes):
    """Generate comprehensive report."""
    print("\n" + "=" * 70)
    print("Generating Report")
    print("=" * 70)

    report = f"""# Taxonomy and Source Reconciliation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report documents the reconciliation of taxonomy and source attribution between
Bishop Museum (Randall) and FishBase image collections.

---

## Taxonomy Issues Resolved

### Roa Genus (formerly Chaetodon)

The Bishop Museum collection uses the old genus assignment "Chaetodon" for three species
that are now placed in genus Roa:

| Bishop Name | Accepted Name | Resolution |
|-------------|---------------|------------|
| Chaetodon excelsa | Roa excelsa | Merged (same species) |
| Chaetodon jayakari | Roa jayakari | No Bishop image (FishBase only) |
| Chaetodon modesta | Roa modesta | No Bishop image (FishBase only) |

**Finding:** The Bishop `Chaetodon_excelsa` image and FishBase `Roa_excelsa` image
are the SAME Randall photograph. We now use the Bishop version with updated taxonomy.

### Zanzibar Butterflyfish Spelling

| Variant | Source | Resolution |
|---------|--------|------------|
| Chaetodon zanzibariensis | Bishop Museum | Preferred (correct spelling) |
| Chaetodon zanzibarensis | FishBase | Synonym, removed as duplicate |

**Note:** The correct spelling is "zanzibarensis" (from Zanzibar). We keep the Bishop
image and remove the FishBase duplicate entry.

---

## Image Source Attribution

### Bishop Museum / Randall Images

All 178 images in `images_bishop/` are from the John Randall collection at Bishop Museum.
These are photographed in controlled conditions and represent the gold standard for
color reference.

### FishBase Randall Images

The following species have Randall-credited images on FishBase that are NOT in the
Bishop collection (i.e., FishBase-only Randall images):

| Species | FishBase File | Notes |
|---------|---------------|-------|
| Chaetodon burgessi | Chbur_u0.jpg | Alternative to current exemplar |
| Chaetodon capistratus | Chcap_u0.jpg | Alternative to current exemplar |
| Chaetodon interruptus | Chuni_u2.jpg | **Currently used as exemplar** |
| Chaetodon oxycephalus | Choxy_u0.jpg | Alternative available |
| Chaetodon quadrimaculatus | Chqua_u2.jpg | Currently using juvenile (j0) |
| Chaetodon sedentarius | Chsed_u1.jpg | **Currently used as exemplar** |
| Chaetodon striatus | Chstr_u0.jpg | Alternative available |
| Chaetodon triangulum | Chtri_ui.jpg | Alternative available |
| Hemitaurichthys thompsoni | Hetho_u0.jpg | **Currently used as exemplar** |
| Johnrandallia nigrirostris | Jonig_u0.jpg | Alternative available |
| Prognathodes aculeatus | Chacu_u1.jpg | Alternative available |
| Roa excelsa | Chexc_u0.jpg | Same as Bishop image |
| Roa jayakari | Chjay_u1.jpg | Randall, no Bishop equivalent |
| Roa modesta | Chmod_u1.jpg | **Currently used as exemplar** |

---

## Duplicate Images

Images that appear in both Bishop and FishBase collections (same Randall photograph):

"""

    if len(duplicates_df) > 0:
        report += "| Species | Bishop File | FishBase File | Hash Distance |\n"
        report += "|---------|-------------|---------------|---------------|\n"
        for _, row in duplicates_df.iterrows():
            report += f"| {row['species']} | {row['bishop_file']} | {row['fishbase_file']} | {row['hash_distance']} |\n"
    else:
        report += "*No exact duplicates detected via hash comparison.*\n"

    report += f"""
---

## Changes Made

### Exemplar File Updates

"""
    for change in exemplar_changes:
        report += f"- {change}\n"

    report += """
### Gestalt K File Updates

"""
    for change in gestalt_changes:
        report += f"- {change}\n"

    report += f"""
---

## Image Counts by Source

| Source | Total Images | Randall? |
|--------|--------------|----------|
"""

    source_counts = inventory_df.groupby('source').agg({
        'filename': 'count',
        'is_randall': 'sum'
    }).rename(columns={'filename': 'count', 'is_randall': 'randall_count'})

    for source, row in source_counts.iterrows():
        randall_status = "All" if source == 'bishop' else f"{int(row['randall_count'])} images"
        report += f"| {source} | {int(row['count'])} | {randall_status} |\n"

    report += """
---

## Recommendations

1. **Use Bishop images when available** - Better color fidelity for analysis
2. **Mark FishBase-only Randall images** - Useful for species without Bishop coverage
3. **Consider updating exemplars** - For species where Randall alternative exists
4. **Run color correction on non-Randall images** - Especially iNaturalist (57% underwater)

---

## Technical Notes

1. **Hash comparison** uses perceptual hashing (16x16 grayscale, binary threshold)
2. **Duplicate threshold**: Hamming distance < 52 (20% of 256 bits)
3. **Taxonomy follows FishBase accepted names** as of February 2025

"""

    # Save report
    report_file = DATA_DIR / "taxonomy_reconciliation_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_file}")

    return report

def main():
    print("=" * 70)
    print("TAXONOMY AND SOURCE RECONCILIATION")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Build inventory
    inventory_df = build_image_inventory()

    # Find duplicates
    duplicates_df = find_duplicate_images(inventory_df)

    # Update files
    exemplar_changes = update_exemplar_file(inventory_df, duplicates_df)
    gestalt_changes = update_gestalt_k_file()

    # Generate report
    report = generate_report(inventory_df, duplicates_df, exemplar_changes, gestalt_changes)

    # Save inventory
    inventory_file = DATA_DIR / "image_inventory_with_sources.csv"
    inventory_df.to_csv(inventory_file, index=False)
    print(f"\nInventory saved to: {inventory_file}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
