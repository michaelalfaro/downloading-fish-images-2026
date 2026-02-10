# Image Directory Organization Guide

## Overview

This repository contains butterflyfish (Chaetodontidae) images from multiple sources, organized for color pattern analysis. Images range from professionally photographed specimens (Randall) to underwater citizen science photos (iNaturalist).

**Last updated:** February 2025

---

## Directory Structure

```
downloading-fish-images-2026/
│
├── scripts/                       # Processing scripts (organized by function)
│   ├── download/                  # Image acquisition scripts
│   ├── processing/                # Segmentation, orientation, color correction
│   ├── quality/                   # QC, filtering, duplicate detection
│   ├── analysis/                  # Color analysis scripts
│   └── utils/                     # Shared utilities (analysis_utils.py)
│
├── apps/                          # Interactive applications
│   ├── review_app.py              # Main curation interface
│   └── review_duplicates_app.py   # Duplicate detection UI
│
├── data/                          # Metadata and inventories
│   ├── all_images_inventory.csv   # Master image list
│   └── *.csv                      # Other metadata files
│
├── images/                        # Primary FishBase + FishPix images (n=131)
├── images_bishop/                 # Bishop Museum / Randall photos (n=178)
├── images_fishbase_extra/         # Additional FishBase images (n=568)
├── images_fishbase_usercontrib/   # FishBase user-contributed (n=722)
├── images_inaturalist/            # iNaturalist community photos (n=1,125)
├── images_quarantine/             # Problematic images (GIFs, duplicates) (n=42)
│
├── analysis/                      # Processing outputs
│   └── approach_1_gmm/            # Main pipeline results
│       ├── segmented/             # Background-removed images
│       ├── oriented/              # Standardized orientation
│       ├── normalized/            # Size-normalized
│       ├── color_corrected/       # Underwater color corrected
│       └── zone_maps/             # Color classification maps
│
├── pilot_analysis/                # Pilot study outputs
├── tree/                          # Phylogenetic tree files
├── papers/                        # Reference papers
├── manuscript/                    # Manuscript drafts
└── archive/                       # Deprecated/test files
```

---

## Image Sources

### 1. `images/` - Primary Reference Images
**Count:** 131 images
**Sources:** FishBase Official + FishPix
**Quality:** Mixed (FishPix often underwater)
**Color Correction Needed:** Some (41% detected as underwater)

Contains initial reference images for each species, sourced from:
- **FishBase Official** (prefix `_FishBase_`): Curated images from FishBase database
- **FishPix** (prefix `_FishPix_`): Often underwater photographs, higher blue cast

**Filename format:**
```
Genus_species_Source_ID.jpg
Example: Chaetodon_auriga_FishPix_12061AF.jpg
```

---

### 2. `images_bishop/` - Bishop Museum / Randall Photos
**Count:** 178 images
**Source:** Bishop Museum archive (John Randall collection)
**Quality:** Excellent (controlled lighting, out-of-water)
**Color Correction Needed:** Minimal (17% detected, mostly false positives)

These are the **gold standard** reference images:
- Photographed in controlled conditions
- Fish placed in small freshwater tank against glass
- Studio lighting, color-corrected
- Best for color reference and `imposeColors()` templates

**Filename format:**
```
Genus_species_Bishop_ID.jpg
Genus_species_Bishop_ID Background Removed.png  # Pre-segmented versions
Example: Chaetodon_auriga_Bishop_472868867.jpg
```

**Note:** Some images have "Background Removed" versions (`.png`) with transparent backgrounds.

---

### 3. `images_fishbase_extra/` - Additional FishBase Images
**Count:** 568 images
**Source:** FishBase database (official images)
**Quality:** Variable
**Color Correction Needed:** Some (23% detected as underwater)

Additional images from FishBase when multiple photos exist per species. Generally similar quality to `images/` FishBase entries.

**Filename format:**
```
Genus_species_FishBase_Code_uN.jpg
Example: Chaetodon_auriga_FishBase_Chaur_u2.jpg
```

The `uN` suffix indicates variant number (u0, u1, u2, etc.).

---

### 4. `images_fishbase_usercontrib/` - User-Contributed FishBase
**Count:** 722 images
**Source:** FishBase user submissions
**Quality:** Highly variable (many underwater)
**Color Correction Needed:** Yes (29% detected, likely underestimated)

Community-contributed photos uploaded to FishBase. Quality ranges from professional to casual underwater snapshots.

**Filename format:**
```
Genus_species_FishBaseUser_Timestamp_IP.jpg
Example: Chaetodon_auriga_FishBaseUser_1610273317_172.68.243.13.jpg
```

The timestamp and IP indicate upload metadata.

---

### 5. `images_inaturalist/` - iNaturalist Photos
**Count:** 1,125 images
**Source:** iNaturalist citizen science platform
**Quality:** Variable (most underwater)
**Color Correction Needed:** Yes (57% detected as underwater)

Community photos from iNaturalist, primarily underwater photography by divers and snorkelers. **Highest proportion of underwater color distortion.**

**Filename format:**
```
Genus_species_iNaturalist_ObservationID.jpg
Example: Chaetodon_auriga_iNaturalist_123456789.jpg
```

---

### 6. `images_quarantine/` - Problematic Images
**Count:** 42 images
**Status:** Excluded from analysis

Images moved here due to:
- GIF format (not processable): `*.gif`
- Duplicates
- Quality issues
- Misidentifications

**Do not use these images for analysis.**

---

## Processed Image Directories

### `analysis/approach_1_gmm/segmented/`
Background-removed versions of images. Each species has its own subdirectory.

```
segmented/
├── Chaetodon_auriga/
│   ├── Chaetodon_auriga_Bishop_472868867.png
│   ├── Chaetodon_auriga_FishPix_12061AF.png
│   └── ...
└── ...
```

### `analysis/approach_1_gmm/oriented/`
Images with standardized orientation (head facing right). Used for exemplar selection.

### `analysis/approach_1_gmm/color_corrected/`
Images that have undergone underwater color correction (Gray World + Red Compensation + CLAHE).

### `analysis/approach_1_gmm/zone_maps/`
Color classification maps showing k-means clustering results.

---

## Color Correction Priority

Based on automated underwater detection analysis:

| Source | % Underwater | Recommendation |
|--------|-------------|----------------|
| iNaturalist | 57% | **NEEDS CORRECTION** |
| FishPix | 51% | **NEEDS CORRECTION** |
| FishBase UserContrib | 29% | RECOMMENDED |
| FishBase Official | 25% | OPTIONAL |
| Bishop (Randall) | 17% | NOT NEEDED (false positives) |

### Correction Pipeline

1. **Detect:** Score > 0.4 indicates underwater distortion
2. **Correct:** Apply Gray World → Red Compensation → CLAHE
3. **Reference:** Optionally histogram-match to Bishop/Randall reference
4. **Verify:** Re-score to confirm improvement

---

## Key Data Files

| File | Description |
|------|-------------|
| `data/all_images_inventory.csv` | Master list of all downloaded images |
| `analysis/approach_1_gmm/species_exemplar.csv` | Selected best image per species |
| `analysis/approach_1_gmm/species_gestalt_k.csv` | Human-annotated color class counts |
| `analysis/approach_1_gmm/segmentation_report.csv` | Background removal quality metrics |
| `analysis/approach_1_gmm/color_correction_report.csv` | Color correction applied |

---

## Exemplar Selection Hierarchy

When selecting best representative image per species, priority order:

1. **Bishop/Randall** - Best color fidelity
2. **FishBase Official** - Curated quality
3. **Oriented images** - Standardized pose
4. **Segmented images** - Background removed
5. **FishPix** - Often underwater but available
6. **FishBase UserContrib** - Variable quality
7. **iNaturalist** - Underwater, but high coverage

---

## File Naming Convention Summary

| Source | Pattern | Example |
|--------|---------|---------|
| FishBase Official | `Genus_species_FishBase_Code_uN.ext` | `Chaetodon_auriga_FishBase_Chaur_u0.jpg` |
| FishPix | `Genus_species_FishPix_IDAF.ext` | `Chaetodon_auriga_FishPix_12061AF.jpg` |
| Bishop/Randall | `Genus_species_Bishop_ID.ext` | `Chaetodon_auriga_Bishop_472868867.jpg` |
| FishBase User | `Genus_species_FishBaseUser_Time_IP.ext` | `Chaetodon_auriga_FishBaseUser_1610273317_172.68.243.13.jpg` |
| iNaturalist | `Genus_species_iNaturalist_ObsID.ext` | `Chaetodon_auriga_iNaturalist_123456789.jpg` |

---

## Analysis Workflow

```
Raw Images (5 source directories)
        ↓
    Segmentation (background removal)
        ↓
    Orientation (standardize pose)
        ↓
    Exemplar Selection (best per species)
        ↓
    Color Correction (if underwater)
        ↓
    pavo Analysis (color pattern metrics)
```

---

## Notes

- **Species coverage:** ~130 Chaetodontidae species
- **Total images:** ~2,700 across all sources
- **Recommended references:** Bishop/Randall images for color templates
- **Analysis exemplars:** See `analysis/approach_1_gmm/species_exemplar.csv`

---

## Contact

Questions about image provenance or processing: [Project maintainer]
