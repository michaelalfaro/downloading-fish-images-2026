# Chaetodontidae Image Collection & Processing

This repository contains the image acquisition, processing, and curation pipeline for butterflyfish (Chaetodontidae) color pattern analysis.

## Purpose

1. **Image Collection**: Download reference images from multiple sources (FishBase, Bishop Museum, iNaturalist)
2. **Image Processing**: Segment backgrounds, standardize orientation, correct underwater color distortion
3. **Quality Control**: Score image quality, detect duplicates, filter problematic images
4. **Curation**: Interactive review app for exemplar selection and gestalt k annotation

## Quick Start

```bash
# Create conda environment
conda env create -f environment.yml
conda activate fish-images

# Run the review app (main curation interface)
cd apps
streamlit run review_app.py
```

## Directory Structure

```
├── scripts/                       # Processing scripts
│   ├── download/                  # Image acquisition
│   ├── processing/                # Segmentation, orientation, color correction
│   ├── quality/                   # QC and filtering
│   ├── analysis/                  # Color analysis
│   └── utils/                     # Shared utilities
│
├── apps/                          # Interactive applications
│   ├── review_app.py              # Main curation interface
│   └── review_duplicates_app.py   # Duplicate detection
│
├── data/                          # Metadata and inventories
│
├── images/                        # Primary FishBase + FishPix images
├── images_bishop/                 # Bishop Museum / Randall photos (gold standard)
├── images_fishbase_extra/         # Additional FishBase images
├── images_fishbase_usercontrib/   # User-contributed FishBase
├── images_inaturalist/            # iNaturalist community photos
├── images_quarantine/             # Excluded images (GIFs, problems)
│
├── analysis/                      # Processing outputs
│   └── approach_1_gmm/            # Main pipeline results
│       ├── segmented/             # Background-removed images
│       ├── oriented/              # Standardized orientation
│       ├── color_corrected/       # Underwater color corrected
│       └── zone_maps/             # k-means color classifications
│
├── tree/                          # Phylogenetic tree files
├── pilot_analysis/                # Pilot study outputs
├── papers/                        # Reference papers
└── manuscript/                    # Manuscript drafts
```

## Image Sources

| Source | Count | Quality | Color Correction Needed |
|--------|-------|---------|------------------------|
| Bishop/Randall | 178 | Excellent | No (reference standard) |
| FishBase Official | ~200 | Good | Optional (25%) |
| FishPix | 73 | Variable | Yes (51% underwater) |
| FishBase UserContrib | 722 | Variable | Recommended (29%) |
| iNaturalist | 1,125 | Variable | Yes (57% underwater) |

See `IMAGE_DIRECTORY_GUIDE.md` for detailed provenance information.

## Processing Pipeline

```
Raw Images → Segmentation → Orientation → Color Correction → pavo Analysis
     ↓
  [QC filtering: outliers, duplicates, larvae]
     ↓
  [Curation: exemplar selection, gestalt k annotation]
```

## Key Scripts

### Download
- `scripts/download/download_chaetodontidae.py` - FishBase images
- `scripts/download/download_bishop_museum.py` - Randall photos
- `scripts/download/download_fishbase_usercontrib.py` - User submissions

### Processing
- `scripts/processing/segment_fish.py` - Background removal (rembg)
- `scripts/processing/segment_dinosam.py` - Alternative segmentation (DinoSAM)
- `scripts/processing/orient_fish.py` - Standardize orientation
- `scripts/processing/correct_color_cast.py` - Underwater color correction

### Quality Control
- `scripts/quality/score_masks.py` - Segmentation quality metrics
- `scripts/quality/detect_outliers.py` - Identify problematic images
- `scripts/quality/detect_randall_duplicates.py` - Find duplicate Randall images

### Analysis
- `scripts/analysis/fit_gmm_species.py` - GMM color clustering
- `scripts/analysis/visualize_color_morphospace.py` - PCA visualization

## Related Repository

The analysis pipeline uses images from this repository:
- **chaets-divergence-2026**: Phylogenetic comparative analysis of color patterns

## Data Files

| File | Description |
|------|-------------|
| `data/all_images_inventory.csv` | Master list of all downloaded images |
| `analysis/approach_1_gmm/species_exemplar.csv` | Selected best image per species |
| `analysis/approach_1_gmm/species_gestalt_k.csv` | Human-annotated color class counts |

## Requirements

- Python 3.9+
- streamlit (for review app)
- rembg (for segmentation)
- opencv-python
- numpy, pandas, matplotlib

See `environment.yml` for full dependencies.
