#!/usr/bin/env python3
"""
Initialize image_annotations.csv with known larvae and multi-fish flags.
These annotations are used by downstream pipeline steps:
  - Larvae are excluded from color analysis
  - Multi-fish images are processed (largest fish selected) but flagged

Run once to initialize; later scripts append outlier flags.
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis_utils import save_annotations, ANNOTATIONS_CSV, ensure_dir


TODAY = date.today().isoformat()


# User-identified larvae
LARVAE = [
    {
        "filename": "Chaetodon_litus_Bishop_2072919420.jpg",
        "species": "Chaetodon litus",
        "directory": "images_bishop",
        "annotation_type": "larva",
        "annotation_detail": "Larval stage identified by visual inspection",
        "n_fish_noted": "",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
    {
        "filename": "Chaetodon_argentatus_FishBase_Charg_u1.jpg",
        "species": "Chaetodon argentatus",
        "directory": "images_fishbase_extra",
        "annotation_type": "larva",
        "annotation_detail": "Larval stage identified by visual inspection",
        "n_fish_noted": "",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
    {
        "filename": "Chaetodon_argentatus_Bishop_1101542975.jpg",
        "species": "Chaetodon argentatus",
        "directory": "images_bishop",
        "annotation_type": "larva",
        "annotation_detail": "Larval stage identified by visual inspection",
        "n_fish_noted": "",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
]

# User-identified multi-fish images
MULTI_FISH = [
    {
        "filename": "Chaetodon_semilarvatus_FishPix_32790AF.jpg",
        "species": "Chaetodon semilarvatus",
        "directory": "images",
        "annotation_type": "multi_fish",
        "annotation_detail": "2 fish in image",
        "n_fish_noted": "2",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
    {
        "filename": "Prognathodes_marcellae_FishBase_Prmar_u0.jpg",
        "species": "Prognathodes marcellae",
        "directory": "images",
        "annotation_type": "multi_fish",
        "annotation_detail": "2 fish in image",
        "n_fish_noted": "2",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
    {
        "filename": "Chaetodon_austriacus_FishBase_Chaus_u0.jpg",
        "species": "Chaetodon austriacus",
        "directory": "images_fishbase_extra",
        "annotation_type": "multi_fish",
        "annotation_detail": "2+ fish in image",
        "n_fish_noted": "2",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
    {
        "filename": "Chaetodon_collare_FishBase_Chcol_u1.jpg",
        "species": "Chaetodon collare",
        "directory": "images_fishbase_extra",
        "annotation_type": "multi_fish",
        "annotation_detail": "3 fish in image",
        "n_fish_noted": "3",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
    {
        "filename": "Chaetodon_daedalma_FishBase_Chdae_u0.jpg",
        "species": "Chaetodon daedalma",
        "directory": "images_fishbase_extra",
        "annotation_type": "multi_fish",
        "annotation_detail": "2 fish in image",
        "n_fish_noted": "2",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
    {
        "filename": "Chaetodon_guentheri_FishBase_Chgue_u2.jpg",
        "species": "Chaetodon guentheri",
        "directory": "images_fishbase_extra",
        "annotation_type": "multi_fish",
        "annotation_detail": "3+ fish in image",
        "n_fish_noted": "3",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
    {
        "filename": "Chaetodon_hoefleri_FishBase_Chhoe_u2.jpg",
        "species": "Chaetodon hoefleri",
        "directory": "images_fishbase_extra",
        "annotation_type": "multi_fish",
        "annotation_detail": "2 fish in image",
        "n_fish_noted": "2",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
    {
        "filename": "Chaetodon_kleinii_FishBase_Chkle_u3.jpg",
        "species": "Chaetodon kleinii",
        "directory": "images_fishbase_extra",
        "annotation_type": "multi_fish",
        "annotation_detail": "3+ fish in image",
        "n_fish_noted": "3",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
    {
        "filename": "Chaetodon_lunula_FishBase_Chlun_uf.jpg",
        "species": "Chaetodon lunula",
        "directory": "images_fishbase_extra",
        "annotation_type": "multi_fish",
        "annotation_detail": "2 fish in image",
        "n_fish_noted": "2",
        "annotated_by": "user",
        "date_annotated": TODAY,
    },
]


def main():
    all_annotations = LARVAE + MULTI_FISH
    print(f"[1] Writing {len(all_annotations)} image annotations...")
    print(f"    Larvae: {len(LARVAE)}")
    print(f"    Multi-fish: {len(MULTI_FISH)}")

    save_annotations(all_annotations)

    print(f"    Saved to {ANNOTATIONS_CSV}")
    print("[2] Done.")


if __name__ == "__main__":
    main()
