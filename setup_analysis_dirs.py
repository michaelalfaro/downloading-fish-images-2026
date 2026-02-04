#!/usr/bin/env python3
"""
Create directory structure for the color pattern analysis pipeline.
Reads species list from all_images_inventory.csv and creates per-species
subdirectories under each analysis output folder.
"""

import os
import sys
from analysis_utils import (
    SCRIPT_DIR, ANALYSIS_DIR, GMM_DIR, SEGMENTED_DIR, NORMALIZED_DIR,
    ZONE_MAPS_DIR, K_SELECTION_DIR, COLOR_COMP_DIR,
    load_inventory, all_species, species_to_dirname, ensure_dir,
)


def main():
    print("[1] Reading species list from inventory...")
    inventory = load_inventory()
    species_list = all_species(inventory)
    print(f"    Found {len(species_list)} species")

    print("[2] Creating analysis directory structure...")

    # Top-level analysis dir
    ensure_dir(ANALYSIS_DIR)

    # Approach 1: GMM
    for d in [SEGMENTED_DIR, NORMALIZED_DIR, ZONE_MAPS_DIR,
              K_SELECTION_DIR, COLOR_COMP_DIR,
              os.path.join(K_SELECTION_DIR, "bic_curves"),
              os.path.join(GMM_DIR, "adjacency")]:
        ensure_dir(d)

    # Per-species subdirs
    for sp in species_list:
        sp_dir = species_to_dirname(sp)
        for parent in [SEGMENTED_DIR, NORMALIZED_DIR, ZONE_MAPS_DIR]:
            ensure_dir(os.path.join(parent, sp_dir))

    # Approach 2 & 3 placeholders
    for approach in ["approach_2_recolorize", "approach_3_charisma"]:
        ensure_dir(os.path.join(ANALYSIS_DIR, approach))

    print(f"    Created dirs for {len(species_list)} species under:")
    print(f"      {SEGMENTED_DIR}")
    print(f"      {NORMALIZED_DIR}")
    print(f"      {ZONE_MAPS_DIR}")
    print(f"    Plus placeholders for approach_2_recolorize, approach_3_charisma")

    print("[3] Done.")


if __name__ == "__main__":
    main()
