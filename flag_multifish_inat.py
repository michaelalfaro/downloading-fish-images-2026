#!/usr/bin/env python3
"""
Flag iNaturalist images with multiple fish detected during segmentation.

These images risk containing the wrong species (mixed-species flocks)
so they are excluded from the color analysis pipeline.

Reads: analysis/approach_1_gmm/segmentation_report.csv
Updates: analysis/image_annotations.csv

Run after segment_fish.py completes.
"""

import os
import csv
from datetime import date

from analysis_utils import (
    GMM_DIR, append_annotations, load_annotations,
)

REPORT_CSV = os.path.join(GMM_DIR, "segmentation_report.csv")


def main():
    print("[1] Reading segmentation report...")
    annotations = load_annotations()

    multi_inat = []
    total_inat = 0
    total_multi = 0

    with open(REPORT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "success":
                continue
            if row["directory"] != "images_inaturalist":
                continue
            total_inat += 1

            n_fish = int(row["n_fish_detected"])
            if n_fish > 1:
                total_multi += 1
                # Don't overwrite existing user annotations
                if row["filename"] not in annotations:
                    multi_inat.append({
                        "filename": row["filename"],
                        "species": row["species"],
                        "directory": row["directory"],
                        "annotation_type": "multi_fish_inat",
                        "annotation_detail": (
                            f"{n_fish} foreground objects detected; "
                            f"excluding iNaturalist multi-fish to avoid "
                            f"wrong species contamination"
                        ),
                        "n_fish_noted": str(n_fish),
                        "annotated_by": "pipeline",
                        "date_annotated": date.today().isoformat(),
                    })

    print(f"    Total iNaturalist images: {total_inat}")
    print(f"    Multi-fish iNaturalist: {total_multi}")
    print(f"    New annotations to add: {len(multi_inat)}")

    if multi_inat:
        append_annotations(multi_inat)
        print(f"[2] Added {len(multi_inat)} multi-fish iNaturalist flags")
    else:
        print("[2] No new annotations needed")

    print("[3] Done.")


if __name__ == "__main__":
    main()
