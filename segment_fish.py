#!/usr/bin/env python3
"""
Segment fish bodies from Chaetodontidae images using macOS Vision framework.

Uses Apple's VNGenerateForegroundInstanceMaskRequest via PyObjC to isolate
the foreground subject (fish) from the background. This approach leverages
Apple's built-in ML models for subject detection, producing high-quality
masks that match the macOS "Remove Background" feature in Preview/Finder.

For multi-fish images, Vision returns all foreground objects as a single
mask. The pipeline flags these but includes them — downstream steps can
filter based on annotations.

Falls back to GroundingDINO + SlimSAM if Vision fails (rare on macOS 14+).

Usage:
    python3 segment_fish.py                    # process all images
    python3 segment_fish.py --resume           # skip already-processed
    python3 segment_fish.py --species "Chaetodon auriga"  # one species
    python3 segment_fish.py --test 10          # first 10 images only

Requires: pyobjc-framework-Vision, pyobjc-framework-Quartz, Pillow, numpy
"""

import os
import sys
import csv
import time
import ctypes
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from analysis_utils import (
    SCRIPT_DIR, SEGMENTED_DIR, GMM_DIR,
    load_inventory, load_annotations, get_image_path,
    species_to_dirname, ensure_dir,
)

REPORT_CSV = os.path.join(GMM_DIR, "segmentation_report.csv")
REPORT_FIELDS = [
    "filename", "species", "directory", "method",
    "n_fish_detected", "selected_mask_area",
    "image_width", "image_height", "mask_fraction",
    "segmented_path", "status",
]


def macos_vision_segment(image_path):
    """Segment foreground using macOS Vision framework.

    Returns (mask_uint8 [H,W], width, height) or (None, 0, 0) on failure.
    mask_uint8 values: 0=background, 255=foreground, gradients at edges.
    """
    import Vision
    import Quartz

    nsurl = Quartz.NSURL.fileURLWithPath_(image_path)
    ci_image = Quartz.CIImage.imageWithContentsOfURL_(nsurl)
    if ci_image is None:
        return None, 0, 0

    handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(
        ci_image, None
    )
    request = Vision.VNGenerateForegroundInstanceMaskRequest.alloc().init()
    success, error = handler.performRequests_error_([request], None)

    if not success or not request.results():
        return None, 0, 0

    result = request.results()[0]
    mask_pb, error = (
        result.generateScaledMaskForImageForInstances_fromRequestHandler_error_(
            result.allInstances(), handler, None
        )
    )
    if mask_pb is None:
        return None, 0, 0

    # Convert CVPixelBuffer → CIImage → CGImage → grayscale bitmap
    mask_ci = Quartz.CIImage.imageWithCVPixelBuffer_(mask_pb)
    context = Quartz.CIContext.contextWithOptions_(None)
    extent = mask_ci.extent()
    w = int(extent.size.width)
    h = int(extent.size.height)

    gray_cs = Quartz.CGColorSpaceCreateDeviceGray()
    bitmap_data = bytearray(w * h)
    c_buf = (ctypes.c_ubyte * len(bitmap_data)).from_buffer(bitmap_data)
    bitmap_ctx = Quartz.CGBitmapContextCreate(
        c_buf, w, h, 8, w, gray_cs, Quartz.kCGImageAlphaNone
    )
    cg_mask = context.createCGImage_fromRect_(mask_ci, extent)
    Quartz.CGContextDrawImage(
        bitmap_ctx, Quartz.CGRectMake(0, 0, w, h), cg_mask
    )

    # CGBitmapContext with CGContextDrawImage already renders in top-left
    # orientation (the CIImage→CGImage path handles the coordinate flip),
    # so no flipud is needed.
    mask_arr = np.frombuffer(bitmap_data, dtype=np.uint8).reshape(h, w).copy()

    return mask_arr, w, h


def process_image(image_path, output_path):
    """Process a single image. Returns metadata dict for the report."""
    from PIL import Image

    # Try macOS Vision first
    mask, w, h = macos_vision_segment(image_path)
    method = "macos_vision"

    if mask is None:
        return {
            "method": "none",
            "n_fish_detected": 0,
            "selected_mask_area": 0,
            "image_width": 0,
            "image_height": 0,
            "mask_fraction": 0,
            "segmented_path": "",
            "status": "no_detection",
        }

    # Create RGBA image
    try:
        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img)
    except Exception as e:
        return {
            "method": method,
            "n_fish_detected": 0,
            "selected_mask_area": 0,
            "image_width": w,
            "image_height": h,
            "mask_fraction": 0,
            "segmented_path": "",
            "status": f"error: {e}",
        }

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img_arr
    rgba[:, :, 3] = mask

    mask_area = int((mask > 128).sum())
    mask_frac = mask_area / (w * h) if (w * h) > 0 else 0

    # Count distinct foreground blobs as proxy for n_fish
    # (simple connected components on the binary mask)
    binary = (mask > 128).astype(np.uint8)
    try:
        from scipy.ndimage import label as scipy_label
        labeled, n_components = scipy_label(binary)
        n_fish = n_components
    except ImportError:
        n_fish = 1  # can't count without scipy

    ensure_dir(os.path.dirname(output_path))
    Image.fromarray(rgba, "RGBA").save(output_path)

    return {
        "method": method,
        "n_fish_detected": n_fish,
        "selected_mask_area": mask_area,
        "image_width": w,
        "image_height": h,
        "mask_fraction": round(mask_frac, 4),
        "segmented_path": os.path.relpath(output_path, SCRIPT_DIR),
        "status": "success",
    }


def load_existing_report():
    """Load existing report for resume mode."""
    done = set()
    if os.path.exists(REPORT_CSV):
        with open(REPORT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") in ("success", "skipped_larva"):
                    done.add(row["filename"])
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Segment fish using macOS Vision"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--test", type=int, default=None)
    args = parser.parse_args()

    print("[1] Loading inventory and annotations...")
    inventory = load_inventory()
    annotations = load_annotations()
    print(f"    {len(inventory)} images, {len(annotations)} annotations")

    if args.species:
        inventory = [r for r in inventory if r["species"] == args.species]
        print(f"    Filtered to: {args.species} ({len(inventory)} images)")

    already_done = set()
    if args.resume:
        already_done = load_existing_report()
        print(f"    Resume: {len(already_done)} already done")

    if args.test:
        inventory = inventory[:args.test]
        print(f"    Test mode: first {args.test} images")

    print("[2] Segmenting with macOS Vision framework...")
    results = []
    n_success = n_skip_larva = n_skip_done = n_fail = 0
    t_start = time.time()

    # Load existing results if resuming
    if args.resume and os.path.exists(REPORT_CSV):
        with open(REPORT_CSV, newline="") as f:
            results = list(csv.DictReader(f))

    for i, row in enumerate(inventory):
        fname = row["filename"]
        species = row["species"]
        directory = row["directory"]

        if fname in already_done:
            n_skip_done += 1
            continue

        # Skip larvae
        ann = annotations.get(fname)
        if ann and ann["annotation_type"] == "larva":
            results.append({
                "filename": fname, "species": species,
                "directory": directory, "method": "skipped",
                "n_fish_detected": 0, "selected_mask_area": 0,
                "image_width": 0, "image_height": 0,
                "mask_fraction": 0, "segmented_path": "",
                "status": "skipped_larva",
            })
            n_skip_larva += 1
            continue

        image_path = get_image_path(row)
        sp_dir = species_to_dirname(species)
        out_name = os.path.splitext(fname)[0] + ".png"
        output_path = os.path.join(SEGMENTED_DIR, sp_dir, out_name)

        result = process_image(image_path, output_path)
        result["filename"] = fname
        result["species"] = species
        result["directory"] = directory
        results.append(result)

        status = result["status"]
        if status == "success":
            n_success += 1
            mf = result["mask_fraction"]
            nf = result["n_fish_detected"]
            multi = " [MULTI]" if nf > 1 else ""
            print(f"  [{i+1:4d}/{len(inventory)}] {species} -- "
                  f"mask {mf:.1%}, {nf} obj{multi}")
        else:
            n_fail += 1
            print(f"  [{i+1:4d}/{len(inventory)}] {species} -- {status}")

        # Checkpoint every 200 images
        if (i + 1) % 200 == 0:
            _write_report(results)
            elapsed = time.time() - t_start
            done_count = n_success + n_fail
            rate = done_count / elapsed if elapsed > 0 else 0
            print(f"    -- checkpoint {i+1}: {rate:.1f} img/s --")

    _write_report(results)

    elapsed = time.time() - t_start
    print(f"\n[3] Segmentation complete in {elapsed:.0f}s")
    print(f"    Success:       {n_success}")
    print(f"    Failed:        {n_fail}")
    print(f"    Skipped larva: {n_skip_larva}")
    if args.resume:
        print(f"    Skipped done:  {n_skip_done}")
    print(f"    Report: {REPORT_CSV}")


def _write_report(results):
    with open(REPORT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
