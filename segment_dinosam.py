#!/usr/bin/env python3
"""
DINO+SAM segmentation with fish-specific refinements for Chaetodontidae.

This module serves as a fallback for images where Apple Vision fails
(fragmented masks on striped fish, background inclusions of coral/shells).

Key refinements over naive DINO+SAM:
  1. Fish-specific text prompts with GroundingDINO
  2. If DINO returns multiple boxes, merge overlapping + pick most central
  3. Morphological cleanup: close gaps from stripes, fill holes
  4. SAM re-prompting: box prompt + interior positive points from distance
     transform + ring of negative background points in margins
  5. Final mask selection: largest connected component, boundary smoothness
  6. Center-bias: fish are typically centered in the image; background is
     definitely in the margins

Usage:
    python3 segment_dinosam.py                          # all flagged images
    python3 segment_dinosam.py --image path/to/img.jpg  # single image test
    python3 segment_dinosam.py --all                    # run on ALL images

Reads:  analysis/approach_1_gmm/mask_quality_report.csv (for flagged images)
Writes: analysis/approach_1_gmm/segmented_dinosam/{species}/
        analysis/approach_1_gmm/dinosam_report.csv
"""

import os
import sys
import csv
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from analysis_utils import (
    SCRIPT_DIR, GMM_DIR, SEGMENTED_DIR,
    load_inventory, load_annotations, get_image_path,
    species_to_dirname, ensure_dir,
)

DINOSAM_DIR = os.path.join(GMM_DIR, "segmented_dinosam")
DINOSAM_REPORT = os.path.join(GMM_DIR, "dinosam_report.csv")
QUALITY_CSV = os.path.join(GMM_DIR, "mask_quality_report.csv")

REPORT_FIELDS = [
    "filename", "species", "directory", "method",
    "dino_n_boxes", "sam_mask_area",
    "image_width", "image_height", "mask_fraction",
    "segmented_path", "status",
]

# --- Model loading (lazy singletons) ---
_dino_model = None
_dino_processor = None
_sam_model = None
_sam_processor = None


def load_dino():
    """Load GroundingDINO model (lazy singleton)."""
    global _dino_model, _dino_processor
    if _dino_model is None:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        model_id = "IDEA-Research/grounding-dino-tiny"
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        print("    Loading GroundingDINO...")
        _dino_processor = AutoProcessor.from_pretrained(model_id)
        _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(device)
        _dino_model.eval()
        print(f"    GroundingDINO loaded on {device}")
    return _dino_model, _dino_processor


def load_sam():
    """Load SlimSAM model (lazy singleton, on CPU for float64 compat)."""
    global _sam_model, _sam_processor
    if _sam_model is None:
        from transformers import SamModel, SamProcessor

        model_id = "Zigeng/SlimSAM-uniform-77"
        print("    Loading SlimSAM...")
        _sam_processor = SamProcessor.from_pretrained(model_id)
        _sam_model = SamModel.from_pretrained(model_id).to("cpu")
        _sam_model.eval()
        print("    SlimSAM loaded on CPU")
    return _sam_model, _sam_processor


def detect_fish_dino(image, threshold=0.15):
    """Use GroundingDINO to detect fish in an image.

    Args:
        image: PIL Image (RGB)
        threshold: detection confidence threshold

    Returns:
        list of boxes [[x1, y1, x2, y2], ...] in pixel coords, sorted by
        centrality (most central first)
    """
    import torch

    model, processor = load_dino()
    device = next(model.parameters()).device

    # Fish-specific prompts — try several to maximize recall
    prompts = [
        "a fish.",
        "a butterflyfish.",
        "a reef fish.",
    ]

    all_boxes = []
    all_scores = []

    for text in prompts:
        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        for box, score in zip(boxes, scores):
            all_boxes.append(box)
            all_scores.append(score)

    if len(all_boxes) == 0:
        return []

    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)

    # NMS: merge overlapping boxes
    boxes, scores = _nms(all_boxes, all_scores, iou_thresh=0.5)

    # Sort by centrality (closest to image center first)
    w, h = image.size
    cx, cy = w / 2, h / 2
    centralities = []
    for box in boxes:
        bx = (box[0] + box[2]) / 2
        by = (box[1] + box[3]) / 2
        dist = np.sqrt(((bx - cx) / w)**2 + ((by - cy) / h)**2)
        centralities.append(dist)

    order = np.argsort(centralities)
    boxes = boxes[order]

    return boxes.tolist()


def _nms(boxes, scores, iou_thresh=0.5):
    """Simple non-maximum suppression."""
    if len(boxes) == 0:
        return boxes, scores
    order = np.argsort(-scores)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = _compute_ious(boxes[i], boxes[rest])
        order = rest[ious < iou_thresh]
    keep = np.array(keep)
    return boxes[keep], scores[keep]


def _compute_ious(box, boxes):
    """IoU of one box against multiple boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def segment_with_sam(image, boxes, image_np=None):
    """Run SAM with box prompts + positive/negative point prompts.

    Two-pass strategy:
      Pass 1: box-only SAM → get initial mask candidates
      Pass 2: re-prompt SAM with box + positive points from initial mask's
              distance-transform maxima + dense negative margin ring

    The two-pass approach is critical for dark-background images where
    the initial box-only mask may leak into background.

    Returns binary mask [H, W] as uint8 (0/255).
    """
    import torch
    from PIL import Image as PILImage
    from scipy.ndimage import (
        distance_transform_edt, label as scipy_label,
        binary_fill_holes, binary_closing, binary_dilation,
        binary_erosion,
    )

    model, processor = load_sam()
    w_img, h_img = image.size

    if image_np is None:
        image_np = np.array(image)

    if not boxes:
        return None

    best_box = boxes[0]

    # Expand box by 5% for SAM
    bw = best_box[2] - best_box[0]
    bh = best_box[3] - best_box[1]
    pad_x = bw * 0.05
    pad_y = bh * 0.05
    expanded_box = [
        max(0, best_box[0] - pad_x),
        max(0, best_box[1] - pad_y),
        min(w_img, best_box[2] + pad_x),
        min(h_img, best_box[3] + pad_y),
    ]

    # ===== PASS 1: box-only SAM to get initial mask =====
    inputs_p1 = processor(
        image,
        input_boxes=[[[int(c) for c in expanded_box]]],
        return_tensors="pt",
    )
    inputs_p1 = {k: v.to("cpu") if hasattr(v, 'to') else v
                 for k, v in inputs_p1.items()}

    with torch.no_grad():
        outputs_p1 = model(**inputs_p1)

    masks_p1 = processor.image_processor.post_process_masks(
        outputs_p1.pred_masks.cpu(),
        inputs_p1["original_sizes"].cpu(),
        inputs_p1["reshaped_input_sizes"].cpu(),
    )[0]
    iou_p1 = outputs_p1.iou_scores.cpu().numpy()[0, 0]  # shape [3]

    # masks_p1 shape: [1, 3, H, W] — batch=1, 3 candidate masks
    masks_p1_np = masks_p1[0].numpy()  # [3, H, W]

    # Evaluate all 3 SAM masks: pick by combined score of IoU and
    # "not touching edges" (edge-touching masks are likely bg leaks)
    best_mask_p1 = None
    best_score_p1 = -1
    for idx in range(masks_p1_np.shape[0]):
        m = masks_p1_np[idx].astype(bool)  # [H, W]
        iou_val = float(iou_p1[idx])

        # Penalize masks that touch image edges extensively
        margin_px = 3
        edge_frac = (
            m[:margin_px, :].sum() + m[-margin_px:, :].sum() +
            m[:, :margin_px].sum() + m[:, -margin_px:].sum()
        ) / max(1, m.sum())

        # Penalize masks that extend far beyond the DINO box
        box_mask = np.zeros_like(m)
        bx1, by1, bx2, by2 = [int(c) for c in best_box]
        box_mask[by1:by2, bx1:bx2] = True
        outside_frac = (m & ~box_mask).sum() / max(1, m.sum())

        # Penalize very large masks (> 70% of image is suspicious)
        size_frac = m.sum() / (h_img * w_img)
        size_penalty = max(0, size_frac - 0.6) * 2.0

        score = iou_val - 0.3 * edge_frac - 0.2 * outside_frac - size_penalty
        if score > best_score_p1:
            best_score_p1 = score
            best_mask_p1 = m

    if best_mask_p1 is None or best_mask_p1.sum() < 100:
        return None

    # ===== PASS 2: refine with point prompts =====
    # Positive points: from distance transform of pass-1 mask interior
    # This places points at the "most interior" locations of the fish
    pos_points = []

    # Erode the pass-1 mask to find safe interior points
    eroded = binary_erosion(best_mask_p1, iterations=3)
    if eroded is not None and eroded.sum() > 50:
        dt = distance_transform_edt(eroded)
    else:
        dt = distance_transform_edt(best_mask_p1)

    # Pick top 5-7 distance-transform peaks as positive points
    from scipy.ndimage import maximum_filter
    local_max = (dt == maximum_filter(dt, size=max(20, min(h_img, w_img) // 15)))
    local_max = local_max & (dt > dt.max() * 0.3)  # only strong peaks
    peak_ys, peak_xs = np.where(local_max)
    if len(peak_ys) > 7:
        # Keep the 7 most interior peaks
        peak_dists = dt[peak_ys, peak_xs]
        top_idx = np.argsort(-peak_dists)[:7]
        peak_ys, peak_xs = peak_ys[top_idx], peak_xs[top_idx]
    for y, x in zip(peak_ys, peak_xs):
        pos_points.append([float(x), float(y)])

    # Always include box center
    box_cx = (best_box[0] + best_box[2]) / 2
    box_cy = (best_box[1] + best_box[3]) / 2
    pos_points.insert(0, [box_cx, box_cy])

    # Negative points: dense ring around image margins
    # These are DEFINITELY background — the fish is centered
    margin = max(5, int(min(h_img, w_img) * 0.03))
    neg_points = []
    # Sample every ~30px along all 4 edges
    step = max(20, min(h_img, w_img) // 20)
    for x in range(margin, w_img - margin, step):
        neg_points.append([float(x), float(margin)])         # top edge
        neg_points.append([float(x), float(h_img - margin)]) # bottom edge
    for y in range(margin, h_img - margin, step):
        neg_points.append([float(margin), float(y)])          # left edge
        neg_points.append([float(w_img - margin), float(y)])  # right edge
    # Corners
    for corner in [[margin, margin], [w_img-margin, margin],
                   [margin, h_img-margin], [w_img-margin, h_img-margin]]:
        neg_points.append([float(corner[0]), float(corner[1])])

    # Filter out negative points inside the DINO box
    neg_points = [p for p in neg_points
                  if not (best_box[0] - pad_x < p[0] < best_box[2] + pad_x and
                          best_box[1] - pad_y < p[1] < best_box[3] + pad_y)]

    if not pos_points:
        pos_points = [[box_cx, box_cy]]
    if not neg_points:
        # Fallback: 4 corners
        neg_points = [[5, 5], [w_img-5, 5], [5, h_img-5], [w_img-5, h_img-5]]

    all_points = pos_points + neg_points
    all_labels = [1] * len(pos_points) + [0] * len(neg_points)

    # Run SAM pass 2
    inputs_p2 = processor(
        image,
        input_boxes=[[[int(c) for c in expanded_box]]],
        input_points=[[all_points]],
        input_labels=[[all_labels]],
        return_tensors="pt",
    )
    inputs_p2 = {k: v.to("cpu") if hasattr(v, 'to') else v
                 for k, v in inputs_p2.items()}

    with torch.no_grad():
        outputs_p2 = model(**inputs_p2)

    masks_p2 = processor.image_processor.post_process_masks(
        outputs_p2.pred_masks.cpu(),
        inputs_p2["original_sizes"].cpu(),
        inputs_p2["reshaped_input_sizes"].cpu(),
    )[0]
    iou_p2 = outputs_p2.iou_scores.cpu().numpy()[0, 0]  # shape [3]
    masks_p2_np = masks_p2[0].numpy()  # [3, H, W]

    # Pick best mask from pass 2 (same scoring)
    best_mask = None
    best_score = -1
    for idx in range(masks_p2_np.shape[0]):
        m = masks_p2_np[idx].astype(bool)  # [H, W]
        iou_val = float(iou_p2[idx])
        margin_px = 3
        edge_frac = (
            m[:margin_px, :].sum() + m[-margin_px:, :].sum() +
            m[:, :margin_px].sum() + m[:, -margin_px:].sum()
        ) / max(1, m.sum())
        size_frac = m.sum() / (h_img * w_img)
        size_penalty = max(0, size_frac - 0.6) * 2.0
        score = iou_val - 0.3 * edge_frac - size_penalty
        if score > best_score:
            best_score = score
            best_mask = m

    if best_mask is None or best_mask.sum() < 100:
        # Fall back to pass 1
        best_mask = best_mask_p1

    mask = best_mask

    # --- Morphological refinement ---
    # 1. Moderate closing to bridge small stripe gaps
    close_size = max(3, int(min(h_img, w_img) * 0.01))
    struct = np.ones((close_size, close_size), dtype=bool)
    mask = binary_closing(mask, structure=struct, iterations=1)

    # 2. Fill holes
    mask = binary_fill_holes(mask)

    # 3. Largest connected component only
    labeled, n_comp = scipy_label(mask.astype(np.uint8))
    if n_comp > 1:
        sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
        largest = np.argmax(sizes) + 1
        mask = (labeled == largest)

    # Convert to uint8
    mask_uint8 = (mask.astype(np.uint8) * 255)

    return mask_uint8


def merge_dino_boxes(boxes, image_size_wh):
    """Merge DINO boxes that overlap or are very close, with center bias.

    If multiple boxes detected, tries to merge them into one fish region
    by union + dilation. Also merges vertically adjacent fragments
    that likely come from stripe-fragmented fish.
    """
    if len(boxes) <= 1:
        return boxes

    w, h = image_size_wh
    cx, cy = w / 2, h / 2

    # Convert to numpy
    boxes_arr = np.array(boxes)

    # Compute pairwise IoU and merge overlapping
    merged = []
    used = set()
    for i in range(len(boxes_arr)):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(boxes_arr)):
            if j in used:
                continue
            iou = _compute_ious(boxes_arr[i], boxes_arr[j:j+1])[0]
            # Also merge if boxes are close (within 20% of box size)
            gap = _box_gap(boxes_arr[i], boxes_arr[j])
            box_size = max(boxes_arr[i][2] - boxes_arr[i][0],
                          boxes_arr[i][3] - boxes_arr[i][1])
            if iou > 0.1 or gap < box_size * 0.2:
                group.append(j)
                used.add(j)
        used.add(i)

        # Union of group
        group_boxes = boxes_arr[np.array(group)]
        merged_box = [
            group_boxes[:, 0].min(),
            group_boxes[:, 1].min(),
            group_boxes[:, 2].max(),
            group_boxes[:, 3].max(),
        ]
        merged.append(merged_box)

    # Pick the most central merged box
    centralities = []
    for box in merged:
        bx = (box[0] + box[2]) / 2
        by = (box[1] + box[3]) / 2
        dist = np.sqrt(((bx - cx) / w)**2 + ((by - cy) / h)**2)
        centralities.append(dist)

    best = np.argmin(centralities)
    return [merged[best]]


def _box_gap(box_a, box_b):
    """Minimum gap between two boxes (0 if overlapping)."""
    dx = max(0, max(box_a[0], box_b[0]) - min(box_a[2], box_b[2]))
    dy = max(0, max(box_a[1], box_b[1]) - min(box_a[3], box_b[3]))
    return np.sqrt(dx**2 + dy**2)


def process_image_dinosam(image_path, output_path, apple_seg_path=None):
    """Full DINO+SAM pipeline for one image.

    If apple_seg_path is provided, uses the hybrid approach:
      1. Load the Apple Vision RGBA (foreground already extracted)
      2. Composite onto white background (makes DINO+SAM's job easier)
      3. Run DINO "a fish" detection on the composited image
      4. Run SAM to isolate just the fish from the foreground objects
      5. Intersect with original Apple Vision mask (can only REMOVE pixels)

    Returns metadata dict for reporting.
    """
    from PIL import Image

    try:
        img_orig = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "method": "dinosam",
            "dino_n_boxes": 0, "sam_mask_area": 0,
            "image_width": 0, "image_height": 0,
            "mask_fraction": 0, "segmented_path": "",
            "status": f"error_load: {e}",
        }

    w, h = img_orig.size
    img_orig_np = np.array(img_orig)

    # --- Hybrid mode: use Apple Vision output as input ---
    apple_mask = None
    if apple_seg_path and os.path.exists(apple_seg_path):
        try:
            apple_rgba = np.array(Image.open(apple_seg_path).convert("RGBA"))
            apple_mask = apple_rgba[:, :, 3]  # original alpha channel

            # Composite Apple foreground onto white background
            # This gives DINO a clean image: fish + coral on white, no reef bg
            fg = apple_rgba[:, :, :3].astype(float)
            alpha_f = (apple_mask / 255.0)[:, :, np.newaxis]
            composited = (fg * alpha_f + 255.0 * (1 - alpha_f)).astype(np.uint8)
            img_for_dino = Image.fromarray(composited)
        except Exception:
            img_for_dino = img_orig
            apple_mask = None
    else:
        img_for_dino = img_orig

    img_for_dino_np = np.array(img_for_dino)

    # Step 1: DINO detection
    boxes = detect_fish_dino(img_for_dino, threshold=0.15)
    n_boxes = len(boxes)

    if n_boxes == 0:
        boxes = detect_fish_dino(img_for_dino, threshold=0.08)
        n_boxes = len(boxes)

    if n_boxes == 0:
        # If hybrid mode and we have apple mask, just keep apple mask as-is
        if apple_mask is not None:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_orig_np
            rgba[:, :, 3] = apple_mask
            ensure_dir(os.path.dirname(output_path))
            Image.fromarray(rgba, "RGBA").save(output_path)
            mask_area = int((apple_mask > 128).sum())
            return {
                "method": "hybrid_apple_only",
                "dino_n_boxes": 0, "sam_mask_area": mask_area,
                "image_width": w, "image_height": h,
                "mask_fraction": round(mask_area / (w * h), 4),
                "segmented_path": os.path.relpath(output_path, SCRIPT_DIR),
                "status": "success",
            }
        return {
            "method": "dinosam",
            "dino_n_boxes": 0, "sam_mask_area": 0,
            "image_width": w, "image_height": h,
            "mask_fraction": 0, "segmented_path": "",
            "status": "no_detection",
        }

    # Step 2: Merge overlapping boxes
    merged_boxes = merge_dino_boxes(boxes, (w, h))

    # Step 3: SAM segmentation
    mask = segment_with_sam(img_for_dino, merged_boxes, img_for_dino_np)

    if mask is None or mask.max() == 0:
        if apple_mask is not None:
            # Fallback: keep apple mask
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_orig_np
            rgba[:, :, 3] = apple_mask
            ensure_dir(os.path.dirname(output_path))
            Image.fromarray(rgba, "RGBA").save(output_path)
            mask_area = int((apple_mask > 128).sum())
            return {
                "method": "hybrid_apple_only",
                "dino_n_boxes": n_boxes, "sam_mask_area": mask_area,
                "image_width": w, "image_height": h,
                "mask_fraction": round(mask_area / (w * h), 4),
                "segmented_path": os.path.relpath(output_path, SCRIPT_DIR),
                "status": "success",
            }
        return {
            "method": "dinosam",
            "dino_n_boxes": n_boxes, "sam_mask_area": 0,
            "image_width": w, "image_height": h,
            "mask_fraction": 0, "segmented_path": "",
            "status": "sam_failed",
        }

    # Step 4: If hybrid, intersect SAM fish mask with Apple Vision mask
    # This ensures we can only REMOVE background objects from the Apple
    # output, never ADD pixels that Apple excluded (background).
    method = "dinosam"
    if apple_mask is not None:
        final_mask = np.minimum(mask, apple_mask)
        method = "hybrid_apple_dinosam"
    else:
        final_mask = mask

    # Step 5: Build RGBA output using ORIGINAL image pixels
    mask_area = int((final_mask > 128).sum())
    mask_frac = mask_area / (w * h)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img_orig_np
    rgba[:, :, 3] = final_mask

    ensure_dir(os.path.dirname(output_path))
    Image.fromarray(rgba, "RGBA").save(output_path)

    return {
        "method": method,
        "dino_n_boxes": n_boxes,
        "sam_mask_area": mask_area,
        "image_width": w,
        "image_height": h,
        "mask_fraction": round(mask_frac, 4),
        "segmented_path": os.path.relpath(output_path, SCRIPT_DIR),
        "status": "success",
    }


def get_flagged_images():
    """Read mask quality report and return filenames flagged for resegmentation."""
    flagged = set()
    if not os.path.exists(QUALITY_CSV):
        print("    Warning: mask_quality_report.csv not found. "
              "Run score_masks.py first.")
        return flagged
    with open(QUALITY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["recommended_action"] in ("resegment_dinosam", "review"):
                flagged.add(row["filename"])
    return flagged


def main():
    parser = argparse.ArgumentParser(
        description="DINO+SAM segmentation for flagged images"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Process single image (path)")
    parser.add_argument("--apple-mask", type=str, default=None,
                        help="Apple Vision mask for hybrid mode (single image)")
    parser.add_argument("--all", action="store_true",
                        help="Run on ALL images (not just flagged)")
    parser.add_argument("--flagged-only", action="store_true", default=True,
                        help="Only process flagged images (default)")
    parser.add_argument("--test", type=int, default=None,
                        help="Limit to first N images")
    args = parser.parse_args()

    # Single image mode
    if args.image:
        print(f"[1] Processing single image: {args.image}")
        load_dino()
        load_sam()
        out_path = "/tmp/dinosam_test.png"
        # Check if there's an Apple Vision mask for hybrid mode
        apple_path = args.apple_mask
        result = process_image_dinosam(
            os.path.abspath(args.image), out_path,
            apple_seg_path=apple_path,
        )
        print(f"    Result: {result}")
        if result["status"] == "success":
            print(f"    Saved: {out_path}")
        return

    print("[1] Loading inventory...")
    inventory = load_inventory()
    annotations = load_annotations()

    # Determine which images to process
    if args.all:
        to_process = inventory
        print(f"    Processing ALL {len(to_process)} images")
    else:
        flagged = get_flagged_images()
        # Also include images where Apple Vision failed (no_detection)
        seg_report_path = os.path.join(GMM_DIR, "segmentation_report.csv")
        if os.path.exists(seg_report_path):
            with open(seg_report_path, newline="") as f:
                for row in csv.DictReader(f):
                    if row["status"] == "no_detection":
                        flagged.add(row["filename"])
        to_process = [r for r in inventory if r["filename"] in flagged]
        print(f"    Flagged/no-detection images: {len(to_process)}")

    # Skip larvae
    to_process = [r for r in to_process
                  if annotations.get(r["filename"], {}).get(
                      "annotation_type") != "larva"]

    if args.test:
        to_process = to_process[:args.test]
        print(f"    Test mode: first {args.test}")

    if not to_process:
        print("    No images to process.")
        return

    print("[2] Loading models...")
    load_dino()
    load_sam()

    print(f"[3] Processing {len(to_process)} images with DINO+SAM...")
    results = []
    n_success = n_fail = 0
    t_start = time.time()

    for i, row in enumerate(to_process):
        fname = row["filename"]
        species = row["species"]
        directory = row["directory"]
        image_path = get_image_path(row)
        sp_dir = species_to_dirname(species)
        out_name = os.path.splitext(fname)[0] + ".png"
        output_path = os.path.join(DINOSAM_DIR, sp_dir, out_name)

        # Check for existing Apple Vision mask (hybrid mode)
        apple_path = os.path.join(SEGMENTED_DIR, sp_dir, out_name)
        if not os.path.exists(apple_path):
            apple_path = None

        result = process_image_dinosam(image_path, output_path,
                                       apple_seg_path=apple_path)
        result["filename"] = fname
        result["species"] = species
        result["directory"] = directory
        results.append(result)

        if result["status"] == "success":
            n_success += 1
            print(f"  [{i+1:4d}/{len(to_process)}] {species} -- "
                  f"dino:{result['dino_n_boxes']}box, "
                  f"mask:{result['mask_fraction']:.1%}")
        else:
            n_fail += 1
            print(f"  [{i+1:4d}/{len(to_process)}] {species} -- "
                  f"{result['status']}")

        # Checkpoint
        if (i + 1) % 50 == 0:
            _write_report(results)

    _write_report(results)

    elapsed = time.time() - t_start
    print(f"\n[4] DINO+SAM complete in {elapsed:.0f}s")
    print(f"    Success: {n_success}")
    print(f"    Failed:  {n_fail}")
    print(f"    Report:  {DINOSAM_REPORT}")


def _write_report(results):
    with open(DINOSAM_REPORT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
