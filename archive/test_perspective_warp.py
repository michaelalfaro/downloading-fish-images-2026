#!/usr/bin/env python3
"""
Headless test to debug perspective warp with and without offset (click-drag).
Tests whether applying offset before perspective warp causes issues.
"""

import numpy as np
import cv2
from PIL import Image
import os

# Test image
TEST_IMAGE = "/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/analysis/approach_1_gmm/segmented/Amphichaetodon_howensis/Amphichaetodon_howensis_Bishop_472868867.png"
OUTPUT_DIR = "/Users/michaelalfaro/Dropbox/git/downloading-fish-images-2026/test_perspective_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def apply_perspective_correction(img, dorsal_ventral_angle, head_tail_angle=0):
    """Apply perspective correction - same logic as review_app.py

    Maps FROM a trapezoid source TO a rectangle destination.
    This keeps the center fixed while warping the edges.
    """
    import math

    if dorsal_ventral_angle == 0 and head_tail_angle == 0:
        return img

    arr = np.array(img)
    h, w = arr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Warp strength
    max_warp = 0.20
    dv_norm = max(-1, min(1, dorsal_ventral_angle / 30.0))
    ht_norm = max(-1, min(1, head_tail_angle / 30.0))

    dv_squeeze = dv_norm * max_warp
    ht_squeeze = ht_norm * max_warp

    # Calculate insets for source trapezoid
    top_inset = cx * dv_squeeze
    bottom_inset = -cx * dv_squeeze
    left_inset = cy * ht_squeeze
    right_inset = -cy * ht_squeeze

    # SOURCE: Trapezoid (where we sample from)
    src_pts = np.float32([
        [top_inset, left_inset],
        [w - 1 - top_inset, right_inset],
        [bottom_inset, h - 1 - left_inset],
        [w - 1 - bottom_inset, h - 1 - right_inset]
    ])

    # DESTINATION: Rectangle (output shape)
    dst_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])

    print(f"  Perspective: DV={dorsal_ventral_angle}° HT={head_tail_angle}°")
    print(f"  Src trapezoid: TL={src_pts[0]}, TR={src_pts[1]}, BL={src_pts[2]}, BR={src_pts[3]}")

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(arr, matrix, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))

    return Image.fromarray(result)


def apply_offset(img, offset_x, offset_y):
    """Apply translation offset - same logic as review_app.py"""
    arr = np.array(img)
    h, w = arr.shape[:2]
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    translated = cv2.warpAffine(arr, M, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
    print(f"  Offset: X={offset_x}, Y={offset_y}")
    return Image.fromarray(translated)


def save_preview(img, filename):
    """Save image with gray background for visibility"""
    arr = np.array(img, dtype=np.float32)
    rgb = arr[:, :, :3]
    alpha = (arr[:, :, 3] / 255.0)[:, :, np.newaxis]
    bg = 180.0
    result = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
    out = Image.fromarray(result)
    out.save(os.path.join(OUTPUT_DIR, filename))
    print(f"  Saved: {filename}")


def test_pipeline():
    """Test different orderings of perspective and offset"""
    print(f"Loading: {TEST_IMAGE}")
    img = Image.open(TEST_IMAGE).convert("RGBA")
    print(f"Image size: {img.size}")

    # Test parameters
    dv_tilt = 20  # Dorsal-ventral tilt
    ht_tilt = 15  # Head-tail tilt
    offset_x = 50  # Simulated drag offset
    offset_y = -30

    print("\n" + "="*60)
    print("TEST 1: No offset, just perspective warp")
    print("="*60)
    result1 = apply_perspective_correction(img, dv_tilt, ht_tilt)
    save_preview(result1, "test1_perspective_only.png")

    print("\n" + "="*60)
    print("TEST 2: Perspective FIRST, then offset (current app order)")
    print("="*60)
    result2a = apply_perspective_correction(img, dv_tilt, ht_tilt)
    result2b = apply_offset(result2a, offset_x, offset_y)
    save_preview(result2b, "test2_perspective_then_offset.png")

    print("\n" + "="*60)
    print("TEST 3: Offset FIRST, then perspective")
    print("="*60)
    result3a = apply_offset(img, offset_x, offset_y)
    result3b = apply_perspective_correction(result3a, dv_tilt, ht_tilt)
    save_preview(result3b, "test3_offset_then_perspective.png")

    print("\n" + "="*60)
    print("TEST 4: Combined in single transform matrix")
    print("="*60)
    # Build a combined perspective + translation matrix
    arr = np.array(img)
    h, w = arr.shape[:2]

    max_warp = 0.25
    dv_norm = dv_tilt / 30.0
    ht_norm = ht_tilt / 30.0
    dv_inset = (w / 2.0) * dv_norm * max_warp
    ht_inset = (h / 2.0) * ht_norm * max_warp

    # Source: rectangle
    src_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])

    # Destination: warped + offset (add offset to all corners)
    dst_pts = np.float32([
        [0 + dv_inset + offset_x, 0 + ht_inset + offset_y],
        [w - 1 - dv_inset + offset_x, 0 - ht_inset + offset_y],
        [0 - dv_inset + offset_x, h - 1 - ht_inset + offset_y],
        [w - 1 + dv_inset + offset_x, h - 1 + ht_inset + offset_y]
    ])

    print(f"  Combined: perspective + offset in one matrix")
    print(f"  Dst corners: TL={dst_pts[0]}, TR={dst_pts[1]}, BL={dst_pts[2]}, BR={dst_pts[3]}")

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result4 = cv2.warpPerspective(arr, matrix, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0, 0))
    save_preview(Image.fromarray(result4), "test4_combined_transform.png")

    print("\n" + "="*60)
    print("TEST 5: No perspective, just offset (for comparison)")
    print("="*60)
    result5 = apply_offset(img, offset_x, offset_y)
    save_preview(result5, "test5_offset_only.png")

    print("\n" + "="*60)
    print("COMPARISON: Check if Test 1 and Test 2 have same warp shape")
    print("="*60)
    # Compare the actual fish pixels (ignoring position)
    arr1 = np.array(result1)
    arr2a = np.array(result2a)  # Before offset was applied

    # They should be identical
    if np.array_equal(arr1, arr2a):
        print("  ✓ Test 1 and Test 2 (before offset) are IDENTICAL")
    else:
        diff = np.abs(arr1.astype(float) - arr2a.astype(float)).mean()
        print(f"  ✗ Test 1 and Test 2 differ! Mean diff: {diff:.4f}")

    print("\n" + "="*60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*60)


def test_canvas_placement():
    """Test the actual canvas placement logic from review_app.py"""
    print("\n" + "="*60)
    print("TEST 6: Simulating review_app.py canvas placement")
    print("="*60)

    img = Image.open(TEST_IMAGE).convert("RGBA")

    # Simulate rotation (just use original for simplicity)
    rotated = img

    dv_tilt = 20
    ht_tilt = 15
    offset_x = 50
    offset_y = -30
    scale = 100
    canvas_size = 800

    print(f"\n--- Test 6a: Perspective, then place on canvas with offset ---")
    # This is what review_app.py currently does
    fish_with_perspective = apply_perspective_correction(rotated, dv_tilt, ht_tilt)

    # Scale
    fish_img = fish_with_perspective
    if scale != 100:
        w, h = fish_img.size
        scale_factor = scale / 100.0
        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))
        fish_img = fish_img.resize((new_w, new_h), Image.LANCZOS)

    # Create canvas and place fish
    canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
    fish_w, fish_h = fish_img.size
    base_x = (canvas_size - fish_w) // 2
    base_y = (canvas_size - fish_h) // 2
    paste_x = base_x + int(offset_x)
    paste_y = base_y + int(offset_y)

    print(f"  Canvas: {canvas_size}x{canvas_size}")
    print(f"  Fish: {fish_w}x{fish_h}")
    print(f"  Base position: ({base_x}, {base_y})")
    print(f"  With offset: ({paste_x}, {paste_y})")

    # Paste fish onto canvas
    fish_arr = np.array(fish_img)
    canvas_arr = np.array(canvas)

    src_x1 = max(0, -paste_x)
    src_y1 = max(0, -paste_y)
    src_x2 = min(fish_w, canvas_size - paste_x)
    src_y2 = min(fish_h, canvas_size - paste_y)

    dst_x1 = max(0, paste_x)
    dst_y1 = max(0, paste_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        fish_region = fish_arr[src_y1:src_y2, src_x1:src_x2]
        canvas_arr[dst_y1:dst_y2, dst_x1:dst_x2] = fish_region

    save_preview(Image.fromarray(canvas_arr), "test6a_perspective_then_canvas_offset.png")

    print(f"\n--- Test 6b: Place on canvas with offset, THEN perspective ---")
    # Alternative: place first, then warp entire canvas
    canvas2 = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
    fish_arr2 = np.array(rotated)
    canvas_arr2 = np.array(canvas2)

    fish_w2, fish_h2 = rotated.size
    base_x2 = (canvas_size - fish_w2) // 2
    base_y2 = (canvas_size - fish_h2) // 2
    paste_x2 = base_x2 + int(offset_x)
    paste_y2 = base_y2 + int(offset_y)

    src_x1 = max(0, -paste_x2)
    src_y1 = max(0, -paste_y2)
    src_x2 = min(fish_w2, canvas_size - paste_x2)
    src_y2 = min(fish_h2, canvas_size - paste_y2)

    dst_x1 = max(0, paste_x2)
    dst_y1 = max(0, paste_y2)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        fish_region = fish_arr2[src_y1:src_y2, src_x1:src_x2]
        canvas_arr2[dst_y1:dst_y2, dst_x1:dst_x2] = fish_region

    canvas_with_fish = Image.fromarray(canvas_arr2)

    # NOW apply perspective to the entire canvas
    result = apply_perspective_correction(canvas_with_fish, dv_tilt, ht_tilt)
    save_preview(result, "test6b_canvas_offset_then_perspective.png")

    print(f"\n--- Test 6c: No offset, perspective on canvas ---")
    # Place fish at center (no offset), then perspective
    canvas3 = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
    fish_arr3 = np.array(rotated)
    canvas_arr3 = np.array(canvas3)

    fish_w3, fish_h3 = rotated.size
    paste_x3 = (canvas_size - fish_w3) // 2  # Centered, no offset
    paste_y3 = (canvas_size - fish_h3) // 2

    src_x1 = max(0, -paste_x3)
    src_y1 = max(0, -paste_y3)
    src_x2 = min(fish_w3, canvas_size - paste_x3)
    src_y2 = min(fish_h3, canvas_size - paste_y3)

    dst_x1 = max(0, paste_x3)
    dst_y1 = max(0, paste_y3)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        fish_region = fish_arr3[src_y1:src_y2, src_x1:src_x2]
        canvas_arr3[dst_y1:dst_y2, dst_x1:dst_x2] = fish_region

    canvas_centered = Image.fromarray(canvas_arr3)
    result_centered = apply_perspective_correction(canvas_centered, dv_tilt, ht_tilt)
    save_preview(result_centered, "test6c_centered_then_perspective.png")

    print("\n" + "="*60)
    print("ANALYSIS:")
    print("- Test 6a: Current app behavior - warp fish, then place with offset")
    print("- Test 6b: Alternative - place with offset, then warp entire canvas")
    print("- Test 6c: Centered fish, then warp canvas (warp around canvas center)")
    print("")
    print("If 6b and 6c show proper warping around canvas center,")
    print("then we need to apply perspective AFTER placing on canvas.")
    print("="*60)


def test_sequential_drag_and_warp():
    """
    Test the real-world scenario:
    1. User drags to position fish (offset1)
    2. User applies warp (tilt1)
    3. User drags again to reposition (offset2)
    4. User adjusts warp (tilt2)

    Each preview request sends the CUMULATIVE state, not incremental changes.
    So we simulate what the app does for each preview.
    """
    print("\n" + "="*60)
    print("TEST 7: Sequential drag and warp (simulating user interaction)")
    print("="*60)

    img = Image.open(TEST_IMAGE).convert("RGBA")
    canvas_size = 800

    def simulate_preview(fish_img, offset_x, offset_y, dv_tilt, ht_tilt, label):
        """Simulate what review_app.py does for each preview request"""
        print(f"\n--- {label} ---")
        print(f"  State: offset=({offset_x}, {offset_y}), DV={dv_tilt}°, HT={ht_tilt}°")

        # 1. Create canvas
        canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))

        # 2. Place fish on canvas with offset
        fish_arr = np.array(fish_img)
        canvas_arr = np.array(canvas)
        fish_w, fish_h = fish_img.size

        base_x = (canvas_size - fish_w) // 2
        base_y = (canvas_size - fish_h) // 2
        paste_x = base_x + int(offset_x)
        paste_y = base_y + int(offset_y)

        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        src_x2 = min(fish_w, canvas_size - paste_x)
        src_y2 = min(fish_h, canvas_size - paste_y)

        dst_x1 = max(0, paste_x)
        dst_y1 = max(0, paste_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            fish_region = fish_arr[src_y1:src_y2, src_x1:src_x2]
            canvas_arr[dst_y1:dst_y2, dst_x1:dst_x2] = fish_region

        canvas_with_fish = Image.fromarray(canvas_arr)

        # 3. Apply perspective to entire canvas (around canvas center)
        if dv_tilt != 0 or ht_tilt != 0:
            canvas_with_fish = apply_perspective_correction(canvas_with_fish, dv_tilt, ht_tilt)

        return canvas_with_fish

    # Simulate user interaction sequence:

    # Step 1: Initial preview (centered, no warp)
    result1 = simulate_preview(img, 0, 0, 0, 0, "Step 1: Initial (centered, no warp)")
    save_preview(result1, "test7_step1_initial.png")

    # Step 2: User drags fish to new position
    result2 = simulate_preview(img, 80, -50, 0, 0, "Step 2: After first drag (offset only)")
    save_preview(result2, "test7_step2_first_drag.png")

    # Step 3: User applies warp at that position
    result3 = simulate_preview(img, 80, -50, 20, 10, "Step 3: After first warp (DV=20, HT=10)")
    save_preview(result3, "test7_step3_first_warp.png")

    # Step 4: User drags again (warp values unchanged, position changes)
    result4 = simulate_preview(img, -30, 60, 20, 10, "Step 4: After second drag (new position, same warp)")
    save_preview(result4, "test7_step4_second_drag.png")

    # Step 5: User adjusts warp at new position
    result5 = simulate_preview(img, -30, 60, -15, 25, "Step 5: After second warp adjustment")
    save_preview(result5, "test7_step5_second_warp.png")

    print("\n" + "="*60)
    print("EXPECTED BEHAVIOR:")
    print("- Step 1-2: Fish moves, no warp visible")
    print("- Step 3: Warp applied around CANVAS center (red axes)")
    print("         Fish is offset, so it warps asymmetrically relative to itself")
    print("         but symmetrically relative to canvas center")
    print("- Step 4: Fish moves to new position, SAME warp around canvas center")
    print("- Step 5: Different warp, still around canvas center")
    print("")
    print("KEY INSIGHT: The warp is ALWAYS around canvas center (400,400),")
    print("not around the fish. So moving the fish changes how the warp")
    print("affects it visually, but the warp axes stay fixed.")
    print("="*60)


if __name__ == "__main__":
    test_pipeline()
    test_canvas_placement()
    test_sequential_drag_and_warp()
