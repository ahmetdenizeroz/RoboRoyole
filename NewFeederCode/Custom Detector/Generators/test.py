import cv2
import numpy as np
import re
import random
import os
import math

# ============================================================
# Settings
# ============================================================
TXT_PATH = "selected_markers.txt"
OUT_PATH = "random_aruco_scene.png"

IMG_W = 1920
IMG_H = 1080

MARKER_SIDE_FLOAT = (1080 * 2) / 130.0
MARKER_SIDE_PX = round(MARKER_SIDE_FLOAT)   # 17 px
BACKGROUND_VALUE = 255                      # white background
ROTATE_RANDOMLY = True
RANDOM_SEED = None                          # set an int for reproducible output
EDGE_MARGIN = 20                            # margin from image border
MIN_GAP = 20                                # min gap between markers

# ============================================================
# TXT parsing
# ============================================================
def parse_marker_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    marker_size = None
    num_markers = None

    for line in lines:
        m = re.match(r"marker_size\s*=\s*(\d+)x(\d+)", line)
        if m:
            r = int(m.group(1))
            c = int(m.group(2))
            if r != c:
                raise ValueError("Only square markers are supported.")
            marker_size = r

        m = re.match(r"num_markers\s*=\s*(\d+)", line)
        if m:
            num_markers = int(m.group(1))

    if marker_size is None:
        raise ValueError("Could not parse marker_size from txt file.")

    markers = {}
    i = 0
    while i < len(lines):
        m = re.match(r"ID\s+(\d+):", lines[i])
        if m:
            marker_id = int(m.group(1))
            bits = []

            for j in range(1, marker_size + 1):
                if i + j >= len(lines):
                    raise ValueError(f"Not enough rows for marker ID {marker_id}")

                row_vals = lines[i + j].split()
                if len(row_vals) != marker_size:
                    raise ValueError(f"Wrong row length for marker ID {marker_id}")

                row = [int(x) for x in row_vals]
                if any(x not in (0, 1) for x in row):
                    raise ValueError(f"Non-binary value in marker ID {marker_id}")

                bits.append(row)

            markers[marker_id] = np.array(bits, dtype=np.uint8)
            i += marker_size + 1
        else:
            i += 1

    ordered_ids = sorted(markers.keys())
    ordered_markers = [markers[mid] for mid in ordered_ids]

    if num_markers is not None and len(ordered_markers) != num_markers:
        print(
            f"Warning: txt says num_markers={num_markers}, "
            f"but parsed {len(ordered_markers)} markers."
        )

    return marker_size, ordered_ids, ordered_markers

# ============================================================
# Dictionary construction
# ============================================================
def get_byte_list_from_bits(bits):
    if hasattr(cv2.aruco.Dictionary, "getByteListFromBits"):
        return cv2.aruco.Dictionary.getByteListFromBits(bits)
    return cv2.aruco.Dictionary_getByteListFromBits(bits)

def build_custom_dictionary(marker_size, marker_bits_list, maxcorr=1):
    rows = [get_byte_list_from_bits(bits) for bits in marker_bits_list]
    bytes_list = np.concatenate(rows, axis=0)
    return cv2.aruco.Dictionary(bytes_list, marker_size, maxcorr)

# ============================================================
# Geometry helpers
# ============================================================
def boxes_overlap(box1, box2, extra_gap=0):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return not (
        (x1 + w1 + extra_gap) <= x2 or
        (x2 + w2 + extra_gap) <= x1 or
        (y1 + h1 + extra_gap) <= y2 or
        (y2 + h2 + extra_gap) <= y1
    )

def find_non_overlapping_positions(num_boxes, marker_sizes, img_w, img_h,
                                   edge_margin=20, min_gap=10, max_tries=10000):
    placed = []
    for box_w, box_h in marker_sizes:
        success = False
        for _try in range(max_tries):
            x = random.randint(edge_margin, img_w - edge_margin - box_w)
            y = random.randint(edge_margin, img_h - edge_margin - box_h)
            new_box = (x, y, box_w, box_h)
            conflict = any(boxes_overlap(new_box, old_box, extra_gap=min_gap) for old_box in placed)
            if not conflict:
                placed.append(new_box)
                success = True
                break
        if not success:
            raise RuntimeError("Could not place all markers without overlap.")
    return placed

def rotate_image_full(image, angle, background_color):
    """
    Rotates image and expands canvas so no corners are cut.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box dimensions
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    # Adjust transformation matrix to center the rotated image
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=background_color)
    return rotated

# ============================================================
# Main
# ============================================================
def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    marker_size, marker_ids, marker_bits = parse_marker_txt(TXT_PATH)
    aruco_dict = build_custom_dictionary(marker_size, marker_bits, maxcorr=1)

    # Prepare rotated marker images first to know their actual sizes
    processed_markers = []
    for marker_id in marker_ids:
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIDE_PX, borderBits=1)
        angle = random.randint(0, 11) * 30 if ROTATE_RANDOMLY else 0
        img_rot = rotate_image_full(img, angle, BACKGROUND_VALUE)
        processed_markers.append((img_rot, angle))

    # Get the bounding box sizes for placement
    marker_bb_sizes = [(m[0].shape[1], m[0].shape[0]) for m in processed_markers]

    # Create white background
    canvas = np.full((IMG_H, IMG_W), BACKGROUND_VALUE, dtype=np.uint8)

    # Find random non-overlapping positions
    positions = find_non_overlapping_positions(
        num_boxes=len(marker_ids),
        marker_sizes=marker_bb_sizes,
        img_w=IMG_W,
        img_h=IMG_H,
        edge_margin=EDGE_MARGIN,
        min_gap=MIN_GAP
    )

    # Draw markers
    for idx, (marker_img, angle) in enumerate(processed_markers):
        x, y, w, h = positions[idx]
        canvas[y:y+h, x:x+w] = marker_img
        print(f"Placed ID {marker_ids[idx]} at x={x}, y={y}, box={w}x{h}, angle={angle}")

    # Save image
    cv2.imwrite(OUT_PATH, canvas)
    print(f"\nSaved: {OUT_PATH}")

if __name__ == "__main__":
    main()