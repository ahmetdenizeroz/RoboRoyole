import cv2
import numpy as np
import re
import sys
import math

# =========================================================
# USER SETTINGS
# =========================================================
TXT_PATH = "4x4_20.txt"   # your 4x4 custom dictionary txt
CAMERA_INDEX = 1
CAMERA_BACKEND = cv2.CAP_DSHOW
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Marker model
BORDER_BITS = 1
MARKER_SIZE = None   # loaded from txt
TOTAL_CELLS = None   # MARKER_SIZE + 2 * BORDER_BITS
CELL_PIXELS = 40

# Decode tolerance
# For the selected 4x4_20 set from DICT_4X4_50, d_min = 5,
# so Hamming <= 2 is theoretically still separable.
# Start with 1 if you want to be conservative.
MAX_HAMMING = 0
MAX_BORDER_ERRORS = 10

# Candidate filtering
MIN_AREA = 100
MIN_SOLIDITY = 0.6
POLY_EPS_RATIO = 0.05
MAX_CONTOUR_AREA = 500

# Warp / sampling tweaks
QUAD_EXPAND_SCALE = 1.18
SAMPLE_INSET_RATIO = 0.20

# Debug
SHOW_ID0_DEBUG = True


# =========================================================
# TXT PARSER
# =========================================================
def load_custom_markers_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    marker_size = None
    for line in lines:
        m = re.match(r"marker_size\s*=\s*(\d+)x(\d+)", line)
        if m:
            rows = int(m.group(1))
            cols = int(m.group(2))
            if rows != cols:
                raise ValueError("Only square markers are supported.")
            marker_size = rows
            break

    if marker_size is None:
        raise ValueError("Could not find marker_size in txt file.")

    markers = {}
    i = 0
    while i < len(lines):
        m = re.match(r"ID\s+(\d+):", lines[i])
        if not m:
            i += 1
            continue

        marker_id = int(m.group(1))
        i += 1

        rows_data = []
        while i < len(lines) and len(rows_data) < marker_size:
            if not lines[i]:
                i += 1
                continue
            if re.match(r"^[01](\s+[01])+$", lines[i]):
                vals = [int(x) for x in lines[i].split()]
                if len(vals) != marker_size:
                    raise ValueError(f"Invalid row length in marker ID {marker_id}.")
                rows_data.append(vals)
            i += 1

        if len(rows_data) != marker_size:
            raise ValueError(f"Incomplete marker definition for ID {marker_id}.")

        markers[marker_id] = np.array(rows_data, dtype=np.uint8)

    if not markers:
        raise ValueError("No marker definitions found in txt file.")

    return marker_size, markers


# =========================================================
# ROI HELPERS
# =========================================================
def clamp_roi(x, y, w, h, frame_w, frame_h):
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return int(x), int(y), int(w), int(h)


def select_roi_from_frame(frame):
    roi = cv2.selectROI(
        "Select ROI around marker area",
        frame,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyWindow("Select ROI around marker area")

    x, y, w, h = roi
    if w == 0 or h == 0:
        return None

    frame_h, frame_w = frame.shape[:2]
    return clamp_roi(x, y, w, h, frame_w, frame_h)


def point_in_roi(pt, roi):
    if roi is None:
        return True
    x, y = pt
    rx, ry, rw, rh = roi
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)


# =========================================================
# GEOMETRY / DRAWING
# =========================================================
def order_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def expand_quad(quad, scale=QUAD_EXPAND_SCALE):
    quad = np.asarray(quad, dtype=np.float32)
    center = quad.mean(axis=0)
    return (quad - center) * scale + center


def angle_of_quad_deg(quad):
    q = order_quad(quad)
    tl, tr = q[0], q[1]
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    ang = math.degrees(math.atan2(-dy, dx))
    return (ang + 360.0) % 360.0


def draw_label(img, quad, lines, color=(0, 255, 0)):
    q = np.asarray(quad, dtype=np.int32).reshape(4, 1, 2)
    cv2.polylines(img, [q], True, color, 2, cv2.LINE_AA)

    center = np.mean(q.reshape(4, 2), axis=0).astype(int)
    x = int(center[0]) + 8
    y = int(center[1]) - 8

    for i, line in enumerate(lines):
        yy = y + i * 20
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)


# =========================================================
# CANDIDATE FINDING
# =========================================================
def find_square_candidates(red_channel, roi=None):
    """
    Finds square-ish dark candidates from the red channel.
    Keeps only candidates whose center is inside ROI.
    """
    _, bw_inv = cv2.threshold(
        red_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(bw_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    quads = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_CONTOUR_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(cnt, POLY_EPS_RATIO * peri, True)

        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue

        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:
            continue

        quad = approx.reshape(4, 2).astype(np.float32)
        ordered = order_quad(quad)

        side_lengths = [
            np.linalg.norm(ordered[0] - ordered[1]),
            np.linalg.norm(ordered[1] - ordered[2]),
            np.linalg.norm(ordered[2] - ordered[3]),
            np.linalg.norm(ordered[3] - ordered[0]),
        ]
        min_side = min(side_lengths)
        max_side = max(side_lengths)

        if min_side < 6:
            continue
        if max_side / max(min_side, 1e-6) > 1.8:
            continue

        center = ordered.mean(axis=0)
        if not point_in_roi(center, roi):
            continue

        quads.append(ordered)

    return deduplicate_quads(quads)


def deduplicate_quads(quads, center_thresh=12.0):
    if not quads:
        return []

    items = []
    for q in quads:
        center = q.mean(axis=0)
        area = abs(cv2.contourArea(q))
        items.append((q, center, area))

    items.sort(key=lambda x: -x[2])

    kept = []
    for q, center, area in items:
        duplicate = False
        for kq, kc, ka in kept:
            if np.linalg.norm(center - kc) < center_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append((q, center, area))

    return [x[0] for x in kept]


# =========================================================
# WARP / SAMPLE / DECODE
# =========================================================
def warp_marker(gray_img, quad):
    global TOTAL_CELLS

    side = TOTAL_CELLS * CELL_PIXELS

    src = order_quad(expand_quad(quad, QUAD_EXPAND_SCALE))
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray_img, M, (side, side), flags=cv2.INTER_LINEAR)
    return warped


def sample_grid_from_warp(warp_gray, inset_ratio=SAMPLE_INSET_RATIO):
    """
    1 = white
    0 = black
    """
    global TOTAL_CELLS

    _, bw = cv2.threshold(
        warp_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    h, w = bw.shape
    cell_w = w / TOTAL_CELLS
    cell_h = h / TOTAL_CELLS

    bits = np.zeros((TOTAL_CELLS, TOTAL_CELLS), dtype=np.uint8)

    for r in range(TOTAL_CELLS):
        for c in range(TOTAL_CELLS):
            x0 = int((c + inset_ratio) * cell_w)
            x1 = int((c + 1 - inset_ratio) * cell_w)
            y0 = int((r + inset_ratio) * cell_h)
            y1 = int((r + 1 - inset_ratio) * cell_h)

            patch = bw[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            white_ratio = patch.mean() / 255.0
            bits[r, c] = 1 if white_ratio >= 0.5 else 0

    return bits, bw


def count_border_errors(full_bits):
    top = full_bits[0, :]
    bottom = full_bits[-1, :]
    left = full_bits[1:-1, 0]
    right = full_bits[1:-1, -1]

    border = np.concatenate([top, bottom, left, right])

    # black border expected -> 0
    return int(np.count_nonzero(border == 1))


def decode_custom_marker(warp_gray, custom_markers):
    full_bits, bw = sample_grid_from_warp(warp_gray)
    border_errors = count_border_errors(full_bits)

    inner = full_bits[1:-1, 1:-1]

    best = None
    for marker_id, pattern in custom_markers.items():
        for rot_cw in (0, 90, 180, 270):
            k = (-rot_cw // 90) % 4
            expected = np.rot90(pattern, k=k)

            hamming = int(np.count_nonzero(inner != expected))

            if best is None or hamming < best["hamming"]:
                best = {
                    "id": marker_id,
                    "rotation_cw_deg": rot_cw,
                    "hamming": hamming,
                    "inner": inner.copy(),
                    "expected": expected.copy(),
                    "full_bits": full_bits.copy(),
                    "border_errors": border_errors,
                    "bw": bw
                }

    if best is None:
        return None

    if best["hamming"] <= MAX_HAMMING and best["border_errors"] <= MAX_BORDER_ERRORS:
        return best

    if SHOW_ID0_DEBUG and best["id"] == 0:
        print("REJECTED ID 0")
        print("full_bits:")
        print(full_bits)
        print("inner:")
        print(inner)
        print("expected:")
        print(best["expected"])
        print("best id:", best["id"], "rot:", best["rotation_cw_deg"])
        print("hamming:", best["hamming"], "border_errors:", best["border_errors"])
        print("---")
        cv2.imshow("debug_warp_gray", warp_gray)
        cv2.imshow("debug_bw", bw)
        cv2.waitKey(1)

    return None


# =========================================================
# MAIN
# =========================================================
def main():
    global MARKER_SIZE, TOTAL_CELLS

    try:
        MARKER_SIZE, custom_markers = load_custom_markers_from_txt(TXT_PATH)
        TOTAL_CELLS = MARKER_SIZE + 2 * BORDER_BITS
    except Exception as e:
        print(f"Could not load marker txt file: {e}")
        sys.exit(1)

    print(f"Loaded custom dictionary. marker_size = {MARKER_SIZE}x{MARKER_SIZE}")
    print(f"TOTAL_CELLS = {TOTAL_CELLS}")
    for mid in sorted(custom_markers.keys()):
        print(f"ID {mid}:")
        print(custom_markers[mid])

    #cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    cap = cv2.VideoCapture("bee_chunk_000.mp4")
    if not cap.isOpened():
        print("Could not open camera/video.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        print("Could not read first frame.")
        cap.release()
        sys.exit(1)

    roi = select_roi_from_frame(first_frame)
    if roi is None:
        print("ROI selection canceled.")
        cap.release()
        sys.exit(1)

    print(f"ROI selected: {roi}")

    prev_t = cv2.getTickCount()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Frame read failed.")
            break

        # red channel only
        red = frame[:, :, 2]

        # candidate finding inside ROI
        quads = find_square_candidates(red, roi=roi)

        decoded = []
        for quad in quads:
            warped = warp_marker(red, quad)
            result = decode_custom_marker(warped, custom_markers)
            if result is not None:
                result["quad"] = quad
                result["angle_deg"] = angle_of_quad_deg(quad)
                decoded.append(result)

        vis = frame.copy()

        # draw ROI
        if roi is not None:
            rx, ry, rw, rh = roi
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)
            cv2.putText(vis, "ROI filter", (rx, max(20, ry - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, "ROI filter", (rx, max(20, ry - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv2.LINE_AA)

        # draw decoded markers
        for d in decoded:
            '''
            draw_label(
                vis,
                d["quad"],
                [
                    f"ID {d['id']} | rot {d['rotation_cw_deg']} CW",
                    f"angle {d['angle_deg']:.1f} | ham {d['hamming']} | be {d['border_errors']}"
                ],
                color=(0, 255, 0)
            )
            '''
            draw_label(
                vis,
                d["quad"],
                [
                    f"ID {d['id']}"
                ],
                color=(0, 255, 0)
            )

        # draw non-decoded candidates in orange
        decoded_centers = [np.mean(d["quad"], axis=0) for d in decoded]
        for q in quads:
            c = np.mean(q, axis=0)
            matched = any(np.linalg.norm(c - dc) < 8 for dc in decoded_centers)
            if not matched:
                qq = np.asarray(q, dtype=np.int32).reshape(4, 1, 2)
                cv2.polylines(vis, [qq], True, (0, 165, 255), 1, cv2.LINE_AA)

        # FPS
        now_t = cv2.getTickCount()
        dt = (now_t - prev_t) / cv2.getTickFrequency()
        prev_t = now_t
        fps_inst = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = fps_inst if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps_inst

        lines = [
            f"Custom markers loaded: {len(custom_markers)}",
            f"Marker size: {MARKER_SIZE}x{MARKER_SIZE}",
            f"Candidates in ROI: {len(quads)}",
            f"Decoded: {len(decoded)}",
            f"FPS: {fps_smooth:.2f}",
            "q: quit   r: redraw ROI"
        ]

        y0 = 25
        for i, line in enumerate(lines):
            y = y0 + i * 24
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Custom Marker Detection", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            new_roi = select_roi_from_frame(frame)
            if new_roi is not None:
                roi = new_roi
                print(f"New ROI selected: {roi}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
