import cv2
import numpy as np

from bee_detector import BeeDetector, BeeDrawer


VIDEO_PATH = "feed_output_20260410_171655_1920x1080.mp4"
BG_IMAGE_PATH = "median_background.png"
MARKER_TXT_PATH = "3x3_10.txt"

THRESHOLD_VALUE = 50
MIN_BLOB_AREA = 1300
SQUARE_SCALE = 1.4
MIN_SQUARE_SIZE = 24

DILATE_KERNEL_SIZE = 7
DILATE_ITERATIONS = 1
MORPH_OPEN_ITER = 0

ANGLE_LINE_LENGTH = 50

# BGR colors
ID_COLORS = {
    0: (0, 255, 0),        # green
    1: (255, 0, 255),      # purple
    2: (180, 105, 255),    # pink
    3: (0, 100, 0),        # dark green
    9: (255, 0, 0),        # blue
}

DEFAULT_DETECTED_COLOR = (0, 255, 255)   # fallback for recognized but unmapped ids
UNRECOGNIZED_COLOR = (0, 165, 255)       # orange


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def get_marker_color(marker_id):
    return ID_COLORS.get(marker_id, DEFAULT_DETECTED_COLOR)


def make_square_from_rect(x, y, w, h, img_w, img_h, scale=1.0, min_size=16):
    cx = x + w / 2.0
    cy = y + h / 2.0

    side = int(max(w, h) * scale)
    side = max(side, min_size)

    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > img_w:
        shift = x1 - img_w
        x0 -= shift
        x1 = img_w
    if y1 > img_h:
        shift = y1 - img_h
        y0 -= shift
        y1 = img_h

    x0 = clamp(x0, 0, img_w)
    y0 = clamp(y0, 0, img_h)
    x1 = clamp(x1, 0, img_w)
    y1 = clamp(y1, 0, img_h)

    return x0, y0, x1, y1


def detector_angle_to_horizontal_ccw(angle_deg):
    """
    Detector convention:
        0   = up
        90  = left
        180 = down
        270 = right

    Convert to:
        angle from horizontal axis
        CCW positive
        range [0, 360)
    """
    return (angle_deg + 90.0) % 360.0


def draw_angle_line(display, corners, angle_deg, color, length=50):
    center = np.mean(corners, axis=0)

    angle_h = detector_angle_to_horizontal_ccw(angle_deg)
    rad = np.deg2rad(angle_h)

    end_x = int(round(center[0] + length * np.cos(rad)))
    end_y = int(round(center[1] - length * np.sin(rad)))

    c = tuple(np.round(center).astype(int))
    e = (end_x, end_y)

    cv2.line(display, c, e, color, 2, cv2.LINE_AA)
    cv2.circle(display, c, 3, color, -1, cv2.LINE_AA)


def draw_unrecognized_square(display, sx0, sy0, sx1, sy1):
    cv2.rectangle(
        display,
        (sx0, sy0),
        (sx1, sy1),
        UNRECOGNIZED_COLOR,
        2,
        cv2.LINE_AA,
    )


def main():
    detector = BeeDetector(MARKER_TXT_PATH)

    drawer = BeeDrawer(
        default_border_color=DEFAULT_DETECTED_COLOR,
        default_text_color=DEFAULT_DETECTED_COLOR,
        candidate_color=UNRECOGNIZED_COLOR,
        thickness=2,
        font_scale=0.5,
    )
    drawer.set_id_colors(
        border_colors=ID_COLORS,
        text_colors=ID_COLORS,
        clear_existing=True,
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {VIDEO_PATH}")

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError("Could not read first frame from video.")

    bg_bgr = cv2.imread(BG_IMAGE_PATH, cv2.IMREAD_COLOR)
    if bg_bgr is None:
        cap.release()
        raise ValueError(f"Could not load background image: {BG_IMAGE_PATH}")

    if bg_bgr.shape[:2] != first_frame.shape[:2]:
        bg_bgr = cv2.resize(
            bg_bgr,
            (first_frame.shape[1], first_frame.shape[0]),
            interpolation=cv2.INTER_AREA
        )

    roi = cv2.selectROI("Draw ROI", first_frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Draw ROI")

    rx, ry, rw, rh = roi
    if rw <= 0 or rh <= 0:
        cap.release()
        raise RuntimeError("No ROI selected.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE)
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        roi_frame_bgr = frame_bgr[ry:ry + rh, rx:rx + rw]
        roi_bg_bgr = bg_bgr[ry:ry + rh, rx:rx + rw]

        diff_bgr = cv2.absdiff(roi_frame_bgr, roi_bg_bgr)
        diff_scalar = np.max(diff_bgr, axis=2).astype(np.uint8)

        _, binary = cv2.threshold(diff_scalar, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        binary = cv2.dilate(binary, dilate_kernel, iterations=DILATE_ITERATIONS)

        if MORPH_OPEN_ITER > 0:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                open_kernel,
                iterations=MORPH_OPEN_ITER
            )

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display = frame_bgr.copy()
        binary_preview = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1, cv2.LINE_AA)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_BLOB_AREA:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx_local = int(M["m10"] / M["m00"])
            cy_local = int(M["m01"] / M["m00"])

            x, y, w, h = cv2.boundingRect(cnt)

            sx0_l, sy0_l, sx1_l, sy1_l = make_square_from_rect(
                x, y, w, h,
                img_w=rw,
                img_h=rh,
                scale=SQUARE_SCALE,
                min_size=MIN_SQUARE_SIZE
            )

            sx0 = rx + sx0_l
            sy0 = ry + sy0_l
            sx1 = rx + sx1_l
            sy1 = ry + sy1_l

            crop_bgr = frame_bgr[sy0:sy1, sx0:sx1]
            if crop_bgr.size == 0:
                continue

            crop_for_detector = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

            result = detector.detect(crop_for_detector)
            detections = result.get("detections", [])

            cv2.drawContours(binary_preview, [cnt], -1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(binary_preview, (cx_local, cy_local), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.rectangle(binary_preview, (sx0_l, sy0_l), (sx1_l, sy1_l), (255, 0, 0), 1, cv2.LINE_AA)

            if len(detections) == 0:
                draw_unrecognized_square(display, sx0, sy0, sx1, sy1)
                continue

            detections_for_draw = []

            for det in detections:
                marker_id = det["id"]
                angle_deg = float(det["angle_deg"])

                corners = det["marker_corners"].copy().astype(np.float32)
                corners[:, 0] += sx0
                corners[:, 1] += sy0

                angle_h = detector_angle_to_horizontal_ccw(angle_deg)

                detections_for_draw.append({
                    "id": marker_id,
                    "angle_deg": angle_deg,
                    "marker_corners": corners,
                    "label_text": f"ID {marker_id} | {angle_h:.1f} deg",
                })

            display = drawer.draw_detected_markers(
                display,
                detections_for_draw,
                draw_ids=False,
                draw_angles=False,
                draw_center=True,
                draw_first_corner=True,
                draw_orientation=False,
                copy=False,
            )

            for det in detections_for_draw:
                marker_id = det["id"]
                color = get_marker_color(marker_id)
                corners = det["marker_corners"]
                angle_deg = det["angle_deg"]

                draw_angle_line(
                    display,
                    corners,
                    angle_deg,
                    color,
                    length=ANGLE_LINE_LENGTH,
                )

        cv2.imshow("Binary ROI", binary_preview)
        cv2.imshow("Blob + BeeDrawer ArUco Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()