import cv2
import numpy as np

from bee_detector import BeeDetector


VIDEO_PATH = "feed_output_20260410_171655_1920x1080.mp4"
BG_IMAGE_PATH = "median_background.png"
MARKER_TXT_PATH = "3x3_10.txt"

THRESHOLD_VALUE = 50
MIN_BLOB_AREA = 500
SQUARE_SCALE = 1.4
MIN_SQUARE_SIZE = 24

MORPH_OPEN_ITER = 1
MORPH_DILATE_ITER = 1


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


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


def choose_best_detection(detections):
    if not detections:
        return None
    return min(
        detections,
        key=lambda d: (d.get("hamming", 9999), d.get("border_errors", 9999))
    )


def main():
    detector = BeeDetector(MARKER_TXT_PATH)

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

    kernel = np.ones((3, 3), np.uint8)

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        roi_frame_bgr = frame_bgr[ry:ry + rh, rx:rx + rw]
        roi_bg_bgr = bg_bgr[ry:ry + rh, rx:rx + rw]

        # ORIGINAL BGR frame vs ORIGINAL BGR background
        diff_bgr = cv2.absdiff(roi_frame_bgr, roi_bg_bgr)

        # Collapse color difference to one mask image
        diff_scalar = np.max(diff_bgr, axis=2).astype(np.uint8)

        _, binary = cv2.threshold(diff_scalar, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        if MORPH_OPEN_ITER > 0:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITER)
        if MORPH_DILATE_ITER > 0:
            binary = cv2.dilate(binary, kernel, iterations=MORPH_DILATE_ITER)

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

            # Crop comes from ORIGINAL BGR frame.
            # BeeDetector itself requires 2D single-channel input.
            crop_for_detector = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

            result = detector.detect(crop_for_detector)
            best_det = choose_best_detection(result["detections"])

            # Binary preview drawings
            cv2.drawContours(binary_preview, [cnt], -1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(binary_preview, (cx_local, cy_local), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.rectangle(binary_preview, (sx0_l, sy0_l), (sx1_l, sy1_l), (255, 0, 0), 1, cv2.LINE_AA)

            # Original frame drawings
            cv2.rectangle(display, (sx0, sy0), (sx1, sy1), (255, 0, 0), 2, cv2.LINE_AA)

            cx_full = rx + cx_local
            cy_full = ry + cy_local
            cv2.circle(display, (cx_full, cy_full), 4, (0, 0, 255), -1, cv2.LINE_AA)

            if best_det is not None:
                marker_id = best_det["id"]
                angle_deg = best_det["angle_deg"]

                label = f"ID {marker_id} | {angle_deg:.1f} deg"
                text_x = sx0
                text_y = max(20, sy0 - 8)

                cv2.putText(
                    display,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                corners = best_det["marker_corners"].copy()
                corners[:, 0] += sx0
                corners[:, 1] += sy0
                pts = corners.astype(np.int32).reshape((-1, 1, 2))

                cv2.polylines(display, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)

                first_corner = tuple(corners[0].astype(int))
                center = tuple(np.mean(corners, axis=0).astype(int))

                cv2.circle(display, first_corner, 4, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.line(display, center, first_corner, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Binary ROI", binary_preview)
        cv2.imshow("Blob + ArUco Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()