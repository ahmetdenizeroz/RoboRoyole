import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog

from bee_follower import BeeFollower
from quad_debug import BeeQuadDetector


VIDEO_PATH = "test2/trimmed.mp4"
BACKGROUND_PATH = "test2/background.png"

MIN_AREA = 2000
MAX_AREA = 50000
PLAY_DELAY_MS = 30


def show_resized(window_name: str, image, width: int = 600, height: int = 400) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, image)


def read_frame_at_index(cap, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def prompt_frame_number(current_frame_idx: int, total_frames: int):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        frame_idx = simpledialog.askinteger(
            "Jump to frame",
            f"Enter frame number (0 - {total_frames - 1}):",
            initialvalue=current_frame_idx,
            minvalue=0,
            maxvalue=total_frames - 1,
            parent=root,
        )
    finally:
        root.destroy()

    return frame_idx


def select_processing_roi(frame):
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise RuntimeError("No valid ROI selected.")

    return x, y, w, h


def apply_roi(image, roi):
    x, y, w, h = roi
    return image[y:y + h, x:x + w]


def crop_and_threshold_otsu(gray_frame, bbox):
    """
    Returns:
        crop_gray
        crop_otsu
        crop_origin = (x_clipped, y_clipped)
    """
    x, y, w, h = bbox
    H, W = gray_frame.shape[:2]

    x = max(0, x)
    y = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    if x2 <= x or y2 <= y:
        return None, None, None

    crop_gray = gray_frame[y:y2, x:x2]
    _, crop_otsu = cv2.threshold(
        crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return crop_gray, crop_otsu, (x, y)


def local_points_to_global(points_local, crop_origin):
    ox, oy = crop_origin
    pts = np.asarray(points_local, dtype=np.float32).copy()
    pts[:, 0] += ox
    pts[:, 1] += oy
    return pts


def draw_blob_bbox(frame, blob):
    x, y, w, h = blob["bbox"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)


def draw_quad_debug(frame, global_quad, label=None):
    q = np.asarray(global_quad, dtype=np.float32)
    poly = q.astype(np.int32).reshape((-1, 1, 2))
    center = np.mean(q, axis=0)
    cx, cy = int(round(center[0])), int(round(center[1]))

    cv2.polylines(frame, [poly], True, (0, 255, 255), 2)

    for j, pt in enumerate(q):
        px, py = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            str(j),
            (px + 4, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    if label is not None:
        cv2.putText(
            frame,
            label,
            (cx + 8, max(20, cy - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def process_frame(frame_roi, bg_gray_roi, follower, quad_detector):
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, bg_gray_roi)

    _, binary = cv2.threshold(
        diff,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    blobs = follower.detect(binary_closed)

    draw_frame = frame_roi.copy()
    crop_window_payloads = []

    quad_fail = False
    fail_messages = []

    for i, blob in enumerate(blobs):
        draw_blob_bbox(draw_frame, blob)

        crop_gray, crop_otsu, crop_origin = crop_and_threshold_otsu(gray, blob["bbox"])
        if crop_gray is None:
            continue

        quad_result = quad_detector.detect(crop_gray)
        quads = quad_result["quads"]

        gray_name = f"Crop Gray {i}"
        otsu_name = f"Crop Otsu {i}"
        crop_window_payloads.append((gray_name, crop_gray, otsu_name, crop_otsu))

        if len(quads) == 0:
            quad_fail = True
            fail_messages.append(f"Blob {i}: no quad")

        for q_idx, local_quad in enumerate(quads):
            global_quad = local_points_to_global(local_quad, crop_origin)
            draw_quad_debug(draw_frame, global_quad, label=f"B{i} Q{q_idx}")

    cv2.putText(
        draw_frame,
        f"Blob count: {len(blobs)}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return draw_frame, diff, binary, binary_closed, crop_window_payloads, quad_fail, fail_messages


def is_right_key(key: int) -> bool:
    return key in (2555904, 65363, ord('d'), ord('D'))


def is_left_key(key: int) -> bool:
    return key in (2424832, 65361, ord('a'), ord('A'))


def main():
    bg = cv2.imread(BACKGROUND_PATH, cv2.IMREAD_COLOR)
    if bg is None:
        raise RuntimeError(f"Could not load background image: {BACKGROUND_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError("Could not read total frame count from video.")

    first_frame = read_frame_at_index(cap, 0)
    if first_frame is None:
        raise RuntimeError("Could not read the first frame for ROI selection.")

    if bg.shape[:2] != first_frame.shape[:2]:
        raise RuntimeError(
            "Background image size does not match the video frame size. "
            "ROI-based cropping requires matching dimensions."
        )

    roi = select_processing_roi(first_frame)
    bg_roi = apply_roi(bg, roi)
    bg_gray_roi = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)

    follower = BeeFollower(
        min_area=MIN_AREA,
        max_area=MAX_AREA,
    )

    quad_detector = BeeQuadDetector()
    quad_detector.set_debug(True)
    quad_detector.set_candidate_settings(
        remove_small_group_area=60,
        remove_large_group_area=800,
    )

    current_frame_idx = 0
    previous_crop_window_names = set()
    paused = False
    auto_paused_due_to_fail = False

    while True:
        frame_full = read_frame_at_index(cap, current_frame_idx)
        if frame_full is None:
            break

        frame_roi = apply_roi(frame_full, roi)

        draw_frame, diff, binary, binary_closed, crop_window_payloads, quad_fail, fail_messages = process_frame(
            frame_roi, bg_gray_roi, follower, quad_detector
        )

        current_crop_window_names = set()
        for gray_name, crop_gray, otsu_name, crop_otsu in crop_window_payloads:
            current_crop_window_names.add(gray_name)
            current_crop_window_names.add(otsu_name)
            show_resized(gray_name, crop_gray, 400, 300)
            show_resized(otsu_name, crop_otsu, 400, 300)

        for old_name in previous_crop_window_names - current_crop_window_names:
            try:
                cv2.destroyWindow(old_name)
            except cv2.error:
                pass
        previous_crop_window_names = current_crop_window_names

        if quad_fail and not paused:
            paused = True
            auto_paused_due_to_fail = True

        cv2.putText(
            draw_frame,
            f"Frame: {current_frame_idx}/{total_frames - 1}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if paused:
            pause_text = "PAUSED"
            if auto_paused_due_to_fail:
                pause_text = "PAUSED - QUAD FAIL"

            cv2.putText(
                draw_frame,
                pause_text,
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                draw_frame,
                "Space: play/pause  Right: next  Left: previous  U: jump  Q/ESC: quit",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                draw_frame,
                "Space: pause  Q/ESC: quit",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for idx, msg in enumerate(fail_messages[:5]):
            cv2.putText(
                draw_frame,
                msg,
                (20, 170 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        show_resized("Video Feed", draw_frame, 1280, 720)
        show_resized("Diff", diff, 600, 400)
        show_resized("Binary", binary, 600, 400)
        show_resized("Binary Closed", binary_closed, 600, 400)

        wait_ms = 0 if paused else PLAY_DELAY_MS
        key = cv2.waitKeyEx(wait_ms)

        if key in (27, ord('q'), ord('Q')):
            break

        elif key == ord(' '):
            paused = not paused
            if not paused:
                auto_paused_due_to_fail = False
                current_frame_idx = min(current_frame_idx + 1, total_frames - 1)

        elif paused and key in (ord('u'), ord('U')):
            jump_frame_idx = prompt_frame_number(current_frame_idx, total_frames)
            if jump_frame_idx is not None:
                current_frame_idx = jump_frame_idx
                paused = True
                auto_paused_due_to_fail = False

        elif paused and is_right_key(key):
            auto_paused_due_to_fail = False
            current_frame_idx = min(current_frame_idx + 1, total_frames - 1)

        elif paused and is_left_key(key):
            auto_paused_due_to_fail = False
            current_frame_idx = max(current_frame_idx - 1, 0)

        elif not paused:
            current_frame_idx += 1
            if current_frame_idx >= total_frames:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
