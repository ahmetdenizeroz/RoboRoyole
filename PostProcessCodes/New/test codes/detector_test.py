import cv2
import numpy as np

from bee_follower import BeeFollower
from bee_detector import BeeDetector
from kalman_filter import KalmanFilter


VIDEO_PATH = "test2/trimmed.mp4"
BACKGROUND_PATH = "test2/background.png"
DICT_PATH = "3x3_5.txt"

MIN_AREA = 100
MAX_AREA = 50000

# Kalman / tracking settings
MAX_MISSED_FRAMES = 10
PROCESS_NOISE = 1.0
MEASUREMENT_NOISE = 0.0

# Dilation settings
DILATE_KERNEL_SIZE = 5
DILATE_ITERATIONS = 1

# Playback / controls
PLAY_DELAY_MS = 30

# Candidate visualization
CANDIDATE_MATCH_ATOL = 1.5


def show_resized(window_name: str, image, width: int = 600, height: int = 400) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, image)


def select_initial_roi(frame):
    """
    Let the user draw a ROI once at the beginning.
    Returns (x, y, w, h).
    """
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select ROI", 1280, 720)
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        raise RuntimeError("No valid ROI was selected.")

    return x, y, w, h


def crop_to_roi(image, roi):
    x, y, w, h = roi
    return image[y:y + h, x:x + w].copy()


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


def detection_score(det):
    """
    Lower is better.
    Helps choose one best detection if the same ID appears more than once in one frame.
    """
    return (
        int(det.get("hamming", 10**9)),
        int(det.get("border_errors", 10**9)),
    )


def quads_match(quad_a, quad_b, atol=CANDIDATE_MATCH_ATOL) -> bool:
    qa = np.asarray(quad_a, dtype=np.float32).reshape(4, 2)
    qb = np.asarray(quad_b, dtype=np.float32).reshape(4, 2)
    return float(np.max(np.abs(qa - qb))) <= float(atol)


def analyze_blobs_with_detector(gray, blobs, detector):
    """
    Runs BeeDetector on each blob crop and returns:
      - best detection per marker ID across the whole frame
      - all candidate squares with accepted / failed status

    Output:
        best_by_id, candidate_statuses

    candidate_statuses item format:
    {
        "global_quad": ...,
        "accepted": bool,
        "id": int | None,
        "center": (cx, cy),
    }
    """
    best_by_id = {}
    candidate_statuses = []

    for blob in blobs:
        crop_gray, _, crop_origin = crop_and_threshold_otsu(gray, blob["bbox"])
        if crop_gray is None:
            continue

        det_result = detector.detect(crop_gray)
        detections = det_result["detections"]
        candidates = det_result["candidates"]

        for cand in candidates:
            cand_local = np.asarray(cand, dtype=np.float32)
            matched_det = None

            for det in detections:
                if quads_match(cand_local, det["quad"]):
                    matched_det = det
                    break

            global_quad = local_points_to_global(cand_local, crop_origin)
            center = global_quad.mean(axis=0)

            candidate_statuses.append(
                {
                    "global_quad": global_quad,
                    "accepted": matched_det is not None,
                    "id": None if matched_det is None else int(matched_det["id"]),
                    "center": (float(center[0]), float(center[1])),
                }
            )

        for det in detections:
            local_corners = np.asarray(det["marker_corners"], dtype=np.float32)
            global_corners = local_points_to_global(local_corners, crop_origin)
            center = global_corners.mean(axis=0)

            candidate = {
                "id": int(det["id"]),
                "angle_deg": float(det["angle_deg"]),
                "global_corners": global_corners,
                "center": (float(center[0]), float(center[1])),
                "hamming": int(det["hamming"]),
                "border_errors": int(det["border_errors"]),
            }

            marker_id = candidate["id"]
            if marker_id not in best_by_id or detection_score(candidate) < detection_score(best_by_id[marker_id]):
                best_by_id[marker_id] = candidate

    return best_by_id, candidate_statuses


def draw_measured_marker(frame, det):
    """
    Draw measured marker info from BeeDetector result.
    """
    corners = np.asarray(det["global_corners"], dtype=np.float32)
    center = corners.mean(axis=0)
    cx, cy = int(round(center[0])), int(round(center[1]))

    top_mid = 0.5 * (corners[0] + corners[1])
    tx, ty = int(round(top_mid[0])), int(round(top_mid[1]))

    marker_id = det["id"]
    angle_deg = det["angle_deg"]

    poly = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [poly], True, (0, 255, 255), 2)

    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 0), 2)

    text1 = f"ID={marker_id} Ang={angle_deg:.1f}"
    text2 = f"Pos=({cx},{cy})"

    text_x = cx + 8
    text_y = max(20, cy - 10)

    cv2.putText(frame, text1, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text2, (text_x, text_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)


def draw_predicted_marker(frame, marker_id, pred_pos, last_angle_deg, missed_frames):
    """
    Draw predicted marker position when current-frame detection is missing.
    """
    px, py = int(round(pred_pos[0])), int(round(pred_pos[1]))

    cv2.circle(frame, (px, py), 6, (255, 0, 255), 2)
    cv2.putText(
        frame,
        f"PRED ID={marker_id} Ang={last_angle_deg:.1f} Miss={missed_frames}",
        (px + 8, max(20, py - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_candidate_status(frame, candidate_statuses):
    """
    Draw all candidate squares:
      - green = accepted by BeeDetector
      - red   = square candidate found but decode failed / rejected
    """
    vis = frame.copy()

    failed_count = 0
    accepted_count = 0

    for item in candidate_statuses:
        quad = np.asarray(item["global_quad"], dtype=np.float32)
        poly = quad.astype(np.int32).reshape((-1, 1, 2))
        center = quad.mean(axis=0)
        cx, cy = int(round(center[0])), int(round(center[1]))

        if item["accepted"]:
            accepted_count += 1
            color = (0, 255, 0)
            label = f"OK {item['id']}"
            thickness = 2
        else:
            failed_count += 1
            color = (0, 0, 255)
            label = "MISS"
            thickness = 2

        cv2.polylines(vis, [poly], True, color, thickness)
        cv2.circle(vis, (cx, cy), 3, color, -1)
        cv2.putText(
            vis,
            label,
            (cx + 6, max(18, cy - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        f"Accepted squares: {accepted_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"Failed squares: {failed_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return vis, accepted_count, failed_count


def main():
    bg = cv2.imread(BACKGROUND_PATH, cv2.IMREAD_COLOR)
    if bg is None:
        raise RuntimeError(f"Could not load background image: {BACKGROUND_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        raise RuntimeError("Could not read the first frame from the video.")

    frame_h, frame_w = first_frame.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    if (bg_h, bg_w) != (frame_h, frame_w):
        raise RuntimeError(
            f"Background size {bg_w}x{bg_h} does not match video frame size {frame_w}x{frame_h}."
        )

    roi = select_initial_roi(first_frame)
    roi_x, roi_y, roi_w, roi_h = roi
    print(f"Using ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

    # Restart the video so processing starts again from frame 0.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    bg_roi = crop_to_roi(bg, roi)
    bg_gray = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2GRAY)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    dt = 1.0 / fps

    follower = BeeFollower(
        min_area=MIN_AREA,
        max_area=MAX_AREA,
    )

    detector = BeeDetector(DICT_PATH)
    # None => show internal reject debug for every rejected decode candidate.
    detector.set_debug(show_debug=True, debug_marker_id=None)
    detector.set_candidate_settings(
        remove_small_group_area=60,
        remove_large_group_area=800,
    )

    # One Kalman filter per marker ID
    tracks = {}

    paused = False
    current_frame = None
    last_rendered = None
    last_candidate_view = None
    last_diff = None
    last_binary = None
    last_binary_dilated = None

    if DILATE_KERNEL_SIZE > 1:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE)
        )
    else:
        dilate_kernel = None

    while True:
        advanced = False

        if not paused or current_frame is None:
            ok, frame_full = cap.read()
            if not ok or frame_full is None:
                break

            frame = crop_to_roi(frame_full, roi)
            current_frame = frame.copy()
            advanced = True
        else:
            frame = current_frame.copy()

        if advanced:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, bg_gray)

            _, binary = cv2.threshold(
                diff,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            if dilate_kernel is not None:
                binary_dilated = cv2.dilate(binary, dilate_kernel, iterations=DILATE_ITERATIONS)
            else:
                binary_dilated = binary.copy()

            blobs = follower.detect(binary_dilated)
            detections_by_id, candidate_statuses = analyze_blobs_with_detector(gray, blobs, detector)

            # Predict all existing tracks once for this new frame
            for marker_id, track in list(tracks.items()):
                px, py = track["kf"].predict()
                track["pred_pos"] = (px, py)
                track["missed"] += 1
                track["updated_this_frame"] = False

            # Update tracks with real detections
            for marker_id, det in detections_by_id.items():
                mx, my = det["center"]

                if marker_id not in tracks:
                    kf = KalmanFilter(
                        dt=dt,
                        process_noise=PROCESS_NOISE,
                        measurement_noise=MEASUREMENT_NOISE,
                    )
                    cx, cy = kf.update(mx, my)
                    tracks[marker_id] = {
                        "kf": kf,
                        "pred_pos": (cx, cy),
                        "missed": 0,
                        "updated_this_frame": True,
                        "last_angle_deg": det["angle_deg"],
                    }
                else:
                    cx, cy = tracks[marker_id]["kf"].update(mx, my)
                    tracks[marker_id]["pred_pos"] = (cx, cy)
                    tracks[marker_id]["missed"] = 0
                    tracks[marker_id]["updated_this_frame"] = True
                    tracks[marker_id]["last_angle_deg"] = det["angle_deg"]

            # Remove stale tracks
            stale_ids = [marker_id for marker_id, track in tracks.items()
                         if track["missed"] > MAX_MISSED_FRAMES]
            for marker_id in stale_ids:
                del tracks[marker_id]

            draw_frame = frame.copy()

            # Draw failed candidate squares first so successful detections stay visible on top.
            for item in candidate_statuses:
                if item["accepted"]:
                    continue
                quad = np.asarray(item["global_quad"], dtype=np.float32)
                poly = quad.astype(np.int32).reshape((-1, 1, 2))
                center = quad.mean(axis=0)
                cx, cy = int(round(center[0])), int(round(center[1]))
                cv2.polylines(draw_frame, [poly], True, (0, 0, 255), 2)
                cv2.putText(
                    draw_frame,
                    "MISS",
                    (cx + 6, max(18, cy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Draw measured detections on top
            for det in detections_by_id.values():
                draw_measured_marker(draw_frame, det)

            # Draw predicted positions for IDs that were not detected this frame
            for marker_id, track in tracks.items():
                if not track["updated_this_frame"] and track["missed"] <= MAX_MISSED_FRAMES:
                    draw_predicted_marker(
                        draw_frame,
                        marker_id=marker_id,
                        pred_pos=track["pred_pos"],
                        last_angle_deg=track["last_angle_deg"],
                        missed_frames=track["missed"],
                    )

            candidate_view, accepted_count, failed_count = draw_candidate_status(frame, candidate_statuses)

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
            cv2.putText(
                draw_frame,
                f"Active tracks: {len(tracks)}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                draw_frame,
                f"Candidate OK: {accepted_count}  MISS: {failed_count}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                draw_frame,
                f"ROI: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                draw_frame,
                "Multiple markers per blob: enabled",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 255, 180),
                2,
                cv2.LINE_AA,
            )

            last_rendered = draw_frame.copy()
            last_candidate_view = candidate_view.copy()
            last_diff = diff.copy()
            last_binary = binary.copy()
            last_binary_dilated = binary_dilated.copy()

        else:
            draw_frame = last_rendered.copy() if last_rendered is not None else frame.copy()
            candidate_view = last_candidate_view.copy() if last_candidate_view is not None else frame.copy()

        if paused:
            cv2.putText(
                draw_frame,
                "PAUSED",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                draw_frame,
                "Space: resume   Q/ESC: quit",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                draw_frame,
                "Space: pause   Q/ESC: quit",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        show_resized("Video Feed (ROI only)", draw_frame, 1280, 720)
        show_resized("Candidate Status (ROI only)", candidate_view, 1280, 720)

        if last_diff is not None:
            show_resized("Diff (ROI only)", last_diff, 600, 400)
        if last_binary is not None:
            show_resized("Binary (ROI only)", last_binary, 600, 400)
        if last_binary_dilated is not None:
            show_resized("Binary Dilated (ROI only)", last_binary_dilated, 600, 400)

        wait_ms = 0 if paused else PLAY_DELAY_MS
        key = cv2.waitKeyEx(wait_ms)

        if key in (27, ord("q"), ord("Q")):
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
