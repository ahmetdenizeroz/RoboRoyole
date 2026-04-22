import csv
from pathlib import Path

import cv2
import numpy as np

from bee_detector import BeeDetector


VIDEO_PATH = Path("test.mkv")
DICT_PATH = Path("3x3_10.txt")

OUTPUT_VIDEO_PATH = None          # e.g. Path("detected_output.mp4")
OUTPUT_CSV_PATH = None

INPUT_MODE = "red"                # "red", "green", "blue", "gray"
LABEL_MODE = "id_angle"           # "id", "id_rot", "full", "id_angle"
DRAW_REJECTED = True
SHOW_WINDOW = True

START_FRAME = 0
MAX_FRAMES = None                 # e.g. 500, or None for whole video
WAIT_KEY_MS = 1


def configure_detector(detector: BeeDetector) -> None:
    detector.set_decode_settings(max_hamming=0, max_border_errors=10)
    detector.set_candidate_settings(
        min_area=100,
        max_contour_area=1000,
        min_solidity=0.6,
        poly_eps_ratio=0.05,
        min_side_pixels=6,
        max_side_ratio=1.8,
        dedup_center_thresh=12.0,
    )
    detector.set_sampling_settings(
        cell_pixels=24,
        quad_expand_scale=1.18,
        sample_inset_ratio=0.20,
    )
    detector.set_debug(show_debug=False)


def prepare_input_frame(frame_bgr: np.ndarray, mode: str = "red") -> np.ndarray:
    mode = mode.lower()

    if frame_bgr is None:
        raise ValueError("Input frame is None.")

    if frame_bgr.ndim == 2:
        return frame_bgr

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image or single-channel image.")

    if mode == "red":
        return frame_bgr[:, :, 2]
    if mode == "green":
        return frame_bgr[:, :, 1]
    if mode == "blue":
        return frame_bgr[:, :, 0]
    if mode == "gray":
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported INPUT_MODE: {mode}")


def build_label_lines(detection: dict, label_mode: str) -> list[str]:
    mode = label_mode.lower()

    if mode == "id":
        return [f"ID {detection['id']}"]

    if mode == "id_rot":
        return [f"ID {detection['id']} | rot {detection['rotation_cw_deg']} CW"]

    if mode == "full":
        return [
            f"ID {detection['id']} | rot {detection['rotation_cw_deg']} CW",
            f"angle {detection['angle_deg']:.1f} | raw {detection.get('raw_angle_deg', detection['angle_deg']):.1f}",
            f"ham {detection['hamming']} | be {detection['border_errors']}",
        ]

    if mode == "id_angle":
        return [f"ID {detection['id']} | angle {detection['angle_deg']:.1f}"]

    return [f"ID {detection['id']}"]


def draw_label(
    img: np.ndarray,
    quad: np.ndarray,
    lines: list[str],
    color=(0, 255, 0),
) -> None:
    q = np.asarray(quad, dtype=np.int32).reshape(4, 1, 2)
    cv2.polylines(img, [q], True, color, 2, cv2.LINE_AA)

    center = np.mean(q.reshape(4, 2), axis=0).astype(int)
    x = int(center[0]) + 8
    y = int(center[1]) - 8

    for i, line in enumerate(lines):
        yy = y + i * 20
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_info_block(img: np.ndarray, lines: list[str]) -> None:
    y0 = 25
    for i, line in enumerate(lines):
        y = y0 + i * 24
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def make_visualization(
    original_frame: np.ndarray,
    detections: list[dict],
    candidates: list[np.ndarray],
    label_mode: str,
    draw_rejected: bool,
) -> np.ndarray:
    if original_frame.ndim == 2:
        vis = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = original_frame.copy()

    for det in detections:
        lines = build_label_lines(det, label_mode)
        draw_label(vis, det["quad"], lines, color=(0, 255, 0))

    if draw_rejected:
        decoded_centers = [np.mean(det["quad"], axis=0) for det in detections]
        for q in candidates:
            c = np.mean(q, axis=0)
            matched = any(np.linalg.norm(c - dc) < 8 for dc in decoded_centers)
            if not matched:
                qq = np.asarray(q, dtype=np.int32).reshape(4, 1, 2)
                cv2.polylines(vis, [qq], True, (0, 165, 255), 1, cv2.LINE_AA)

    return vis


def run_detection(detector: BeeDetector, frame_bgr: np.ndarray):
    prepared = prepare_input_frame(frame_bgr, INPUT_MODE)
    result = detector.detect(prepared)

    detections = result.get("detections", [])
    candidates = result.get("candidates", [])

    vis = make_visualization(
        original_frame=frame_bgr,
        detections=detections,
        candidates=candidates,
        label_mode=LABEL_MODE,
        draw_rejected=DRAW_REJECTED,
    )

    return detections, candidates, vis, prepared


def main() -> None:
    detector = BeeDetector(str(DICT_PATH))
    configure_detector(detector)

    #"cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    if START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError("Could not read first frame from video.")

    frame_h, frame_w = first_frame.shape[:2]

    writer = None
    if OUTPUT_VIDEO_PATH is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (frame_w, frame_h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open output video writer: {OUTPUT_VIDEO_PATH}")

    csv_file = None
    csv_writer = None
    if OUTPUT_CSV_PATH is not None:
        csv_file = open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame_index",
            "id",
            "rotation_cw_deg",
            "angle_deg",
            "hamming",
            "border_errors",
            "center_x",
            "center_y",
        ])

    current_frame_index = START_FRAME
    processed_count = 0
    paused = False

    def handle_frame(frame, frame_index: int):
        nonlocal processed_count

        detections, candidates, vis, prepared = run_detection(detector, frame)

        info_lines = [
            f"Frame: {frame_index}",
            f"Input mode: {INPUT_MODE}",
            f"Candidates: {len(candidates)}",
            f"Decoded: {len(detections)}",
            "q: quit   space: pause",
        ]
        draw_info_block(vis, info_lines)

        if csv_writer is not None:
            for det in detections:
                quad = det.get("quad")
                if quad is not None:
                    center = quad.mean(axis=0)
                    cx, cy = round(float(center[0]), 3), round(float(center[1]), 3)
                else:
                    cx, cy = "", ""

                csv_writer.writerow([
                    frame_index,
                    det.get("id", ""),
                    det.get("rotation_cw_deg", ""),
                    round(float(det.get("angle_deg", 0.0)), 3) if "angle_deg" in det else "",
                    det.get("hamming", ""),
                    det.get("border_errors", ""),
                    cx,
                    cy,
                ])

        if writer is not None:
            writer.write(vis)

        if SHOW_WINDOW:
            cv2.imshow("BeeDetector Video", vis)
            # Optional debug view of the prepared input:
            # cv2.imshow("Prepared Input", prepared)

        processed_count += 1

    handle_frame(first_frame, current_frame_index)

    while True:
        key = cv2.waitKey(0 if paused else WAIT_KEY_MS) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
            continue

        if MAX_FRAMES is not None and processed_count >= MAX_FRAMES:
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        current_frame_index += 1
        handle_frame(frame, current_frame_index)

    cap.release()
    if writer is not None:
        writer.release()
    if csv_file is not None:
        csv_file.close()
    cv2.destroyAllWindows()

    print("Done.")
    print(f"Processed frames: {processed_count}")
    if total_frames > 0:
        print(f"Video total frames: {total_frames}")
    if OUTPUT_VIDEO_PATH is not None:
        print(f"Saved annotated video to: {OUTPUT_VIDEO_PATH}")
    if OUTPUT_CSV_PATH is not None:
        print(f"Saved detections CSV to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()