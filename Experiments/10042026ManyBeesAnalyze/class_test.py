import csv
from pathlib import Path

import cv2

from bee_detector import BeeDetector


VIDEO_PATH = Path("2026-04-08 17-12-34.mkv")
DICT_PATH = Path("4x4_20.txt")

OUTPUT_VIDEO_PATH = None          # e.g. Path("detected_output.mp4")
OUTPUT_CSV_PATH = None

LABEL_MODE = "id_angle"           # "id", "id_rot", "full"
DRAW_REJECTED = True
SHOW_WINDOW = True

START_FRAME = 0
MAX_FRAMES = None                 # e.g. 500, or None for whole video
WAIT_KEY_MS = 1


def configure_detector(detector: BeeDetector) -> None:
    # Works with both the older and newer class API.
    if hasattr(detector, "set_source_settings"):
        detector.set_source_settings(detection_channel="red")
    elif hasattr(detector, "set_input_settings"):
        detector.set_input_settings(detection_channel="red")
    else:
        detector.detection_channel = "red"

    detector.set_decode_settings(max_hamming=0, max_border_errors=10)
    detector.set_candidate_settings(
        min_area=100,
        max_contour_area=500,
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


def run_detection(detector: BeeDetector, frame, draw=True):
    """
    Support both BeeDetector APIs:
    - process_frame(...)
    - detect_markers(...) + draw_detected_markers(...)
    """
    if hasattr(detector, "process_frame"):
        result = detector.process_frame(
            frame,
            draw=draw,
            draw_rejected_candidates=DRAW_REJECTED,
            label_mode=LABEL_MODE,
        )
        detections = result.get("detections", [])
        candidates = result.get("candidates", result.get("rejected_candidates", []))
        vis = result.get("visualization", frame.copy() if draw else None)
        return detections, candidates, vis

    if hasattr(detector, "detect_markers"):
        result = detector.detect_markers(frame)
        detections = result.get("detections", [])
        candidates = result.get("candidates", result.get("rejected_candidates", []))
        vis = None
        if draw and hasattr(detector, "draw_detected_markers"):
            vis = detector.draw_detected_markers(frame, result, label_mode=LABEL_MODE)
        elif draw:
            vis = frame.copy()
        return detections, candidates, vis

    raise RuntimeError("BeeDetector does not expose a supported detection method.")


def main() -> None:
    detector = BeeDetector(str(DICT_PATH))
    configure_detector(detector)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
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

    def add_info_block(img, lines):
        if hasattr(detector, "_draw_info_block"):
            detector._draw_info_block(img, lines)
            return

        y0 = 25
        for i, line in enumerate(lines):
            y = y0 + i * 24
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 1, cv2.LINE_AA)

    def handle_frame(frame, frame_index: int):
        nonlocal processed_count

        detections, candidates, vis = run_detection(detector, frame, draw=True)

        if vis is not None:
            info_lines = [
                f"Frame: {frame_index}",
                f"Candidates: {len(candidates)}",
                f"Decoded: {len(detections)}",
                "q: quit   space: pause",
            ]
            add_info_block(vis, info_lines)

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

        if writer is not None and vis is not None:
            writer.write(vis)

        if SHOW_WINDOW and vis is not None:
            cv2.imshow("BeeDetector Video", vis)

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