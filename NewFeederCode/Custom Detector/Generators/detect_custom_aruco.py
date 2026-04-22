import cv2
import numpy as np
import re
import sys
import os
import math


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
                    raise ValueError(f"Marker ID {marker_id} row has wrong length.")
                row = [int(x) for x in row_vals]
                if any(x not in (0, 1) for x in row):
                    raise ValueError(f"Marker ID {marker_id} contains non-binary values.")
                bits.append(row)

            markers[marker_id] = np.array(bits, dtype=np.uint8)
            i += marker_size + 1
        else:
            i += 1

    if not markers:
        raise ValueError("No marker bit patterns found in txt file.")

    # Reorder by ID
    ordered_ids = sorted(markers.keys())
    ordered_markers = [markers[mid] for mid in ordered_ids]

    if num_markers is not None and len(ordered_markers) != num_markers:
        print(
            f"Warning: txt says num_markers={num_markers}, "
            f"but parsed {len(ordered_markers)} marker patterns."
        )

    return marker_size, ordered_ids, ordered_markers


def get_byte_list_from_bits(bits):
    """
    Compatibility helper for different OpenCV versions.
    """
    if hasattr(cv2.aruco.Dictionary, "getByteListFromBits"):
        return cv2.aruco.Dictionary.getByteListFromBits(bits)
    return cv2.aruco.Dictionary_getByteListFromBits(bits)


def build_custom_dictionary(marker_size, marker_bits_list, maxcorr=1):
    rows = [get_byte_list_from_bits(bits) for bits in marker_bits_list]
    bytes_list = np.concatenate(rows, axis=0)
    return cv2.aruco.Dictionary(bytes_list, marker_size, maxcorr)


def create_detector(dictionary):
    """
    Compatibility helper for OpenCV versions.
    """
    params = cv2.aruco.DetectorParameters()

    # These can help for tiny 3x3 markers if needed
    params.minMarkerPerimeterRate = 0.01
    params.polygonalApproxAccuracyRate = 0.05

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params), None
    else:
        return None, params


def detect_markers(image, dictionary):
    detector, params = create_detector(dictionary)

    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            image, dictionary, parameters=params
        )

    return corners, ids, rejected


def compute_2d_angle_deg(marker_corners):
    """
    marker_corners shape: (1,4,2) or (4,2)
    OpenCV corner order:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = marker_corners.reshape(4, 2)
    tl, tr, br, bl = pts

    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]

    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360.0
    return angle


def draw_results(image, corners, ids):
    out = image.copy()

    if ids is None or len(ids) == 0:
        return out

    cv2.aruco.drawDetectedMarkers(out, corners, ids)

    for i, marker_id in enumerate(ids.flatten()):
        pts = corners[i].reshape(4, 2).astype(np.int32)
        center = np.mean(pts, axis=0).astype(int)

        angle_deg = compute_2d_angle_deg(corners[i])

        # Orientation arrow: top-left -> top-right
        tl = tuple(pts[0])
        tr = tuple(pts[1])
        cv2.arrowedLine(out, tl, tr, (255, 0, 0), 2, tipLength=0.2)

        text1 = f"ID {marker_id}"
        text2 = f"{angle_deg:.1f} deg"

        cv2.putText(
            out, text1,
            (center[0] - 30, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.putText(
            out, text2,
            (center[0] - 40, center[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 200, 255), 2, cv2.LINE_AA
        )

    return out


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python detect_custom_aruco.py selected_markers.txt input.png")
        sys.exit(1)

    txt_path = sys.argv[1]
    image_path = sys.argv[2]

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"TXT file not found: {txt_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    marker_size, ordered_ids, marker_bits = parse_marker_txt(txt_path)

    print("Parsed markers:")
    for i, bits in zip(ordered_ids, marker_bits):
        print(f"\nID {i}:")
        print(bits)

    # best_min_rotational_hamming_distance = 4 -> maxcorr = 1 is reasonable
    custom_dict = build_custom_dictionary(marker_size, marker_bits, maxcorr=1)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    corners, ids, rejected = detect_markers(image, custom_dict)

    if ids is None or len(ids) == 0:
        print("No markers detected.")
        annotated = image.copy()
    else:
        print("\nDetected markers:")
        for i, marker_id in enumerate(ids.flatten()):
            angle_deg = compute_2d_angle_deg(corners[i])
            print(f"  ID {marker_id}, image angle = {angle_deg:.1f} deg")

        annotated = draw_results(image, corners, ids)

    out_path = "detected_output.png"
    cv2.imwrite(out_path, annotated)
    print(f"\nSaved annotated result to: {out_path}")

    # Optional display
    cv2.imshow("Detection Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()