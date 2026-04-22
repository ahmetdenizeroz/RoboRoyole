import cv2
import numpy as np
import re
import math
from typing import Any, Dict, List, Optional, Tuple

class BeeDetector:
    """
    Custom square-marker detector for bee experiments.

    Responsibilities of this class:
    - Load a custom marker dictionary from a TXT file
    - Detect square candidates in a single-channel input image
    - Warp candidates to a canonical square
    - Sample marker bits
    - Decode against all markers and all 90-degree rotations

    Debug behavior:
    - If self.show_debug is True, detect() returns a "debug" field
      containing intermediate frames and candidate decode info.
    """

    # -----------------------------------------------------
    # Construction / configuration
    # -----------------------------------------------------
    def __init__(self, txt_path: str) -> None:
        # Detection settings
        self.max_hamming = 0
        self.max_border_errors = 10

        self.min_area = 100.0
        self.max_contour_area = 500.0
        self.min_solidity = 0.5
        self.poly_eps_ratio = 0.05
        self.min_side_pixels = 6.0
        self.max_side_ratio = 1.8
        self.dedup_center_thresh = 12.0

        # Warp / sampling settings
        self.cell_pixels = 24
        self.quad_expand_scale = 1.18
        self.sample_inset_ratio = 0.20

        # Debug settings
        self.show_debug = False
        self.debug_marker_id: Optional[int] = None

        # Dictionary / metadata state
        self.txt_path = txt_path
        self.marker_size: Optional[int] = None
        self.num_markers: Optional[int] = None
        self.border_bits: Optional[int] = None
        self.border_color: Optional[str] = None
        self.bit_to_color: Dict[int, str] = {}
        self.white_bit_value: Optional[int] = None
        self.black_bit_value: Optional[int] = None
        self.expected_border_bit: Optional[int] = None

        self.total_cells: Optional[int] = None
        self.side_pixels: Optional[int] = None

        self.custom_markers: Dict[int, np.ndarray] = {}

        # Cached fast structures
        self._dst_quad: Optional[np.ndarray] = None
        self._sample_x0: Optional[np.ndarray] = None
        self._sample_x1: Optional[np.ndarray] = None
        self._sample_y0: Optional[np.ndarray] = None
        self._sample_y1: Optional[np.ndarray] = None
        self._sample_areas: Optional[np.ndarray] = None
        self._border_mask: Optional[np.ndarray] = None
        self._inner_slice = None

        self._template_ids: Optional[np.ndarray] = None
        self._template_rotations: Optional[np.ndarray] = None
        self._template_payloads_flat: Optional[np.ndarray] = None

        self.load_dictionary(txt_path)

    # -----------------------------------------------------
    # Public configuration methods
    # -----------------------------------------------------
    def set_decode_settings(
        self,
        *,
        max_hamming: Optional[int] = None,
        max_border_errors: Optional[int] = None,
    ) -> None:
        if max_hamming is not None:
            self.max_hamming = int(max_hamming)
        if max_border_errors is not None:
            self.max_border_errors = int(max_border_errors)

    def set_candidate_settings(
        self,
        *,
        min_area: Optional[float] = None,
        max_contour_area: Optional[float] = None,
        min_solidity: Optional[float] = None,
        poly_eps_ratio: Optional[float] = None,
        min_side_pixels: Optional[float] = None,
        max_side_ratio: Optional[float] = None,
        dedup_center_thresh: Optional[float] = None,
    ) -> None:
        if min_area is not None:
            self.min_area = float(min_area)
        if max_contour_area is not None:
            self.max_contour_area = float(max_contour_area)
        if min_solidity is not None:
            self.min_solidity = float(min_solidity)
        if poly_eps_ratio is not None:
            self.poly_eps_ratio = float(poly_eps_ratio)
        if min_side_pixels is not None:
            self.min_side_pixels = float(min_side_pixels)
        if max_side_ratio is not None:
            self.max_side_ratio = float(max_side_ratio)
        if dedup_center_thresh is not None:
            self.dedup_center_thresh = float(dedup_center_thresh)

    def set_sampling_settings(
        self,
        *,
        cell_pixels: Optional[int] = None,
        quad_expand_scale: Optional[float] = None,
        sample_inset_ratio: Optional[float] = None,
    ) -> None:
        rebuild = False

        if cell_pixels is not None and int(cell_pixels) != self.cell_pixels:
            self.cell_pixels = int(cell_pixels)
            rebuild = True

        if sample_inset_ratio is not None and float(sample_inset_ratio) != self.sample_inset_ratio:
            self.sample_inset_ratio = float(sample_inset_ratio)
            rebuild = True

        if quad_expand_scale is not None:
            self.quad_expand_scale = float(quad_expand_scale)

        if rebuild:
            self._rebuild_geometry_cache()

    def set_debug(
        self,
        *,
        show_debug: Optional[bool] = None,
        debug_marker_id: Optional[int] = None,
    ) -> None:
        if show_debug is not None:
            self.show_debug = bool(show_debug)
        if debug_marker_id is not None:
            self.debug_marker_id = int(debug_marker_id)

    def get_settings(self) -> Dict[str, Any]:
        return {
            "txt_path": self.txt_path,
            "marker_size": self.marker_size,
            "num_markers": self.num_markers,
            "border_bits": self.border_bits,
            "bit_to_color": dict(self.bit_to_color),
            "border_color": self.border_color,
            "expected_border_bit": self.expected_border_bit,
            "cell_pixels": self.cell_pixels,
            "quad_expand_scale": self.quad_expand_scale,
            "sample_inset_ratio": self.sample_inset_ratio,
            "max_hamming": self.max_hamming,
            "max_border_errors": self.max_border_errors,
            "min_area": self.min_area,
            "max_contour_area": self.max_contour_area,
            "min_solidity": self.min_solidity,
            "poly_eps_ratio": self.poly_eps_ratio,
            "min_side_pixels": self.min_side_pixels,
            "max_side_ratio": self.max_side_ratio,
            "dedup_center_thresh": self.dedup_center_thresh,
            "show_debug": self.show_debug,
            "debug_marker_id": self.debug_marker_id,
        }

    # -----------------------------------------------------
    # Dictionary loading / metadata parsing
    # -----------------------------------------------------
    def load_dictionary(self, txt_path: Optional[str] = None) -> None:
        if txt_path is not None:
            self.txt_path = txt_path

        metadata, markers = self._parse_dictionary_file(self.txt_path)

        self.marker_size = metadata["marker_size"]
        self.num_markers = metadata["num_markers"]
        self.border_bits = metadata["border_bits"]
        self.bit_to_color = metadata["bit_to_color"]
        self.border_color = metadata["border_color"]

        self.white_bit_value = next(bit for bit, color in self.bit_to_color.items() if color == "white")
        self.black_bit_value = next(bit for bit, color in self.bit_to_color.items() if color == "black")
        self.expected_border_bit = (
            self.white_bit_value if self.border_color == "white" else self.black_bit_value
        )

        self.custom_markers = markers
        self.total_cells = self.marker_size + 2 * self.border_bits

        self._rebuild_geometry_cache()
        self._precompute_rotated_templates()

    def compute_min_rotational_distance(self) -> int:
        ids = sorted(self.custom_markers.keys())
        if len(ids) < 2:
            return 0

        d_min = 10**9
        for i in range(len(ids)):
            p1 = self.custom_markers[ids[i]]
            for j in range(i + 1, len(ids)):
                p2 = self.custom_markers[ids[j]]
                best = 10**9
                for rot_cw in (0, 90, 180, 270):
                    expected = self._rotate_pattern_cw(p2, rot_cw)
                    dist = int(np.count_nonzero(p1 != expected))
                    if dist < best:
                        best = dist
                d_min = min(d_min, best)
        return int(d_min)

    def recommend_max_hamming(self) -> int:
        d_min = self.compute_min_rotational_distance()
        return max(0, (d_min - 1) // 2)

    # -----------------------------------------------------
    # Main API
    # -----------------------------------------------------
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect markers in a preprocessed single-channel frame.

        Expected input:
        - 2D numpy array
        - already prepared by caller
          (for example grayscale, red channel, cropped image, etc.)

        Returns:
        {
            "detections": [...],
            "candidates": [...],
            "debug": { ... }   # only if self.show_debug is True
        }
        """
        if frame is None:
            raise ValueError("Input frame is None.")

        if frame.ndim != 2:
            raise ValueError(
                "BeeDetector expects a single-channel 2D image. "
                "Prepare the input outside the detector."
            )

        work_img = frame
        quads, bw_inv = self._find_square_candidates(work_img)

        detections: List[Dict[str, Any]] = []

        debug_data = None
        if self.show_debug:
            debug_data = {
                "bw_inv": bw_inv.copy(),
                "candidates_warped": [],
                "candidates_bw": [],
                "candidate_results": [],
            }

        for quad in quads:
            warped = self._warp_marker(work_img, quad)
            decode = self._decode_candidate(warped)

            if self.show_debug:
                include_this_candidate = (
                    self.debug_marker_id is None or decode["id"] == self.debug_marker_id
                )

                if include_this_candidate:
                    debug_data["candidates_warped"].append(warped.copy())
                    debug_data["candidates_bw"].append(decode["bw"].copy())
                    debug_data["candidate_results"].append({
                        "accepted": decode["accepted"],
                        "id": decode["id"],
                        "rotation_cw_deg": decode["rotation_cw_deg"],
                        "hamming": decode["hamming"],
                        "border_errors": decode["border_errors"],
                        "full_bits": decode["full_bits"].copy(),
                        "inner": decode["inner"].copy(),
                        "expected": decode["expected"].copy(),
                        "quad": quad.copy(),
                    })

            if decode["accepted"]:
                result = {
                    "id": decode["id"],
                    "rotation_cw_deg": decode["rotation_cw_deg"],
                    "hamming": decode["hamming"],
                    "inner": decode["inner"],
                    "expected": decode["expected"],
                    "full_bits": decode["full_bits"],
                    "border_errors": decode["border_errors"],
                    "bw": decode["bw"],
                    "quad": quad,
                    "raw_angle_deg": self._angle_of_quad_deg(quad),
                    "angle_deg": self._marker_angle_deg(quad, decode["rotation_cw_deg"]),
                    "marker_corners": self._marker_ordered_corners(
                        quad, decode["rotation_cw_deg"]
                    ),
                }
                detections.append(result)

        output = {
            "detections": detections,
            "candidates": quads,
        }

        if self.show_debug:
            output["debug"] = debug_data

        return output

    # -----------------------------------------------------
    # Internal parsers / precompute
    # -----------------------------------------------------
    @staticmethod
    def _parse_dictionary_file(txt_path: str) -> Tuple[Dict[str, Any], Dict[int, np.ndarray]]:
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        lines = [line.strip() for line in raw_lines]

        marker_size = None
        num_markers = None
        border_bits = None
        bit_to_color = None
        border_color = None

        for line in lines:
            if not line or line.startswith("#"):
                continue

            m = re.match(r"marker_size\s*=\s*(\d+)\s*x\s*(\d+)", line, flags=re.IGNORECASE)
            if m:
                rows = int(m.group(1))
                cols = int(m.group(2))
                if rows != cols:
                    raise ValueError("Only square markers are supported.")
                marker_size = rows
                continue

            m = re.match(r"num_markers\s*=\s*(\d+)", line, flags=re.IGNORECASE)
            if m:
                num_markers = int(m.group(1))
                continue

            m = re.match(r"border_bits\s*=\s*(\d+)", line, flags=re.IGNORECASE)
            if m:
                border_bits = int(m.group(1))
                continue

            m = re.match(r"bit_meaning\s*=\s*(.+)", line, flags=re.IGNORECASE)
            if m:
                bit_to_color = BeeDetector._parse_bit_meaning(m.group(1))
                continue

            m = re.match(r"border_color\s*=\s*(black|white)", line, flags=re.IGNORECASE)
            if m:
                border_color = m.group(1).lower()
                continue

        if marker_size is None:
            raise ValueError("Could not find marker_size in txt file.")
        if num_markers is None:
            raise ValueError("Could not find num_markers in txt file.")
        if border_bits is None:
            raise ValueError("Could not find border_bits in txt file.")
        if bit_to_color is None:
            raise ValueError("Could not find bit_meaning in txt file.")
        if border_color is None:
            raise ValueError("Could not find border_color in txt file.")

        row_re = re.compile(rf"^[01](?:\s+[01]){{{marker_size - 1}}}$")
        id_re = re.compile(r"^ID\s+(\d+)\s*:\s*$", flags=re.IGNORECASE)

        markers: Dict[int, np.ndarray] = {}
        i = 0
        while i < len(lines):
            line = lines[i]
            m = id_re.match(line)
            if not m:
                i += 1
                continue

            marker_id = int(m.group(1))
            i += 1
            rows_data = []

            while i < len(lines) and len(rows_data) < marker_size:
                line = lines[i]
                if not line or line.startswith("#"):
                    i += 1
                    continue
                if row_re.match(line):
                    vals = [int(x) for x in line.split()]
                    rows_data.append(vals)
                i += 1

            if len(rows_data) != marker_size:
                raise ValueError(f"Incomplete marker definition for ID {marker_id}.")

            markers[marker_id] = np.array(rows_data, dtype=np.uint8)

        if len(markers) != num_markers:
            raise ValueError(
                f"num_markers says {num_markers}, but parsed {len(markers)} marker definitions."
            )

        metadata = {
            "marker_size": marker_size,
            "num_markers": num_markers,
            "border_bits": border_bits,
            "bit_to_color": bit_to_color,
            "border_color": border_color,
        }
        return metadata, markers

    @staticmethod
    def _parse_bit_meaning(text: str) -> Dict[int, str]:
        cleaned = text.lower().replace(";", ",")
        pairs = re.findall(r"([01])\s*[:=]\s*(white|black)", cleaned)
        if len(pairs) != 2:
            raise ValueError(
                "bit_meaning must look like '1:white,0:black' or '1=white,0=black'."
            )
        bit_to_color = {int(bit): color for bit, color in pairs}
        if set(bit_to_color.keys()) != {0, 1}:
            raise ValueError("bit_meaning must define both 0 and 1.")
        if sorted(bit_to_color.values()) != ["black", "white"]:
            raise ValueError("bit_meaning must map one bit to black and the other to white.")
        return bit_to_color

    def _rebuild_geometry_cache(self) -> None:
        if self.total_cells is None:
            return

        self.side_pixels = int(self.total_cells * self.cell_pixels)
        side = self.side_pixels

        self._dst_quad = np.array(
            [
                [0, 0],
                [side - 1, 0],
                [side - 1, side - 1],
                [0, side - 1],
            ],
            dtype=np.float32,
        )

        cell_w = side / self.total_cells
        cell_h = side / self.total_cells

        x0 = np.array(
            [int((c + self.sample_inset_ratio) * cell_w) for c in range(self.total_cells)],
            dtype=np.int32,
        )
        x1 = np.array(
            [int((c + 1 - self.sample_inset_ratio) * cell_w) for c in range(self.total_cells)],
            dtype=np.int32,
        )
        y0 = np.array(
            [int((r + self.sample_inset_ratio) * cell_h) for r in range(self.total_cells)],
            dtype=np.int32,
        )
        y1 = np.array(
            [int((r + 1 - self.sample_inset_ratio) * cell_h) for r in range(self.total_cells)],
            dtype=np.int32,
        )

        self._sample_x0 = x0
        self._sample_x1 = x1
        self._sample_y0 = y0
        self._sample_y1 = y1

        widths = (x1 - x0).astype(np.float32)
        heights = (y1 - y0).astype(np.float32)
        self._sample_areas = np.outer(heights, widths)

        border_mask = np.zeros((self.total_cells, self.total_cells), dtype=bool)
        b = self.border_bits
        border_mask[:b, :] = True
        border_mask[-b:, :] = True
        border_mask[:, :b] = True
        border_mask[:, -b:] = True
        self._border_mask = border_mask
        self._inner_slice = (slice(b, self.total_cells - b), slice(b, self.total_cells - b))

    def _precompute_rotated_templates(self) -> None:
        ids: List[int] = []
        rots: List[int] = []
        payloads: List[np.ndarray] = []

        for marker_id in sorted(self.custom_markers.keys()):
            pattern = self.custom_markers[marker_id]
            for rot_cw in (0, 90, 180, 270):
                rotated = self._rotate_pattern_cw(pattern, rot_cw)
                ids.append(marker_id)
                rots.append(rot_cw)
                payloads.append(rotated.reshape(-1).astype(np.uint8))

        self._template_ids = np.asarray(ids, dtype=np.int32)
        self._template_rotations = np.asarray(rots, dtype=np.int32)
        self._template_payloads_flat = np.stack(payloads, axis=0)

    @staticmethod
    def _rotate_pattern_cw(pattern: np.ndarray, rot_cw: int) -> np.ndarray:
        k = (-rot_cw // 90) % 4
        return np.rot90(pattern, k=k).copy()

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------
    @staticmethod
    def _order_quad(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _expand_quad(self, quad: np.ndarray) -> np.ndarray:
        quad = np.asarray(quad, dtype=np.float32)
        center = quad.mean(axis=0)
        return (quad - center) * self.quad_expand_scale + center

    def _angle_of_quad_deg(self, quad: np.ndarray) -> float:
        q = self._order_quad(quad)
        tl, tr = q[0], q[1]
        v = tr - tl
        ang = math.degrees(math.atan2(-float(v[1]), float(v[0])))
        return (ang + 360.0) % 360.0

    def _marker_ordered_corners(self, quad: np.ndarray, rotation_cw_deg: int) -> np.ndarray:
        q = self._order_quad(quad)
        shift = (int(rotation_cw_deg) % 360) // 90
        return np.roll(q, -shift, axis=0)

    def _marker_angle_deg(self, quad: np.ndarray, rotation_cw_deg: int) -> float:
        q = self._marker_ordered_corners(quad, rotation_cw_deg)
        tl, tr = q[0], q[1]
        v = tr - tl
        ang = math.degrees(math.atan2(-float(v[1]), float(v[0])))
        return (ang + 360.0) % 360.0

    def _find_square_candidates(
        self,
        channel_img: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        _, bw_inv = cv2.threshold(
            channel_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(bw_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        quads: List[np.ndarray] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_contour_area:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue

            approx = cv2.approxPolyDP(cnt, self.poly_eps_ratio * peri, True)
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue

            solidity = area / hull_area
            if solidity < self.min_solidity:
                continue

            quad = approx.reshape(4, 2).astype(np.float32)
            ordered = self._order_quad(quad)

            side_lengths = [
                np.linalg.norm(ordered[0] - ordered[1]),
                np.linalg.norm(ordered[1] - ordered[2]),
                np.linalg.norm(ordered[2] - ordered[3]),
                np.linalg.norm(ordered[3] - ordered[0]),
            ]
            min_side = float(min(side_lengths))
            max_side = float(max(side_lengths))

            if min_side < self.min_side_pixels:
                continue
            if max_side / max(min_side, 1e-6) > self.max_side_ratio:
                continue

            quads.append(ordered)

        quads = self._deduplicate_quads(quads)
        return quads, bw_inv

    def _deduplicate_quads(self, quads: List[np.ndarray]) -> List[np.ndarray]:
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
            for _, kc, _ in kept:
                if np.linalg.norm(center - kc) < self.dedup_center_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((q, center, area))

        return [x[0] for x in kept]

    def _warp_marker(self, channel_img: np.ndarray, quad: np.ndarray) -> np.ndarray:
        src = self._order_quad(self._expand_quad(quad))
        M = cv2.getPerspectiveTransform(src, self._dst_quad)
        warped = cv2.warpPerspective(
            channel_img,
            M,
            (self.side_pixels, self.side_pixels),
            flags=cv2.INTER_LINEAR,
        )
        return warped

    def _sample_grid_from_warp(self, warp_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, bw = cv2.threshold(warp_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        white_mask = (bw == 255).astype(np.uint8)
        integ = cv2.integral(white_mask, sdepth=cv2.CV_32S)

        x0 = self._sample_x0
        x1 = self._sample_x1
        y0 = self._sample_y0
        y1 = self._sample_y1

        sums = (
            integ[np.ix_(y1, x1)]
            - integ[np.ix_(y0, x1)]
            - integ[np.ix_(y1, x0)]
            + integ[np.ix_(y0, x0)]
        ).astype(np.float32)

        white_ratio = sums / np.maximum(self._sample_areas, 1.0)

        bits = np.empty((self.total_cells, self.total_cells), dtype=np.uint8)
        if self.white_bit_value == 1:
            bits[:, :] = (white_ratio >= 0.5).astype(np.uint8)
        else:
            bits[:, :] = (white_ratio < 0.5).astype(np.uint8)

        return bits, bw

    def _count_border_errors(self, full_bits: np.ndarray) -> int:
        border_vals = full_bits[self._border_mask]
        return int(np.count_nonzero(border_vals != self.expected_border_bit))

    def _decode_candidate(self, warp_gray: np.ndarray) -> Dict[str, Any]:
        full_bits, bw = self._sample_grid_from_warp(warp_gray)
        border_errors = self._count_border_errors(full_bits)
        inner = full_bits[self._inner_slice]
        inner_flat = inner.reshape(1, -1)

        distances = np.count_nonzero(self._template_payloads_flat != inner_flat, axis=1)
        best_idx = int(np.argmin(distances))
        best_hamming = int(distances[best_idx])
        best_id = int(self._template_ids[best_idx])
        best_rot = int(self._template_rotations[best_idx])
        expected_flat = self._template_payloads_flat[best_idx]
        expected = expected_flat.reshape(self.marker_size, self.marker_size)

        accepted = (
            best_hamming <= self.max_hamming
            and border_errors <= self.max_border_errors
        )

        return {
            "accepted": accepted,
            "id": best_id,
            "rotation_cw_deg": best_rot,
            "hamming": best_hamming,
            "inner": inner.copy(),
            "expected": expected.copy(),
            "full_bits": full_bits.copy(),
            "border_errors": border_errors,
            "bw": bw.copy(),
        }
    
class BeeDrawer:
    """
    Drawing utility for BeeDetector results.

    Features
    --------
    - draw accepted decoded markers
    - draw raw candidate quads
    - draw any externally provided 4-corner squares
    - configurable default colors
    - configurable per-ID border/text colors
    """

    def __init__(
        self,
        *,
        default_border_color: Tuple[int, int, int] = (0, 255, 0),
        default_text_color: Tuple[int, int, int] = (0, 255, 0),
        first_corner_color: Tuple[int, int, int] = (0, 0, 255),
        center_color: Tuple[int, int, int] = (255, 0, 0),
        candidate_color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
        font_scale: float = 0.5,
        id_border_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        id_text_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> None:
        self.default_border_color = default_border_color
        self.default_text_color = default_text_color
        self.first_corner_color = first_corner_color
        self.center_color = center_color
        self.candidate_color = candidate_color
        self.thickness = thickness
        self.font_scale = font_scale

        self.id_border_colors: Dict[int, Tuple[int, int, int]] = dict(id_border_colors or {})
        self.id_text_colors: Dict[int, Tuple[int, int, int]] = dict(id_text_colors or {})

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------
    def set_style(
        self,
        *,
        default_border_color: Optional[Tuple[int, int, int]] = None,
        default_text_color: Optional[Tuple[int, int, int]] = None,
        first_corner_color: Optional[Tuple[int, int, int]] = None,
        center_color: Optional[Tuple[int, int, int]] = None,
        candidate_color: Optional[Tuple[int, int, int]] = None,
        thickness: Optional[int] = None,
        font_scale: Optional[float] = None,
    ) -> None:
        if default_border_color is not None:
            self.default_border_color = default_border_color
        if default_text_color is not None:
            self.default_text_color = default_text_color
        if first_corner_color is not None:
            self.first_corner_color = first_corner_color
        if center_color is not None:
            self.center_color = center_color
        if candidate_color is not None:
            self.candidate_color = candidate_color
        if thickness is not None:
            self.thickness = int(thickness)
        if font_scale is not None:
            self.font_scale = float(font_scale)

    def set_id_colors(
        self,
        *,
        border_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        text_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        clear_existing: bool = False,
    ) -> None:
        if clear_existing:
            self.id_border_colors.clear()
            self.id_text_colors.clear()

        if border_colors is not None:
            for marker_id, color in border_colors.items():
                self.id_border_colors[int(marker_id)] = tuple(color)

        if text_colors is not None:
            for marker_id, color in text_colors.items():
                self.id_text_colors[int(marker_id)] = tuple(color)

    def get_style(self) -> Dict[str, Any]:
        return {
            "default_border_color": self.default_border_color,
            "default_text_color": self.default_text_color,
            "first_corner_color": self.first_corner_color,
            "center_color": self.center_color,
            "candidate_color": self.candidate_color,
            "thickness": self.thickness,
            "font_scale": self.font_scale,
            "id_border_colors": dict(self.id_border_colors),
            "id_text_colors": dict(self.id_text_colors),
        }

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------
    @staticmethod
    def _ensure_bgr(image: np.ndarray, copy: bool) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None.")

        if image.ndim == 2:
            out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return out.copy() if copy else out

        if image.ndim == 3 and image.shape[2] == 3:
            return image.copy() if copy else image

        raise ValueError("BeeDrawer expects a grayscale or BGR image.")

    @staticmethod
    def _as_detections(detections_or_result: Any) -> List[Dict[str, Any]]:
        if detections_or_result is None:
            return []

        if isinstance(detections_or_result, dict):
            return list(detections_or_result.get("detections", []))

        return list(detections_or_result)

    @staticmethod
    def _as_candidates(candidates_or_result: Any) -> List[np.ndarray]:
        if candidates_or_result is None:
            return []

        if isinstance(candidates_or_result, dict):
            return list(candidates_or_result.get("candidates", []))

        return list(candidates_or_result)

    @staticmethod
    def _normalize_quad(corners: Any) -> np.ndarray:
        q = np.asarray(corners, dtype=np.float32)
        if q.shape != (4, 2):
            q = q.reshape(4, 2)
        return q

    def _border_color_for_id(
        self,
        marker_id: Optional[int],
        override_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
        fallback: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[int, int, int]:
        if marker_id is not None:
            if override_map is not None and marker_id in override_map:
                return tuple(override_map[marker_id])
            if marker_id in self.id_border_colors:
                return tuple(self.id_border_colors[marker_id])

        return self.default_border_color if fallback is None else fallback

    def _text_color_for_id(
        self,
        marker_id: Optional[int],
        override_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
        fallback: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[int, int, int]:
        if marker_id is not None:
            if override_map is not None and marker_id in override_map:
                return tuple(override_map[marker_id])
            if marker_id in self.id_text_colors:
                return tuple(self.id_text_colors[marker_id])

        return self.default_text_color if fallback is None else fallback

    def _draw_one_quad(
        self,
        out: np.ndarray,
        quad: np.ndarray,
        *,
        border_color: Tuple[int, int, int],
        text: Optional[str] = None,
        text_color: Optional[Tuple[int, int, int]] = None,
        draw_center: bool = False,
        center_color: Optional[Tuple[int, int, int]] = None,
        draw_first_corner: bool = False,
        first_corner_color: Optional[Tuple[int, int, int]] = None,
        draw_orientation: bool = False,
    ) -> None:
        quad = self._normalize_quad(quad)
        pts = np.round(quad).astype(np.int32).reshape((-1, 1, 2))

        cv2.polylines(out, [pts], True, border_color, self.thickness, cv2.LINE_AA)

        center_xy = quad.mean(axis=0)
        center = tuple(np.round(center_xy).astype(int))
        p0 = tuple(np.round(quad[0]).astype(int))

        if draw_center:
            cv2.circle(
                out,
                center,
                max(2, self.thickness + 1),
                self.center_color if center_color is None else center_color,
                -1,
                cv2.LINE_AA,
            )

        if draw_first_corner:
            cv2.circle(
                out,
                p0,
                max(2, self.thickness + 1),
                self.first_corner_color if first_corner_color is None else first_corner_color,
                -1,
                cv2.LINE_AA,
            )

        if draw_orientation:
            cv2.line(
                out,
                center,
                p0,
                self.first_corner_color if first_corner_color is None else first_corner_color,
                self.thickness,
                cv2.LINE_AA,
            )

        if text:
            anchor_idx = int(np.argmin(quad[:, 0] + quad[:, 1]))
            anchor = tuple(np.round(quad[anchor_idx]).astype(int))
            text_org = (anchor[0] + 4, anchor[1] - 6)

            cv2.putText(
                out,
                text,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.default_text_color if text_color is None else text_color,
                max(1, self.thickness),
                cv2.LINE_AA,
            )

    # -----------------------------------------------------
    # Public drawing methods
    # -----------------------------------------------------
    def draw_detected_markers(
        self,
        image: np.ndarray,
        detections_or_result: Any,
        *,
        draw_ids: bool = True,
        draw_angles: bool = False,
        draw_center: bool = True,
        draw_first_corner: bool = True,
        draw_orientation: bool = True,
        copy: bool = False,
        border_color_overrides: Optional[Dict[int, Tuple[int, int, int]]] = None,
        text_color_overrides: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Draw accepted decoded markers.

        Accepts either:
        - full result dict from detector.detect(...)
        - result["detections"]
        """
        out = self._ensure_bgr(image, copy=copy)
        detections = self._as_detections(detections_or_result)

        for det in detections:
            marker_id = det.get("id", None)
            corners = det.get("marker_corners", det.get("quad"))
            if corners is None:
                continue

            border_color = self._border_color_for_id(marker_id, border_color_overrides)
            text_color = self._text_color_for_id(marker_id, text_color_overrides)

            custom_label = det.get("label_text", None)

            if custom_label is None:
                labels = []
                if draw_ids and marker_id is not None:
                    labels.append(f"ID {marker_id}")
                if draw_angles and "angle_deg" in det:
                    labels.append(f"{float(det['angle_deg']):.1f} deg")
                text_to_draw = " | ".join(labels) if labels else None
            else:
                text_to_draw = str(custom_label)

            self._draw_one_quad(
                out,
                corners,
                border_color=border_color,
                text=text_to_draw,
                text_color=text_color,
                draw_center=draw_center,
                draw_first_corner=draw_first_corner,
                draw_orientation=draw_orientation,
)

        return out

    def draw_candidates(
        self,
        image: np.ndarray,
        candidates_or_result: Any,
        *,
        draw_index: bool = False,
        copy: bool = False,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Draw raw candidate quads.

        Accepts either:
        - full result dict from detector.detect(...)
        - result["candidates"]
        - or any iterable of (4,2) corner arrays
        """
        out = self._ensure_bgr(image, copy=copy)
        candidates = self._as_candidates(candidates_or_result)
        border_color = self.candidate_color if color is None else color

        for i, quad in enumerate(candidates):
            label = str(i) if draw_index else None
            self._draw_one_quad(
                out,
                quad,
                border_color=border_color,
                text=label,
                text_color=border_color,
                draw_center=False,
                draw_first_corner=False,
                draw_orientation=False,
            )

        return out

    def draw_quads(
        self,
        image: np.ndarray,
        quads: List[np.ndarray],
        *,
        labels: Optional[List[str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        draw_center: bool = False,
        draw_first_corner: bool = False,
        draw_orientation: bool = False,
        copy: bool = False,
    ) -> np.ndarray:
        """
        Draw any externally supplied 4-corner squares.

        This is the most generic method when you already have corners.
        """
        out = self._ensure_bgr(image, copy=copy)

        for i, quad in enumerate(quads):
            label = None if labels is None or i >= len(labels) else labels[i]
            color = self.candidate_color if colors is None or i >= len(colors) else colors[i]

            self._draw_one_quad(
                out,
                quad,
                border_color=color,
                text=label,
                text_color=color,
                draw_center=draw_center,
                draw_first_corner=draw_first_corner,
                draw_orientation=draw_orientation,
            )

        return out