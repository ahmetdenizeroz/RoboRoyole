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
        self.min_solidity = 0.2
        self.poly_eps_ratio = 0.05
        self.min_side_pixels = 3.0
        self.max_side_ratio = 2.2
        self.dedup_center_thresh = 12.0

        # Warp / sampling settings
        self.cell_pixels = 64
        self.quad_expand_scale = 1.0
        self.sample_inset_ratio = 0.05

        self.candidate_thresh = 90
        self.remove_small_group_area: Optional[float] = None
        self.remove_large_group_area: Optional[float] = None

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
        remove_small_group_area: Optional[float] = None,
        remove_large_group_area: Optional[float] = None,
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
        if remove_small_group_area is not None:
            self.remove_small_group_area = float(remove_small_group_area)
        if remove_large_group_area is not None:
            self.remove_large_group_area = float(remove_large_group_area)

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
    
    # Debug Helpers
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

    def _make_debug_grid_overlay(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()

        side = int(self.side_pixels)
        n = int(self.total_cells)

        # Full cell divider lines
        for i in range(1, n):
            p = int(round(i * side / n))
            cv2.line(vis, (p, 0), (p, side - 1), (0, 255, 255), 1)   # vertical
            cv2.line(vis, (0, p), (side - 1, p), (0, 255, 255), 1)   # horizontal

        # Sampling rectangles that actually use sample_inset_ratio
        for r in range(n):
            for c in range(n):
                x0 = int(self._sample_x0[c])
                x1 = int(self._sample_x1[c] - 1)
                y0 = int(self._sample_y0[r])
                y1 = int(self._sample_y1[r] - 1)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 1)

        return vis
    
    def _make_candidate_debug_view(
        self,
        channel_img: np.ndarray,
        raw_quads: List[np.ndarray],
        kept_quads: List[np.ndarray],
    ) -> np.ndarray:
        if channel_img.ndim == 2:
            vis = cv2.cvtColor(channel_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = channel_img.copy()

        # Raw quads that passed geometric filters
        for i, q in enumerate(raw_quads):
            q = np.asarray(q, dtype=np.float32)
            poly = q.astype(np.int32).reshape((-1, 1, 2))
            center = np.mean(q, axis=0).astype(int)

            cv2.polylines(vis, [poly], True, (0, 255, 0), 1)  # green outline

            # red corner dots + corner indices
            for j, pt in enumerate(q):
                px, py = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(vis, (px, py), 3, (0, 0, 255), -1)
                cv2.putText(
                    vis,
                    str(j),
                    (px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                vis,
                f"R{i}",
                (int(center[0]) + 4, int(center[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Final quads after deduplication
        for i, q in enumerate(kept_quads):
            q = np.asarray(q, dtype=np.float32)
            poly = q.astype(np.int32).reshape((-1, 1, 2))
            center = np.mean(q, axis=0).astype(int)

            cv2.polylines(vis, [poly], True, (0, 255, 255), 2)  # yellow outline
            cv2.circle(vis, (int(center[0]), int(center[1])), 2, (255, 0, 0), -1)

            # red corner dots + corner indices
            for j, pt in enumerate(q):
                px, py = int(round(pt[0])), int(round(pt[1]))
                cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)
                cv2.putText(
                    vis,
                    str(j),
                    (px + 4, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                vis,
                f"K{i}",
                (int(center[0]) + 4, int(center[1]) + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return vis

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
            "remove_large_group_area": self.remove_large_group_area,
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

        This version keeps all accepted marker detections found in the crop.
        The caller may still choose to deduplicate detections across blobs or IDs.
        """
        if frame is None:
            raise ValueError("Input frame is None.")

        if frame.ndim != 2:
            raise ValueError(
                "BeeDetector expects a single-channel 2D image. "
                "Prepare the input outside the detector."
            )

        work_img = frame
        proposals = self._find_square_candidates(work_img)

        H, W = work_img.shape[:2]
        crop_center = np.array([(W - 1) * 0.5, (H - 1) * 0.5], dtype=np.float32)

        detections: List[Dict[str, Any]] = []

        for p in proposals:
            quad = p["quad"]
            warped = self._warp_marker(work_img, quad)
            result = self._decode_candidate(warped)

            if result is None:
                continue

            _, _, _, quad_area, quad_center = self._quad_geometry(quad)
            center_dist = float(np.linalg.norm(quad_center - crop_center))
            method_rank = 0 if p["method"] == "approx" else 1

            result["quad"] = quad
            result["candidate_method"] = p["method"]
            result["raw_angle_deg"] = self._angle_of_quad_deg(quad)
            result["angle_deg"] = self._marker_angle_deg(quad, result["rotation_cw_deg"])
            result["marker_corners"] = self._marker_ordered_corners(
                quad, result["rotation_cw_deg"]
            )

            # Lower is better
            result["_sort_key"] = (
                int(result["hamming"]),
                int(result["border_errors"]),
                method_rank,          # prefer approx when decode quality ties
                center_dist,          # prefer marker near crop center
                -quad_area,           # prefer larger candidate
            )

            detections.append(result)

        detections.sort(key=lambda d: d["_sort_key"])

        for d in detections:
            d.pop("_sort_key", None)

        return {
            "detections": detections,
            "candidates": [p["quad"] for p in proposals],
        }

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
    def _filter_white_groups_by_area(self, bw_inv: np.ndarray) -> np.ndarray:
        """
        Remove white connected components whose area is:
        - smaller than self.remove_small_group_area, or
        - larger than self.remove_large_group_area
        """
        min_area = self.remove_small_group_area
        max_area = self.remove_large_group_area

        if min_area is None and max_area is None:
            return bw_inv.copy()

        filtered = bw_inv.copy()

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bw_inv, connectivity=4
        )

        for label_id in range(1, num_labels):  # 0 = background
            area = float(stats[label_id, cv2.CC_STAT_AREA])

            if min_area is not None and area < float(min_area):
                filtered[labels == label_id] = 0
                continue

            if max_area is not None and area > float(max_area):
                filtered[labels == label_id] = 0
                continue

        return filtered
    @staticmethod
    def _order_quad(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

        # 1) sort points by angle around centroid
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]

        # 2) ensure clockwise order
        def signed_area(poly: np.ndarray) -> float:
            x = poly[:, 0]
            y = poly[:, 1]
            return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

        if signed_area(pts) < 0:
            pts = pts[::-1]

        # 3) rotate so the first point is top-left
        start = np.argmin(pts[:, 0] + pts[:, 1])
        pts = np.roll(pts, -start, axis=0)

        return pts.astype(np.float32)

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
        return (ang + 450.0) % 360.0
    
    def _extract_component_contour(self, component_mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)
    
    def _quad_geometry(
        self, quad: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
        ordered = self._order_quad(quad)
        side_lengths = [
            np.linalg.norm(ordered[0] - ordered[1]),
            np.linalg.norm(ordered[1] - ordered[2]),
            np.linalg.norm(ordered[2] - ordered[3]),
            np.linalg.norm(ordered[3] - ordered[0]),
        ]
        min_side = float(min(side_lengths))
        max_side = float(max(side_lengths))
        area = float(abs(cv2.contourArea(ordered.astype(np.float32))))
        center = ordered.mean(axis=0)
        return ordered, min_side, max_side, area, center
    
    def _is_usable_quad(self, quad: np.ndarray) -> bool:
        ordered, min_side, max_side, _, _ = self._quad_geometry(quad)

        if min_side < self.min_side_pixels:
            return False
        if max_side / max(min_side, 1e-6) > self.max_side_ratio:
            return False

        return True
    
    def _propose_quads_from_contour(
        self,
        cnt: np.ndarray,
        component_area: float,
    ) -> List[Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []

        peri = cv2.arcLength(cnt, True)
        used_approx = False

        if peri > 0:
            approx = cv2.approxPolyDP(cnt, self.poly_eps_ratio * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                if self._is_usable_quad(quad):
                    proposals.append(
                        {
                            "quad": self._order_quad(quad),
                            "method": "approx",
                            "component_area": float(component_area),
                        }
                    )
                    used_approx = True

        # Fallback only if approx path did not produce a usable 4-corner quad
        if not used_approx:
            rect = cv2.minAreaRect(cnt.astype(np.float32))
            (_, _), (rect_w, rect_h), _ = rect

            if rect_w > 0 and rect_h > 0:
                rect_area = float(rect_w * rect_h)
                fill_ratio = float(component_area) / max(rect_area, 1e-6)

                # Prevent very loose rectangles around messy blobs
                if fill_ratio >= 0.20:
                    quad = cv2.boxPoints(rect).astype(np.float32)
                    if self._is_usable_quad(quad):
                        proposals.append(
                            {
                                "quad": self._order_quad(quad),
                                "method": "rect",
                                "component_area": float(component_area),
                            }
                        )

        return proposals
    
    def _deduplicate_proposals(
        self, proposals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not proposals:
            return []

        items = []
        for p in proposals:
            q = p["quad"]
            _, _, _, area, center = self._quad_geometry(q)
            method_rank = 0 if p["method"] == "approx" else 1
            items.append((p, center, area, method_rank))

        # Prefer approx over rect, and larger area over smaller
        items.sort(key=lambda x: (x[3], -x[2]))

        kept = []
        for p, center, area, method_rank in items:
            duplicate = False
            for kp, kc, ka, km in kept:
                if np.linalg.norm(center - kc) < self.dedup_center_thresh:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((p, center, area, method_rank))

        return [x[0] for x in kept]

    def _find_square_candidates(self, channel_img: np.ndarray) -> List[Dict[str, Any]]:
        _, bw_inv = cv2.threshold(
            channel_img, self.candidate_thresh, 255, cv2.THRESH_BINARY_INV
        )

        # Filter white groups by area before candidate extraction
        bw_work = self._filter_white_groups_by_area(bw_inv)

        if self.show_debug:
            cv2.imshow("debug_candidate_binary_raw", bw_inv)
            cv2.imshow("debug_candidate_binary_filtered", bw_work)
            cv2.waitKey(1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bw_work, connectivity=4
        )

        raw_proposals: List[Dict[str, Any]] = []

        for label_id in range(1, num_labels):  # 0 = background
            area = float(stats[label_id, cv2.CC_STAT_AREA])
            if area < self.min_area or area > self.max_contour_area:
                continue

            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            if w <= 0 or h <= 0:
                continue

            roi_labels = labels[y:y + h, x:x + w]
            component_mask = np.zeros((h, w), dtype=np.uint8)
            component_mask[roi_labels == label_id] = 255

            cnt_local = self._extract_component_contour(component_mask)
            if cnt_local is None:
                continue

            cnt = cnt_local + np.array([[[x, y]]], dtype=np.int32)

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue

            solidity = area / hull_area
            if solidity < self.min_solidity:
                continue

            proposals = self._propose_quads_from_contour(cnt, area)
            raw_proposals.extend(proposals)

        kept_proposals = self._deduplicate_proposals(raw_proposals)

        if self.show_debug:
            raw_quads = [p["quad"] for p in raw_proposals]
            kept_quads = [p["quad"] for p in kept_proposals]
            candidate_vis = self._make_candidate_debug_view(
                channel_img, raw_quads, kept_quads
            )
            cv2.imshow("debug_candidate_quads", candidate_vis)
            cv2.waitKey(1)

        return kept_proposals

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
        value, bw = cv2.threshold(warp_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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

    def _decode_candidate(self, warp_gray: np.ndarray) -> Optional[Dict[str, Any]]:
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

        result = {
            "id": best_id,
            "rotation_cw_deg": best_rot,
            "hamming": best_hamming,
            "inner": inner.copy(),
            "expected": expected.copy(),
            "full_bits": full_bits.copy(),
            "border_errors": border_errors,
            "bw": bw,
        }

        if best_hamming <= self.max_hamming and border_errors <= self.max_border_errors:
            return result

        if self.show_debug and (self.debug_marker_id is None or best_id == self.debug_marker_id):
            print("REJECTED MARKER")
            print("full_bits:")
            print(full_bits)
            print("inner:")
            print(inner)
            print("expected:")
            print(expected)
            print("best id:", best_id, "rot:", best_rot)
            print("hamming:", best_hamming, "border_errors:", border_errors)
            print("---")

            warp_vis = self._make_debug_grid_overlay(warp_gray)
            bw_vis = self._make_debug_grid_overlay(bw)

            cv2.imshow("debug_warp_gray", warp_vis)
            cv2.imshow("debug_bw", bw_vis)
            cv2.waitKey(1)

        return None