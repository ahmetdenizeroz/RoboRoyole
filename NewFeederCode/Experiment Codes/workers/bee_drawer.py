import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

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

            labels = []
            if draw_ids and marker_id is not None:
                labels.append(f"ID {marker_id}")
            if draw_angles and "angle_deg" in det:
                labels.append(f"{float(det['angle_deg']):.1f} deg")

            self._draw_one_quad(
                out,
                corners,
                border_color=border_color,
                text=" | ".join(labels) if labels else None,
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