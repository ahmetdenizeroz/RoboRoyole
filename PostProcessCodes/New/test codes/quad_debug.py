import cv2
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class BeeQuadDetector:
    """
    Quad-only detector for debugging square candidate detection.

    Input:
        - single-channel grayscale image

    Output:
        detect(...) returns:
        {
            "quads": [quad0, quad1, ...],              # kept quads only
            "proposals": [proposal0, proposal1, ...],  # kept proposal dicts
            "raw_proposals": [...],                    # all raw proposal dicts
        }

    Each quad is a 4x2 float32 array ordered as:
        [top-left, top-right, bottom-right, bottom-left]
    """

    def __init__(self) -> None:
        # Candidate detection settings
        self.min_area = 100.0
        self.max_contour_area = 500.0
        self.min_solidity = 0.2
        self.poly_eps_ratio = 0.05
        self.min_side_pixels = 3.0
        self.max_side_ratio = 2.2
        self.dedup_center_thresh = 12.0
        self.candidate_thresh = 60

        self.remove_small_group_area: Optional[float] = None
        self.remove_large_group_area: Optional[float] = None

        # Debug
        self.show_debug = False

    # -----------------------------------------------------
    # Public configuration
    # -----------------------------------------------------
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
        candidate_thresh: Optional[int] = None,
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
        if candidate_thresh is not None:
            self.candidate_thresh = int(candidate_thresh)
        if remove_small_group_area is not None:
            self.remove_small_group_area = float(remove_small_group_area)
        if remove_large_group_area is not None:
            self.remove_large_group_area = float(remove_large_group_area)

    def set_debug(self, show_debug: bool) -> None:
        self.show_debug = bool(show_debug)

    def get_settings(self) -> Dict[str, Any]:
        return {
            "min_area": self.min_area,
            "max_contour_area": self.max_contour_area,
            "min_solidity": self.min_solidity,
            "poly_eps_ratio": self.poly_eps_ratio,
            "min_side_pixels": self.min_side_pixels,
            "max_side_ratio": self.max_side_ratio,
            "dedup_center_thresh": self.dedup_center_thresh,
            "candidate_thresh": self.candidate_thresh,
            "remove_large_group_area": self.remove_large_group_area,
            "remove_small_group_area": self.remove_small_group_area,
            "show_debug": self.show_debug,
        }

    # -----------------------------------------------------
    # Main API
    # -----------------------------------------------------
    def detect(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Detect quads from a single-channel grayscale image.
        """
        if gray is None:
            raise ValueError("Input image is None.")
        if gray.ndim != 2:
            raise ValueError("BeeQuadDetector expects a single-channel grayscale image.")
        if gray.dtype != np.uint8:
            raise ValueError("Input grayscale image must have dtype uint8.")

        raw_proposals, kept_proposals = self._find_square_candidates(gray)

        return {
            "quads": [p["quad"] for p in kept_proposals],
            "proposals": kept_proposals,
            "raw_proposals": raw_proposals,
        }

    # -----------------------------------------------------
    # Debug helpers
    # -----------------------------------------------------
    def _filter_white_groups_by_area(self, bw_inv: np.ndarray) -> np.ndarray:
        """
        Remove white connected components whose area is:
        - smaller than self.remove_small_group_area, or
        - larger than self.remove_large_group_area

        Uses 4-connectivity to match the main candidate extraction stage.
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
    
    def _check_quad(self, quad: np.ndarray) -> Dict[str, Any]:
        ordered, min_side, max_side, area, center = self._quad_geometry(quad)
        side_ratio = max_side / max(min_side, 1e-6)

        ok = True
        reason = "ok"

        if min_side < self.min_side_pixels:
            ok = False
            reason = "min_side"
        elif side_ratio > self.max_side_ratio:
            ok = False
            reason = "side_ratio"

        return {
            "ordered": ordered,
            "min_side": float(min_side),
            "max_side": float(max_side),
            "side_ratio": float(side_ratio),
            "area": float(area),
            "center": center,
            "ok": ok,
            "reason": reason,
        }


    def _status_color(self, status: str):
        if status == "kept":
            return (0, 255, 255)   # yellow
        if status in ("proposed", "raw_only"):
            return (0, 255, 0)     # green
        return (0, 0, 255)         # red


    def _make_components_debug_view(
        self,
        gray_img: np.ndarray,
        component_infos: List[Dict[str, Any]],
    ) -> np.ndarray:
        vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        for info in component_infos:
            x, y, w, h = info["bbox"]
            status = info["status"]
            color = self._status_color(status)

            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)

            text = f"L{info['label_id']} A={info['area']:.0f} {status}"
            cv2.putText(
                vis,
                text,
                (x, max(15, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

        return vis


    def _make_contours_debug_view(
        self,
        gray_img: np.ndarray,
        component_infos: List[Dict[str, Any]],
    ) -> np.ndarray:
        vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        for info in component_infos:
            cnt = info.get("contour")
            if cnt is None:
                continue

            color = self._status_color(info["status"])
            cv2.drawContours(vis, [cnt], -1, color, 1)

            center = info.get("component_center")
            if center is not None:
                cx, cy = int(round(center[0])), int(round(center[1]))
                cv2.circle(vis, (cx, cy), 2, (255, 0, 0), -1)
                cv2.putText(
                    vis,
                    f"L{info['label_id']}",
                    (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return vis


    def _make_approx_rect_debug_view(
        self,
        gray_img: np.ndarray,
        component_infos: List[Dict[str, Any]],
    ) -> np.ndarray:
        vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        for info in component_infos:
            cnt = info.get("contour")
            if cnt is not None:
                cv2.drawContours(vis, [cnt], -1, (255, 0, 0), 1)   # blue contour

            approx_pts = info.get("approx_points")
            if approx_pts is not None and len(approx_pts) >= 2:
                approx_poly = np.asarray(approx_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [approx_poly], True, (0, 255, 0), 1)  # green approx

            rect_quad = info.get("rect_quad")
            if rect_quad is not None:
                rect_poly = np.asarray(rect_quad, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [rect_poly], True, (255, 0, 255), 1)  # magenta rect

            x, y, w, h = info["bbox"]
            cv2.putText(
                vis,
                f"L{info['label_id']}",
                (x, y + h + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self._status_color(info["status"]),
                1,
                cv2.LINE_AA,
            )

        return vis


    def _print_component_debug_info(self, component_infos: List[Dict[str, Any]]) -> None:
        print("\n[DEBUG] COMPONENT SUMMARY")

        def fmt(v, digits=2):
            if v is None:
                return "NA"
            if isinstance(v, (int, np.integer)):
                return str(int(v))
            return f"{float(v):.{digits}f}"

        def fmt_pts(pts):
            if pts is None:
                return "NA"
            arr = np.asarray(pts, dtype=np.float32)
            return np.array2string(np.round(arr, 2), separator=", ")

        for info in component_infos:
            print(
                " | ".join([
                    f"L{info['label_id']}",
                    f"status={info['status']}",
                    f"area={fmt(info.get('area'), 1)}",
                    f"solidity={fmt(info.get('solidity'), 3)}",
                    f"peri={fmt(info.get('peri'), 2)}",
                    f"approx_pts={fmt(info.get('approx_pts'), 0)}",
                    f"min_side={fmt(info.get('min_side'), 2)}",
                    f"max_side={fmt(info.get('max_side'), 2)}",
                    f"side_ratio={fmt(info.get('side_ratio'), 3)}",
                    f"fill_ratio={fmt(info.get('fill_ratio'), 3)}",
                    f"method={info.get('method', 'NA')}",
                ])
            )

            if info.get("approx_points") is not None:
                print("  approx_points:")
                print(f"  {fmt_pts(info['approx_points'])}")

            if info.get("rect_quad") is not None:
                print("  rect_quad:")
                print(f"  {fmt_pts(info['rect_quad'])}")

            if info.get("candidate_quad") is not None:
                print("  candidate_quad:")
                print(f"  {fmt_pts(info['candidate_quad'])}")

            if info.get("ordered_candidate_quad") is not None:
                print("  ordered_candidate_quad:")
                print(f"  {fmt_pts(info['ordered_candidate_quad'])}")

    def _make_candidate_debug_view(
        self,
        gray_img: np.ndarray,
        raw_quads: List[np.ndarray],
        kept_quads: List[np.ndarray],
    ) -> np.ndarray:
        vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        # Raw quads
        for i, q in enumerate(raw_quads):
            q = np.asarray(q, dtype=np.float32)
            poly = q.astype(np.int32).reshape((-1, 1, 2))
            center = np.mean(q, axis=0).astype(int)

            cv2.polylines(vis, [poly], True, (0, 255, 0), 1)

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

        # Kept quads
        for i, q in enumerate(kept_quads):
            q = np.asarray(q, dtype=np.float32)
            poly = q.astype(np.int32).reshape((-1, 1, 2))
            center = np.mean(q, axis=0).astype(int)

            cv2.polylines(vis, [poly], True, (0, 255, 255), 2)
            cv2.circle(vis, (int(center[0]), int(center[1])), 2, (255, 0, 0), -1)

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

    # -----------------------------------------------------
    # Geometry helpers
    # -----------------------------------------------------
    @staticmethod
    def _order_quad(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

        # 1) sort points by angle around centroid
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]

        # 2) ensure clockwise order in image coordinates
        #    (with y increasing downward)
        def signed_area(poly):
            x = poly[:, 0]
            y = poly[:, 1]
            return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))

        if signed_area(pts) < 0:
            pts = pts[::-1]

        # 3) rotate so top-left comes first
        #    use smallest (x + y)
        start = np.argmin(pts[:, 0] + pts[:, 1])
        pts = np.roll(pts, -start, axis=0)

        # final order: TL, TR, BR, BL
        return pts.astype(np.float32)

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

    # -----------------------------------------------------
    # Contour/component helpers
    # -----------------------------------------------------
    def _extract_component_contour(self, component_mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _propose_quads_from_contour(
        self,
        cnt: np.ndarray,
        component_area: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        proposals: List[Dict[str, Any]] = []

        debug_info: Dict[str, Any] = {
            "contour": cnt.copy(),
            "peri": None,
            "approx_pts": None,
            "approx_points": None,
            "rect_quad": None,
            "fill_ratio": None,
            "min_side": None,
            "max_side": None,
            "side_ratio": None,
            "method": None,
            "candidate_quad": None,       
            "ordered_candidate_quad": None 
        }

        peri = cv2.arcLength(cnt, True)
        debug_info["peri"] = float(peri)

        used_approx = False

        if peri > 0:
            approx = cv2.approxPolyDP(cnt, self.poly_eps_ratio * peri, True)
            approx_pts = approx.reshape(-1, 2).astype(np.float32)
            debug_info["approx_points"] = approx_pts
            debug_info["approx_pts"] = int(len(approx))

            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                quad_check = self._check_quad(quad)

                debug_info["candidate_quad"] = quad.copy()
                debug_info["ordered_candidate_quad"] = quad_check["ordered"].copy()
                debug_info["min_side"] = quad_check["min_side"]
                debug_info["max_side"] = quad_check["max_side"]
                debug_info["side_ratio"] = quad_check["side_ratio"]

                if quad_check["ok"]:
                    proposals.append(
                        {
                            "quad": quad_check["ordered"],
                            "method": "approx",
                            "component_area": float(component_area),
                        }
                    )
                    debug_info["method"] = "approx"
                    used_approx = True

        if not used_approx:
            rect = cv2.minAreaRect(cnt.astype(np.float32))
            (_, _), (rect_w, rect_h), _ = rect

            if rect_w > 0 and rect_h > 0:
                rect_area = float(rect_w * rect_h)
                fill_ratio = float(component_area) / max(rect_area, 1e-6)
                rect_quad = cv2.boxPoints(rect).astype(np.float32)

                debug_info["rect_quad"] = rect_quad
                debug_info["fill_ratio"] = float(fill_ratio)

                quad_check = self._check_quad(rect_quad)
                debug_info["min_side"] = quad_check["min_side"]
                debug_info["max_side"] = quad_check["max_side"]
                debug_info["side_ratio"] = quad_check["side_ratio"]
                debug_info["candidate_quad"] = rect_quad.copy()
                debug_info["ordered_candidate_quad"] = quad_check["ordered"].copy()

                if fill_ratio >= 0.20 and quad_check["ok"]:
                    proposals.append(
                        {
                            "quad": quad_check["ordered"],
                            "method": "rect",
                            "component_area": float(component_area),
                        }
                    )
                    debug_info["method"] = "rect"

        return proposals, debug_info

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

        # Prefer approx over rect, then prefer larger quads
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

    # -----------------------------------------------------
    # Main candidate finder
    # -----------------------------------------------------
    def _find_square_candidates(
        self, gray: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Returns:
            raw_proposals, kept_proposals
        """
        self.candidate_thresh, bw_inv = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        #_, bw_inv = cv2.threshold(
        #   gray, self.candidate_thresh - 10 , 255, cv2.THRESH_BINARY_INV
        #

        # NEW: remove large white connected groups before candidate extraction
        bw_work = self._filter_white_groups_by_area(bw_inv)

        if self.show_debug:
            cv2.imshow("debug_candidate_binary_raw", bw_inv)
            cv2.imshow("debug_candidate_binary_filtered", bw_work)
            cv2.waitKey(1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bw_work, connectivity=4
        )

        raw_proposals: List[Dict[str, Any]] = []
        component_infos: List[Dict[str, Any]] = []
        proposal_id_counter = 0

        for label_id in range(1, num_labels):  # label 0 = background
            area = float(stats[label_id, cv2.CC_STAT_AREA])
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            info: Dict[str, Any] = {
                "label_id": label_id,
                "area": area,
                "bbox": (x, y, w, h),
                "component_center": tuple(centroids[label_id]),
                "solidity": None,
                "peri": None,
                "approx_pts": None,
                "approx_points": None,
                "rect_quad": None,
                "fill_ratio": None,
                "min_side": None,
                "max_side": None,
                "side_ratio": None,
                "method": None,
                "contour": None,
                "status": "reject_area",
                "proposal_ids": [],
                "candidate_quad": None,
                "ordered_candidate_quad": None,
            }

            if area < self.min_area or area > self.max_contour_area:
                component_infos.append(info)
                continue

            if w <= 0 or h <= 0:
                info["status"] = "reject_bbox"
                component_infos.append(info)
                continue

            roi_labels = labels[y:y + h, x:x + w]
            component_mask = np.zeros((h, w), dtype=np.uint8)
            component_mask[roi_labels == label_id] = 255

            cnt_local = self._extract_component_contour(component_mask)
            if cnt_local is None:
                info["status"] = "reject_no_contour"
                component_infos.append(info)
                continue

            cnt = cnt_local + np.array([[[x, y]]], dtype=np.int32)
            info["contour"] = cnt

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                info["status"] = "reject_hull"
                component_infos.append(info)
                continue

            solidity = area / hull_area
            info["solidity"] = float(solidity)

            if solidity < self.min_solidity:
                info["status"] = "reject_solidity"
                component_infos.append(info)
                continue

            proposals, prop_debug = self._propose_quads_from_contour(cnt, area)

            info["peri"] = prop_debug.get("peri")
            info["approx_pts"] = prop_debug.get("approx_pts")
            info["approx_points"] = prop_debug.get("approx_points")
            info["rect_quad"] = prop_debug.get("rect_quad")
            info["fill_ratio"] = prop_debug.get("fill_ratio")
            info["min_side"] = prop_debug.get("min_side")
            info["max_side"] = prop_debug.get("max_side")
            info["side_ratio"] = prop_debug.get("side_ratio")
            info["method"] = prop_debug.get("method")
            info["candidate_quad"] = prop_debug.get("candidate_quad")
            info["ordered_candidate_quad"] = prop_debug.get("ordered_candidate_quad")

            if not proposals:
                info["status"] = "reject_no_quad"
                component_infos.append(info)
                continue

            for p in proposals:
                p["proposal_id"] = proposal_id_counter
                p["label_id"] = label_id
                info["proposal_ids"].append(proposal_id_counter)
                proposal_id_counter += 1
                raw_proposals.append(p)

            info["status"] = "proposed"
            component_infos.append(info)

        kept_proposals = self._deduplicate_proposals(raw_proposals)
        kept_ids = {p["proposal_id"] for p in kept_proposals}

        for info in component_infos:
            raw_ids = info.get("proposal_ids", [])
            kept_local = [pid for pid in raw_ids if pid in kept_ids]

            if kept_local:
                info["status"] = "kept"
            elif raw_ids:
                info["status"] = "raw_only"

        if self.show_debug:
            raw_quads = [p["quad"] for p in raw_proposals]
            kept_quads = [p["quad"] for p in kept_proposals]

            self._print_component_debug_info(component_infos)

            components_vis = self._make_components_debug_view(gray, component_infos)
            contours_vis = self._make_contours_debug_view(gray, component_infos)
            approx_rect_vis = self._make_approx_rect_debug_view(gray, component_infos)
            candidate_vis = self._make_candidate_debug_view(gray, raw_quads, kept_quads)

            cv2.imshow("debug_components", components_vis)
            cv2.imshow("debug_contours", contours_vis)
            cv2.imshow("debug_approx_vs_rect", approx_rect_vis)
            cv2.imshow("debug_candidate_quads", candidate_vis)
            cv2.waitKey(1)

        return raw_proposals, kept_proposals