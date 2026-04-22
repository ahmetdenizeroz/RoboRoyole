import cv2
import math
import numpy as np


class BeeFollower:
    """
    Detects white blobs from a binary image and computes:
    - area
    - centroid
    - major/minor principal axes based on second central moments

    Input image must be a single-channel binary image:
    background = 0, blob = 255
    """

    def __init__(
        self,
        min_area: float = 50,
        max_area: float = 5000,
        retrieval_mode: int = cv2.RETR_EXTERNAL,
        approx_method: int = cv2.CHAIN_APPROX_NONE,
    ):
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.retrieval_mode = retrieval_mode
        self.approx_method = approx_method

    # ------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------
    def set_detection_settings(
        self,
        *,
        min_area=None,
        max_area=None,
        retrieval_mode=None,
        approx_method=None,
    ) -> None:
        if min_area is not None:
            self.min_area = float(min_area)
        if max_area is not None:
            self.max_area = float(max_area)
        if retrieval_mode is not None:
            self.retrieval_mode = retrieval_mode
        if approx_method is not None:
            self.approx_method = approx_method

    def get_detection_settings(self) -> dict:
        return {
            "min_area": self.min_area,
            "max_area": self.max_area,
            "retrieval_mode": self.retrieval_mode,
            "approx_method": self.approx_method,
        }

    # ------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------
    def detect(self, binary_img: np.ndarray) -> list:
        """
        Detect valid blobs from a binary image.

        Returns a list of dictionaries. Each dictionary contains:
        - contour
        - area
        - centroid
        - bbox
        - major_axis_vector
        - minor_axis_vector
        - major_axis_angle_deg
        - minor_axis_angle_deg
        - major_moment
        - minor_moment
        - major_axis_endpoints
        - minor_axis_endpoints
        """
        self._validate_binary_image(binary_img)

        contours, _ = cv2.findContours(
            binary_img,
            self.retrieval_mode,
            self.approx_method
        )

        blobs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue

            blob_data = self._analyze_contour(contour)
            if blob_data is not None:
                blobs.append(blob_data)

        return blobs

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _validate_binary_image(self, binary_img: np.ndarray) -> None:
        if binary_img is None:
            raise ValueError("binary_img is None.")
        if binary_img.ndim != 2:
            raise ValueError("binary_img must be a single-channel image.")
        if binary_img.dtype != np.uint8:
            raise ValueError("binary_img must have dtype uint8.")

    def _analyze_contour(self, contour: np.ndarray) -> dict | None:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Central moments normalized by mass
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]

        # Second moment matrix around centroid
        cov = np.array([
            [mu20, mu11],
            [mu11, mu02]
        ], dtype=np.float64)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort so eigvals[0] becomes major, eigvals[1] becomes minor
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        major_moment = float(eigvals[0])
        minor_moment = float(eigvals[1])

        major_vec = eigvecs[:, 0]
        minor_vec = eigvecs[:, 1]

        # Angles from +x axis, CCW positive
        major_angle_deg = math.degrees(math.atan2(major_vec[1], major_vec[0]))
        minor_angle_deg = math.degrees(math.atan2(minor_vec[1], minor_vec[0]))

        # Use sqrt(moment) as a practical drawing scale
        # This is not "full object length", just a useful inertia-based axis size.
        major_len = float(2.0 * math.sqrt(max(major_moment, 0.0)))
        minor_len = float(2.0 * math.sqrt(max(minor_moment, 0.0)))

        center = np.array([cx, cy], dtype=np.float64)

        major_p1 = tuple((center - major_vec * major_len).astype(int))
        major_p2 = tuple((center + major_vec * major_len).astype(int))
        minor_p1 = tuple((center - minor_vec * minor_len).astype(int))
        minor_p2 = tuple((center + minor_vec * minor_len).astype(int))

        x, y, w, h = cv2.boundingRect(contour)

        return {
            "contour": contour,
            "area": float(cv2.contourArea(contour)),
            "centroid": (float(cx), float(cy)),
            "bbox": (int(x), int(y), int(w), int(h)),
            "major_axis_vector": (float(major_vec[0]), float(major_vec[1])),
            "minor_axis_vector": (float(minor_vec[0]), float(minor_vec[1])),
            "major_axis_angle_deg": float(major_angle_deg),
            "minor_axis_angle_deg": float(minor_angle_deg),
            "major_moment": major_moment,
            "minor_moment": minor_moment,
            "major_axis_length": major_len,
            "minor_axis_length": minor_len,
            "major_axis_endpoints": (major_p1, major_p2),
            "minor_axis_endpoints": (minor_p1, minor_p2),
        }