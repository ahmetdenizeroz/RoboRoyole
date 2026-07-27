
"""
single_bee_detector_core.py

Shared core for the no-ArUco single-bee tracker and the visual tuner.

Important design choices
------------------------
- The same SingleBeeDetector class is used by the tuner and by full tracking.
- The detector can use static, adaptive, or no background.
- Adaptive background is represented as a bounded rolling single-channel buffer.
- Detection can run on an arena bounding-box crop instead of the full frame.
- Detection can run on a downscaled image, then refine the selected candidate on
  a small original-resolution ROI.
- Expensive contour properties are calculated only after cheap component
  rejection by area/bbox/aspect/border/jump.
"""

from __future__ import annotations

import configparser
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]
Color = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Settings dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreprocessSettings:
    # adaptive/static/none
    background_mode: str = "adaptive"
    # Number of previous frames used by AdaptiveRollingBackground.
    adaptive_history_length: int = 40
    # median/average
    adaptive_background_method: str = "median"

    # red/green/blue/grayscale/hsv_value/hsv_saturation
    channel: str = "red"
    # absdiff: abs(channel - background_channel)
    # none: use extracted channel directly
    difference_mode: str = "absdiff"
    # 0 or 1 means no blur. Values > 1 are forced to odd numbers.
    blur_kernel: int = 3

    # Speed options.
    # 1.0 = full resolution. 0.5 = process at half width/height.
    detection_scale: float = 1.0
    # If True and detection_scale < 1.0, re-segment only the selected candidate
    # bbox at original resolution to improve x/y/angle.
    refine_on_original: bool = True
    # If an arena mask is given, crop to its bounding rectangle before processing.
    crop_to_arena_bbox: bool = True
    # Padding around the mapped low-res bbox during original-resolution refinement.
    refine_padding_px: int = 12


@dataclass
class ThresholdSettings:
    # manual/otsu/adaptive
    mode: str = "otsu"
    manual_threshold: int = 45
    adaptive_block_size: int = 51
    adaptive_c: float = -5.0
    invert_binary: bool = False


@dataclass
class MorphologySettings:
    open_kernel: int = 3
    close_kernel: int = 5
    erode_kernel: int = 0
    erode_iterations: int = 0
    dilate_kernel: int = 3
    dilate_iterations: int = 1
    # ellipse/rect/cross
    kernel_shape: str = "ellipse"


@dataclass
class BlobFilterSettings:
    # These values are always interpreted in ORIGINAL-resolution pixels, even
    # when detection_scale < 1.
    min_area: float = 50.0
    max_area: float = 5000.0
    min_solidity: float = 0.10
    min_extent: float = 0.0
    max_aspect_ratio: float = 6.0
    min_width: float = 0.0
    max_width: float = 10_000.0
    min_height: float = 0.0
    max_height: float = 10_000.0
    reject_touching_frame_border: bool = False


@dataclass
class CandidateSelectionSettings:
    # largest_area / closest_to_previous / score
    method: str = "largest_area"
    max_jump_px: float = 0.0
    allow_largest_if_no_previous: bool = True
    area_weight: float = 1.0
    solidity_weight: float = 0.5
    distance_weight: float = 1.0


@dataclass
class KalmanSettings:
    enabled: bool = True
    max_missed_frames: int = 10
    process_noise: float = 1.0
    measurement_noise: float = 5.0


@dataclass
class DebugSettings:
    draw_rejected_candidates: bool = True
    draw_candidate_labels: bool = True
    draw_axes: bool = True


@dataclass
class SingleBeeDetectorSettings:
    version: int = 1
    preprocess: PreprocessSettings = field(default_factory=PreprocessSettings)
    threshold: ThresholdSettings = field(default_factory=ThresholdSettings)
    morphology: MorphologySettings = field(default_factory=MorphologySettings)
    blob_filter: BlobFilterSettings = field(default_factory=BlobFilterSettings)
    candidate_selection: CandidateSelectionSettings = field(default_factory=CandidateSelectionSettings)
    kalman: KalmanSettings = field(default_factory=KalmanSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CandidateBlob:
    index: int
    contour: Optional[np.ndarray]
    area: float
    centroid: Point
    bbox: BBox
    width: int
    height: int
    solidity: float
    extent: float
    aspect_ratio: float
    major_axis_angle_deg: float
    major_axis_vector: Point
    minor_axis_vector: Point
    major_axis_endpoints: Tuple[Point, Point]
    minor_axis_endpoints: Tuple[Point, Point]
    reject_reasons: List[str] = field(default_factory=list)
    score: Optional[float] = None
    distance_to_previous: Optional[float] = None
    selected: bool = False

    @property
    def is_valid(self) -> bool:
        return len(self.reject_reasons) == 0


@dataclass
class DebugImages:
    channel: np.ndarray
    difference: np.ndarray
    threshold: np.ndarray
    cleaned: np.ndarray
    background_used: Optional[np.ndarray] = None
    crop_bbox: Optional[BBox] = None
    processing_scale: float = 1.0
    background_history_count: int = 0


@dataclass
class SingleBeeDetectionResult:
    detected: bool
    state: str
    x: float
    y: float
    angle_deg: float
    area: float
    bbox: Optional[BBox]
    selected_candidate: Optional[CandidateBlob]
    selected_index: Optional[int]
    candidates: List[CandidateBlob]
    debug_images: DebugImages
    message: str = ""


# ---------------------------------------------------------------------------
# Settings load/save helpers
# ---------------------------------------------------------------------------


def default_settings() -> SingleBeeDetectorSettings:
    return SingleBeeDetectorSettings()


def _bool_to_str(value: bool) -> str:
    return "true" if bool(value) else "false"


def _get_str(cfg: configparser.ConfigParser, section: str, key: str, default: str) -> str:
    if cfg.has_option(section, key):
        return cfg.get(section, key)
    return default


def _get_int(cfg: configparser.ConfigParser, section: str, key: str, default: int) -> int:
    try:
        return cfg.getint(section, key)
    except Exception:
        return int(default)


def _get_float(cfg: configparser.ConfigParser, section: str, key: str, default: float) -> float:
    try:
        return cfg.getfloat(section, key)
    except Exception:
        return float(default)


def _get_bool(cfg: configparser.ConfigParser, section: str, key: str, default: bool) -> bool:
    try:
        return cfg.getboolean(section, key)
    except Exception:
        return bool(default)


def settings_to_config(settings: SingleBeeDetectorSettings) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str

    cfg["preset"] = {"version": str(settings.version)}

    cfg["preprocess"] = {
        "background_mode": settings.preprocess.background_mode,
        "adaptive_history_length": str(settings.preprocess.adaptive_history_length),
        "adaptive_background_method": settings.preprocess.adaptive_background_method,
        "channel": settings.preprocess.channel,
        "difference_mode": settings.preprocess.difference_mode,
        "blur_kernel": str(settings.preprocess.blur_kernel),
        "detection_scale": str(settings.preprocess.detection_scale),
        "refine_on_original": _bool_to_str(settings.preprocess.refine_on_original),
        "crop_to_arena_bbox": _bool_to_str(settings.preprocess.crop_to_arena_bbox),
        "refine_padding_px": str(settings.preprocess.refine_padding_px),
    }

    cfg["threshold"] = {
        "mode": settings.threshold.mode,
        "manual_threshold": str(settings.threshold.manual_threshold),
        "adaptive_block_size": str(settings.threshold.adaptive_block_size),
        "adaptive_c": str(settings.threshold.adaptive_c),
        "invert_binary": _bool_to_str(settings.threshold.invert_binary),
    }

    cfg["morphology"] = {
        "open_kernel": str(settings.morphology.open_kernel),
        "close_kernel": str(settings.morphology.close_kernel),
        "erode_kernel": str(settings.morphology.erode_kernel),
        "erode_iterations": str(settings.morphology.erode_iterations),
        "dilate_kernel": str(settings.morphology.dilate_kernel),
        "dilate_iterations": str(settings.morphology.dilate_iterations),
        "kernel_shape": settings.morphology.kernel_shape,
    }

    cfg["blob_filter"] = {
        "min_area": str(settings.blob_filter.min_area),
        "max_area": str(settings.blob_filter.max_area),
        "min_solidity": str(settings.blob_filter.min_solidity),
        "min_extent": str(settings.blob_filter.min_extent),
        "max_aspect_ratio": str(settings.blob_filter.max_aspect_ratio),
        "min_width": str(settings.blob_filter.min_width),
        "max_width": str(settings.blob_filter.max_width),
        "min_height": str(settings.blob_filter.min_height),
        "max_height": str(settings.blob_filter.max_height),
        "reject_touching_frame_border": _bool_to_str(settings.blob_filter.reject_touching_frame_border),
    }

    cfg["candidate_selection"] = {
        "method": settings.candidate_selection.method,
        "max_jump_px": str(settings.candidate_selection.max_jump_px),
        "allow_largest_if_no_previous": _bool_to_str(settings.candidate_selection.allow_largest_if_no_previous),
        "area_weight": str(settings.candidate_selection.area_weight),
        "solidity_weight": str(settings.candidate_selection.solidity_weight),
        "distance_weight": str(settings.candidate_selection.distance_weight),
    }

    cfg["kalman"] = {
        "enabled": _bool_to_str(settings.kalman.enabled),
        "max_missed_frames": str(settings.kalman.max_missed_frames),
        "process_noise": str(settings.kalman.process_noise),
        "measurement_noise": str(settings.kalman.measurement_noise),
    }

    cfg["debug"] = {
        "draw_rejected_candidates": _bool_to_str(settings.debug.draw_rejected_candidates),
        "draw_candidate_labels": _bool_to_str(settings.debug.draw_candidate_labels),
        "draw_axes": _bool_to_str(settings.debug.draw_axes),
    }

    return cfg


def config_to_settings(cfg: configparser.ConfigParser) -> SingleBeeDetectorSettings:
    d = default_settings()

    d.version = _get_int(cfg, "preset", "version", d.version)

    p = d.preprocess
    p.background_mode = _get_str(cfg, "preprocess", "background_mode", p.background_mode)
    p.adaptive_history_length = _get_int(cfg, "preprocess", "adaptive_history_length", p.adaptive_history_length)
    p.adaptive_background_method = _get_str(cfg, "preprocess", "adaptive_background_method", p.adaptive_background_method)
    p.channel = _get_str(cfg, "preprocess", "channel", p.channel)
    p.difference_mode = _get_str(cfg, "preprocess", "difference_mode", p.difference_mode)
    p.blur_kernel = _get_int(cfg, "preprocess", "blur_kernel", p.blur_kernel)
    p.detection_scale = _get_float(cfg, "preprocess", "detection_scale", p.detection_scale)
    p.refine_on_original = _get_bool(cfg, "preprocess", "refine_on_original", p.refine_on_original)
    p.crop_to_arena_bbox = _get_bool(cfg, "preprocess", "crop_to_arena_bbox", p.crop_to_arena_bbox)
    p.refine_padding_px = _get_int(cfg, "preprocess", "refine_padding_px", p.refine_padding_px)

    t = d.threshold
    t.mode = _get_str(cfg, "threshold", "mode", t.mode)
    t.manual_threshold = _get_int(cfg, "threshold", "manual_threshold", t.manual_threshold)
    t.adaptive_block_size = _get_int(cfg, "threshold", "adaptive_block_size", t.adaptive_block_size)
    t.adaptive_c = _get_float(cfg, "threshold", "adaptive_c", t.adaptive_c)
    t.invert_binary = _get_bool(cfg, "threshold", "invert_binary", t.invert_binary)

    m = d.morphology
    m.open_kernel = _get_int(cfg, "morphology", "open_kernel", m.open_kernel)
    m.close_kernel = _get_int(cfg, "morphology", "close_kernel", m.close_kernel)
    m.erode_kernel = _get_int(cfg, "morphology", "erode_kernel", m.erode_kernel)
    m.erode_iterations = _get_int(cfg, "morphology", "erode_iterations", m.erode_iterations)
    m.dilate_kernel = _get_int(cfg, "morphology", "dilate_kernel", m.dilate_kernel)
    m.dilate_iterations = _get_int(cfg, "morphology", "dilate_iterations", m.dilate_iterations)
    m.kernel_shape = _get_str(cfg, "morphology", "kernel_shape", m.kernel_shape)

    b = d.blob_filter
    b.min_area = _get_float(cfg, "blob_filter", "min_area", b.min_area)
    b.max_area = _get_float(cfg, "blob_filter", "max_area", b.max_area)
    b.min_solidity = _get_float(cfg, "blob_filter", "min_solidity", b.min_solidity)
    b.min_extent = _get_float(cfg, "blob_filter", "min_extent", b.min_extent)
    b.max_aspect_ratio = _get_float(cfg, "blob_filter", "max_aspect_ratio", b.max_aspect_ratio)
    b.min_width = _get_float(cfg, "blob_filter", "min_width", b.min_width)
    b.max_width = _get_float(cfg, "blob_filter", "max_width", b.max_width)
    b.min_height = _get_float(cfg, "blob_filter", "min_height", b.min_height)
    b.max_height = _get_float(cfg, "blob_filter", "max_height", b.max_height)
    b.reject_touching_frame_border = _get_bool(cfg, "blob_filter", "reject_touching_frame_border", b.reject_touching_frame_border)

    c = d.candidate_selection
    c.method = _get_str(cfg, "candidate_selection", "method", c.method)
    c.max_jump_px = _get_float(cfg, "candidate_selection", "max_jump_px", c.max_jump_px)
    c.allow_largest_if_no_previous = _get_bool(cfg, "candidate_selection", "allow_largest_if_no_previous", c.allow_largest_if_no_previous)
    c.area_weight = _get_float(cfg, "candidate_selection", "area_weight", c.area_weight)
    c.solidity_weight = _get_float(cfg, "candidate_selection", "solidity_weight", c.solidity_weight)
    c.distance_weight = _get_float(cfg, "candidate_selection", "distance_weight", c.distance_weight)

    k = d.kalman
    k.enabled = _get_bool(cfg, "kalman", "enabled", k.enabled)
    k.max_missed_frames = _get_int(cfg, "kalman", "max_missed_frames", k.max_missed_frames)
    k.process_noise = _get_float(cfg, "kalman", "process_noise", k.process_noise)
    k.measurement_noise = _get_float(cfg, "kalman", "measurement_noise", k.measurement_noise)

    dbg = d.debug
    dbg.draw_rejected_candidates = _get_bool(cfg, "debug", "draw_rejected_candidates", dbg.draw_rejected_candidates)
    dbg.draw_candidate_labels = _get_bool(cfg, "debug", "draw_candidate_labels", dbg.draw_candidate_labels)
    dbg.draw_axes = _get_bool(cfg, "debug", "draw_axes", dbg.draw_axes)

    return d


def save_settings_txt(path: str | Path, settings: SingleBeeDetectorSettings) -> None:
    path = Path(path)
    cfg = settings_to_config(settings)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Single Bee Detector Settings\n")
        f.write("# This file is shared by the tuner GUI and the tracking GUI.\n")
        f.write("# Edit carefully, or preferably modify it from the tuner GUI.\n\n")
        cfg.write(f)


def load_settings_txt(path: str | Path) -> SingleBeeDetectorSettings:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    read_files = cfg.read(str(path), encoding="utf-8")
    if not read_files:
        raise FileNotFoundError(f"Could not read settings file: {path}")
    return config_to_settings(cfg)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _force_odd(value: int, minimum: int = 1) -> int:
    value = int(round(value))
    value = max(value, minimum)
    if value % 2 == 0:
        value += 1
    return value


def _kernel_shape_code(name: str) -> int:
    name = str(name).strip().lower()
    if name == "rect":
        return cv2.MORPH_RECT
    if name == "cross":
        return cv2.MORPH_CROSS
    return cv2.MORPH_ELLIPSE


def make_morph_kernel(size: int, shape: str = "ellipse") -> Optional[np.ndarray]:
    size = int(size)
    if size <= 1:
        return None
    size = _force_odd(size, minimum=3)
    return cv2.getStructuringElement(_kernel_shape_code(shape), (size, size))


def ensure_uint8_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 2:
        if image.dtype == np.uint8:
            return image.copy()
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 3:
        return cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected grayscale or BGR image")


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("image is None")
    if image.ndim == 2:
        return cv2.cvtColor(ensure_uint8_gray(image), cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError("Expected grayscale/BGR/BGRA image")


def extract_channel(image_bgr_or_gray: np.ndarray, channel: str) -> np.ndarray:
    channel = str(channel).strip().lower()
    if image_bgr_or_gray.ndim == 2:
        return ensure_uint8_gray(image_bgr_or_gray)

    image_bgr = ensure_bgr(image_bgr_or_gray)

    if channel == "blue":
        return image_bgr[:, :, 0].copy()
    if channel == "green":
        return image_bgr[:, :, 1].copy()
    if channel == "red":
        return image_bgr[:, :, 2].copy()
    if channel == "hsv_value":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 2].copy()
    if channel == "hsv_saturation":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1].copy()

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def normalize_mask(mask: Optional[np.ndarray], shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != shape_hw:
        raise ValueError(f"Arena mask shape {mask.shape[:2]} does not match frame shape {shape_hw}.")
    return ((mask > 0).astype(np.uint8) * 255)


def arena_bbox_from_mask(mask: Optional[np.ndarray], image_shape: Tuple[int, int], margin: int = 0) -> BBox:
    h, w = image_shape[:2]
    if mask is None or cv2.countNonZero(mask) == 0:
        return (0, 0, w, h)
    x, y, bw, bh = cv2.boundingRect((mask > 0).astype(np.uint8))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(w, x + bw + margin)
    y1 = min(h, y + bh + margin)
    return (int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0)))


def crop_array(arr: Optional[np.ndarray], bbox: BBox) -> Optional[np.ndarray]:
    if arr is None:
        return None
    x, y, w, h = bbox
    return arr[y:y + h, x:x + w].copy()


def resize_gray_nearest(img: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_NEAREST)


def resize_gray_area(img: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 0.999:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Adaptive background helper
# ---------------------------------------------------------------------------


class AdaptiveRollingBackground:
    """
    Bounded rolling single-channel background.

    - median: stores last N uint8 channel frames; background is pixelwise median.
    - average: stores last N uint8 channel frames plus a uint32 rolling sum.
      This avoids recomputing the full average stack on every frame.
    """

    def __init__(self, channel: str = "red", max_history: int = 40, method: str = "median") -> None:
        self.channel = str(channel)
        self.max_history = max(1, int(max_history))
        self.method = str(method).strip().lower()
        if self.method not in {"median", "average"}:
            self.method = "median"
        self._frames: Deque[np.ndarray] = deque()
        self._sum: Optional[np.ndarray] = None

    @property
    def history_count(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()
        self._sum = None

    def update(self, frame_bgr: np.ndarray) -> None:
        ch = extract_channel(frame_bgr, self.channel).astype(np.uint8, copy=True)

        if self.method == "average":
            if self._sum is None:
                self._sum = np.zeros(ch.shape, dtype=np.uint32)
            if self._sum.shape != ch.shape:
                self.clear()
                self._sum = np.zeros(ch.shape, dtype=np.uint32)
            self._frames.append(ch)
            self._sum += ch.astype(np.uint32)
            while len(self._frames) > self.max_history:
                old = self._frames.popleft()
                self._sum -= old.astype(np.uint32)
        else:
            if self._frames and self._frames[0].shape != ch.shape:
                self.clear()
            self._frames.append(ch)
            while len(self._frames) > self.max_history:
                self._frames.popleft()

    def get_background(self) -> Optional[np.ndarray]:
        if not self._frames:
            return None
        if self.method == "average":
            assert self._sum is not None
            return np.clip(self._sum / max(1, len(self._frames)), 0, 255).astype(np.uint8)
        stack = np.stack(list(self._frames), axis=0)
        return np.median(stack, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class SingleBeeDetector:
    def __init__(self, settings: Optional[SingleBeeDetectorSettings] = None) -> None:
        self.settings = settings if settings is not None else default_settings()

    def set_settings(self, settings: SingleBeeDetectorSettings) -> None:
        self.settings = settings

    def preprocess(
        self,
        frame_bgr: np.ndarray,
        background_bgr: Optional[np.ndarray] = None,
        background_channel: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return (channel_image, difference_or_source_image, background_channel_used)."""
        s = self.settings.preprocess
        channel_img = extract_channel(frame_bgr, s.channel)

        mode = str(s.difference_mode).strip().lower()
        bg_used: Optional[np.ndarray] = None

        if mode == "absdiff":
            if background_channel is not None:
                bg_used = ensure_uint8_gray(background_channel)
            elif background_bgr is not None:
                bg_used = extract_channel(background_bgr, s.channel)

        if mode == "absdiff" and bg_used is not None:
            if bg_used.shape != channel_img.shape:
                raise ValueError(
                    f"Background channel shape {bg_used.shape} does not match frame channel shape {channel_img.shape}."
                )
            work = cv2.absdiff(channel_img, bg_used)
        elif mode == "absdiff" and str(s.background_mode).strip().lower() in {"adaptive", "static"}:
            # No previous/background frame is available. For adaptive mode this
            # normally happens on the first processed frame. Use a black
            # difference image instead of accidentally treating the raw channel
            # as foreground.
            bg_used = np.zeros_like(channel_img, dtype=np.uint8)
            work = np.zeros_like(channel_img, dtype=np.uint8)
        else:
            work = channel_img.copy()

        k = int(s.blur_kernel)
        if k > 1:
            k = _force_odd(k, minimum=3)
            work = cv2.GaussianBlur(work, (k, k), 0)

        return channel_img, work, bg_used

    def threshold(self, source_gray: np.ndarray) -> np.ndarray:
        t = self.settings.threshold
        mode = str(t.mode).strip().lower()

        if mode == "manual":
            _, binary = cv2.threshold(source_gray, int(t.manual_threshold), 255, cv2.THRESH_BINARY)
        elif mode == "adaptive":
            block = _force_odd(int(t.adaptive_block_size), minimum=3)
            binary = cv2.adaptiveThreshold(
                source_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                float(t.adaptive_c),
            )
        else:
            _, binary = cv2.threshold(source_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if t.invert_binary:
            binary = cv2.bitwise_not(binary)

        return binary

    def apply_arena_mask(self, binary_img: np.ndarray, arena_mask: Optional[np.ndarray]) -> np.ndarray:
        if arena_mask is None:
            return binary_img
        mask = normalize_mask(arena_mask, binary_img.shape[:2])
        return cv2.bitwise_and(binary_img, binary_img, mask=mask)

    def apply_morphology(self, binary_img: np.ndarray, arena_mask: Optional[np.ndarray] = None) -> np.ndarray:
        m = self.settings.morphology
        cleaned = binary_img.copy()

        open_kernel = make_morph_kernel(m.open_kernel, m.kernel_shape)
        if open_kernel is not None:
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

        close_kernel = make_morph_kernel(m.close_kernel, m.kernel_shape)
        if close_kernel is not None:
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

        erode_kernel = make_morph_kernel(m.erode_kernel, m.kernel_shape)
        if erode_kernel is not None and int(m.erode_iterations) > 0:
            cleaned = cv2.erode(cleaned, erode_kernel, iterations=int(m.erode_iterations))

        dilate_kernel = make_morph_kernel(m.dilate_kernel, m.kernel_shape)
        if dilate_kernel is not None and int(m.dilate_iterations) > 0:
            cleaned = cv2.dilate(cleaned, dilate_kernel, iterations=int(m.dilate_iterations))

        return self.apply_arena_mask(cleaned, arena_mask)

    def detect(
        self,
        frame_bgr: np.ndarray,
        background_bgr: Optional[np.ndarray] = None,
        background_channel: Optional[np.ndarray] = None,
        background_history_count: int = 0,
        arena_mask: Optional[np.ndarray] = None,
        previous_position: Optional[Point] = None,
    ) -> SingleBeeDetectionResult:
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        frame_bgr = ensure_bgr(frame_bgr)
        h_full, w_full = frame_bgr.shape[:2]
        arena_mask_full = normalize_mask(arena_mask, (h_full, w_full)) if arena_mask is not None else None

        s = self.settings.preprocess
        crop_bbox = arena_bbox_from_mask(arena_mask_full, (h_full, w_full), margin=2) if bool(s.crop_to_arena_bbox) else (0, 0, w_full, h_full)
        crop_x, crop_y, crop_w, crop_h = crop_bbox

        frame_crop = crop_array(frame_bgr, crop_bbox)
        mask_crop = crop_array(arena_mask_full, crop_bbox) if arena_mask_full is not None else None

        bg_bgr_crop = None
        if background_bgr is not None:
            background_bgr = ensure_bgr(background_bgr)
            if background_bgr.shape[:2] != frame_bgr.shape[:2]:
                raise ValueError(f"Static background size {background_bgr.shape[:2]} does not match frame size {frame_bgr.shape[:2]}.")
            bg_bgr_crop = crop_array(background_bgr, crop_bbox)

        bg_channel_crop = None
        if background_channel is not None:
            bg_channel = ensure_uint8_gray(background_channel)
            if bg_channel.shape[:2] == frame_bgr.shape[:2]:
                # Caller supplied a full-frame adaptive/static channel.
                bg_channel_crop = crop_array(bg_channel, crop_bbox)
            elif bg_channel.shape[:2] == (crop_h, crop_w):
                # Caller already supplied the arena-bbox crop. This is the fast
                # path used by the full tracker to avoid full-frame adaptive
                # background maintenance.
                bg_channel_crop = bg_channel.copy()
            else:
                raise ValueError(
                    f"Background channel size {bg_channel.shape[:2]} matches neither full frame "
                    f"{frame_bgr.shape[:2]} nor detector crop {(crop_h, crop_w)}."
                )

        channel_full, diff_full, bg_used_full = self.preprocess(
            frame_crop,
            background_bgr=bg_bgr_crop,
            background_channel=bg_channel_crop,
        )

        scale = float(s.detection_scale)
        if not np.isfinite(scale):
            scale = 1.0
        scale = float(np.clip(scale, 0.05, 1.0))
        inv_scale = 1.0 / scale

        if scale < 0.999:
            diff_proc = resize_gray_area(diff_full, scale)
            channel_proc = resize_gray_area(channel_full, scale)
            mask_proc = resize_gray_nearest(mask_crop, scale) if mask_crop is not None else None
            bg_used_proc = resize_gray_area(bg_used_full, scale) if bg_used_full is not None else None
        else:
            diff_proc = diff_full
            channel_proc = channel_full
            mask_proc = mask_crop
            bg_used_proc = bg_used_full

        binary = self.threshold(diff_proc)
        binary = self.apply_arena_mask(binary, mask_proc)
        cleaned = self.apply_morphology(binary, mask_proc)

        prev_for_candidates = previous_position
        candidates = self.find_candidates(
            cleaned,
            previous_position=prev_for_candidates,
            inv_scale=inv_scale,
            offset=(crop_x, crop_y),
        )
        selected = self.select_candidate(candidates, previous_position=prev_for_candidates)

        if selected is not None and scale < 0.999 and bool(s.refine_on_original):
            refined = self._refine_selected_on_original(
                diff_full=diff_full,
                mask_full=mask_crop,
                selected=selected,
                crop_offset=(crop_x, crop_y),
            )
            if refined is not None:
                refined.index = selected.index
                refined.score = selected.score
                refined.selected = False
                selected = refined
                # Replace selected candidate in list so overlays use refined contour.
                for i, cand in enumerate(candidates):
                    if cand.index == refined.index:
                        candidates[i] = refined
                        break

        if selected is None:
            return SingleBeeDetectionResult(
                detected=False,
                state="missing",
                x=-1.0,
                y=-1.0,
                angle_deg=-1.0,
                area=0.0,
                bbox=None,
                selected_candidate=None,
                selected_index=None,
                candidates=candidates,
                debug_images=DebugImages(
                    channel=channel_proc,
                    difference=diff_proc,
                    threshold=binary,
                    cleaned=cleaned,
                    background_used=bg_used_proc,
                    crop_bbox=crop_bbox,
                    processing_scale=scale,
                    background_history_count=int(background_history_count),
                ),
                message="No valid bee candidate found.",
            )

        selected.selected = True
        return SingleBeeDetectionResult(
            detected=True,
            state="detected",
            x=float(selected.centroid[0]),
            y=float(selected.centroid[1]),
            angle_deg=float(selected.major_axis_angle_deg),
            area=float(selected.area),
            bbox=selected.bbox,
            selected_candidate=selected,
            selected_index=selected.index,
            candidates=candidates,
            debug_images=DebugImages(
                channel=channel_proc,
                difference=diff_proc,
                threshold=binary,
                cleaned=cleaned,
                background_used=bg_used_proc,
                crop_bbox=crop_bbox,
                processing_scale=scale,
                background_history_count=int(background_history_count),
            ),
            message="Detected.",
        )

    def find_candidates(
        self,
        cleaned_binary: np.ndarray,
        previous_position: Optional[Point] = None,
        *,
        inv_scale: float = 1.0,
        offset: Tuple[int, int] = (0, 0),
    ) -> List[CandidateBlob]:
        if cleaned_binary is None:
            raise ValueError("cleaned_binary is None")
        if cleaned_binary.ndim != 2:
            raise ValueError("cleaned_binary must be single-channel")
        if cleaned_binary.dtype != np.uint8:
            cleaned_binary = cleaned_binary.astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (cleaned_binary > 0).astype(np.uint8),
            connectivity=8,
        )

        candidates: List[CandidateBlob] = []
        for label_id in range(1, num_labels):
            cand = self._candidate_from_component(
                label_id=label_id,
                labels=labels,
                stats=stats,
                centroids=centroids,
                image_shape=cleaned_binary.shape,
                previous_position=previous_position,
                inv_scale=inv_scale,
                offset=offset,
            )
            candidates.append(cand)

        candidates.sort(key=lambda c: c.area, reverse=True)
        for new_idx, cand in enumerate(candidates):
            cand.index = new_idx
        return candidates

    def _candidate_from_component(
        self,
        *,
        label_id: int,
        labels: np.ndarray,
        stats: np.ndarray,
        centroids: np.ndarray,
        image_shape: Tuple[int, int],
        previous_position: Optional[Point],
        inv_scale: float,
        offset: Tuple[int, int],
    ) -> CandidateBlob:
        h_img, w_img = image_shape[:2]
        f = self.settings.blob_filter
        sel = self.settings.candidate_selection
        off_x, off_y = offset

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        pixel_area = float(stats[label_id, cv2.CC_STAT_AREA])

        # Estimate values in original-resolution coordinates before doing any
        # expensive contour work.
        x_full = int(round(off_x + x * inv_scale))
        y_full = int(round(off_y + y * inv_scale))
        w_full = max(1, int(round(w * inv_scale)))
        h_full = max(1, int(round(h * inv_scale)))
        bbox_full = (x_full, y_full, w_full, h_full)

        area_est_full = pixel_area * inv_scale * inv_scale
        cx_full = float(off_x + centroids[label_id][0] * inv_scale)
        cy_full = float(off_y + centroids[label_id][1] * inv_scale)
        aspect_ratio = max(w_full / max(h_full, 1), h_full / max(w_full, 1))
        extent_est = area_est_full / max(1.0, float(w_full * h_full))

        reasons: List[str] = []
        if area_est_full < float(f.min_area):
            reasons.append(f"area < min_area ({area_est_full:.1f} < {f.min_area:.1f})")
        if area_est_full > float(f.max_area):
            reasons.append(f"area > max_area ({area_est_full:.1f} > {f.max_area:.1f})")
        if w_full < float(f.min_width):
            reasons.append(f"width < min_width ({w_full} < {f.min_width:.1f})")
        if w_full > float(f.max_width):
            reasons.append(f"width > max_width ({w_full} > {f.max_width:.1f})")
        if h_full < float(f.min_height):
            reasons.append(f"height < min_height ({h_full} < {f.min_height:.1f})")
        if h_full > float(f.max_height):
            reasons.append(f"height > max_height ({h_full} > {f.max_height:.1f})")
        if aspect_ratio > float(f.max_aspect_ratio):
            reasons.append(f"aspect_ratio > max_aspect_ratio ({aspect_ratio:.2f} > {f.max_aspect_ratio:.2f})")
        if f.reject_touching_frame_border and (x <= 0 or y <= 0 or x + w >= w_img - 1 or y + h >= h_img - 1):
            reasons.append("touching processing-frame border")

        distance_to_previous: Optional[float] = None
        if previous_position is not None:
            distance_to_previous = math.hypot(cx_full - previous_position[0], cy_full - previous_position[1])
            if float(sel.max_jump_px) > 0.0 and distance_to_previous > float(sel.max_jump_px):
                reasons.append(f"jump > max_jump_px ({distance_to_previous:.1f} > {sel.max_jump_px:.1f})")

        # Cheap rejection succeeded or failed. If already failed, do not spend
        # time calculating contour, convex hull, solidity, or PCA axis.
        if reasons:
            return self._make_simple_candidate(
                index=label_id,
                area=area_est_full,
                centroid=(cx_full, cy_full),
                bbox=bbox_full,
                width=w_full,
                height=h_full,
                extent=extent_est,
                aspect_ratio=aspect_ratio,
                reasons=reasons,
                distance_to_previous=distance_to_previous,
            )

        component_roi = (labels[y:y + h, x:x + w] == label_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            reasons.append("no contour after component extraction")
            return self._make_simple_candidate(
                index=label_id,
                area=area_est_full,
                centroid=(cx_full, cy_full),
                bbox=bbox_full,
                width=w_full,
                height=h_full,
                extent=extent_est,
                aspect_ratio=aspect_ratio,
                reasons=reasons,
                distance_to_previous=distance_to_previous,
            )

        contour_proc = max(contours, key=cv2.contourArea)
        contour_proc = contour_proc.astype(np.float32)
        contour_proc[:, 0, 0] += float(x)
        contour_proc[:, 0, 1] += float(y)

        contour_full = contour_proc.copy()
        contour_full[:, 0, 0] = contour_full[:, 0, 0] * float(inv_scale) + float(off_x)
        contour_full[:, 0, 1] = contour_full[:, 0, 1] * float(inv_scale) + float(off_y)
        contour_full_i = np.round(contour_full).astype(np.int32)

        return self._candidate_from_full_contour(
            index=label_id,
            contour_full=contour_full_i,
            bbox_hint=bbox_full,
            previous_position=previous_position,
            initial_reasons=[],
            distance_to_previous=distance_to_previous,
        )

    def _make_simple_candidate(
        self,
        *,
        index: int,
        area: float,
        centroid: Point,
        bbox: BBox,
        width: int,
        height: int,
        extent: float,
        aspect_ratio: float,
        reasons: List[str],
        distance_to_previous: Optional[float],
    ) -> CandidateBlob:
        cx, cy = centroid
        return CandidateBlob(
            index=index,
            contour=None,
            area=float(area),
            centroid=(float(cx), float(cy)),
            bbox=bbox,
            width=int(width),
            height=int(height),
            solidity=0.0,
            extent=float(extent),
            aspect_ratio=float(aspect_ratio),
            major_axis_angle_deg=0.0,
            major_axis_vector=(1.0, 0.0),
            minor_axis_vector=(0.0, 1.0),
            major_axis_endpoints=((cx - 10, cy), (cx + 10, cy)),
            minor_axis_endpoints=((cx, cy - 10), (cx, cy + 10)),
            reject_reasons=list(reasons),
            distance_to_previous=distance_to_previous,
        )

    def _candidate_from_full_contour(
        self,
        *,
        index: int,
        contour_full: np.ndarray,
        bbox_hint: Optional[BBox] = None,
        previous_position: Optional[Point] = None,
        initial_reasons: Optional[List[str]] = None,
        distance_to_previous: Optional[float] = None,
    ) -> CandidateBlob:
        f = self.settings.blob_filter
        sel = self.settings.candidate_selection
        reasons = list(initial_reasons or [])

        area = float(cv2.contourArea(contour_full))
        x, y, w, h = cv2.boundingRect(contour_full)
        bbox = bbox_hint if bbox_hint is not None else (int(x), int(y), int(w), int(h))
        width = int(bbox[2])
        height = int(bbox[3])
        aspect_ratio = max(width / max(height, 1), height / max(width, 1))

        hull = cv2.convexHull(contour_full)
        hull_area = float(cv2.contourArea(hull)) if hull is not None and len(hull) >= 3 else 0.0
        solidity = area / hull_area if hull_area > 1e-9 else 0.0
        extent = area / max(1.0, float(width * height))

        if area < float(f.min_area):
            reasons.append(f"area < min_area ({area:.1f} < {f.min_area:.1f})")
        if area > float(f.max_area):
            reasons.append(f"area > max_area ({area:.1f} > {f.max_area:.1f})")
        if solidity < float(f.min_solidity):
            reasons.append(f"solidity < min_solidity ({solidity:.3f} < {f.min_solidity:.3f})")
        if extent < float(f.min_extent):
            reasons.append(f"extent < min_extent ({extent:.3f} < {f.min_extent:.3f})")
        if aspect_ratio > float(f.max_aspect_ratio):
            reasons.append(f"aspect_ratio > max_aspect_ratio ({aspect_ratio:.2f} > {f.max_aspect_ratio:.2f})")

        M = cv2.moments(contour_full)
        if M["m00"] == 0:
            cx, cy = float(x + w / 2.0), float(y + h / 2.0)
            reasons.append("zero contour moment")
        else:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

        if previous_position is not None:
            distance_to_previous = math.hypot(cx - previous_position[0], cy - previous_position[1])
            if float(sel.max_jump_px) > 0.0 and distance_to_previous > float(sel.max_jump_px):
                reasons.append(f"jump > max_jump_px ({distance_to_previous:.1f} > {sel.max_jump_px:.1f})")

        major_angle, major_vec, minor_vec, major_endpoints, minor_endpoints = self._contour_axes(contour_full, (cx, cy), M)

        return CandidateBlob(
            index=index,
            contour=contour_full,
            area=float(area),
            centroid=(cx, cy),
            bbox=bbox,
            width=width,
            height=height,
            solidity=float(solidity),
            extent=float(extent),
            aspect_ratio=float(aspect_ratio),
            major_axis_angle_deg=float(major_angle),
            major_axis_vector=major_vec,
            minor_axis_vector=minor_vec,
            major_axis_endpoints=major_endpoints,
            minor_axis_endpoints=minor_endpoints,
            reject_reasons=reasons,
            distance_to_previous=distance_to_previous,
        )

    def _contour_axes(
        self,
        contour: np.ndarray,
        centroid: Point,
        moments: Dict[str, float],
    ) -> Tuple[float, Point, Point, Tuple[Point, Point], Tuple[Point, Point]]:
        cx, cy = centroid
        if moments["m00"] == 0:
            return 0.0, (1.0, 0.0), (0.0, 1.0), ((cx - 10, cy), (cx + 10, cy)), ((cx, cy - 10), (cx, cy + 10))

        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]
        cov = np.array([[mu20, mu11], [mu11, mu02]], dtype=np.float64)

        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
        except Exception:
            eigvals = np.array([100.0, 25.0])
            eigvecs = np.eye(2)

        major_vec_np = eigvecs[:, 0]
        minor_vec_np = eigvecs[:, 1]
        angle = math.degrees(math.atan2(float(major_vec_np[1]), float(major_vec_np[0]))) % 180.0

        major_len = max(10.0, 2.0 * math.sqrt(max(float(eigvals[0]), 1e-9)))
        minor_len = max(6.0, 2.0 * math.sqrt(max(float(eigvals[1]), 1e-9)))

        major_vec = (float(major_vec_np[0]), float(major_vec_np[1]))
        minor_vec = (float(minor_vec_np[0]), float(minor_vec_np[1]))

        major_endpoints = (
            (cx - major_vec[0] * major_len, cy - major_vec[1] * major_len),
            (cx + major_vec[0] * major_len, cy + major_vec[1] * major_len),
        )
        minor_endpoints = (
            (cx - minor_vec[0] * minor_len, cy - minor_vec[1] * minor_len),
            (cx + minor_vec[0] * minor_len, cy + minor_vec[1] * minor_len),
        )

        return angle, major_vec, minor_vec, major_endpoints, minor_endpoints

    def _refine_selected_on_original(
        self,
        *,
        diff_full: np.ndarray,
        mask_full: Optional[np.ndarray],
        selected: CandidateBlob,
        crop_offset: Tuple[int, int],
    ) -> Optional[CandidateBlob]:
        pad = max(0, int(self.settings.preprocess.refine_padding_px))
        crop_x, crop_y = crop_offset
        x, y, w, h = selected.bbox

        # Convert full-frame bbox back to detector-crop-local coordinates.
        lx0 = max(0, int(x - crop_x - pad))
        ly0 = max(0, int(y - crop_y - pad))
        lx1 = min(diff_full.shape[1], int(x - crop_x + w + pad))
        ly1 = min(diff_full.shape[0], int(y - crop_y + h + pad))
        if lx1 <= lx0 or ly1 <= ly0:
            return None

        sub_diff = diff_full[ly0:ly1, lx0:lx1]
        sub_mask = mask_full[ly0:ly1, lx0:lx1] if mask_full is not None else None

        try:
            sub_binary = self.threshold(sub_diff)
            sub_binary = self.apply_arena_mask(sub_binary, sub_mask)
            sub_cleaned = self.apply_morphology(sub_binary, sub_mask)
            sub_candidates = self.find_candidates(
                sub_cleaned,
                previous_position=selected.centroid,
                inv_scale=1.0,
                offset=(crop_x + lx0, crop_y + ly0),
            )
            refined = self.select_candidate(sub_candidates, previous_position=selected.centroid)
            return refined if refined is not None and refined.is_valid else None
        except Exception:
            return None

    def select_candidate(
        self,
        candidates: List[CandidateBlob],
        previous_position: Optional[Point] = None,
    ) -> Optional[CandidateBlob]:
        valid = [c for c in candidates if c.is_valid]
        if not valid:
            return None

        sel = self.settings.candidate_selection
        method = str(sel.method).strip().lower()

        max_area = max((c.area for c in valid), default=1.0)
        max_dist = max((c.distance_to_previous or 0.0 for c in valid), default=1.0)
        max_dist = max(max_dist, 1.0)

        for c in valid:
            if method == "closest_to_previous" and previous_position is not None:
                c.score = -(c.distance_to_previous or 0.0)
            elif method == "score":
                area_score = c.area / max(max_area, 1e-9)
                solidity_score = c.solidity
                if previous_position is not None:
                    distance_score = 1.0 - min((c.distance_to_previous or 0.0) / max_dist, 1.0)
                else:
                    distance_score = 0.0
                c.score = (
                    float(sel.area_weight) * area_score
                    + float(sel.solidity_weight) * solidity_score
                    + float(sel.distance_weight) * distance_score
                )
            else:
                c.score = c.area

        if method == "closest_to_previous":
            if previous_position is not None:
                return min(valid, key=lambda c: ((c.distance_to_previous or 1e12), -c.area))
            if not bool(sel.allow_largest_if_no_previous):
                return None

        return max(valid, key=lambda c: (c.score if c.score is not None else -1e12, c.area))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def _pt(p: Point) -> Tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def draw_arena_mask_outline(image_bgr: np.ndarray, arena_mask: Optional[np.ndarray]) -> np.ndarray:
    out = ensure_bgr(image_bgr)
    if arena_mask is None:
        return out
    mask = arena_mask
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 180, 0), 2)
    return out


def draw_detection_overlay(
    frame_bgr: np.ndarray,
    result: SingleBeeDetectionResult,
    arena_mask: Optional[np.ndarray] = None,
    *,
    draw_rejected: bool = True,
    draw_labels: bool = True,
    draw_axes: bool = True,
) -> np.ndarray:
    out = ensure_bgr(frame_bgr)
    out = draw_arena_mask_outline(out, arena_mask)

    for cand in result.candidates:
        if cand.selected:
            color: Color = (0, 255, 0)
            thickness = 2
        elif cand.is_valid:
            color = (0, 220, 255)
            thickness = 1
        else:
            if not draw_rejected:
                continue
            color = (0, 0, 255)
            thickness = 1

        if cand.contour is not None:
            cv2.drawContours(out, [cand.contour], -1, color, thickness)

        x, y, w, h = cand.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        cv2.circle(out, _pt(cand.centroid), 3, color, -1)

        if draw_axes and cand.is_valid:
            cv2.line(out, _pt(cand.major_axis_endpoints[0]), _pt(cand.major_axis_endpoints[1]), (255, 0, 255), 2)
            cv2.line(out, _pt(cand.minor_axis_endpoints[0]), _pt(cand.minor_axis_endpoints[1]), (255, 255, 0), 1)

        if draw_labels:
            label = f"#{cand.index} A={cand.area:.0f}"
            if cand.selected:
                label = f"SELECTED {label} ang={cand.major_axis_angle_deg:.1f}"
            elif not cand.is_valid:
                if cand.reject_reasons:
                    label = f"REJ {label}: {cand.reject_reasons[0]}"
                else:
                    label = f"REJ {label}"
            cv2.putText(out, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if result.detected and result.selected_candidate is not None:
        cv2.circle(out, _pt(result.selected_candidate.centroid), 6, (0, 255, 0), 2)
        cv2.putText(
            out,
            f"bee: x={result.x:.1f}, y={result.y:.1f}, angle={result.angle_deg:.1f}, scale={result.debug_images.processing_scale:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(out, "bee: missing", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    if result.debug_images.crop_bbox is not None:
        x, y, w, h = result.debug_images.crop_bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 180, 0), 1)

    return out


def mask_from_rect(image_shape: Tuple[int, int], rect: Optional[BBox]) -> Optional[np.ndarray]:
    if rect is None:
        return None
    h_img, w_img = image_shape[:2]
    x, y, w, h = [int(v) for v in rect]
    x0 = max(0, min(w_img - 1, x))
    y0 = max(0, min(h_img - 1, y))
    x1 = max(0, min(w_img, x + max(1, w)))
    y1 = max(0, min(h_img, y + max(1, h)))
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def summarize_result(result: SingleBeeDetectionResult, max_candidates: int = 12) -> str:
    lines: List[str] = []
    lines.append(f"State: {result.state}")
    lines.append(
        f"crop_bbox={result.debug_images.crop_bbox}, processing_scale={result.debug_images.processing_scale:.3f}, "
        f"adaptive_history_count={result.debug_images.background_history_count}"
    )
    if result.detected:
        lines.append(
            f"Selected: #{result.selected_index}  x={result.x:.2f}, y={result.y:.2f}, "
            f"angle={result.angle_deg:.2f} deg, area={result.area:.1f}, bbox={result.bbox}"
        )
    else:
        lines.append(result.message)

    valid_count = sum(1 for c in result.candidates if c.is_valid)
    lines.append(f"Candidates: {len(result.candidates)} total, {valid_count} valid")
    lines.append("")

    for cand in result.candidates[:max_candidates]:
        status = "SELECTED" if cand.selected else ("valid" if cand.is_valid else "rejected")
        dist = f", dist_prev={cand.distance_to_previous:.1f}" if cand.distance_to_previous is not None else ""
        score = f", score={cand.score:.3f}" if cand.score is not None else ""
        lines.append(
            f"#{cand.index}: {status}, area={cand.area:.1f}, centroid=({cand.centroid[0]:.1f}, {cand.centroid[1]:.1f}), "
            f"bbox={cand.bbox}, solidity={cand.solidity:.3f}, extent={cand.extent:.3f}, aspect={cand.aspect_ratio:.2f}{dist}{score}"
        )
        if cand.reject_reasons:
            for reason in cand.reject_reasons[:5]:
                lines.append(f"    - {reason}")

    if len(result.candidates) > max_candidates:
        lines.append(f"... {len(result.candidates) - max_candidates} more candidates not shown")

    return "\n".join(lines)
