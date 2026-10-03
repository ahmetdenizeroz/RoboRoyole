
"""
yolo_worker.py

Standalone worker around an ultralytics YOLO model: model loading/caching
and running inference. This is the ONLY place that talks to `ultralytics`
directly.

bee_detector_core.py (tuner + shared detector) and tracking_core.py
(full tracking runs) both import YoloWorker / get_cached_worker from here
instead of loading/calling YOLO themselves. This keeps model-loading and
inference logic in one place and lets a long tracking run preload the model
once, up front, instead of lazily on the first processed frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class YoloDetection:
    """One raw YOLO box, in the coordinate space of the image passed to predict()."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: Optional[int]


class YoloWorker:
    """
    Loads (and caches) one YOLO model and runs inference on frames.

    Usage:
        worker = YoloWorker(model_path="weights/best.pt", device="cuda")
        worker.load()  # optional - predict() lazy-loads on first call too
        detections = worker.predict(frame_bgr, confidence=0.25, iou_threshold=0.45,
                                     imgsz=640, max_detections=50)
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self._model = None
        self.load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def matches(self, model_path: str, device: str) -> bool:
        return self.model_path == str(model_path) and self.device == str(device)

    def _resolve_model_path(self) -> str:
        path = self.model_path
        if not Path(path).is_absolute():
            candidate = Path(__file__).parent / path
            if candidate.exists():
                return str(candidate)
        return path

    def load(self) -> bool:
        """Load the model if not already loaded. Returns True on success.

        Safe to call up front (e.g. before a long tracking run starts) so the
        first processed frame does not pay the model-load latency.
        """
        if self._model is not None:
            return True
        if not self.model_path.strip():
            self.load_error = "model_path is empty."
            return False

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self.load_error = f"ultralytics package not installed: {exc}"
            return False

        try:
            model = YOLO(self._resolve_model_path())
            try:
                model.to(self.device)
            except Exception:
                pass  # some exported formats ignore .to(); predict(device=...) still applies below
            self._model = model
            self.load_error = None
            return True
        except Exception as exc:
            self.load_error = f"Failed to load YOLO model '{self.model_path}': {exc}"
            self._model = None
            return False

    def unload(self) -> None:
        self._model = None

    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        max_detections: int = 50,
        class_filter: Optional[Sequence[int]] = None,
    ) -> List[YoloDetection]:
        """Run inference on a single BGR frame. Returns [] (and sets
        self.load_error) instead of raising, so callers can keep running on
        the next frame/setting change."""
        if self._model is None and not self.load():
            return []

        try:
            results = self._model.predict(
                source=frame_bgr,
                conf=float(confidence),
                iou=float(iou_threshold),
                imgsz=int(imgsz),
                device=self.device,
                max_det=int(max_detections),
                classes=list(class_filter) if class_filter else None,
                verbose=False,
            )
        except Exception as exc:
            self.load_error = f"YOLO inference failed: {exc}"
            return []

        detections: List[YoloDetection] = []
        if not results:
            return detections
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return detections

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
            cls_id = int(boxes.cls[i]) if boxes.cls is not None else None
            detections.append(YoloDetection(xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls_id))
        return detections


# ---------------------------------------------------------------------------
# Process-wide worker cache
# ---------------------------------------------------------------------------
# Both the tuner GUI (frame-by-frame, settings change often) and the tracking
# engine (one long run) can end up wanting "the worker for this model_path +
# device". Caching by that key means a model already loaded by one of them
# is reused instead of being loaded into memory twice.

_worker_cache: Dict[Tuple[str, str], YoloWorker] = {}


def get_cached_worker(model_path: str, device: str) -> YoloWorker:
    key = (str(model_path), str(device))
    worker = _worker_cache.get(key)
    if worker is None:
        worker = YoloWorker(model_path, device)
        _worker_cache[key] = worker
    return worker


def clear_worker_cache() -> None:
    _worker_cache.clear()
