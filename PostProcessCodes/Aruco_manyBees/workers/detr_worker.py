"""
detr_worker.py

Standalone worker around an RF-DETR model: model loading/caching and
running inference. This is the ONLY place that talks to `rfdetr` directly.

Mirrors yolo_worker.py's structure/contract on purpose, so
bee_detector_core.py and tracking_core.py can treat "yolo" and
"detr" as interchangeable detection backends.

NOTE ON THE rfdetr API: this is written against the current rfdetr usage
pattern - `from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium,
RFDETRLarge` (RFDETRBase was deprecated/removed upstream; RFDETRSmall is
its documented replacement), `model = RFDETRSmall(pretrain_weights=path)`,
`model.predict(image_rgb, threshold=...)` returning a
supervision.Detections-like object with .xyxy / .confidence / .class_id
arrays. If the installed rfdetr version's API differs, load()/predict()
catch the real exception into self.load_error instead of crashing -
report that message back so the import/call pattern here can be adjusted
to match your exact version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class DetrDetection:
    """One raw RF-DETR box, in the coordinate space of the image passed to predict()."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: Optional[int]


class DetrWorker:
    """
    Loads (and caches) one RF-DETR model and runs inference on frames.

    Usage:
        worker = DetrWorker(model_path="weights/rfdetr_bees.pth", device="cuda", variant="base")
        worker.load()  # optional - predict() lazy-loads on first call too
        detections = worker.predict(frame_bgr, confidence=0.5)
    """

    def __init__(self, model_path: str, device: str = "cpu", variant: str = "base") -> None:
        self.model_path = str(model_path)
        self.device = str(device)
        self.variant = (str(variant).strip().lower() or "base")
        self._model = None
        self.load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def matches(self, model_path: str, device: str, variant: str) -> bool:
        return (
            self.model_path == str(model_path)
            and self.device == str(device)
            and self.variant == (str(variant).strip().lower() or "base")
        )

    def load(self) -> bool:
        """Load the model if not already loaded. Returns True on success.

        Safe to call up front (e.g. before a long tracking run starts) so the
        first processed frame does not pay the model-load latency.
        """
        if self._model is not None:
            return True
        if not self.model_path.strip():
            # Deliberately fatal: instantiating an RFDETR* class with no
            # pretrain_weights makes rfdetr download its own generic
            # COCO-pretrained checkpoint from Roboflow's servers on the
            # calling thread - a multi-minute, UI-freezing surprise the
            # first time someone selects "DETR" before browsing for a
            # checkpoint. Require an explicit path instead (same contract
            # as YoloWorker.load()).
            self.load_error = "model_path is empty."
            return False

        try:
            from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
        except ImportError as exc:
            self.load_error = f"rfdetr package not installed: {exc}"
            return False

        # RFDETRBase was deprecated and removed in favor of size-specific
        # classes; RFDETRBase's old default behaviour maps to RFDETRSmall.
        variant_classes = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "base": RFDETRSmall,  # backward-compat alias for older saved settings
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        model_cls = variant_classes.get(self.variant, RFDETRSmall)

        try:
            model = model_cls(pretrain_weights=self.model_path)
            try:
                # .optimize_for_inference() was renamed to .inference() in
                # newer rfdetr releases; try the current name first and
                # fall back for older installs. Purely an optional speedup -
                # safe to skip entirely if neither is available.
                if hasattr(model, "inference"):
                    model.inference()
                else:
                    model.optimize_for_inference()
            except Exception:
                pass
            self._model = model
            self.load_error = None
            return True
        except Exception as exc:
            self.load_error = f"Failed to load RF-DETR model (variant='{self.variant}', path='{self.model_path}'): {exc}"
            self._model = None
            return False

    def unload(self) -> None:
        self._model = None

    def predict(
        self,
        frame_bgr: np.ndarray,
        *,
        confidence: float = 0.5,
        class_filter: Optional[Sequence[int]] = None,
    ) -> List[DetrDetection]:
        """Run inference on a single BGR frame. Returns [] (and sets
        self.load_error) instead of raising, so callers can keep running on
        the next frame/setting change."""
        if self._model is None and not self.load():
            return []

        try:
            # RF-DETR (like most COCO-trained detectors) expects RGB input;
            # OpenCV frames are BGR.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._model.predict(frame_rgb, threshold=float(confidence))
        except Exception as exc:
            self.load_error = f"RF-DETR inference failed: {exc}"
            return []

        try:
            xyxy = np.asarray(result.xyxy)
            raw_conf = getattr(result, "confidence", None)
            raw_cls = getattr(result, "class_id", None)
            conf = np.asarray(raw_conf) if raw_conf is not None else None
            cls = np.asarray(raw_cls) if raw_cls is not None else None
        except Exception as exc:
            self.load_error = f"Unexpected RF-DETR result format: {exc}"
            return []

        detections: List[DetrDetection] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            c = float(conf[i]) if conf is not None else 0.0
            cid = int(cls[i]) if cls is not None else None
            if class_filter and cid is not None and cid not in class_filter:
                continue
            detections.append(DetrDetection(x1, y1, x2, y2, c, cid))
        return detections


# ---------------------------------------------------------------------------
# Process-wide worker cache (same rationale as yolo_worker.get_cached_worker)
# ---------------------------------------------------------------------------

_worker_cache: Dict[Tuple[str, str, str], DetrWorker] = {}


def get_cached_worker(model_path: str, device: str, variant: str = "base") -> DetrWorker:
    key = (str(model_path), str(device), str(variant).strip().lower() or "base")
    worker = _worker_cache.get(key)
    if worker is None:
        worker = DetrWorker(model_path, device, variant)
        _worker_cache[key] = worker
    return worker


def clear_worker_cache() -> None:
    _worker_cache.clear()