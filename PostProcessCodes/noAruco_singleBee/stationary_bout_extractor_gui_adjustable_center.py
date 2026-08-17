"""
stationary_bout_extractor_gui_adjustable_center.py

Standalone GUI for extracting stationary bouts near a stimulus area from RAW
single-bee tracking coordinates.

Key points
----------
- Uses RAW coordinates only, not filtered/Kalman coordinates.
- Raw rows with -1,-1,-1 are treated as non-detections/skipped frames.
- Timing uses the original row/frame index, so frame skipping does not compress time.
- Loads the background image path from *_info.txt when available.
- Shows the stimulus on the background image.
- Lets you adjust the target center by editing X/Y or clicking on the preview.
- The checked area is a user-defined circle around the adjusted center.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from PySide6 import QtCore, QtGui, QtWidgets


Point = Tuple[float, float]


@dataclass
class StimulusArea:
    stim_id: str
    shape: str
    category: str
    geom: object
    center: Point
    original_radius_px: Optional[float] = None


@dataclass
class AnalysisSettings:
    fps: float = 30.0
    min_duration_s: float = 1.0
    target_radius_px: float = 150.0
    stationary_radius_px: float = 10.0
    max_gap_frames: int = 15
    bee_id: str = "ID_1"
    output_path: str = "stationary_bouts.csv"


@dataclass
class StationaryBout:
    stimulus_id: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    point_count: int
    anchor_x: float
    anchor_y: float
    mean_x: float
    mean_y: float
    max_anchor_distance_px: float
    target_center_x: float
    target_center_y: float
    target_radius_px: float
    stationary_radius_px: float

    def as_row(self) -> Dict[str, object]:
        return {
            "stimulus_id": self.stimulus_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": f"{self.start_time_s:.6f}",
            "end_time_s": f"{self.end_time_s:.6f}",
            "duration_s": f"{self.duration_s:.6f}",
            "start_time_hms": seconds_to_hms(self.start_time_s),
            "end_time_hms": seconds_to_hms(self.end_time_s),
            "point_count": self.point_count,
            "anchor_x": f"{self.anchor_x:.3f}",
            "anchor_y": f"{self.anchor_y:.3f}",
            "mean_x": f"{self.mean_x:.3f}",
            "mean_y": f"{self.mean_y:.3f}",
            "max_anchor_distance_px": f"{self.max_anchor_distance_px:.3f}",
            "target_center_x": f"{self.target_center_x:.3f}",
            "target_center_y": f"{self.target_center_y:.3f}",
            "target_radius_px": f"{self.target_radius_px:.3f}",
            "stationary_radius_px": f"{self.stationary_radius_px:.3f}",
        }


class OpenBoutState:
    def __init__(self, stimulus: StimulusArea, settings: AnalysisSettings):
        self.stimulus = stimulus
        self.settings = settings
        self.active = False
        self.start_frame = -1
        self.end_frame = -1
        self.last_valid_frame = -1
        self.anchor_x = math.nan
        self.anchor_y = math.nan
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.count = 0
        self.max_anchor_distance_px = 0.0

    def start(self, frame_index: int, x: float, y: float) -> None:
        self.active = True
        self.start_frame = int(frame_index)
        self.end_frame = int(frame_index)
        self.last_valid_frame = int(frame_index)
        self.anchor_x = float(x)
        self.anchor_y = float(y)
        self.sum_x = float(x)
        self.sum_y = float(y)
        self.count = 1
        self.max_anchor_distance_px = 0.0

    def add(self, frame_index: int, x: float, y: float) -> None:
        if not self.active:
            self.start(frame_index, x, y)
            return
        d = math.hypot(float(x) - self.anchor_x, float(y) - self.anchor_y)
        self.end_frame = int(frame_index)
        self.last_valid_frame = int(frame_index)
        self.sum_x += float(x)
        self.sum_y += float(y)
        self.count += 1
        self.max_anchor_distance_px = max(self.max_anchor_distance_px, d)

    def too_large_gap(self, frame_index: int) -> bool:
        if not self.active:
            return False
        return int(frame_index) - self.last_valid_frame > int(self.settings.max_gap_frames)

    def point_fits_stationary_anchor(self, x: float, y: float) -> bool:
        if not self.active:
            return True
        return math.hypot(float(x) - self.anchor_x, float(y) - self.anchor_y) <= float(self.settings.stationary_radius_px)

    def finalize(self) -> Optional[StationaryBout]:
        if not self.active or self.count <= 0:
            self.reset()
            return None

        fps = max(float(self.settings.fps), 1e-9)
        duration_s = (self.end_frame - self.start_frame) / fps
        bout = None
        if duration_s >= float(self.settings.min_duration_s):
            bout = StationaryBout(
                stimulus_id=self.stimulus.stim_id,
                start_frame=self.start_frame,
                end_frame=self.end_frame,
                start_time_s=self.start_frame / fps,
                end_time_s=self.end_frame / fps,
                duration_s=duration_s,
                point_count=self.count,
                anchor_x=self.anchor_x,
                anchor_y=self.anchor_y,
                mean_x=self.sum_x / self.count,
                mean_y=self.sum_y / self.count,
                max_anchor_distance_px=self.max_anchor_distance_px,
                target_center_x=self.stimulus.center[0],
                target_center_y=self.stimulus.center[1],
                target_radius_px=self.settings.target_radius_px,
                stationary_radius_px=self.settings.stationary_radius_px,
            )
        self.reset()
        return bout

    def reset(self) -> None:
        self.active = False
        self.start_frame = -1
        self.end_frame = -1
        self.last_valid_frame = -1
        self.anchor_x = math.nan
        self.anchor_y = math.nan
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.count = 0
        self.max_anchor_distance_px = 0.0


def seconds_to_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_geom(geom_str: str):
    numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", geom_str or "")]
    if len(numbers) == 3:
        return [int(numbers[0]), int(numbers[1]), int(numbers[2])]
    if len(numbers) == 4:
        return [int(numbers[0]), int(numbers[1]), int(numbers[2]), int(numbers[3])]
    if len(numbers) > 4 and len(numbers) % 2 == 0:
        return [(int(numbers[i]), int(numbers[i + 1])) for i in range(0, len(numbers), 2)]
    return None


def shape_center_and_radius(shape: str, geom) -> Tuple[Point, Optional[float]]:
    if not geom:
        return (math.nan, math.nan), None
    if shape == "circle" and len(geom) >= 3:
        return (float(geom[0]), float(geom[1])), float(geom[2])
    if shape == "rect" and len(geom) >= 4:
        return (float(geom[0] + geom[2] / 2.0), float(geom[1] + geom[3] / 2.0)), None
    if shape == "poly" and len(geom) >= 3:
        xs = [float(p[0]) for p in geom]
        ys = [float(p[1]) for p in geom]
        return (sum(xs) / len(xs), sum(ys) / len(ys)), None
    return (math.nan, math.nan), None


def _parse_scalar(value: str):
    value = value.strip()
    if value == "":
        return ""
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return value


def parse_info_file(path: str | Path) -> Dict[str, object]:
    info: Dict[str, object] = {"stimuli": [], "arenas": []}
    current_section = None
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                label = line.lower()
                if "stimulus areas" in label:
                    current_section = "stimuli"
                elif "arenas" in label:
                    current_section = "arenas"
                else:
                    current_section = None
                continue

            parts = raw_line.rstrip("\n").split("\t")
            if current_section in {"stimuli", "arenas"}:
                if len(parts) >= 4 and parts[0] != "id":
                    sid, shape, category, geom_text = parts[0], parts[1], parts[2], parts[3]
                    geom = parse_geom(geom_text)
                    center, radius = shape_center_and_radius(shape, geom)
                    area = StimulusArea(
                        stim_id=sid,
                        shape=shape,
                        category=category,
                        geom=geom,
                        center=center,
                        original_radius_px=radius,
                    )
                    if current_section == "stimuli":
                        info["stimuli"].append(area)
                    else:
                        info["arenas"].append(area)
                continue

            if len(parts) >= 2:
                info[parts[0].strip()] = _parse_scalar("\t".join(parts[1:]))

    if "fps" not in info:
        info["fps"] = 30.0
    return info


def detect_bee_ids_from_header(raw_path: str | Path) -> List[str]:
    with Path(raw_path).open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
    ids = []
    for col in header:
        if col.endswith("_X"):
            ids.append(col[:-2])
    return ids


def coordinate_indices(header: List[str], bee_id: str) -> Tuple[int, int, int]:
    return header.index(f"{bee_id}_X"), header.index(f"{bee_id}_Y"), header.index(f"{bee_id}_Ang")


def read_raw_point(parts: List[str], x_idx: int, y_idx: int, ang_idx: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        x = float(parts[x_idx])
        y = float(parts[y_idx])
        ang = float(parts[ang_idx])
    except Exception:
        return None, None, None
    if x < 0 or y < 0:
        return None, None, None
    return x, y, ang


def point_inside_circle(x: float, y: float, center: Point, radius: float) -> bool:
    return math.hypot(float(x) - float(center[0]), float(y) - float(center[1])) <= float(radius)


def extract_stationary_bouts(
    raw_path: str | Path,
    stimuli: List[StimulusArea],
    settings: AnalysisSettings,
    progress_callback=None,
) -> List[StationaryBout]:
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not stimuli:
        return []

    file_size = raw_path.stat().st_size
    bouts: List[StationaryBout] = []
    states = [OpenBoutState(stim, settings) for stim in stimuli]

    with raw_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        try:
            x_idx, y_idx, ang_idx = coordinate_indices(header, settings.bee_id)
        except ValueError as exc:
            raise ValueError(f"Could not find {settings.bee_id}_X/Y/Ang columns in raw file.") from exc

        last_percent = -1
        for frame_idx, raw_line in enumerate(f):
            parts = raw_line.strip().split("\t")
            if len(parts) <= max(x_idx, y_idx, ang_idx):
                x = y = None
            else:
                x, y, _ang = read_raw_point(parts, x_idx, y_idx, ang_idx)

            if x is None or y is None:
                for state in states:
                    if state.too_large_gap(frame_idx):
                        bout = state.finalize()
                        if bout is not None:
                            bouts.append(bout)
                continue

            for state in states:
                stim = state.stimulus
                if not point_inside_circle(x, y, stim.center, settings.target_radius_px):
                    bout = state.finalize()
                    if bout is not None:
                        bouts.append(bout)
                    continue

                if state.too_large_gap(frame_idx):
                    bout = state.finalize()
                    if bout is not None:
                        bouts.append(bout)
                    state.start(frame_idx, x, y)
                    continue

                if not state.active:
                    state.start(frame_idx, x, y)
                elif state.point_fits_stationary_anchor(x, y):
                    state.add(frame_idx, x, y)
                else:
                    bout = state.finalize()
                    if bout is not None:
                        bouts.append(bout)
                    state.start(frame_idx, x, y)

            if progress_callback is not None and frame_idx % 5000 == 0:
                pos = f.tell()
                percent = int(100 * pos / max(file_size, 1))
                if percent != last_percent:
                    progress_callback(percent)
                    last_percent = percent

    for state in states:
        bout = state.finalize()
        if bout is not None:
            bouts.append(bout)
    if progress_callback is not None:
        progress_callback(100)
    return bouts


def save_bouts_csv(path: str | Path, bouts: List[StationaryBout]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stimulus_id",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "start_time_hms",
        "end_time_hms",
        "point_count",
        "anchor_x",
        "anchor_y",
        "mean_x",
        "mean_y",
        "max_anchor_distance_px",
        "target_center_x",
        "target_center_y",
        "target_radius_px",
        "stationary_radius_px",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for b in bouts:
            writer.writerow(b.as_row())


class PreviewLabel(QtWidgets.QLabel):
    center_clicked = QtCore.Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 420)
        self.setStyleSheet("QLabel { background: #111; border: 1px solid #444; color: #DDD; }")
        self.image_bgr = None
        self.original_stimulus: Optional[StimulusArea] = None
        self.adjusted_center: Optional[Point] = None
        self.target_radius_px = 0.0
        self._scaled_size: Optional[Tuple[int, int]] = None
        self._offset: Tuple[float, float] = (0.0, 0.0)

    def set_data(self, image_bgr, original_stimulus: Optional[StimulusArea], adjusted_center: Optional[Point], target_radius_px: float):
        self.image_bgr = image_bgr
        self.original_stimulus = original_stimulus
        self.adjusted_center = adjusted_center
        self.target_radius_px = float(target_radius_px)
        self.update_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_pixmap()

    def mousePressEvent(self, event):
        if self.image_bgr is None or self._scaled_size is None:
            return
        sw, sh = self._scaled_size
        ox, oy = self._offset
        px = float(event.position().x())
        py = float(event.position().y())
        if px < ox or py < oy or px > ox + sw or py > oy + sh:
            return
        h, w = self.image_bgr.shape[:2]
        x = (px - ox) * (w / max(sw, 1))
        y = (py - oy) * (h / max(sh, 1))
        self.center_clicked.emit(float(x), float(y))

    def update_pixmap(self):
        if self.image_bgr is None:
            self.setText("No background image loaded")
            self.setPixmap(QtGui.QPixmap())
            return

        img = self.image_bgr.copy()
        if self.original_stimulus is not None:
            ox, oy = self.original_stimulus.center
            cv2.drawMarker(img, (int(round(ox)), int(round(oy))), (255, 0, 0), cv2.MARKER_CROSS, 18, 2)
            if self.original_stimulus.original_radius_px is not None:
                cv2.circle(
                    img,
                    (int(round(ox)), int(round(oy))),
                    int(round(self.original_stimulus.original_radius_px)),
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    img,
                    f"original {self.original_stimulus.id if hasattr(self.original_stimulus, 'id') else self.original_stimulus.stim_id}",
                    (int(round(ox)) + 8, max(20, int(round(oy)) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if self.adjusted_center is not None:
            cx, cy = self.adjusted_center
            cv2.drawMarker(img, (int(round(cx)), int(round(cy))), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            if self.target_radius_px > 0:
                cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(self.target_radius_px)), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"target center=({cx:.1f},{cy:.1f}) r={self.target_radius_px:.1f}",
                (int(round(cx)) + 8, min(img.shape[0] - 10, int(round(cy)) + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self._scaled_size = (scaled.width(), scaled.height())
        self._offset = ((self.width() - scaled.width()) / 2.0, (self.height() - scaled.height()) / 2.0)
        self.setPixmap(scaled)


class Worker(QtCore.QThread):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal(object, object)

    def __init__(self, raw_path: str, stimuli: List[StimulusArea], settings: AnalysisSettings):
        super().__init__()
        self.raw_path = raw_path
        self.stimuli = stimuli
        self.settings = settings

    def run(self):
        try:
            bouts = extract_stationary_bouts(self.raw_path, self.stimuli, self.settings, progress_callback=self.progress.emit)
            if self.settings.output_path:
                save_bouts_csv(self.settings.output_path, bouts)
            self.finished.emit(bouts, None)
        except Exception as exc:
            self.finished.emit(None, exc)


class StationaryBoutExtractorWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raw Coordinate Stationary Bout Extractor")
        self.resize(1250, 760)
        self.info_data: Dict[str, object] = {}
        self.stimuli: List[StimulusArea] = []
        self.background_image = None
        self.worker: Optional[Worker] = None
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)

        left = QtWidgets.QVBoxLayout()
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(470)
        root.addWidget(left_widget, 0)

        files_group = QtWidgets.QGroupBox("Input / Output")
        files_layout = QtWidgets.QFormLayout(files_group)
        self.edit_info = QtWidgets.QLineEdit()
        self.btn_info = QtWidgets.QPushButton("Browse...")
        self.edit_raw = QtWidgets.QLineEdit()
        self.btn_raw = QtWidgets.QPushButton("Browse...")
        self.edit_bg = QtWidgets.QLineEdit()
        self.btn_bg = QtWidgets.QPushButton("Browse...")
        self.edit_out = QtWidgets.QLineEdit()
        self.btn_out = QtWidgets.QPushButton("Browse...")

        row = QtWidgets.QHBoxLayout(); row.addWidget(self.edit_info); row.addWidget(self.btn_info)
        files_layout.addRow("Info TXT:", row)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.edit_raw); row.addWidget(self.btn_raw)
        files_layout.addRow("Raw coordinates TXT:", row)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.edit_bg); row.addWidget(self.btn_bg)
        files_layout.addRow("Background image:", row)
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.edit_out); row.addWidget(self.btn_out)
        files_layout.addRow("Output CSV:", row)
        left.addWidget(files_group)

        settings_group = QtWidgets.QGroupBox("Analysis Settings")
        settings_layout = QtWidgets.QFormLayout(settings_group)

        self.combo_bee = QtWidgets.QComboBox()
        self.combo_stim = QtWidgets.QComboBox()
        self.spin_fps = QtWidgets.QDoubleSpinBox(minimum=0.001, maximum=1000.0, value=30.0, decimals=4)
        self.spin_min_dur = QtWidgets.QDoubleSpinBox(minimum=0.01, maximum=3600.0, value=1.0, decimals=3)
        self.spin_target_radius = QtWidgets.QDoubleSpinBox(minimum=1.0, maximum=100000.0, value=150.0, decimals=2)
        self.spin_stationary_radius = QtWidgets.QDoubleSpinBox(minimum=0.1, maximum=100000.0, value=10.0, decimals=2)
        self.spin_max_gap = QtWidgets.QSpinBox(minimum=1, maximum=100000, value=15)
        self.spin_center_x = QtWidgets.QDoubleSpinBox(minimum=0.0, maximum=100000.0, value=0.0, decimals=1)
        self.spin_center_y = QtWidgets.QDoubleSpinBox(minimum=0.0, maximum=100000.0, value=0.0, decimals=1)
        self.spin_center_x.setSingleStep(1.0)
        self.spin_center_y.setSingleStep(1.0)
        self.btn_reset_center = QtWidgets.QPushButton("Reset center to original")

        center_row = QtWidgets.QHBoxLayout()
        center_row.addWidget(QtWidgets.QLabel("X:")); center_row.addWidget(self.spin_center_x)
        center_row.addWidget(QtWidgets.QLabel("Y:")); center_row.addWidget(self.spin_center_y)

        settings_layout.addRow("Bee ID:", self.combo_bee)
        settings_layout.addRow("Stimulus:", self.combo_stim)
        settings_layout.addRow("FPS:", self.spin_fps)
        settings_layout.addRow("Adjusted target center (px):", center_row)
        settings_layout.addRow("", self.btn_reset_center)
        settings_layout.addRow("Minimum still duration (s):", self.spin_min_dur)
        settings_layout.addRow("Target circle radius (px):", self.spin_target_radius)
        settings_layout.addRow("Stationary movement radius (px):", self.spin_stationary_radius)
        settings_layout.addRow("Max valid-detection gap (frames):", self.spin_max_gap)
        left.addWidget(settings_group)

        self.lbl_stim = QtWidgets.QLabel("No info file loaded.")
        self.lbl_stim.setWordWrap(True)
        left.addWidget(self.lbl_stim)

        buttons = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Extract Stationary Bouts")
        self.progress = QtWidgets.QProgressBar()
        buttons.addWidget(self.btn_run)
        buttons.addWidget(self.progress, 1)
        left.addLayout(buttons)

        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Stimulus", "Start", "End", "Duration (s)", "Frames", "Mean X", "Mean Y", "Max move px"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        left.addWidget(self.table, 1)

        self.lbl_status = QtWidgets.QLabel("")
        left.addWidget(self.lbl_status)

        right = QtWidgets.QVBoxLayout()
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right)
        root.addWidget(right_widget, 1)

        self.preview = PreviewLabel()
        right.addWidget(self.preview, 1)
        note = QtWidgets.QLabel(
            "Yellow = original stimulus circle. Blue cross = original center.\n"
            "Green = adjusted target circle. Red cross = adjusted target center.\n"
            "Click on the background preview to set the adjusted center."
        )
        note.setWordWrap(True)
        right.addWidget(note)

        self.btn_info.clicked.connect(self.browse_info)
        self.btn_raw.clicked.connect(self.browse_raw)
        self.btn_bg.clicked.connect(self.browse_bg)
        self.btn_out.clicked.connect(self.browse_out)
        self.btn_run.clicked.connect(self.run_analysis)
        self.combo_stim.currentIndexChanged.connect(self.stimulus_changed)
        self.spin_center_x.valueChanged.connect(self.update_preview)
        self.spin_center_y.valueChanged.connect(self.update_preview)
        self.spin_target_radius.valueChanged.connect(self.update_preview)
        self.btn_reset_center.clicked.connect(self.reset_center_to_original)
        self.preview.center_clicked.connect(self.set_center_from_preview_click)

    def browse_info(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open info TXT", "", "Text files (*.txt);;All files (*)")
        if not path:
            return
        self.edit_info.setText(path)
        self.load_info(path)
        self._suggest_output_path()

    def browse_raw(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open raw coordinates TXT", "", "Text files (*.txt);;All files (*)")
        if not path:
            return
        self.edit_raw.setText(path)
        self.load_raw_header(path)
        self._suggest_output_path()

    def browse_bg(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open background image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)")
        if not path:
            return
        self.edit_bg.setText(path)
        self.load_background(path)

    def browse_out(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save output CSV", "", "CSV files (*.csv);;All files (*)")
        if path:
            if not path.lower().endswith(".csv"):
                path += ".csv"
            self.edit_out.setText(path)

    def load_background(self, path: str):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Background Error", f"Could not open image:\n{path}")
            return
        self.background_image = img
        self.update_preview()

    def load_info(self, path: str):
        try:
            self.info_data = parse_info_file(path)
            self.stimuli = list(self.info_data.get("stimuli", []))
            fps = float(self.info_data.get("fps", 30.0))
            self.spin_fps.setValue(fps)
            n = int(self.info_data.get("detect_every_n_frames", 1))
            self.spin_max_gap.setValue(max(2 * n + 1, n + 2, 3))

            self.combo_stim.blockSignals(True)
            self.combo_stim.clear()
            for s in self.stimuli:
                self.combo_stim.addItem(f"{s.stim_id} center=({s.center[0]:.1f},{s.center[1]:.1f})")
            self.combo_stim.blockSignals(False)

            text = [f"FPS: {fps:.4f}", f"detect_every_n_frames: {n}"]
            if self.stimuli:
                text.append("Stimuli: " + ", ".join(
                    f"{s.stim_id} center=({s.center[0]:.1f},{s.center[1]:.1f})" for s in self.stimuli
                ))
                self.combo_stim.setCurrentIndex(0)
                self.reset_center_to_original(update_only=True)
                if self.stimuli[0].original_radius_px is not None:
                    self.spin_target_radius.setValue(float(self.stimuli[0].original_radius_px))
            else:
                text.append("No stimulus areas found in info file.")
            self.lbl_stim.setText(" | ".join(text))

            bg_path = str(self.info_data.get("background_path", "")).strip()
            if bg_path and Path(bg_path).exists():
                self.edit_bg.setText(bg_path)
                self.load_background(bg_path)
            else:
                self.update_preview()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Info Error", str(exc))

    def load_raw_header(self, path: str):
        try:
            ids = detect_bee_ids_from_header(path)
            self.combo_bee.clear()
            self.combo_bee.addItems(ids or ["ID_1"])
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Raw File Error", str(exc))

    def _suggest_output_path(self):
        if self.edit_out.text().strip():
            return
        raw = self.edit_raw.text().strip()
        if raw:
            p = Path(raw)
            suggested = p.with_name(p.stem.replace("_coordinates_raw", "") + "_stationary_bouts.csv")
            self.edit_out.setText(str(suggested))

    def selected_original_stimulus(self) -> Optional[StimulusArea]:
        idx = self.combo_stim.currentIndex()
        if idx < 0 or idx >= len(self.stimuli):
            return None
        return self.stimuli[idx]

    def selected_adjusted_stimulus(self) -> Optional[StimulusArea]:
        s = self.selected_original_stimulus()
        if s is None:
            return None
        return StimulusArea(
            stim_id=s.stim_id,
            shape=s.shape,
            category=s.category,
            geom=s.geom,
            center=(float(self.spin_center_x.value()), float(self.spin_center_y.value())),
            original_radius_px=s.original_radius_px,
        )

    def stimulus_changed(self):
        s = self.selected_original_stimulus()
        if s is None:
            self.update_preview()
            return
        self.reset_center_to_original(update_only=True)
        if s.original_radius_px is not None:
            self.spin_target_radius.setValue(float(s.original_radius_px))
        self.update_preview()

    def reset_center_to_original(self, update_only: bool = False):
        s = self.selected_original_stimulus()
        if s is None:
            return
        self.spin_center_x.blockSignals(True)
        self.spin_center_y.blockSignals(True)
        self.spin_center_x.setValue(float(s.center[0]))
        self.spin_center_y.setValue(float(s.center[1]))
        self.spin_center_x.blockSignals(False)
        self.spin_center_y.blockSignals(False)
        if not update_only:
            self.update_preview()

    def set_center_from_preview_click(self, x: float, y: float):
        self.spin_center_x.setValue(float(x))
        self.spin_center_y.setValue(float(y))
        self.lbl_status.setText(f"Adjusted center set to ({x:.1f}, {y:.1f})")

    def update_preview(self):
        self.preview.set_data(
            self.background_image,
            self.selected_original_stimulus(),
            (float(self.spin_center_x.value()), float(self.spin_center_y.value())),
            float(self.spin_target_radius.value()),
        )

    def run_analysis(self):
        raw_path = self.edit_raw.text().strip()
        info_path = self.edit_info.text().strip()
        out_path = self.edit_out.text().strip()
        if not raw_path or not Path(raw_path).exists():
            QtWidgets.QMessageBox.warning(self, "Missing Input", "Select the raw coordinate TXT file.")
            return
        if not info_path or not Path(info_path).exists():
            QtWidgets.QMessageBox.warning(self, "Missing Input", "Select the info TXT file.")
            return
        if not self.stimuli:
            self.load_info(info_path)
        adjusted = self.selected_adjusted_stimulus()
        if adjusted is None:
            QtWidgets.QMessageBox.warning(self, "No Stimulus", "No stimulus area was found/selected.")
            return
        if not out_path:
            self._suggest_output_path()
            out_path = self.edit_out.text().strip()

        settings = AnalysisSettings(
            fps=float(self.spin_fps.value()),
            min_duration_s=float(self.spin_min_dur.value()),
            target_radius_px=float(self.spin_target_radius.value()),
            stationary_radius_px=float(self.spin_stationary_radius.value()),
            max_gap_frames=int(self.spin_max_gap.value()),
            bee_id=self.combo_bee.currentText() or "ID_1",
            output_path=out_path,
        )
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_status.setText(
            f"Running with adjusted center ({adjusted.center[0]:.1f}, {adjusted.center[1]:.1f})..."
        )
        self.worker = Worker(raw_path, [adjusted], settings)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.start()

    def analysis_finished(self, bouts, error):
        self.btn_run.setEnabled(True)
        if error is not None:
            QtWidgets.QMessageBox.critical(self, "Analysis Error", str(error))
            self.lbl_status.setText("Failed.")
            return
        self.populate_table(bouts)
        self.lbl_status.setText(f"Found {len(bouts)} stationary bout(s). Saved to: {self.edit_out.text().strip()}")

    def populate_table(self, bouts: List[StationaryBout]):
        self.table.setRowCount(0)
        for b in bouts:
            r = self.table.rowCount()
            self.table.insertRow(r)
            values = [
                b.stimulus_id,
                seconds_to_hms(b.start_time_s),
                seconds_to_hms(b.end_time_s),
                f"{b.duration_s:.3f}",
                f"{b.start_frame}-{b.end_frame}",
                f"{b.mean_x:.1f}",
                f"{b.mean_y:.1f}",
                f"{b.max_anchor_distance_px:.2f}",
            ]
            for c, v in enumerate(values):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        self.table.resizeColumnsToContents()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = StationaryBoutExtractorWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
