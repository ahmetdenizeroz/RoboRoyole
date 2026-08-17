
"""
single_bee_detector_tuner_gui.py

Visual tuner for the shared SingleBeeDetector core.

New speed-tuning additions
--------------------------
- A single large image view controlled by a dropdown.
- detection_scale setting: test 1.0, 0.75, 0.5, 0.33, 0.25, etc.
- refine_on_original setting: low-res detection + original-res selected-candidate refinement.
- crop_to_arena_bbox setting: process only arena bounding box.
- Saved settings are consumed by the final tracking GUI/core.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from single_bee_detector_core import (
    AdaptiveRollingBackground,
    BlobFilterSettings,
    CandidateSelectionSettings,
    DebugSettings,
    KalmanSettings,
    MorphologySettings,
    PreprocessSettings,
    SingleBeeDetector,
    SingleBeeDetectorSettings,
    ThresholdSettings,
    ArucoSettings,
    default_settings,
    draw_detection_overlay,
    extract_channel,
    imread_unicode,
    load_settings_txt,
    mask_from_rect,
    save_settings_txt,
    summarize_result,
)


Rect = Tuple[int, int, int, int]


class ImageLabel(QtWidgets.QLabel):
    rect_drawn = QtCore.Signal(tuple)

    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.setMinimumSize(760, 520)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #171717; color: #dddddd; border: 1px solid #444;")
        self.setText(title)
        self.setMouseTracking(True)

        self._image: Optional[np.ndarray] = None
        self._pixmap_original: Optional[QtGui.QPixmap] = None
        self._draw_rect_enabled = False
        self._dragging = False
        self._drag_start: Optional[QtCore.QPoint] = None
        self._drag_current: Optional[QtCore.QPoint] = None

    def set_draw_rect_enabled(self, enabled: bool) -> None:
        self._draw_rect_enabled = bool(enabled)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor if enabled else QtCore.Qt.CursorShape.ArrowCursor)
        self.update()

    def set_image(self, image: Optional[np.ndarray]) -> None:
        self._image = None if image is None else image.copy()
        if image is None:
            self._pixmap_original = None
            self.clear()
            self.setText(self.title)
            self.update()
            return
        self._pixmap_original = self._to_pixmap(image)
        self._update_scaled_pixmap()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap_original is None:
            return
        scaled = self._pixmap_original.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    @staticmethod
    def _to_pixmap(image: np.ndarray) -> QtGui.QPixmap:
        if image.ndim == 2:
            h, w = image.shape
            q = QtGui.QImage(image.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()
            return QtGui.QPixmap.fromImage(q)
        if image.ndim == 3 and image.shape[2] == 3:
            h, w, ch = image.shape
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888).copy()
            return QtGui.QPixmap.fromImage(q)
        raise ValueError("ImageLabel expects grayscale or BGR image")

    def _pixmap_rect(self) -> Optional[QtCore.QRectF]:
        pix = self.pixmap()
        if pix is None:
            return None
        x0 = (self.width() - pix.width()) / 2.0
        y0 = (self.height() - pix.height()) / 2.0
        return QtCore.QRectF(x0, y0, pix.width(), pix.height())

    def _widget_to_image(self, pos: QtCore.QPoint) -> Optional[Tuple[int, int]]:
        if self._image is None:
            return None
        rect = self._pixmap_rect()
        if rect is None or not rect.contains(QtCore.QPointF(pos)):
            return None
        h, w = self._image.shape[:2]
        x = (pos.x() - rect.x()) / max(1.0, rect.width()) * w
        y = (pos.y() - rect.y()) / max(1.0, rect.height()) * h
        return int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._draw_rect_enabled and event.button() == QtCore.Qt.MouseButton.LeftButton:
            if self._widget_to_image(event.position().toPoint()) is not None:
                self._dragging = True
                self._drag_start = event.position().toPoint()
                self._drag_current = event.position().toPoint()
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            self._drag_current = event.position().toPoint()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._dragging = False
            start = self._widget_to_image(self._drag_start) if self._drag_start else None
            end = self._widget_to_image(event.position().toPoint())
            self._drag_start = None
            self._drag_current = None
            self.update()
            if start is not None and end is not None:
                x0, y0 = start
                x1, y1 = end
                x, y = min(x0, x1), min(y0, y1)
                w, h = abs(x1 - x0), abs(y1 - y0)
                if w >= 5 and h >= 5:
                    self.rect_drawn.emit((x, y, w, h))
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if not self._dragging or self._drag_start is None or self._drag_current is None:
            return
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 200, 0), 2, QtCore.Qt.PenStyle.DashLine))
        painter.drawRect(QtCore.QRect(self._drag_start, self._drag_current).normalized())


class SingleBeeDetectorTuner(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Single Bee Detector Tuner")
        self.resize(1500, 920)

        self.video_path: Optional[str] = None
        self.background_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_frame_index = 0
        self.current_frame: Optional[np.ndarray] = None
        self.background: Optional[np.ndarray] = None
        self.roi_rect: Optional[Rect] = None
        self.previous_position: Optional[Tuple[float, float]] = None
        self.last_settings_path: Optional[str] = None

        self.settings = default_settings()
        self.detector = SingleBeeDetector(self.settings)
        self._last_result = None
        self._last_images: dict[str, Optional[np.ndarray]] = {}
        self._updating_widgets = False
        self._adaptive_cache_key = None
        self._adaptive_cache_bg = None
        self._adaptive_cache_count = 0

        self.play_timer = QtCore.QTimer(self)
        self.play_timer.timeout.connect(self._play_next_frame)

        self._build_ui()
        self._set_widgets_from_settings(self.settings)
        self._update_ui_enabled_state()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)

        file_row = QtWidgets.QHBoxLayout()
        self.btn_load_video = QtWidgets.QPushButton("Load Video...")
        self.btn_load_background = QtWidgets.QPushButton("Load Static Background...")
        self.btn_load_settings = QtWidgets.QPushButton("Load Settings...")
        self.btn_save_settings = QtWidgets.QPushButton("Save Settings As...")
        self.btn_reset_settings = QtWidgets.QPushButton("Reset Settings")
        self.lbl_files = QtWidgets.QLabel("No video loaded.")
        self.lbl_files.setWordWrap(True)
        for w in [self.btn_load_video, self.btn_load_background, self.btn_load_settings, self.btn_save_settings, self.btn_reset_settings]:
            file_row.addWidget(w)
        file_row.addWidget(self.lbl_files, 1)
        root.addLayout(file_row)

        frame_row = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("◀")
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_next = QtWidgets.QPushButton("▶")
        self.slider_frame = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.spin_frame = QtWidgets.QSpinBox()
        self.lbl_time = QtWidgets.QLabel("Frame: 0 / 0   Time: 0.00 s")
        frame_row.addWidget(self.btn_prev)
        frame_row.addWidget(self.btn_play)
        frame_row.addWidget(self.btn_next)
        frame_row.addWidget(self.slider_frame, 1)
        frame_row.addWidget(QtWidgets.QLabel("Frame:"))
        frame_row.addWidget(self.spin_frame)
        frame_row.addWidget(self.lbl_time)
        root.addLayout(frame_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        roi_row = QtWidgets.QHBoxLayout()
        self.chk_use_roi = QtWidgets.QCheckBox("Use rectangular arena ROI")
        self.btn_draw_roi = QtWidgets.QPushButton("Draw ROI on Main View")
        self.btn_clear_roi = QtWidgets.QPushButton("Clear ROI")
        self.lbl_roi = QtWidgets.QLabel("ROI: full frame")
        roi_row.addWidget(self.chk_use_roi)
        roi_row.addWidget(self.btn_draw_roi)
        roi_row.addWidget(self.btn_clear_roi)
        roi_row.addWidget(self.lbl_roi, 1)
        left_layout.addLayout(roi_row)

        view_row = QtWidgets.QHBoxLayout()
        self.combo_view = QtWidgets.QComboBox()
        self.combo_view.addItems([
            "Original + Selected Overlay",
            "All Candidates / Rejection Reasons",
            "Background Used",
            "Background Difference / Source",
            "Threshold Binary",
            "Cleaned Mask",
            "Channel / Crop Source",
        ])
        view_row.addWidget(QtWidgets.QLabel("View:"))
        view_row.addWidget(self.combo_view, 1)
        left_layout.addLayout(view_row)

        self.main_view = ImageLabel("Selected View")
        left_layout.addWidget(self.main_view, 1)

        self.txt_summary = QtWidgets.QPlainTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMinimumHeight(150)
        left_layout.addWidget(self.txt_summary)

        settings_panel = self._build_settings_panel()
        splitter.addWidget(left)
        splitter.addWidget(settings_panel)
        splitter.setSizes([1030, 470])

        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_load_background.clicked.connect(self.load_background)
        self.btn_load_settings.clicked.connect(self.load_settings)
        self.btn_save_settings.clicked.connect(self.save_settings)
        self.btn_reset_settings.clicked.connect(self.reset_settings)
        self.btn_prev.clicked.connect(lambda: self.seek_frame(self.current_frame_index - 1, reset_previous=True))
        self.btn_next.clicked.connect(lambda: self.seek_frame(self.current_frame_index + 1, reset_previous=False))
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider_frame.valueChanged.connect(self._slider_changed)
        self.spin_frame.valueChanged.connect(self._spin_changed)
        self.chk_use_roi.stateChanged.connect(lambda _: self.update_detection())
        self.btn_draw_roi.clicked.connect(self.toggle_draw_roi)
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.main_view.rect_drawn.connect(self.set_roi_rect)
        self.combo_view.currentTextChanged.connect(lambda _: self._refresh_selected_view())

    def _build_settings_panel(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(container)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)

        inner = QtWidgets.QWidget()
        form_root = QtWidgets.QVBoxLayout(inner)
        scroll.setWidget(inner)

        runtime = QtWidgets.QGroupBox("Tuner Runtime Options")
        runtime_layout = QtWidgets.QVBoxLayout(runtime)
        self.chk_use_previous_position = QtWidgets.QCheckBox("Use previous detected position while playing/stepping")
        self.chk_use_previous_position.setChecked(True)
        self.chk_auto_update_previous = QtWidgets.QCheckBox("Update previous position after each valid detection")
        self.chk_auto_update_previous.setChecked(True)
        self.btn_clear_previous = QtWidgets.QPushButton("Clear Previous Position")
        runtime_layout.addWidget(self.chk_use_previous_position)
        runtime_layout.addWidget(self.chk_auto_update_previous)
        runtime_layout.addWidget(self.btn_clear_previous)
        self.btn_clear_previous.clicked.connect(self.clear_previous_position)
        form_root.addWidget(runtime)

        preprocess = QtWidgets.QGroupBox("Preprocess / Speed")
        pf = QtWidgets.QFormLayout(preprocess)
        self.combo_background_mode = QtWidgets.QComboBox(); self.combo_background_mode.addItems(["adaptive", "static", "none"])
        self.spin_adaptive_history = QtWidgets.QSpinBox(); self.spin_adaptive_history.setRange(1, 500)
        self.combo_adaptive_method = QtWidgets.QComboBox(); self.combo_adaptive_method.addItems(["median", "average"])
        self.combo_channel = QtWidgets.QComboBox(); self.combo_channel.addItems(["red", "green", "blue", "grayscale", "hsv_value", "hsv_saturation"])
        self.combo_difference = QtWidgets.QComboBox(); self.combo_difference.addItems(["absdiff", "none"])
        self.spin_blur = QtWidgets.QSpinBox(); self.spin_blur.setRange(0, 99)
        self.double_detection_scale = QtWidgets.QDoubleSpinBox(); self.double_detection_scale.setRange(0.05, 1.0); self.double_detection_scale.setDecimals(3); self.double_detection_scale.setSingleStep(0.05)
        self.chk_refine_on_original = QtWidgets.QCheckBox("refine_on_original")
        self.chk_crop_to_arena_bbox = QtWidgets.QCheckBox("crop_to_arena_bbox")
        self.spin_refine_padding = QtWidgets.QSpinBox(); self.spin_refine_padding.setRange(0, 500)
        pf.addRow("background_mode", self.combo_background_mode)
        pf.addRow("adaptive_history_length", self.spin_adaptive_history)
        pf.addRow("adaptive_background_method", self.combo_adaptive_method)
        pf.addRow("channel", self.combo_channel)
        pf.addRow("difference_mode", self.combo_difference)
        pf.addRow("blur_kernel", self.spin_blur)
        pf.addRow("detection_scale", self.double_detection_scale)
        pf.addRow("", self.chk_refine_on_original)
        pf.addRow("", self.chk_crop_to_arena_bbox)
        pf.addRow("refine_padding_px", self.spin_refine_padding)
        form_root.addWidget(preprocess)

        threshold = QtWidgets.QGroupBox("Threshold")
        tf = QtWidgets.QFormLayout(threshold)
        self.combo_threshold_mode = QtWidgets.QComboBox(); self.combo_threshold_mode.addItems(["otsu", "manual", "adaptive"])
        self.slider_manual_threshold = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.slider_manual_threshold.setRange(0, 255)
        self.spin_manual_threshold = QtWidgets.QSpinBox(); self.spin_manual_threshold.setRange(0, 255)
        thr_row = QtWidgets.QHBoxLayout(); thr_row.addWidget(self.slider_manual_threshold, 1); thr_row.addWidget(self.spin_manual_threshold)
        self.spin_adaptive_block = QtWidgets.QSpinBox(); self.spin_adaptive_block.setRange(3, 501)
        self.double_adaptive_c = self._double_spin(-100.0, 100.0, 2, 0.5)
        self.chk_invert_binary = QtWidgets.QCheckBox("invert_binary")
        tf.addRow("mode", self.combo_threshold_mode)
        tf.addRow("manual_threshold", thr_row)
        tf.addRow("adaptive_block_size", self.spin_adaptive_block)
        tf.addRow("adaptive_c", self.double_adaptive_c)
        tf.addRow("", self.chk_invert_binary)
        form_root.addWidget(threshold)

        morph = QtWidgets.QGroupBox("Morphology")
        mf = QtWidgets.QFormLayout(morph)
        self.combo_kernel_shape = QtWidgets.QComboBox(); self.combo_kernel_shape.addItems(["ellipse", "rect", "cross"])
        self.spin_open_kernel = QtWidgets.QSpinBox(); self.spin_open_kernel.setRange(0, 99)
        self.spin_close_kernel = QtWidgets.QSpinBox(); self.spin_close_kernel.setRange(0, 99)
        self.spin_erode_kernel = QtWidgets.QSpinBox(); self.spin_erode_kernel.setRange(0, 99)
        self.spin_erode_iter = QtWidgets.QSpinBox(); self.spin_erode_iter.setRange(0, 20)
        self.spin_dilate_kernel = QtWidgets.QSpinBox(); self.spin_dilate_kernel.setRange(0, 99)
        self.spin_dilate_iter = QtWidgets.QSpinBox(); self.spin_dilate_iter.setRange(0, 20)
        for label, widget in [
            ("kernel_shape", self.combo_kernel_shape),
            ("open_kernel", self.spin_open_kernel),
            ("close_kernel", self.spin_close_kernel),
            ("erode_kernel", self.spin_erode_kernel),
            ("erode_iterations", self.spin_erode_iter),
            ("dilate_kernel", self.spin_dilate_kernel),
            ("dilate_iterations", self.spin_dilate_iter),
        ]:
            mf.addRow(label, widget)
        form_root.addWidget(morph)

        blob = QtWidgets.QGroupBox("Blob Filter")
        bf = QtWidgets.QFormLayout(blob)
        self.double_min_area = self._double_spin(0, 1_000_000, 2, 10)
        self.double_max_area = self._double_spin(0, 1_000_000, 2, 100)
        self.double_min_solidity = self._double_spin(0, 1, 4, 0.01)
        self.double_min_extent = self._double_spin(0, 1, 4, 0.01)
        self.double_max_aspect = self._double_spin(0, 100, 3, 0.1)
        self.double_min_width = self._double_spin(0, 10000, 2, 1)
        self.double_max_width = self._double_spin(0, 10000, 2, 10)
        self.double_min_height = self._double_spin(0, 10000, 2, 1)
        self.double_max_height = self._double_spin(0, 10000, 2, 10)
        self.chk_reject_border = QtWidgets.QCheckBox("reject_touching_frame_border")
        for label, widget in [
            ("min_area", self.double_min_area),
            ("max_area", self.double_max_area),
            ("min_solidity", self.double_min_solidity),
            ("min_extent", self.double_min_extent),
            ("max_aspect_ratio", self.double_max_aspect),
            ("min_width", self.double_min_width),
            ("max_width", self.double_max_width),
            ("min_height", self.double_min_height),
            ("max_height", self.double_max_height),
        ]:
            bf.addRow(label, widget)
        bf.addRow("", self.chk_reject_border)
        form_root.addWidget(blob)

        select = QtWidgets.QGroupBox("Candidate Selection")
        sf = QtWidgets.QFormLayout(select)
        self.combo_selection_method = QtWidgets.QComboBox(); self.combo_selection_method.addItems(["largest_area", "closest_to_previous", "score"])
        self.spin_max_bees = QtWidgets.QSpinBox(); self.spin_max_bees.setRange(1, 1000)
        self.double_max_jump = self._double_spin(0, 100000, 2, 10)
        self.chk_allow_largest_if_no_previous = QtWidgets.QCheckBox("allow_largest_if_no_previous")
        self.double_area_weight = self._double_spin(0, 100, 3, 0.1)
        self.double_solidity_weight = self._double_spin(0, 100, 3, 0.1)
        self.double_distance_weight = self._double_spin(0, 100, 3, 0.1)
        sf.addRow("method", self.combo_selection_method)
        sf.addRow("max_bees", self.spin_max_bees)
        sf.addRow("max_jump_px", self.double_max_jump)
        sf.addRow("", self.chk_allow_largest_if_no_previous)
        sf.addRow("area_weight", self.double_area_weight)
        sf.addRow("solidity_weight", self.double_solidity_weight)
        sf.addRow("distance_weight", self.double_distance_weight)
        form_root.addWidget(select)

        kalman = QtWidgets.QGroupBox("Kalman Settings Saved for Tracker")
        kf = QtWidgets.QFormLayout(kalman)
        self.chk_kalman_enabled = QtWidgets.QCheckBox("enabled")
        self.spin_max_missed = QtWidgets.QSpinBox(); self.spin_max_missed.setRange(0, 10000)
        self.double_process_noise = self._double_spin(0, 10000, 4, 0.1)
        self.double_measurement_noise = self._double_spin(0, 10000, 4, 0.1)
        kf.addRow("", self.chk_kalman_enabled)
        kf.addRow("max_missed_frames", self.spin_max_missed)
        kf.addRow("process_noise", self.double_process_noise)
        kf.addRow("measurement_noise", self.double_measurement_noise)
        form_root.addWidget(kalman)
        
        aruco = QtWidgets.QGroupBox("ArUco / ID Tracking")
        af = QtWidgets.QFormLayout(aruco)
        self.chk_aruco_enabled = QtWidgets.QCheckBox("enabled")
        
        self.combo_aruco_detect_mode = QtWidgets.QComboBox()
        self.combo_aruco_detect_mode.addItems(["arena", "full_frame"])

        dict_layout = QtWidgets.QHBoxLayout()
        self.combo_aruco_dict = QtWidgets.QComboBox()

        import glob
        script_dir = Path(__file__).parent
        txt_files = [Path(p).name for p in glob.glob(str(script_dir / "*.txt"))]
        if "single_bee_settings.txt" in txt_files:
            txt_files.remove("single_bee_settings.txt")
        if "single_bee_settings_v1.txt" in txt_files:
            txt_files.remove("single_bee_settings_v1.txt")
        self.combo_aruco_dict.addItems(txt_files)
        
        self.btn_browse_dict = QtWidgets.QPushButton("Browse...")
        self.btn_browse_dict.clicked.connect(self._browse_dict)
        dict_layout.addWidget(self.combo_aruco_dict, 1)
        dict_layout.addWidget(self.btn_browse_dict)

        self.spin_max_hamming = QtWidgets.QSpinBox(); self.spin_max_hamming.setRange(0, 5)
        self.spin_max_border_errors = QtWidgets.QSpinBox(); self.spin_max_border_errors.setRange(0, 10)
        self.spin_crop_padding = QtWidgets.QSpinBox(); self.spin_crop_padding.setRange(0, 100)

        af.addRow("", self.chk_aruco_enabled)
        af.addRow("detect_mode", self.combo_aruco_detect_mode)
        af.addRow("dict_path", dict_layout)
        af.addRow("max_hamming", self.spin_max_hamming)
        af.addRow("max_border_errors", self.spin_max_border_errors)
        af.addRow("crop_padding", self.spin_crop_padding)
        form_root.addWidget(aruco)

        debug = QtWidgets.QGroupBox("Debug Drawing")
        df = QtWidgets.QVBoxLayout(debug)
        self.chk_draw_rejected = QtWidgets.QCheckBox("draw_rejected_candidates")
        self.chk_draw_labels = QtWidgets.QCheckBox("draw_candidate_labels")
        self.chk_draw_axes = QtWidgets.QCheckBox("draw_axes")
        df.addWidget(self.chk_draw_rejected); df.addWidget(self.chk_draw_labels); df.addWidget(self.chk_draw_axes)
        form_root.addWidget(debug)

        form_root.addStretch(1)

        # Connect controls. No tuple-based findChildren: PySide6 does not support it.
        for widget_type in (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox):
            for widget in inner.findChildren(widget_type):
                widget.valueChanged.connect(self._settings_widget_changed)
        for widget in inner.findChildren(QtWidgets.QComboBox):
            widget.currentTextChanged.connect(self._settings_widget_changed)
        for widget in inner.findChildren(QtWidgets.QCheckBox):
            if widget not in {self.chk_use_previous_position, self.chk_auto_update_previous}:
                widget.stateChanged.connect(self._settings_widget_changed)

        self.slider_manual_threshold.valueChanged.connect(self.spin_manual_threshold.setValue)
        self.spin_manual_threshold.valueChanged.connect(self.slider_manual_threshold.setValue)

        return container

    def _double_spin(self, mn: float, mx: float, decimals: int, step: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(mn, mx)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setKeyboardTracking(False)
        return spin

    # ------------------------------------------------------------------
    # Settings mapping
    # ------------------------------------------------------------------
    def _set_widgets_from_settings(self, s: SingleBeeDetectorSettings) -> None:
        self._updating_widgets = True
        try:
            p = s.preprocess
            self.combo_background_mode.setCurrentText(p.background_mode)
            self.spin_adaptive_history.setValue(p.adaptive_history_length)
            self.combo_adaptive_method.setCurrentText(p.adaptive_background_method)
            self.combo_channel.setCurrentText(p.channel)
            self.combo_difference.setCurrentText(p.difference_mode)
            self.spin_blur.setValue(p.blur_kernel)
            self.double_detection_scale.setValue(float(p.detection_scale))
            self.chk_refine_on_original.setChecked(p.refine_on_original)
            self.chk_crop_to_arena_bbox.setChecked(p.crop_to_arena_bbox)
            self.spin_refine_padding.setValue(p.refine_padding_px)

            self.combo_threshold_mode.setCurrentText(s.threshold.mode)
            self.slider_manual_threshold.setValue(s.threshold.manual_threshold)
            self.spin_manual_threshold.setValue(s.threshold.manual_threshold)
            self.spin_adaptive_block.setValue(s.threshold.adaptive_block_size)
            self.double_adaptive_c.setValue(s.threshold.adaptive_c)
            self.chk_invert_binary.setChecked(s.threshold.invert_binary)

            self.combo_kernel_shape.setCurrentText(s.morphology.kernel_shape)
            self.spin_open_kernel.setValue(s.morphology.open_kernel)
            self.spin_close_kernel.setValue(s.morphology.close_kernel)
            self.spin_erode_kernel.setValue(s.morphology.erode_kernel)
            self.spin_erode_iter.setValue(s.morphology.erode_iterations)
            self.spin_dilate_kernel.setValue(s.morphology.dilate_kernel)
            self.spin_dilate_iter.setValue(s.morphology.dilate_iterations)

            self.double_min_area.setValue(s.blob_filter.min_area)
            self.double_max_area.setValue(s.blob_filter.max_area)
            self.double_min_solidity.setValue(s.blob_filter.min_solidity)
            self.double_min_extent.setValue(s.blob_filter.min_extent)
            self.double_max_aspect.setValue(s.blob_filter.max_aspect_ratio)
            self.double_min_width.setValue(s.blob_filter.min_width)
            self.double_max_width.setValue(s.blob_filter.max_width)
            self.double_min_height.setValue(s.blob_filter.min_height)
            self.double_max_height.setValue(s.blob_filter.max_height)
            self.chk_reject_border.setChecked(s.blob_filter.reject_touching_frame_border)

            self.combo_selection_method.setCurrentText(s.candidate_selection.method)
            self.spin_max_bees.setValue(getattr(s.candidate_selection, 'max_bees', 1))
            self.double_max_jump.setValue(s.candidate_selection.max_jump_px)
            self.chk_allow_largest_if_no_previous.setChecked(s.candidate_selection.allow_largest_if_no_previous)
            self.double_area_weight.setValue(s.candidate_selection.area_weight)
            self.double_solidity_weight.setValue(s.candidate_selection.solidity_weight)
            self.double_distance_weight.setValue(s.candidate_selection.distance_weight)

            self.chk_kalman_enabled.setChecked(s.kalman.enabled)
            self.spin_max_missed.setValue(s.kalman.max_missed_frames)
            self.double_process_noise.setValue(s.kalman.process_noise)
            self.double_measurement_noise.setValue(s.kalman.measurement_noise)

            self.chk_draw_rejected.setChecked(s.debug.draw_rejected_candidates)
            self.chk_draw_labels.setChecked(s.debug.draw_candidate_labels)
            self.chk_draw_axes.setChecked(s.debug.draw_axes)
            
            a = s.aruco
            self.chk_aruco_enabled.setChecked(a.enabled)
            if self.combo_aruco_dict.findText(a.dict_path) == -1:
                self.combo_aruco_dict.addItem(a.dict_path)
            self.combo_aruco_dict.setCurrentText(a.dict_path)
            self.combo_aruco_detect_mode.setCurrentText(getattr(a, "detect_mode", "arena"))
            self.spin_max_hamming.setValue(a.max_hamming)
            self.spin_max_border_errors.setValue(a.max_border_errors)
            self.spin_crop_padding.setValue(a.crop_padding)
        finally:
            self._updating_widgets = False

    def _settings_from_widgets(self) -> SingleBeeDetectorSettings:
        return SingleBeeDetectorSettings(
            version=1,
            preprocess=PreprocessSettings(
                background_mode=self.combo_background_mode.currentText(),
                adaptive_history_length=self.spin_adaptive_history.value(),
                adaptive_background_method=self.combo_adaptive_method.currentText(),
                channel=self.combo_channel.currentText(),
                difference_mode=self.combo_difference.currentText(),
                blur_kernel=self.spin_blur.value(),
                detection_scale=self.double_detection_scale.value(),
                refine_on_original=self.chk_refine_on_original.isChecked(),
                crop_to_arena_bbox=self.chk_crop_to_arena_bbox.isChecked(),
                refine_padding_px=self.spin_refine_padding.value(),
            ),
            threshold=ThresholdSettings(
                mode=self.combo_threshold_mode.currentText(),
                manual_threshold=self.spin_manual_threshold.value(),
                adaptive_block_size=self.spin_adaptive_block.value(),
                adaptive_c=self.double_adaptive_c.value(),
                invert_binary=self.chk_invert_binary.isChecked(),
            ),
            morphology=MorphologySettings(
                open_kernel=self.spin_open_kernel.value(),
                close_kernel=self.spin_close_kernel.value(),
                erode_kernel=self.spin_erode_kernel.value(),
                erode_iterations=self.spin_erode_iter.value(),
                dilate_kernel=self.spin_dilate_kernel.value(),
                dilate_iterations=self.spin_dilate_iter.value(),
                kernel_shape=self.combo_kernel_shape.currentText(),
            ),
            blob_filter=BlobFilterSettings(
                min_area=self.double_min_area.value(),
                max_area=self.double_max_area.value(),
                min_solidity=self.double_min_solidity.value(),
                min_extent=self.double_min_extent.value(),
                max_aspect_ratio=self.double_max_aspect.value(),
                min_width=self.double_min_width.value(),
                max_width=self.double_max_width.value(),
                min_height=self.double_min_height.value(),
                max_height=self.double_max_height.value(),
                reject_touching_frame_border=self.chk_reject_border.isChecked(),
            ),
            candidate_selection=CandidateSelectionSettings(
                method=self.combo_selection_method.currentText(),
                max_bees=self.spin_max_bees.value(),
                max_jump_px=self.double_max_jump.value(),
                allow_largest_if_no_previous=self.chk_allow_largest_if_no_previous.isChecked(),
                area_weight=self.double_area_weight.value(),
                solidity_weight=self.double_solidity_weight.value(),
                distance_weight=self.double_distance_weight.value(),
            ),
            kalman=KalmanSettings(
                enabled=self.chk_kalman_enabled.isChecked(),
                max_missed_frames=self.spin_max_missed.value(),
                process_noise=self.double_process_noise.value(),
                measurement_noise=self.double_measurement_noise.value(),
            ),
            debug=DebugSettings(
                draw_rejected_candidates=self.chk_draw_rejected.isChecked(),
                draw_candidate_labels=self.chk_draw_labels.isChecked(),
                draw_axes=self.chk_draw_axes.isChecked(),
            ),
            aruco=ArucoSettings(
                enabled=self.chk_aruco_enabled.isChecked(),
                dict_path=self.combo_aruco_dict.currentText(),
                detect_mode=self.combo_aruco_detect_mode.currentText(),
                max_hamming=self.spin_max_hamming.value(),
                max_border_errors=self.spin_max_border_errors.value(),
                crop_padding=self.spin_crop_padding.value(),
            ),
        )

    def _settings_widget_changed(self, *args) -> None:
        if self._updating_widgets:
            return
        self._adaptive_cache_key = None
        self.settings = self._settings_from_widgets()
        self.detector.set_settings(self.settings)
        self.update_detection()

    # ------------------------------------------------------------------
    # File loading/saving
    # ------------------------------------------------------------------
    def load_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v);;All Files (*)")
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Video Error", f"Could not open video:\n{path}")
            return
        if self.cap is not None:
            self.cap.release()
        self.cap = cap
        self.video_path = path
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.current_frame_index = 0
        self.previous_position = None
        self.roi_rect = None
        self.chk_use_roi.setChecked(False)
        max_frame = max(0, self.frame_count - 1)
        self.slider_frame.setRange(0, max_frame)
        self.spin_frame.setRange(0, max_frame)
        self._adaptive_cache_key = None
        self.seek_frame(0, reset_previous=True)
        self._update_file_label()
        self._update_ui_enabled_state()

    def load_background(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Static Background Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)")
        if not path:
            return
        bg = imread_unicode(path, cv2.IMREAD_UNCHANGED)
        if bg is None:
            QtWidgets.QMessageBox.critical(self, "Background Error", f"Could not load background:\n{path}")
            return
        if bg.ndim == 2:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        elif bg.ndim == 3 and bg.shape[2] == 4:
            bg = cv2.cvtColor(bg, cv2.COLOR_BGRA2BGR)
        self.background = bg
        self.background_path = path
        self._update_file_label()
        self.update_detection()

    def load_settings(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Detector Settings", "", "Settings Files (*.txt *.ini);;All Files (*)")
        if not path:
            return
        try:
            s = load_settings_txt(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Settings Error", f"Could not load settings:\n{exc}")
            return
        self.settings = s
        self.detector.set_settings(s)
        self.last_settings_path = path
        self._adaptive_cache_key = None
        self._set_widgets_from_settings(s)
        self._update_file_label()
        self.update_detection()

    def save_settings(self) -> None:
        self.settings = self._settings_from_widgets()
        suggested = self.last_settings_path or "single_bee_settings_v1.txt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Detector Settings", suggested, "Settings Files (*.txt *.ini);;All Files (*)")
        if not path:
            return
        try:
            save_settings_txt(path, self.settings)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Settings Error", f"Could not save settings:\n{exc}")
            return
        self.last_settings_path = path
        self._update_file_label()
        QtWidgets.QMessageBox.information(self, "Saved", f"Settings saved to:\n{path}")

    def reset_settings(self) -> None:
        self.settings = default_settings()
        self.detector.set_settings(self.settings)
        self._adaptive_cache_key = None
        self._set_widgets_from_settings(self.settings)
        self.previous_position = None
        self.update_detection()

    # ------------------------------------------------------------------
    # Frame navigation/playback
    # ------------------------------------------------------------------
    def seek_frame(self, frame_index: int, *, reset_previous: bool = False) -> None:
        if self.cap is None:
            return
        frame_index = int(np.clip(frame_index, 0, max(0, self.frame_count - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return
        self.current_frame_index = frame_index
        self.current_frame = frame
        if reset_previous:
            self.previous_position = None
            self.detector.clear_history()
        self._updating_widgets = True
        try:
            self.slider_frame.setValue(frame_index)
            self.spin_frame.setValue(frame_index)
        finally:
            self._updating_widgets = False
        self.update_detection()

    def _slider_changed(self, value: int) -> None:
        if not self._updating_widgets:
            self.seek_frame(value, reset_previous=True)

    def _spin_changed(self, value: int) -> None:
        if not self._updating_widgets:
            self.seek_frame(value, reset_previous=True)

    def toggle_play(self) -> None:
        if self.cap is None:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.btn_play.setText("Play")
        else:
            self.play_timer.start(max(1, int(1000.0 / max(self.fps, 1.0))))
            self.btn_play.setText("Pause")

    def _play_next_frame(self) -> None:
        if self.current_frame_index >= self.frame_count - 1:
            self.play_timer.stop()
            self.btn_play.setText("Play")
            return
        self.seek_frame(self.current_frame_index + 1, reset_previous=False)

    def _browse_dict(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select ArUco Dictionary", "", "Text Files (*.txt);;All Files (*.*)"
        )
        if path:
            if self.combo_aruco_dict.findText(path) == -1:
                self.combo_aruco_dict.addItem(path)
            self.combo_aruco_dict.setCurrentText(path)

    # ------------------------------------------------------------------
    # ROI / previous point
    # ------------------------------------------------------------------
    def toggle_draw_roi(self) -> None:
        enabled = not self.main_view._draw_rect_enabled
        if enabled:
            self.combo_view.setCurrentText("Original + Selected Overlay")
        self.main_view.set_draw_rect_enabled(enabled)
        self.btn_draw_roi.setText("Finish Drawing ROI" if enabled else "Draw ROI on Main View")

    def set_roi_rect(self, rect: Rect) -> None:
        self.roi_rect = rect
        self.chk_use_roi.setChecked(True)
        self.main_view.set_draw_rect_enabled(False)
        self.btn_draw_roi.setText("Draw ROI on Main View")
        self.previous_position = None
        self._update_roi_label()
        self.update_detection()

    def clear_roi(self) -> None:
        self.roi_rect = None
        self.chk_use_roi.setChecked(False)
        self.previous_position = None
        self._update_roi_label()
        self.update_detection()

    def _arena_mask(self) -> Optional[np.ndarray]:
        if self.current_frame is None or not self.chk_use_roi.isChecked() or self.roi_rect is None:
            return None
        return mask_from_rect(self.current_frame.shape[:2], self.roi_rect)

    def clear_previous_position(self) -> None:
        self.previous_position = None
        self.detector.clear_history()
        self.update_detection()

    # ------------------------------------------------------------------
    # Adaptive background for tuner preview
    # ------------------------------------------------------------------
    def _adaptive_background_for_current_frame(self, settings: SingleBeeDetectorSettings):
        if self.video_path is None or self.current_frame is None:
            return None, 0
        p = settings.preprocess
        key = (
            self.video_path,
            self.current_frame_index,
            p.channel,
            int(p.adaptive_history_length),
            p.adaptive_background_method,
        )
        if key == self._adaptive_cache_key:
            return self._adaptive_cache_bg, self._adaptive_cache_count

        hist = max(1, int(p.adaptive_history_length))
        start = max(0, self.current_frame_index - hist)
        bg = AdaptiveRollingBackground(channel=p.channel, max_history=hist, method=p.adaptive_background_method)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None, 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _frame_idx in range(start, self.current_frame_index):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            bg.update(frame)
        cap.release()

        bg_channel = bg.get_background()
        count = bg.history_count
        self._adaptive_cache_key = key
        self._adaptive_cache_bg = bg_channel
        self._adaptive_cache_count = count
        return bg_channel, count

    # ------------------------------------------------------------------
    # Detection update
    # ------------------------------------------------------------------
    def update_detection(self) -> None:
        if self.current_frame is None:
            return
        self.settings = self._settings_from_widgets()
        self.detector.set_settings(self.settings)

        arena_mask = self._arena_mask()
        prev = self.previous_position if self.chk_use_previous_position.isChecked() else None

        mode = self.settings.preprocess.background_mode.strip().lower()
        static_bg = self.background if mode == "static" else None
        bg_channel = None
        bg_count = 0
        if mode == "adaptive":
            bg_channel, bg_count = self._adaptive_background_for_current_frame(self.settings)

        try:
            result = self.detector.detect(
                self.current_frame,
                background_bgr=static_bg,
                background_channel=bg_channel,
                background_history_count=bg_count,
                arena_mask=arena_mask,
                previous_position=prev,
            )
        except Exception as exc:
            self._last_result = None
            self._last_images = {"Original + Selected Overlay": self.current_frame}
            self.main_view.set_image(self.current_frame)
            self.txt_summary.setPlainText(f"Detection error:\n{exc}")
            return

        self._last_result = result
        if result.detected and self.chk_auto_update_previous.isChecked():
            self.previous_position = (result.x, result.y)

        clean_overlay = draw_detection_overlay(
            self.current_frame,
            result,
            arena_mask=arena_mask,
            draw_rejected=False,
            draw_labels=True,
            draw_axes=self.settings.debug.draw_axes,
        )
        candidate_overlay = draw_detection_overlay(
            self.current_frame,
            result,
            arena_mask=arena_mask,
            draw_rejected=self.settings.debug.draw_rejected_candidates,
            draw_labels=self.settings.debug.draw_candidate_labels,
            draw_axes=self.settings.debug.draw_axes,
        )

        self._last_images = {
            "Original + Selected Overlay": clean_overlay,
            "All Candidates / Rejection Reasons": candidate_overlay,
            "Background Used": result.debug_images.background_used,
            "Background Difference / Source": result.debug_images.difference,
            "Threshold Binary": result.debug_images.threshold,
            "Cleaned Mask": result.debug_images.cleaned,
            "Channel / Crop Source": result.debug_images.channel,
        }

        summary = summarize_result(result)
        if self.previous_position is not None:
            summary += f"\n\nPrevious position used next: ({self.previous_position[0]:.1f}, {self.previous_position[1]:.1f})"
        else:
            summary += "\n\nPrevious position used next: None"
        self.txt_summary.setPlainText(summary)
        self._refresh_selected_view()
        self._update_time_label()
        self._update_roi_label()

    def _refresh_selected_view(self) -> None:
        name = self.combo_view.currentText()
        image = self._last_images.get(name)
        if image is None:
            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"No image for: {name}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
            image = placeholder
        self.main_view.set_image(image)

    # ------------------------------------------------------------------
    # Labels / state
    # ------------------------------------------------------------------
    def _update_file_label(self) -> None:
        video = Path(self.video_path).name if self.video_path else "No video"
        bg = Path(self.background_path).name if self.background_path else "No static background"
        settings = Path(self.last_settings_path).name if self.last_settings_path else "Unsaved/default settings"
        self.lbl_files.setText(f"Video: {video}   |   Static BG: {bg}   |   Settings: {settings}")

    def _update_time_label(self) -> None:
        t = self.current_frame_index / max(self.fps, 1e-9)
        self.lbl_time.setText(f"Frame: {self.current_frame_index} / {max(0, self.frame_count - 1)}   Time: {t:.2f} s   FPS: {self.fps:.2f}")

    def _update_roi_label(self) -> None:
        if self.chk_use_roi.isChecked() and self.roi_rect is not None:
            self.lbl_roi.setText(f"ROI: x={self.roi_rect[0]}, y={self.roi_rect[1]}, w={self.roi_rect[2]}, h={self.roi_rect[3]}")
        else:
            self.lbl_roi.setText("ROI: full frame")

    def _update_ui_enabled_state(self) -> None:
        has_video = self.cap is not None
        for w in [self.btn_prev, self.btn_play, self.btn_next, self.slider_frame, self.spin_frame, self.btn_draw_roi, self.btn_clear_roi]:
            w.setEnabled(has_video)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = SingleBeeDetectorTuner()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
