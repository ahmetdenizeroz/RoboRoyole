import sys
import cv2
import numpy as np
from pathlib import Path
from PySide6 import QtWidgets, QtCore, QtGui
from video_processor_core import VideoProcessorCore

class WorkerThread(QtCore.QThread):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, core, path):
        super().__init__()
        self.core = core
        self.path = path

    def run(self):
        try:
            self.core.calculate_stabilization(self.path, self.update_progress)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def update_progress(self, val):
        self.progress.emit(val)

class ExportThread(QtCore.QThread):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, core, in_path, out_path, settings):
        super().__init__()
        self.core = core
        self.in_path = in_path
        self.out_path = out_path
        self.settings = settings

    def run(self):
        try:
            cap = cv2.VideoCapture(self.in_path)
            if not cap.isOpened():
                raise RuntimeError("Failed to open input video.")
            
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.out_path, fourcc, fps, (w, h), isColor=True)
            
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Force BGR if it's grayscale
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                processed = self.core.apply_filters(frame, idx, self.settings)
                out.write(processed)
                
                idx += 1
                if idx % 10 == 0:
                    self.progress.emit(int((idx / total) * 100))
                    
            cap.release()
            out.release()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class VideoProcessorGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Pre-processor")
        self.resize(1000, 700)
        self.core = VideoProcessorCore()
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left Panel - Controls
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_panel.setFixedWidth(350)
        
        # Load Video
        self.btn_load = QtWidgets.QPushButton("Load Video")
        self.btn_load.clicked.connect(self._load_video)
        self.lbl_path = QtWidgets.QLabel("No video loaded.")
        self.lbl_path.setWordWrap(True)
        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.lbl_path)
        
        # Stabilization
        stab_group = QtWidgets.QGroupBox("Stabilization")
        stab_layout = QtWidgets.QVBoxLayout(stab_group)
        self.btn_calc_stab = QtWidgets.QPushButton("Calculate Stabilization")
        self.btn_calc_stab.setEnabled(False)
        self.btn_calc_stab.clicked.connect(self._calc_stabilization)
        self.chk_stab = QtWidgets.QCheckBox("Enable Stabilization")
        self.chk_stab.setEnabled(False)
        self.chk_stab.stateChanged.connect(self._update_preview)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        
        stab_layout.addWidget(self.btn_calc_stab)
        stab_layout.addWidget(self.progress_bar)
        stab_layout.addWidget(self.chk_stab)
        left_layout.addWidget(stab_group)

        # Brightness / Contrast
        bc_group = QtWidgets.QGroupBox("Brightness & Contrast")
        bc_layout = QtWidgets.QFormLayout(bc_group)
        self.slider_brightness = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_brightness.setRange(-100, 100)
        self.slider_brightness.setValue(0)
        self.slider_brightness.valueChanged.connect(self._update_preview)
        
        self.slider_contrast = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_contrast.setRange(10, 300) # 0.1 to 3.0
        self.slider_contrast.setValue(100) # 1.0
        self.slider_contrast.valueChanged.connect(self._update_preview)
        
        bc_layout.addRow("Brightness:", self.slider_brightness)
        bc_layout.addRow("Contrast:", self.slider_contrast)
        
        self.btn_reset_bc = QtWidgets.QPushButton("Reset Defaults")
        self.btn_reset_bc.clicked.connect(self._reset_bc)
        bc_layout.addRow(self.btn_reset_bc)
        
        left_layout.addWidget(bc_group)

        # CLAHE
        clahe_group = QtWidgets.QGroupBox("CLAHE (Local Contrast)")
        clahe_layout = QtWidgets.QFormLayout(clahe_group)
        self.chk_clahe = QtWidgets.QCheckBox("Enable CLAHE")
        self.chk_clahe.stateChanged.connect(self._update_preview)
        self.slider_clahe_clip = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_clahe_clip.setRange(1, 100) # 0.1 to 10.0
        self.slider_clahe_clip.setValue(20) # 2.0
        self.slider_clahe_clip.valueChanged.connect(self._update_preview)
        self.slider_clahe_grid = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_clahe_grid.setRange(2, 32)
        self.slider_clahe_grid.setValue(8)
        self.slider_clahe_grid.valueChanged.connect(self._update_preview)
        
        clahe_layout.addRow(self.chk_clahe)
        clahe_layout.addRow("Clip Limit:", self.slider_clahe_clip)
        clahe_layout.addRow("Grid Size:", self.slider_clahe_grid)
        left_layout.addWidget(clahe_group)

        # Morphology
        morph_group = QtWidgets.QGroupBox("Morphology (Top-Hat/Black-Hat)")
        morph_layout = QtWidgets.QFormLayout(morph_group)
        self.chk_morph = QtWidgets.QCheckBox("Enable Morphology")
        self.chk_morph.stateChanged.connect(self._update_preview)
        self.combo_morph_type = QtWidgets.QComboBox()
        self.combo_morph_type.addItems(["Top-Hat", "Black-Hat"])
        self.combo_morph_type.currentIndexChanged.connect(self._update_preview)
        self.slider_morph_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_morph_size.setRange(3, 51)
        self.slider_morph_size.setSingleStep(2)
        self.slider_morph_size.setValue(15)
        self.slider_morph_size.valueChanged.connect(self._update_preview)
        
        morph_layout.addRow(self.chk_morph)
        morph_layout.addRow("Type:", self.combo_morph_type)
        morph_layout.addRow("Kernel Size:", self.slider_morph_size)
        left_layout.addWidget(morph_group)

        # Export
        left_layout.addStretch()
        self.btn_export = QtWidgets.QPushButton("Export Video")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_video)
        left_layout.addWidget(self.btn_export)

        # Right Panel - Preview
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        
        self.lbl_preview = QtWidgets.QLabel("Preview")
        self.lbl_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_preview.setStyleSheet("background-color: black; color: white;")
        
        self.slider_frame = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_frame.setEnabled(False)
        self.slider_frame.valueChanged.connect(self._on_frame_slider)
        
        right_layout.addWidget(self.lbl_preview, 1)
        right_layout.addWidget(self.slider_frame)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

    def _get_settings(self):
        return {
            "use_stabilization": self.chk_stab.isChecked(),
            "brightness": self.slider_brightness.value(),
            "contrast": self.slider_contrast.value() / 100.0,
            "use_clahe": self.chk_clahe.isChecked(),
            "clahe_clip": self.slider_clahe_clip.value() / 10.0,
            "clahe_grid": self.slider_clahe_grid.value(),
            "use_morphology": self.chk_morph.isChecked(),
            "morph_type": self.combo_morph_type.currentText(),
            "morph_size": self.slider_morph_size.value()
        }

    def _load_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if not path:
            return
            
        self.video_path = path
        self.lbl_path.setText(Path(path).name)
        
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.slider_frame.setRange(0, max(0, self.total_frames - 1))
        self.slider_frame.setValue(0)
        self.slider_frame.setEnabled(True)
        
        self.btn_calc_stab.setEnabled(True)
        self.chk_stab.setEnabled(False)
        self.chk_stab.setChecked(False)
        self.core.is_calculated = False
        self.btn_export.setEnabled(True)
        self.progress_bar.setValue(0)
        
        self._update_preview()

    def _on_frame_slider(self):
        self._update_preview()

    def _reset_bc(self):
        self.slider_brightness.blockSignals(True)
        self.slider_contrast.blockSignals(True)
        self.slider_brightness.setValue(0)
        self.slider_contrast.setValue(100)
        self.slider_brightness.blockSignals(False)
        self.slider_contrast.blockSignals(False)
        self._update_preview()

    def _update_preview(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        idx = self.slider_frame.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            return
            
        settings = self._get_settings()
        processed = self.core.apply_filters(frame, idx, settings)
        
        # Convert BGR to RGB for PySide
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        
        pixmap = QtGui.QPixmap.fromImage(qimg)
        # scale keeping aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.lbl_preview.width(), 
            self.lbl_preview.height(), 
            QtCore.Qt.KeepAspectRatio, 
            QtCore.Qt.SmoothTransformation
        )
        self.lbl_preview.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview()

    def _calc_stabilization(self):
        if not self.video_path:
            return
            
        self.btn_calc_stab.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = WorkerThread(self.core, self.video_path)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.error.connect(lambda e: QtWidgets.QMessageBox.critical(self, "Error", e))
        self.worker.finished.connect(self._on_stab_finished)
        self.worker.start()

    def _on_stab_finished(self):
        self.chk_stab.setEnabled(True)
        self.progress_bar.setValue(100)
        QtWidgets.QMessageBox.information(self, "Success", "Stabilization calculated!")
        self.btn_calc_stab.setEnabled(True)

    def _export_video(self):
        if not self.video_path:
            return
            
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Processed Video", "", "Video Files (*.mp4)"
        )
        if not out_path:
            return
            
        self.btn_export.setEnabled(False)
        self.progress_bar.setValue(0)
        
        settings = self._get_settings()
        self.exporter = ExportThread(self.core, self.video_path, out_path, settings)
        self.exporter.progress.connect(self.progress_bar.setValue)
        self.exporter.error.connect(lambda e: QtWidgets.QMessageBox.critical(self, "Error", e))
        self.exporter.finished.connect(self._on_export_finished)
        self.exporter.start()

    def _on_export_finished(self):
        self.btn_export.setEnabled(True)
        self.progress_bar.setValue(100)
        QtWidgets.QMessageBox.information(self, "Success", "Video exported successfully!")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VideoProcessorGUI()
    window.show()
    sys.exit(app.exec())
