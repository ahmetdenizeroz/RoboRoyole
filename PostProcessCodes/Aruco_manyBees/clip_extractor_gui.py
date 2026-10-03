import sys
import os
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QSpinBox,
    QProgressBar, QMessageBox, QTabWidget
)
from PySide6.QtCore import QThread, Signal, Qt

from clip_extractor_core import ClipExtractorCore

class ExtractionWorker(QThread):
    progress_updated = Signal(int, int)
    extraction_finished = Signal(str, list)
    error_occurred = Signal(str)

    def __init__(self, video_path, txt_path, output_dir, roi, min_seconds, pad_seconds):
        super().__init__()
        self.video_path = video_path
        self.txt_path = txt_path
        self.output_dir = output_dir
        self.roi = roi
        self.min_seconds = min_seconds
        self.pad_seconds = pad_seconds
        self.core = ClipExtractorCore()

    def run(self):
        try:
            coords = self.core.parse_coordinates(self.txt_path)
            
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            cap.release()
            
            min_frames = int(round(self.min_seconds * fps))
            
            sequences = self.core.find_roi_sequences(
                coords=coords,
                roi=self.roi,
                min_frames=min_frames,
                fps=fps,
                pad_seconds=self.pad_seconds
            )
            
            if not sequences:
                self.extraction_finished.emit("No sequences found matching the criteria.", [])
                return
                
            def report_progress(current, total):
                self.progress_updated.emit(current, total)
                
            output_files = self.core.extract_clips(
                video_path=self.video_path,
                sequences=sequences,
                output_dir=self.output_dir,
                progress_callback=report_progress
            )
            
            self.extraction_finished.emit("Extraction completed successfully!", output_files)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class ClipExtractorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bee ROI Clip Extractor & Timeline Plotter")
        self.resize(700, 450)
        
        self.roi = None
        self._init_ui()
        
    def _init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # --- TAB 1: Extract Clips ---
        extractor_widget = QWidget()
        layout = QVBoxLayout(extractor_widget)
        
        # Video file
        video_layout = QHBoxLayout()
        video_layout.addWidget(QLabel("Video File:"))
        self.video_input = QLineEdit()
        video_layout.addWidget(self.video_input)
        video_btn = QPushButton("Browse")
        video_btn.clicked.connect(self.browse_video)
        video_layout.addWidget(video_btn)
        layout.addLayout(video_layout)
        
        # Coordinates file
        txt_layout = QHBoxLayout()
        txt_layout.addWidget(QLabel("Coordinates (.txt):"))
        self.txt_input = QLineEdit()
        txt_layout.addWidget(self.txt_input)
        txt_btn = QPushButton("Browse")
        txt_btn.clicked.connect(self.browse_txt)
        txt_layout.addWidget(txt_btn)
        layout.addLayout(txt_layout)
        
        # Output dir
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output Directory:"))
        self.out_input = QLineEdit()
        out_layout.addWidget(self.out_input)
        out_btn = QPushButton("Browse")
        out_btn.clicked.connect(self.browse_out)
        out_layout.addWidget(out_btn)
        layout.addLayout(out_layout)
        
        # ROI selection
        roi_layout = QHBoxLayout()
        self.roi_btn = QPushButton("Select ROI on First Frame")
        self.roi_btn.clicked.connect(self.select_roi)
        roi_layout.addWidget(self.roi_btn)
        self.roi_label = QLabel("No ROI selected")
        roi_layout.addWidget(self.roi_label)
        layout.addLayout(roi_layout)
        
        # Settings
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Min Duration (seconds):"))
        self.min_sec_spin = QSpinBox()
        self.min_sec_spin.setRange(1, 3600)
        self.min_sec_spin.setValue(5)
        settings_layout.addWidget(self.min_sec_spin)
        
        settings_layout.addWidget(QLabel("Padding (seconds):"))
        self.pad_sec_spin = QSpinBox()
        self.pad_sec_spin.setRange(0, 60)
        self.pad_sec_spin.setValue(1)
        settings_layout.addWidget(self.pad_sec_spin)
        layout.addLayout(settings_layout)
        
        # Progress and Extract
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.extract_btn = QPushButton("Extract Clips")
        self.extract_btn.clicked.connect(self.start_extraction)
        layout.addWidget(self.extract_btn)
        
        self.tabs.addTab(extractor_widget, "Extract Clips")
        
        
        # --- TAB 2: Feeding Plotter ---
        plotter_widget = QWidget()
        plot_layout = QVBoxLayout(plotter_widget)
        
        # Part 1 Video Folder
        p1_vid_layout = QHBoxLayout()
        p1_vid_layout.addWidget(QLabel("Part 1 Clips Folder:"))
        self.p1_vid_input = QLineEdit()
        p1_vid_layout.addWidget(self.p1_vid_input)
        p1_vid_btn = QPushButton("Browse")
        p1_vid_btn.clicked.connect(lambda: self.browse_dir(self.p1_vid_input))
        p1_vid_layout.addWidget(p1_vid_btn)
        plot_layout.addLayout(p1_vid_layout)
        
        # Part 2 Video Folder
        p2_vid_layout = QHBoxLayout()
        p2_vid_layout.addWidget(QLabel("Part 2 Clips Folder (Optional):"))
        self.p2_vid_input = QLineEdit()
        p2_vid_layout.addWidget(self.p2_vid_input)
        p2_vid_btn = QPushButton("Browse")
        p2_vid_btn.clicked.connect(lambda: self.browse_dir(self.p2_vid_input))
        p2_vid_layout.addWidget(p2_vid_btn)
        plot_layout.addLayout(p2_vid_layout)
        
        # Part 1 Info TXT
        p1_info_layout = QHBoxLayout()
        p1_info_layout.addWidget(QLabel("Part 1 Info TXT:"))
        self.p1_info_input = QLineEdit()
        p1_info_layout.addWidget(self.p1_info_input)
        p1_info_btn = QPushButton("Browse")
        p1_info_btn.clicked.connect(lambda: self.browse_file(self.p1_info_input, "Text Files (*.txt)"))
        p1_info_layout.addWidget(p1_info_btn)
        plot_layout.addLayout(p1_info_layout)
        
        # Part 2 Info TXT
        p2_info_layout = QHBoxLayout()
        p2_info_layout.addWidget(QLabel("Part 2 Info TXT (Optional):"))
        self.p2_info_input = QLineEdit()
        p2_info_layout.addWidget(self.p2_info_input)
        p2_info_btn = QPushButton("Browse")
        p2_info_btn.clicked.connect(lambda: self.browse_file(self.p2_info_input, "Text Files (*.txt)"))
        p2_info_layout.addWidget(p2_info_btn)
        plot_layout.addLayout(p2_info_layout)
        
        # Plot Output Dir
        plot_out_layout = QHBoxLayout()
        plot_out_layout.addWidget(QLabel("Output Directory:"))
        self.plot_out_input = QLineEdit()
        plot_out_layout.addWidget(self.plot_out_input)
        plot_out_btn = QPushButton("Browse")
        plot_out_btn.clicked.connect(lambda: self.browse_dir(self.plot_out_input))
        plot_out_layout.addWidget(plot_out_btn)
        plot_layout.addLayout(plot_out_layout)
        
        self.generate_btn = QPushButton("Generate Timeline Plot")
        self.generate_btn.clicked.connect(self.generate_plot)
        plot_layout.addWidget(self.generate_btn)
        plot_layout.addStretch()
        
        self.tabs.addTab(plotter_widget, "Feeding Plotter")


    # --- Browse Helpers ---
    def browse_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv)")
        if file:
            self.video_input.setText(file)
            
    def browse_txt(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Coordinates", "", "Text Files (*.txt)")
        if file:
            self.txt_input.setText(file)
            
    def browse_out(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.out_input.setText(dir_path)
            
    def browse_dir(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            line_edit.setText(dir_path)
            
    def browse_file(self, line_edit, filter_str):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
        if file:
            line_edit.setText(file)


    # --- Extract Clips Logic ---
    def select_roi(self):
        video_path = self.video_input.text()
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "Error", "Please select a valid video file first.")
            return
            
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            QMessageBox.warning(self, "Error", "Could not read the first frame of the video.")
            return
            
        max_w, max_h = 1280, 720
        h, w = frame.shape[:2]
        scale = 1.0
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            
        if scale < 1.0:
            display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            display_frame = frame
            
        roi = cv2.selectROI("Select ROI (Press ENTER or SPACE to finish)", display_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI (Press ENTER or SPACE to finish)")
        
        if roi == (0, 0, 0, 0):
            QMessageBox.information(self, "Info", "ROI selection cancelled.")
            self.roi = None
            self.roi_label.setText("No ROI selected")
        else:
            if scale < 1.0:
                roi = (
                    int(roi[0] / scale),
                    int(roi[1] / scale),
                    int(roi[2] / scale),
                    int(roi[3] / scale)
                )
            self.roi = roi
            self.roi_label.setText(f"ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
            
    def start_extraction(self):
        video_path = self.video_input.text()
        txt_path = self.txt_input.text()
        out_dir = self.out_input.text()
        
        if not all([video_path, txt_path, out_dir]):
            QMessageBox.warning(self, "Error", "Please fill in all file and directory fields.")
            return
            
        if self.roi is None:
            QMessageBox.warning(self, "Error", "Please select an ROI first.")
            return
            
        self.extract_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.worker = ExtractionWorker(
            video_path=video_path,
            txt_path=txt_path,
            output_dir=out_dir,
            roi=self.roi,
            min_seconds=self.min_sec_spin.value(),
            pad_seconds=self.pad_sec_spin.value()
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.extraction_finished.connect(self.extraction_done)
        self.worker.error_occurred.connect(self.extraction_error)
        self.worker.start()
        
    def update_progress(self, current, total):
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
            
    def extraction_done(self, message, files):
        self.extract_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Finished", f"{message}\nExtracted {len(files)} clips.")
        
    def extraction_error(self, err_msg):
        self.extract_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"An error occurred:\n{err_msg}")


    # --- Plotter Logic ---
    def generate_plot(self):
        p1_vid = self.p1_vid_input.text()
        p1_info = self.p1_info_input.text()
        p2_vid = self.p2_vid_input.text()
        p2_info = self.p2_info_input.text()
        out_dir = self.plot_out_input.text()
        
        if not p1_vid or not p1_info or not out_dir:
            QMessageBox.warning(self, "Error", "Please fill in Part 1 folders and Output Directory at minimum.")
            return
            
        try:
            out_path = ClipExtractorCore.generate_timeline_plot(
                part1_folder=p1_vid, part1_info=p1_info,
                part2_folder=p2_vid, part2_info=p2_info,
                output_dir=out_dir
            )
            QMessageBox.information(self, "Success", f"Plot generated successfully at:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not generate plot:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ClipExtractorGUI()
    window.show()
    sys.exit(app.exec())
