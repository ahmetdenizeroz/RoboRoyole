import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QPushButton, QLineEdit, QFileDialog,
    QLabel, QTextEdit, QMessageBox, QDialog, QFormLayout, QDialogButtonBox
)
from PySide6.QtCore import Qt, QThread
from analyzer_core import AnalyzerCore, AnalyzerWorker

class TrimDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trim Data")
        self.setStyleSheet("""
            QDialog { background-color: #1A1A1A; color: #f9c901; }
            QLabel { color: #f9c901; font-weight: bold; }
            QLineEdit { background-color: #222; border: 1px solid #985b10; color: #f6e000; padding: 4px; }
            QPushButton { background-color: #6b4701; color: #f6e000; padding: 5px; font-weight: bold; border-radius: 3px; }
            QPushButton:hover { background-color: #985b10; }
        """)
        layout = QFormLayout(self)
        self.start_input = QLineEdit("00:00:00")
        self.end_input = QLineEdit("")
        self.end_input.setPlaceholderText("Leave blank to keep to end")
        layout.addRow("Start Time (HH:MM:SS):", self.start_input)
        layout.addRow("End Time (HH:MM:SS):", self.end_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_times(self):
        return self.start_input.text(), self.end_input.text()

class DurationFilterDialog(QDialog):
    def __init__(self, parent=None, default_min="0.0", default_max="999999.0", default_intensity="0.0"):
        super().__init__(parent)
        self.setWindowTitle("Duration Filter Settings")
        self.setStyleSheet("""
            QDialog { background-color: #1A1A1A; color: #f9c901; }
            QLabel { color: #f9c901; font-weight: bold; }
            QLineEdit { background-color: #222; border: 1px solid #985b10; color: #f6e000; padding: 4px; }
            QPushButton { background-color: #6b4701; color: #f6e000; padding: 5px; font-weight: bold; border-radius: 3px; }
            QPushButton:hover { background-color: #985b10; }
        """)
        layout = QFormLayout(self)
        self.min_input = QLineEdit(default_min)
        self.max_input = QLineEdit(default_max)
        self.intensity_input = QLineEdit(default_intensity)
        layout.addRow("Minimum duration (s):", self.min_input)
        layout.addRow("Maximum duration (s):", self.max_input)
        layout.addRow("Minimum avg intensity:", self.intensity_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_settings(self):
        try:
            min_dur = float(self.min_input.text())
        except ValueError:
            min_dur = 0.0
        try:
            max_dur = float(self.max_input.text())
        except ValueError:
            max_dur = float('inf')
        try:
            min_avg = float(self.intensity_input.text())
        except ValueError:
            min_avg = 0.0
        return min_dur, max_dur, min_avg

class ExtractClipsDialog(QDialog):
    def __init__(self, parent=None, default_pre="2.0", default_post="3.0", default_min="0.0", default_max="999999.0", default_intensity="0.0"):
        super().__init__(parent)
        self.setWindowTitle("Clip Extraction Settings")
        self.setStyleSheet("""
            QDialog { background-color: #1A1A1A; color: #f9c901; }
            QLabel { color: #f9c901; font-weight: bold; }
            QLineEdit { background-color: #222; border: 1px solid #985b10; color: #f6e000; padding: 4px; }
            QPushButton { background-color: #6b4701; color: #f6e000; padding: 5px; font-weight: bold; border-radius: 3px; }
            QPushButton:hover { background-color: #985b10; }
        """)
        layout = QFormLayout(self)
        self.pre_input = QLineEdit(default_pre)
        self.post_input = QLineEdit(default_post)
        self.min_input = QLineEdit(default_min)
        self.max_input = QLineEdit(default_max)
        self.intensity_input = QLineEdit(default_intensity)
        layout.addRow("Pre-pad duration (s):", self.pre_input)
        layout.addRow("Post-pad duration (s):", self.post_input)
        layout.addRow("Minimum duration (s):", self.min_input)
        layout.addRow("Maximum duration (s):", self.max_input)
        layout.addRow("Minimum avg intensity:", self.intensity_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_settings(self):
        try:
            pre = float(self.pre_input.text())
        except ValueError:
            pre = 2.0
        try:
            post = float(self.post_input.text())
        except ValueError:
            post = 3.0
        try:
            min_dur = float(self.min_input.text())
        except ValueError:
            min_dur = 0.0
        try:
            max_dur = float(self.max_input.text())
        except ValueError:
            max_dur = float('inf')
        try:
            min_avg = float(self.intensity_input.text())
        except ValueError:
            min_avg = 0.0
        return pre, post, min_dur, max_dur, min_avg

class AnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bee Feeder - Data Analyzer")
        self.setGeometry(100, 100, 900, 700)

        self.core = AnalyzerCore()
        self.core.log_message.connect(self.log_status)

        self.worker_thread = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Inputs Group
        inputs_group = QGroupBox("Inputs & Configuration")
        inputs_layout = QGridLayout()

        self.log_path_input = QLineEdit()
        self.btn_browse_log = QPushButton("Browse Log (.txt)")
        self.btn_browse_log.clicked.connect(self.browse_log)

        self.vid_path_input = QLineEdit()
        self.btn_browse_vid = QPushButton("Browse Video (.mp4/.mkv)")
        self.btn_browse_vid.clicked.connect(self.browse_vid)

        self.tags_input = QLineEdit("1, 2, 5-10")

        inputs_layout.addWidget(QLabel("Log File:"), 0, 0)
        inputs_layout.addWidget(self.log_path_input, 0, 1)
        inputs_layout.addWidget(self.btn_browse_log, 0, 2)

        inputs_layout.addWidget(QLabel("Video File (Optional):"), 1, 0)
        inputs_layout.addWidget(self.vid_path_input, 1, 1)
        inputs_layout.addWidget(self.btn_browse_vid, 1, 2)

        inputs_layout.addWidget(QLabel("Target Tags (e.g. 1, 2, 5-10):"), 2, 0)
        inputs_layout.addWidget(self.tags_input, 2, 1)

        inputs_group.setLayout(inputs_layout)
        main_layout.addWidget(inputs_group)

        # Operations Group
        ops_group = QGroupBox("Analysis Operations")
        ops_layout = QGridLayout()

        self.btn_validate = QPushButton("Validate ArUco Tags")
        self.btn_uptime = QPushButton("Calculate Total Uptime")
        self.btn_lines = QPushButton("Feeds Per Bee Count")
        
        self.btn_csv = QPushButton("Extract Feeding Durations CSV")
        self.btn_csv_possible = QPushButton("Extract Possible Feedings CSV")
        self.btn_interactive = QPushButton("Interactive Bee Plot (HTML)")
        
        self.btn_baselines = QPushButton("Analyze Sensor Baselines")
        self.btn_extract_vid = QPushButton("Extract Feeding Clips from Video")
        self.btn_extract_vid_possible = QPushButton("Extract Possible Feeding Clips")
        self.btn_trim = QPushButton("Trim Video & Log")
        
        self.btn_validate.clicked.connect(self.run_validate)
        self.btn_uptime.clicked.connect(self.run_uptime)
        self.btn_lines.clicked.connect(self.run_lines)
        self.btn_csv.clicked.connect(self.run_csv)
        self.btn_csv_possible.clicked.connect(self.run_csv_possible)
        self.btn_interactive.clicked.connect(self.run_interactive)
        self.btn_baselines.clicked.connect(self.run_baselines)
        self.btn_extract_vid.clicked.connect(self.run_extract_vid)
        self.btn_extract_vid_possible.clicked.connect(self.run_extract_vid_possible)
        self.btn_trim.clicked.connect(self.run_trim)

        ops_layout.addWidget(self.btn_validate, 0, 0)
        ops_layout.addWidget(self.btn_uptime, 0, 1)
        ops_layout.addWidget(self.btn_lines, 0, 2)
        
        ops_layout.addWidget(self.btn_csv, 1, 0)
        ops_layout.addWidget(self.btn_csv_possible, 1, 1)
        ops_layout.addWidget(self.btn_interactive, 1, 2)
        
        ops_layout.addWidget(self.btn_baselines, 2, 0)
        ops_layout.addWidget(self.btn_extract_vid, 2, 1)
        ops_layout.addWidget(self.btn_extract_vid_possible, 2, 2)
        
        ops_layout.addWidget(self.btn_trim, 3, 0, 1, 3)

        ops_group.setLayout(ops_layout)
        main_layout.addWidget(ops_group)

        # Log Output
        log_group = QGroupBox("Analysis Log")
        log_layout = QVBoxLayout()
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet("background-color: #222; color: #FFF; font-family: monospace;")
        
        self.btn_save_log = QPushButton("Save Log to File")
        self.btn_save_log.clicked.connect(self.save_log)

        log_layout.addWidget(self.text_log)
        log_layout.addWidget(self.btn_save_log)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)

        self.set_bee_theme()

    def set_bee_theme(self):
        self.setStyleSheet("""
            QWidget { background-color: #1A1A1A; color: #f9c901; font-size: 12px; }
            QGroupBox { background-color: #896800; color: #1A1A1A; border: 2px solid #6b4701; border-radius: 8px; margin-top: 15px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 2px 10px; background-color: #f6e000; color: #1A1A1A; border-radius: 4px; }
            QPushButton { background-color: #6b4701; color: #f6e000; border: 1px solid #f9c901; padding: 6px 12px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #985b10; color: #f6e000; }
            QPushButton:disabled { background-color: #555; color: #888; border: 1px solid #444; }
            QLineEdit { background-color: #1A1A1A; border: 1px solid #985b10; padding: 5px; border-radius: 3px; color: #f6e000; }
            QGroupBox QLabel { color: #1A1A1A; background: transparent; }
        """)

    def log_status(self, message):
        self.text_log.append(message)

    def browse_log(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Log File", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.log_path_input.setText(path)

    def browse_vid(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.mkv *.avi);;All Files (*)")
        if path:
            self.vid_path_input.setText(path)

    def save_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Log Output", "analysis_log.txt", "Text Files (*.txt)")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(self.text_log.toPlainText())
                QMessageBox.information(self, "Success", "Log saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save log: {e}")

    def get_parsed_tags(self):
        ids_str = self.tags_input.text()
        ids = set()
        if not ids_str: return list(ids)
        try:
            parts = ids_str.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    ids.update(range(start, end + 1))
                else:
                    ids.add(int(part))
        except: pass
        return list(ids)

    def set_ui_enabled(self, enabled):
        self.btn_validate.setEnabled(enabled)
        self.btn_uptime.setEnabled(enabled)
        self.btn_lines.setEnabled(enabled)
        self.btn_csv.setEnabled(enabled)
        self.btn_csv_possible.setEnabled(enabled)
        self.btn_interactive.setEnabled(enabled)
        self.btn_baselines.setEnabled(enabled)
        self.btn_extract_vid.setEnabled(enabled)
        self.btn_extract_vid_possible.setEnabled(enabled)
        self.btn_trim.setEnabled(enabled)

    def run_task_in_bg(self, fn, *args, **kwargs):
        self.set_ui_enabled(False)
        self.worker_thread = AnalyzerWorker(fn, *args, **kwargs)
        self.worker_thread.finished.connect(self.on_task_finished)
        self.worker_thread.error.connect(self.on_task_error)
        self.worker_thread.start()

    def on_task_finished(self):
        self.log_status("Task completed.")
        self.set_ui_enabled(True)

    def on_task_error(self, err):
        self.log_status(f"ERROR during task: {err}")
        self.set_ui_enabled(True)

    def check_log_path(self):
        if not os.path.exists(self.log_path_input.text()):
            QMessageBox.warning(self, "Warning", "Please select a valid log file.")
            return False
        return True

    def run_validate(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.validate_aruco_tags, self.log_path_input.text(), self.get_parsed_tags())

    def run_uptime(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.calculate_total_log_time, self.log_path_input.text())

    def run_lines(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.number_of_lines, self.log_path_input.text(), self.get_parsed_tags())

    def run_csv(self):
        if not self.check_log_path(): return
        out_csv = os.path.join(os.path.dirname(self.log_path_input.text()), "feed_sequence_durations.csv")
        self.run_task_in_bg(self.core.time_data_for_feeding, self.log_path_input.text(), out_csv, self.get_parsed_tags())

    def run_csv_possible(self):
        if not self.check_log_path(): return
        out_csv = os.path.join(os.path.dirname(self.log_path_input.text()), "possible_feedings.csv")
        
        dialog = DurationFilterDialog(self)
        if dialog.exec():
            min_dur, max_dur, min_avg = dialog.get_settings()
            self.run_task_in_bg(self.core.analyze_possible_feeding, self.log_path_input.text(), out_csv, min_dur, max_dur, min_avg)

    def run_interactive(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.plot_complete_bee_analysis, self.log_path_input.text(), self.get_parsed_tags())

    def run_baselines(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.analyze_baselines, self.log_path_input.text())

    def run_extract_vid(self):
        if not self.check_log_path(): return
        if not os.path.exists(self.vid_path_input.text()):
            QMessageBox.warning(self, "Warning", "Please select a valid video file.")
            return
            
        sequence_csv = os.path.join(os.path.dirname(self.log_path_input.text()), "feed_sequence_durations.csv")
        if not os.path.exists(sequence_csv):
            QMessageBox.warning(self, "Warning", "feed_sequence_durations.csv not found! Run 'Extract Feeding Durations CSV' first.")
            return

        out_folder = os.path.join(os.path.dirname(self.vid_path_input.text()), "feed_sequence_clips")
        self.run_task_in_bg(self.core.extract_feed_sequence_clips, sequence_csv, self.log_path_input.text(), self.vid_path_input.text(), out_folder)

    def run_extract_vid_possible(self):
        if not self.check_log_path(): return
        if not os.path.exists(self.vid_path_input.text()):
            QMessageBox.warning(self, "Warning", "Please select a valid video file.")
            return
            
        sequence_csv = os.path.join(os.path.dirname(self.log_path_input.text()), "possible_feedings.csv")
        if not os.path.exists(sequence_csv):
            QMessageBox.warning(self, "Warning", "possible_feedings.csv not found! Run 'Extract Possible Feedings CSV' first.")
            return

        dialog = ExtractClipsDialog(self)
        if dialog.exec():
            pre_pad, post_pad, min_dur, max_dur, min_avg = dialog.get_settings()
            out_folder = os.path.join(os.path.dirname(self.vid_path_input.text()), "possible feedings")
            self.run_task_in_bg(self.core.extract_possible_feeding_clips, sequence_csv, self.log_path_input.text(), self.vid_path_input.text(), out_folder, pre_pad, post_pad, min_dur, max_dur, min_avg)

    def run_trim(self):
        if not self.check_log_path(): return
        if not os.path.exists(self.vid_path_input.text()):
            QMessageBox.warning(self, "Warning", "Please select a valid video file.")
            return

        dialog = TrimDialog(self)
        if dialog.exec():
            start_str, end_str = dialog.get_times()
            self.run_task_in_bg(self.core.trim_data, self.log_path_input.text(), self.vid_path_input.text(), start_str, end_str)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalyzerGUI()
    window.show()
    sys.exit(app.exec())
