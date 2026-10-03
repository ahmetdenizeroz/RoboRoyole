import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QPushButton, QLineEdit, QFileDialog,
    QLabel, QTextEdit, QMessageBox, QDialog, QFormLayout, QDialogButtonBox,
    QSplitter, QListWidget, QListWidgetItem, QTabWidget
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
    def __init__(self, parent=None, default_min="0.0", default_max="999999.0", default_intensity="0.0", default_merge="8.0"):
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
        self.merge_input = QLineEdit(default_merge)
        layout.addRow("Minimum duration (s):", self.min_input)
        layout.addRow("Maximum duration (s):", self.max_input)
        layout.addRow("Minimum avg intensity:", self.intensity_input)
        layout.addRow("Merge gap (s):", self.merge_input)
        
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
        try:
            merge_gap = float(self.merge_input.text())
        except ValueError:
            merge_gap = 1.0
        return min_dur, max_dur, min_avg, merge_gap

class DownsizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downsize Video")
        self.setStyleSheet("""
            QDialog { background-color: #1A1A1A; color: #f9c901; }
            QLabel { color: #f9c901; font-weight: bold; }
            QLineEdit { background-color: #222; border: 1px solid #985b10; color: #f6e000; padding: 4px; }
            QPushButton { background-color: #6b4701; color: #f6e000; padding: 5px; font-weight: bold; border-radius: 3px; }
            QPushButton:hover { background-color: #985b10; }
        """)
        layout = QFormLayout(self)
        self.size_input = QLineEdit("50")
        layout.addRow("Target Size (MB):", self.size_input)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_target_size(self):
        try:
            return float(self.size_input.text())
        except ValueError:
            return 50.0

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
        self.selected_log_path = None
        self.current_folder = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # --- LEFT PANEL ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.btn_select_folder = QPushButton("Select Workspace Folder")
        self.btn_select_folder.clicked.connect(self.browse_folder)
        
        left_layout.addWidget(self.btn_select_folder)
        left_layout.addWidget(QLabel("Available Log Files (.txt):"))
        
        self.file_list = QListWidget()
        self.file_list.itemSelectionChanged.connect(self.on_file_selected)
        left_layout.addWidget(self.file_list)
        
        splitter.addWidget(left_panel)
        
        # --- RIGHT PANEL ---
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)

        # Inputs Group
        inputs_group = QGroupBox("Inputs & Configuration")
        inputs_layout = QGridLayout()

        self.lbl_current_file = QLabel("Currently Analyzing: None")
        self.lbl_current_file.setStyleSheet("color: #f6e000; font-weight: bold; font-size: 14px;")

        self.vid_path_input = QLineEdit()
        self.btn_browse_vid = QPushButton("Browse Video (.mp4/.mkv)")
        self.btn_browse_vid.clicked.connect(self.browse_vid)

        self.tags_input = QLineEdit("1, 2, 5-10")

        inputs_layout.addWidget(self.lbl_current_file, 0, 0, 1, 3)

        inputs_layout.addWidget(QLabel("Video File (Optional):"), 1, 0)
        inputs_layout.addWidget(self.vid_path_input, 1, 1)
        inputs_layout.addWidget(self.btn_browse_vid, 1, 2)

        inputs_layout.addWidget(QLabel("Target Tags (e.g. 1, 2, 5-10):"), 2, 0)
        inputs_layout.addWidget(self.tags_input, 2, 1)

        inputs_group.setLayout(inputs_layout)
        self.right_layout.addWidget(inputs_group)

        # Operations Group
        ops_group = QGroupBox("Analysis Operations")
        ops_layout = QVBoxLayout()
        
        self.tabs = QTabWidget()
        
        tab_basic = QWidget()
        layout_basic = QVBoxLayout(tab_basic)
        self.btn_validate = QPushButton("Validate ArUco Tags")
        self.btn_uptime = QPushButton("Calculate Total Uptime")
        self.btn_lines = QPushButton("Feeds Per Bee Count")
        self.btn_baselines = QPushButton("Analyze Sensor Baselines")
        layout_basic.addWidget(self.btn_validate)
        layout_basic.addWidget(self.btn_uptime)
        layout_basic.addWidget(self.btn_lines)
        layout_basic.addWidget(self.btn_baselines)
        layout_basic.addStretch()
        
        tab_data = QWidget()
        layout_data = QVBoxLayout(tab_data)
        self.btn_csv = QPushButton("Extract Feeding Durations CSV")
        self.btn_csv_possible = QPushButton("Extract Possible Feedings CSV")
        layout_data.addWidget(self.btn_csv)
        layout_data.addWidget(self.btn_csv_possible)
        layout_data.addStretch()
        
        tab_vis = QWidget()
        layout_vis = QVBoxLayout(tab_vis)
        self.btn_interactive = QPushButton("Interactive Bee Plot (HTML)")
        self.btn_plot_cycles = QPushButton("Plot Scaled Feeding Cycles")
        self.btn_plot_possible = QPushButton("Plot Possible Feedings")
        self.btn_plot_electrodes = QPushButton("Plot Electrode Values")
        self.btn_plot_soft_stop = QPushButton("Plot Soft Stop Triggers (HTML)")
        self.btn_plot_steps = QPushButton("Plot Steps Set (HTML)")
        layout_vis.addWidget(self.btn_interactive)
        layout_vis.addWidget(self.btn_plot_cycles)
        layout_vis.addWidget(self.btn_plot_possible)
        layout_vis.addWidget(self.btn_plot_electrodes)
        layout_vis.addWidget(self.btn_plot_soft_stop)
        layout_vis.addWidget(self.btn_plot_steps)
        layout_vis.addStretch()
        
        tab_video = QWidget()
        layout_video = QVBoxLayout(tab_video)
        self.btn_extract_vid_possible = QPushButton("Extract Possible Feeding Clips")
        self.btn_trim = QPushButton("Trim Video & Log")
        self.btn_downsize = QPushButton("Downsize Video")
        layout_video.addWidget(self.btn_extract_vid_possible)
        layout_video.addWidget(self.btn_trim)
        layout_video.addWidget(self.btn_downsize)
        layout_video.addStretch()
        
        self.tabs.addTab(tab_basic, "Basic Analysis")
        self.tabs.addTab(tab_data, "Data Extraction")
        self.tabs.addTab(tab_vis, "Visualization")
        self.tabs.addTab(tab_video, "Video Tools")
        
        ops_layout.addWidget(self.tabs)
        
        self.btn_validate.clicked.connect(self.run_validate)
        self.btn_uptime.clicked.connect(self.run_uptime)
        self.btn_lines.clicked.connect(self.run_lines)
        self.btn_csv.clicked.connect(self.run_csv)
        self.btn_csv_possible.clicked.connect(self.run_csv_possible)
        self.btn_interactive.clicked.connect(self.run_interactive)
        self.btn_plot_cycles.clicked.connect(self.run_plot_cycles)
        self.btn_plot_possible.clicked.connect(self.run_plot_possible)
        self.btn_plot_electrodes.clicked.connect(self.run_plot_electrodes)
        self.btn_plot_soft_stop.clicked.connect(self.run_plot_soft_stop)
        self.btn_plot_steps.clicked.connect(self.run_plot_steps)
        self.btn_baselines.clicked.connect(self.run_baselines)
        self.btn_extract_vid_possible.clicked.connect(self.run_extract_vid_possible)
        self.btn_trim.clicked.connect(self.run_trim)
        self.btn_downsize.clicked.connect(self.run_downsize)

        ops_group.setLayout(ops_layout)
        self.right_layout.addWidget(ops_group)

        # Log Output
        log_group = QGroupBox("Analysis Log")
        log_layout = QVBoxLayout()
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet("background-color: #222; color: #FFF; font-family: monospace;")
        
        self.btn_save_log = QPushButton("Save Log to File")
        self.btn_save_log.clicked.connect(self.save_log)

        self.btn_clear_log = QPushButton("Clear Log")
        self.btn_clear_log.clicked.connect(self.text_log.clear)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_save_log)
        btn_layout.addWidget(self.btn_clear_log)

        log_layout.addWidget(self.text_log)
        log_layout.addLayout(btn_layout)
        log_group.setLayout(log_layout)
        self.right_layout.addWidget(log_group)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 650])

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
            QTabWidget::pane { border: 1px solid #985b10; background: #1A1A1A; border-radius: 4px; }
            QTabBar::tab { background: #333; color: #f9c901; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; border: 1px solid #444; border-bottom: none; }
            QTabBar::tab:selected { background: #6b4701; color: #f6e000; border: 1px solid #985b10; border-bottom: none; font-weight: bold; }
            QTabBar::tab:hover:!selected { background: #555; }
        """)

    def log_status(self, message):
        self.text_log.append(message)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Workspace Folder")
        if folder:
            self.current_folder = folder
            self.refresh_file_list()

    def refresh_file_list(self):
        self.file_list.clear()
        if not hasattr(self, 'current_folder') or not self.current_folder: return
        
        for f in os.listdir(self.current_folder):
            if f.endswith('.txt'):
                item = QListWidgetItem(f)
                item.setData(Qt.UserRole, os.path.join(self.current_folder, f))
                self.file_list.addItem(item)

    def on_file_selected(self):
        items = self.file_list.selectedItems()
        if items:
            path = items[0].data(Qt.UserRole)
            self.selected_log_path = path
            self.lbl_current_file.setText(f"Currently Analyzing: {items[0].text()}")
        else:
            self.selected_log_path = None
            self.lbl_current_file.setText("Currently Analyzing: None")

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
        self.btn_plot_cycles.setEnabled(enabled)
        self.btn_plot_possible.setEnabled(enabled)
        self.btn_plot_electrodes.setEnabled(enabled)
        self.btn_plot_soft_stop.setEnabled(enabled)
        self.btn_plot_steps.setEnabled(enabled)
        self.btn_baselines.setEnabled(enabled)
        self.btn_extract_vid_possible.setEnabled(enabled)
        self.btn_trim.setEnabled(enabled)
        self.btn_downsize.setEnabled(enabled)

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
        if not self.selected_log_path or not os.path.exists(self.selected_log_path):
            QMessageBox.warning(self, "Warning", "Please select a valid log file from the list.")
            return False
        return True

    def run_validate(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.validate_aruco_tags, self.selected_log_path, self.get_parsed_tags())

    def run_uptime(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.calculate_total_log_time, self.selected_log_path)

    def run_lines(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.number_of_lines, self.selected_log_path, self.get_parsed_tags())

    def run_csv(self):
        if not self.check_log_path(): return
        base_name = os.path.splitext(os.path.basename(self.selected_log_path))[0]
        out_csv = os.path.join(os.path.dirname(self.selected_log_path), f"{base_name}_feed_durations.csv")
        self.run_task_in_bg(self.core.time_data_for_feeding, self.selected_log_path, out_csv, self.get_parsed_tags())

    def run_csv_possible(self):
        if not self.check_log_path(): return
        base_name = os.path.splitext(os.path.basename(self.selected_log_path))[0]
        out_csv = os.path.join(os.path.dirname(self.selected_log_path), f"{base_name}_possible_feedings.csv")
        
        dialog = DurationFilterDialog(self)
        if dialog.exec():
            min_dur, max_dur, min_avg, merge_gap = dialog.get_settings()
            self.run_task_in_bg(self.core.analyze_possible_feeding, self.selected_log_path, out_csv, min_dur, max_dur, min_avg, merge_gap)

    def run_interactive(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.plot_complete_bee_analysis, self.selected_log_path, self.get_parsed_tags())

    def run_plot_cycles(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.plot_scaled_feeding_cycles, self.selected_log_path, self.get_parsed_tags())

    def run_plot_possible(self):
        if not self.check_log_path(): return
        base_name = os.path.splitext(os.path.basename(self.selected_log_path))[0]
        sequence_csv = os.path.join(os.path.dirname(self.selected_log_path), f"{base_name}_possible_feedings.csv")
        if not os.path.exists(sequence_csv):
            QMessageBox.warning(self, "Warning", f"{base_name}_possible_feedings.csv not found! Run 'Extract Possible Feedings CSV' first.")
            return
        self.run_task_in_bg(self.core.plot_possible_feedings, sequence_csv)

    def run_plot_electrodes(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.plot_all_electrodes, self.selected_log_path)

    def run_plot_soft_stop(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.plot_soft_stop, self.selected_log_path)

    def run_plot_steps(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.plot_steps_set, self.selected_log_path)

    def run_baselines(self):
        if not self.check_log_path(): return
        self.run_task_in_bg(self.core.analyze_baselines, self.selected_log_path)

    def run_extract_vid_possible(self):
        if not self.check_log_path(): return
        if not os.path.exists(self.vid_path_input.text()):
            QMessageBox.warning(self, "Warning", "Please select a valid video file.")
            return
            
        base_name = os.path.splitext(os.path.basename(self.selected_log_path))[0]
        sequence_csv = os.path.join(os.path.dirname(self.selected_log_path), f"{base_name}_possible_feedings.csv")
        if not os.path.exists(sequence_csv):
            QMessageBox.warning(self, "Warning", f"{base_name}_possible_feedings.csv not found! Run 'Extract Possible Feedings CSV' first.")
            return

        dialog = ExtractClipsDialog(self)
        if dialog.exec():
            pre_pad, post_pad, min_dur, max_dur, min_avg = dialog.get_settings()
            out_folder = os.path.join(os.path.dirname(self.vid_path_input.text()), f"{base_name}_possible_feedings_clips")
            self.run_task_in_bg(self.core.extract_possible_feeding_clips, sequence_csv, self.selected_log_path, self.vid_path_input.text(), out_folder, pre_pad, post_pad, min_dur, max_dur, min_avg)

    def run_trim(self):
        if not self.check_log_path(): return
        if not os.path.exists(self.vid_path_input.text()):
            QMessageBox.warning(self, "Warning", "Please select a valid video file.")
            return

        dialog = TrimDialog(self)
        if dialog.exec():
            start_str, end_str = dialog.get_times()
            self.run_task_in_bg(self.core.trim_data, self.selected_log_path, self.vid_path_input.text(), start_str, end_str)

    def run_downsize(self):
        if not os.path.exists(self.vid_path_input.text()):
            QMessageBox.warning(self, "Warning", "Please select a valid video file.")
            return

        dialog = DownsizeDialog(self)
        if dialog.exec():
            target_mb = dialog.get_target_size()
            self.run_task_in_bg(self.core.downsize_video, self.vid_path_input.text(), target_mb)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalyzerGUI()
    window.show()
    sys.exit(app.exec())
