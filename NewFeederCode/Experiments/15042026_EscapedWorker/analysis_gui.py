# title: "Tracking Data Analysis GUI"
# date: "10/27/2025"
# author: "Babur Erdem / Gemini"

import sys
import os
import subprocess
import importlib
import pandas as pd

# --- Dependency Checker ---
REQUIRED = [
    ("PySide6", "PySide6"),
    ("pandas", "pandas"),
    ("cv2", "opencv-python"),
    ("matplotlib", "matplotlib"),
]


def _ensure(mod_name, pip_name):
    if mod_name == 'analysis_core':
        try:
            importlib.import_module(mod_name)
            return True
        except ImportError:
            print(f"FATAL: Could not find 'analysis_core.py'. Make sure it's in the same directory.")
            return False
    try:
        importlib.import_module(mod_name)
        return True
    except ImportError:
        try:
            print(f"Installing missing package: {pip_name}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--user", pip_name], check=True)
            importlib.import_module(mod_name)
            return True
        except Exception as e:
            print(f"FATAL: Missing package '{pip_name}'. Please install it manually: pip install {pip_name}")
            print(f"Error details: {e}")
            return False


_missing = [p for m, p in REQUIRED if not _ensure(m, p)]
if _missing:
    sys.exit(1)

from PySide6 import QtCore, QtGui, QtWidgets
import analysis_core  # Import the backend logic


# --- Main Application Window ---
class AnalysisWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tracking Data Analyzer")
        self.resize(600, 500)

        self.output_folder_path = None
        self.experiment_name = None
        self.info_data = None
        self.coord_df = None
        self.results_df = None
        self.all_intersections = []  # (animal_id, stim_id)

        # --- GUI Elements ---
        layout = QtWidgets.QVBoxLayout(self)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # --- Left Panel: Settings ---
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        # Folder Selection
        folder_layout = QtWidgets.QHBoxLayout()
        self.lbl_folder = QtWidgets.QLabel("No Output Folder Selected.")
        self.btn_select_folder = QtWidgets.QPushButton("Load Tracking Folder...")
        folder_layout.addWidget(self.lbl_folder)
        folder_layout.addWidget(self.btn_select_folder)
        left_layout.addLayout(folder_layout)

        # Info Display
        self.lbl_info = QtWidgets.QLabel("Load a tracking output folder to begin.")
        self.lbl_info.setStyleSheet("font-style: italic;")
        self.lbl_info.setWordWrap(True)
        left_layout.addWidget(self.lbl_info)

        # Coordinate File Selection
        self.coord_group = QtWidgets.QGroupBox("Coordinate Source")
        coord_layout = QtWidgets.QHBoxLayout(self.coord_group)
        self.radio_filtered = QtWidgets.QRadioButton("Filtered")
        self.radio_raw = QtWidgets.QRadioButton("Raw")
        self.radio_filtered.setChecked(True)
        coord_layout.addWidget(self.radio_filtered)
        coord_layout.addWidget(self.radio_raw)
        left_layout.addWidget(self.coord_group)

        # Time Bin Setting
        bin_layout = QtWidgets.QHBoxLayout()
        bin_layout.addWidget(QtWidgets.QLabel("Analysis Time Bin (seconds):"))
        self.spin_time_bin = QtWidgets.QSpinBox(minimum=1, maximum=3600, value=30)
        bin_layout.addWidget(self.spin_time_bin)
        bin_layout.addStretch()
        left_layout.addLayout(bin_layout)

        # --- Metric Selection Group ---
        metrics_group = QtWidgets.QGroupBox("Select Metrics to Calculate")
        metrics_layout = QtWidgets.QVBoxLayout(metrics_group)
        self.chk_metric_speed = QtWidgets.QCheckBox("Average Speed (mm/s)")
        self.chk_metric_dist_arena = QtWidgets.QCheckBox("Average Distance to Arena Center")
        self.chk_metric_stim_duration = QtWidgets.QCheckBox("Duration in Stimulus Area")
        self.chk_metric_stim_entries = QtWidgets.QCheckBox("Entries/Exits from Stimulus Area")
        self.chk_metric_stim_dist = QtWidgets.QCheckBox("Average Distance to Stimulus Center")

        # Set defaults
        self.chk_metric_speed.setChecked(True)
        self.chk_metric_dist_arena.setChecked(True)
        self.chk_metric_stim_duration.setChecked(True)
        self.chk_metric_stim_entries.setChecked(True)
        self.chk_metric_stim_dist.setChecked(True)

        metrics_layout.addWidget(self.chk_metric_speed)
        metrics_layout.addWidget(self.chk_metric_dist_arena)
        metrics_layout.addWidget(self.chk_metric_stim_duration)
        metrics_layout.addWidget(self.chk_metric_stim_entries)
        metrics_layout.addWidget(self.chk_metric_stim_dist)
        left_layout.addWidget(metrics_group)
        self.stim_metric_checkboxes = [self.chk_metric_stim_duration, self.chk_metric_stim_entries,
                                       self.chk_metric_stim_dist]

        left_layout.addStretch()
        main_splitter.addWidget(left_widget)

        # --- Right Panel: Interactions ---
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        interaction_group = QtWidgets.QGroupBox("Arena-Stimulus Interaction")  # Text Changed
        interaction_layout = QtWidgets.QVBoxLayout(interaction_group)
        # Removed "Select which interaction..." label

        self.list_interactions = QtWidgets.QListWidget()
        self.list_interactions.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        interaction_layout.addWidget(self.list_interactions)

        self.btn_erase_interaction = QtWidgets.QPushButton("Remove Selected from Analysis")
        self.btn_erase_interaction.setToolTip(
            "Removes the selected interaction(s) from the list.\nThey will not be analyzed.")
        interaction_layout.addWidget(self.btn_erase_interaction)

        right_layout.addWidget(interaction_group)
        main_splitter.addWidget(right_widget)

        layout.addWidget(main_splitter)

        # --- Bottom Panel: Actions & Status ---
        bottom_layout = QtWidgets.QVBoxLayout()
        self.lbl_status = QtWidgets.QLabel("")
        bottom_layout.addWidget(self.lbl_status)

        button_layout = QtWidgets.QHBoxLayout()
        self.btn_run_analysis = QtWidgets.QPushButton("Run Analysis")
        self.btn_save_results = QtWidgets.QPushButton("Save Results (.txt)...")
        self.btn_save_plots = QtWidgets.QPushButton("Save Plots (.png)...")
        self.btn_exit = QtWidgets.QPushButton("Exit")

        button_layout.addWidget(self.btn_run_analysis)
        button_layout.addWidget(self.btn_save_results)
        button_layout.addWidget(self.btn_save_plots)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_exit)
        bottom_layout.addLayout(button_layout)

        layout.addLayout(bottom_layout)

        # --- Connect Signals ---
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_erase_interaction.clicked.connect(self.erase_interaction)
        self.btn_run_analysis.clicked.connect(self.run_analysis)
        self.btn_save_results.clicked.connect(self.save_results)
        self.btn_save_plots.clicked.connect(self.save_plots)
        self.btn_exit.clicked.connect(self.close)

        # --- Initial State ---
        self.coord_group.setEnabled(False)
        self.btn_run_analysis.setEnabled(False)
        self.btn_save_results.setEnabled(False)
        self.btn_save_plots.setEnabled(False)
        self.btn_erase_interaction.setEnabled(False)
        for chk in self.stim_metric_checkboxes:
            chk.setEnabled(False)

    def select_folder(self):
        """Opens a dialog to select the tracking output folder."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Tracking Output Folder")
        if not directory:
            return

        self.output_folder_path = directory
        folder_name = os.path.basename(directory)
        if folder_name.endswith("_output"):
            self.experiment_name = folder_name[:-len("_output")]
        else:
            self.experiment_name = folder_name

        self.lbl_folder.setText(f"Folder: ...{os.path.basename(directory)}")
        self.lbl_folder.setToolTip(directory)
        self.lbl_status.setText(f"Selected experiment: {self.experiment_name}")

        # Reset
        self.info_data = None
        self.coord_df = None
        self.results_df = None
        self.btn_save_results.setEnabled(False)
        self.btn_save_plots.setEnabled(False)
        self.list_interactions.clear()

        # Check for files
        info_file = os.path.join(self.output_folder_path, f"{self.experiment_name}_info.txt")
        coord_file_filt = os.path.join(self.output_folder_path, f"{self.experiment_name}_coordinates_filtered.txt")
        coord_file_raw = os.path.join(self.output_folder_path, f"{self.experiment_name}_coordinates_raw.txt")

        if not os.path.exists(info_file):
            QtWidgets.QMessageBox.warning(self, "File Not Found",
                                          f"Could not find '{self.experiment_name}_info.txt' in the selected folder.")
            self.lbl_status.setText("Error: Required _info.txt file not found.")
            return

        if not os.path.exists(coord_file_filt) and not os.path.exists(coord_file_raw):
            QtWidgets.QMessageBox.warning(self, "Files Not Found",
                                          "Could not find any 'coordinates_filtered.txt' or 'coordinates_raw.txt' files.")
            self.lbl_status.setText("Error: No coordinate files found.")
            return

        # Enable coordinate selection
        self.radio_filtered.setEnabled(os.path.exists(coord_file_filt))
        self.radio_raw.setEnabled(os.path.exists(coord_file_raw))
        if not os.path.exists(coord_file_filt):
            self.radio_raw.setChecked(True)
        else:
            self.radio_filtered.setChecked(True)
        self.coord_group.setEnabled(True)

        # Load info data immediately
        self.info_data = analysis_core.parse_info_file(info_file)
        if self.info_data is None:
            QtWidgets.QMessageBox.critical(self, "Error Loading Data", f"Failed to parse '{info_file}'. Check console.")
            self.lbl_status.setText("Error parsing info file.")
            return

        # Populate GUI
        self.lbl_info.setText(f"FPS: {self.info_data.get('fps', 'N/A')} | "
                              f"Scale: {1 / self.info_data.get('pixel_to_mm', 1.0):.2f} px/mm | "
                              f"Arenas: {len(self.info_data.get('arenas', {}))} | "
                              f"Stimuli: {len(self.info_data.get('stimulus_areas', {}))}")

        # Find and display intersections
        self.all_intersections = analysis_core.find_intersections(self.info_data)
        if self.all_intersections:
            for (a_id, s_id) in self.all_intersections:
                self.list_interactions.addItem(f"{a_id} <-> {s_id}")
            self.btn_erase_interaction.setEnabled(True)
            for chk in self.stim_metric_checkboxes:
                chk.setEnabled(True)
        else:
            self.list_interactions.addItem("No Arena/Stimulus intersections found.")
            self.btn_erase_interaction.setEnabled(False)
            for chk in self.stim_metric_checkboxes:
                chk.setEnabled(False)

        self.btn_run_analysis.setEnabled(True)
        self.lbl_status.setText(f"Ready to analyze: {self.experiment_name}")

    def erase_interaction(self):
        """Removes a selected interaction from the list."""
        selected_items = self.list_interactions.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            row = self.list_interactions.row(item)
            self.list_interactions.takeItem(row)

    def run_analysis(self):
        """Gathers settings and runs the time-binned analysis."""
        if not self.output_folder_path or not self.experiment_name or not self.info_data:
            QtWidgets.QMessageBox.critical(self, "Error", "Data not loaded. Please select a folder first.")
            return

        self.lbl_status.setText("Loading coordinate data...")
        QtWidgets.QApplication.processEvents()

        file_type = "filtered" if self.radio_filtered.isChecked() else "raw"
        coord_file = os.path.join(self.output_folder_path, f"{self.experiment_name}_coordinates_{file_type}.txt")

        self.coord_df = analysis_core.load_coordinates(coord_file)
        if self.coord_df is None:
            QtWidgets.QMessageBox.critical(self, "Error Loading Data", f"Failed to load '{coord_file}'. Check console.")
            self.lbl_status.setText("Error loading coordinates.")
            return

        # Get settings from GUI
        bin_seconds = self.spin_time_bin.value()
        selected_metrics = {
            'speed': self.chk_metric_speed.isChecked(),
            'dist_arena': self.chk_metric_dist_arena.isChecked(),
            'stim_duration': self.chk_metric_stim_duration.isChecked(),
            'stim_entries': self.chk_metric_stim_entries.isChecked(),
            'stim_dist': self.chk_metric_stim_dist.isChecked(),
        }

        active_interactions = []
        for i in range(self.list_interactions.count()):
            item_text = self.list_interactions.item(i).text()
            if " <-> " in item_text:
                a_id, s_id = item_text.split(" <-> ")
                active_interactions.append((a_id, s_id))

        self.lbl_status.setText(f"Running analysis on {file_type} data...")
        QtWidgets.QApplication.processEvents()

        # Run analysis
        try:
            self.results_df = analysis_core.analyze_data_in_bins(
                self.experiment_name,
                self.coord_df,
                self.info_data,
                bin_seconds,
                selected_metrics,
                active_interactions
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis:\n{e}")
            self.lbl_status.setText("Analysis failed.")
            self.results_df = None
            self.btn_save_results.setEnabled(False)
            self.btn_save_plots.setEnabled(False)
            import traceback
            traceback.print_exc()
            return

        if self.results_df is not None and not self.results_df.empty:
            self.lbl_status.setText(f"Analysis complete. Found {len(self.results_df)} data points.")
            self.btn_save_results.setEnabled(True)
            self.btn_save_plots.setEnabled(True)
        else:
            self.lbl_status.setText("Analysis finished, but no results generated.")
            self.btn_save_results.setEnabled(False)
            self.btn_save_plots.setEnabled(False)

    def save_results(self):
        """Saves the analyzed data to a tab-delimited file."""
        if self.results_df is None or self.results_df.empty:
            QtWidgets.QMessageBox.warning(self, "No Results", "No analysis results available to save.")
            return

        file_type = "filtered" if self.radio_filtered.isChecked() else "raw"
        default_filename = os.path.join(self.output_folder_path,
                                        f"{self.experiment_name}_analysis_{file_type}_{self.spin_time_bin.value()}s.txt")

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Analysis Results", default_filename, "Tab-Separated Text (*.txt);;CSV Files (*.csv)")

        if save_path:
            try:
                # Save as tab-separated
                self.results_df.to_csv(save_path, index=False, sep='\t', float_format='%.4f')
                self.lbl_status.setText(f"Results saved to {os.path.basename(save_path)}")
                QtWidgets.QMessageBox.information(self, "Save Successful", f"Results saved to:\n{save_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Error", f"Could not save results:\n{e}")
                self.lbl_status.setText("Error saving results.")

    def save_plots(self):
        """Saves the analysis plots and heatmaps to a selected directory."""
        if self.results_df is None or self.results_df.empty or self.coord_df is None:
            QtWidgets.QMessageBox.warning(self, "No Results", "No analysis results available to plot.")
            return

        selected_metrics = {
            'speed': self.chk_metric_speed.isChecked(),
            'dist_arena': self.chk_metric_dist_arena.isChecked(),
            'stim_duration': self.chk_metric_stim_duration.isChecked(),
            'stim_entries': self.chk_metric_stim_entries.isChecked(),
            'stim_dist': self.chk_metric_stim_dist.isChecked(),
        }

        save_directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Plots", self.output_folder_path)

        if save_directory:
            try:
                self.lbl_status.setText("Generating plots...")
                QtWidgets.QApplication.processEvents()

                # --- Generate Line Plots ---
                analysis_core.create_visualizations(
                    self.results_df,
                    self.info_data,
                    save_directory,
                    self.experiment_name,
                    selected_metrics
                )

                # --- Generate Heatmaps ---
                file_type = "filtered" if self.radio_filtered.isChecked() else "raw"
                coord_file_path = os.path.join(self.output_folder_path,
                                               f"{self.experiment_name}_coordinates_{file_type}.txt")
                # Re-load coordinates just for heatmap (or pass self.coord_df)
                coords_for_heatmap = analysis_core.load_coordinates(coord_file_path)

                analysis_core.create_heatmaps(
                    coords_for_heatmap,
                    self.info_data,
                    save_directory,
                    self.experiment_name
                )

                self.lbl_status.setText("Plots saved successfully.")
                QtWidgets.QMessageBox.information(self, "Plots Saved",
                                                  f"Plots and heatmaps saved to:\n{save_directory}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Plotting Error", f"Could not save plots:\n{e}")
                self.lbl_status.setText("Error saving plots.")
                import traceback
                traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = AnalysisWindow()
    window.show()
    sys.exit(app.exec())