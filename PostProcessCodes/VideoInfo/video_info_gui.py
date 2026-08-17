import sys
from PySide6 import QtWidgets, QtCore
from video_info_core import VideoInfoExtractor

class VideoInfoGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Information Extractor")
        self.resize(500, 600)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        # Top section
        top_layout = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Load Video")
        self.btn_load.clicked.connect(self._load_video)
        self.lbl_path = QtWidgets.QLabel("No video loaded.")
        self.lbl_path.setWordWrap(True)
        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.lbl_path, 1)
        main_layout.addLayout(top_layout)

        # Scroll area for details
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(content_widget)

        # General Group
        self.grp_general = QtWidgets.QGroupBox("General Information")
        self.form_general = QtWidgets.QFormLayout(self.grp_general)
        self.content_layout.addWidget(self.grp_general)

        # Video Group
        self.grp_video = QtWidgets.QGroupBox("Video Stream")
        self.form_video = QtWidgets.QFormLayout(self.grp_video)
        self.content_layout.addWidget(self.grp_video)

        # Audio Group
        self.grp_audio = QtWidgets.QGroupBox("Audio Stream")
        self.form_audio = QtWidgets.QFormLayout(self.grp_audio)
        self.content_layout.addWidget(self.grp_audio)

        self.content_layout.addStretch()
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

    def _load_video(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*.*)"
        )
        if not file_path:
            return

        self.lbl_path.setText(file_path)
        
        extractor = VideoInfoExtractor(file_path)
        try:
            info = extractor.extract_info()
            self._populate_ui(info)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            self._clear_forms()

    def _clear_forms(self):
        for form in [self.form_general, self.form_video, self.form_audio]:
            while form.rowCount() > 0:
                form.removeRow(0)

    def _populate_ui(self, info: dict):
        self._clear_forms()

        # Helper to safely display values
        def add_row(form, label, value, suffix=""):
            val_str = str(value)
            if not val_str or val_str == "0" or val_str == "Unknown":
                val_str = "N/A"
            elif suffix and val_str != "N/A":
                val_str += f" {suffix}"
            form.addRow(label + ":", QtWidgets.QLabel(val_str))

        gen = info.get("general", {})
        add_row(self.form_general, "File Name", gen.get("filename"))
        add_row(self.form_general, "Container Format", gen.get("container_format"))
        add_row(self.form_general, "File Size", gen.get("size_mb"), "MB")
        add_row(self.form_general, "Duration", gen.get("duration_sec"), "sec")
        add_row(self.form_general, "Overall Bitrate", gen.get("overall_bitrate_kbps"), "kbps")

        vid = info.get("video", {})
        add_row(self.form_video, "Codec", f"{vid.get('codec')} ({vid.get('profile')})")
        add_row(self.form_video, "Resolution", f"{vid.get('width')}x{vid.get('height')}")
        add_row(self.form_video, "Frame Rate", vid.get("fps"), "FPS")
        add_row(self.form_video, "Total Frames", vid.get("total_frames"))
        add_row(self.form_video, "Pixel Format", vid.get("pixel_format"))
        add_row(self.form_video, "Display Aspect Ratio", vid.get("display_aspect_ratio"))
        add_row(self.form_video, "Bitrate", vid.get("bitrate_kbps"), "kbps")

        aud = info.get("audio", {})
        if not aud:
            self.form_audio.addRow(QtWidgets.QLabel("No audio stream found."))
        else:
            add_row(self.form_audio, "Codec", aud.get("codec"))
            add_row(self.form_audio, "Channels", aud.get("channels"))
            add_row(self.form_audio, "Channel Layout", aud.get("channel_layout"))
            add_row(self.form_audio, "Sample Rate", aud.get("sample_rate_hz"), "Hz")
            add_row(self.form_audio, "Bitrate", aud.get("bitrate_kbps"), "kbps")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VideoInfoGUI()
    window.show()
    sys.exit(app.exec())
