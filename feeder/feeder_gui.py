 # feeder_gui.py
# This file creates the PySide6 GUI for the Feeder Controller.
# It imports and controls the 'FeederController' class from 'feeder_core.py'.
import sys
import time
import numpy as np
import serial.tools.list_ports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QPushButton, QLineEdit, QComboBox,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QMessageBox,
    QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QPen,
    QMouseEvent, QWheelEvent, QKeyEvent
)

# Import the core logic
try:
    from feeder_core import FeederController
except ImportError:
    print("Error: feeder_core.py not found. Make sure it's in the same directory.")
    sys.exit(1)


# --- Worker Thread ---
# This runs blocking tasks (like connecting) in the background
class Worker(QThread):
    finished = Signal(bool, str)  # Emits success(bool) and message(str)

    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args

    def run(self):
        try:
            result = self.fn(*self.args)
            self.finished.emit(result, "")  # Success
        except Exception as e:
            self.finished.emit(False, str(e))  # Failure


# --- Video Canvas ---
# A custom QLabel for displaying video and drawing the zone
class VideoCanvas(QLabel):
    # Emits x, y, radius
    zone_selected = Signal(int, int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #333; color: #888; border: 1px solid #555;")
        self.setText("Camera Disconnected")

        self.img_size = (0, 0)  # w, h of the actual video frame
        self.pixmap_size = (0, 0)  # w, h of the scaled pixmap
        self.pixmap_offset = (0, 0)  # x, y offset of the pixmap

        # --- Zone definition ---
        self.zone_center = None  # (x, y) in image coordinates
        self.zone_radius = 0

        # --- Drawing state ---
        self.is_drawing = False
        self.is_dragging = False
        self.start_point = None  # (x, y) in image coordinates
        self.drag_offset = None  # (x, y) offset from circle center

        # Allow this widget to receive key presses (for 'Del')
        self.setFocusPolicy(Qt.StrongFocus)

    def set_frame(self, rgb_frame: np.ndarray):
        """Update the label with a new video frame."""
        if rgb_frame is None or rgb_frame.size == 0:
            self.clear_frame()
            return

        h, w, ch = rgb_frame.shape
        self.img_size = (w, h)
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale the pixmap to fit the label while maintaining aspect ratio
        self.q_pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = self.q_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.pixmap_size = (scaled_pixmap.width(), scaled_pixmap.height())

        # Calculate offsets for centering the pixmap
        self.pixmap_offset = (
            (self.width() - self.pixmap_size[0]) // 2,
            (self.height() - self.pixmap_size[1]) // 2
        )

        # We need a new pixmap to draw on
        canvas_pixmap = QPixmap(self.size())
        canvas_pixmap.fill(Qt.black)  # Fill background

        painter = QPainter(canvas_pixmap)
        # Draw the video frame
        painter.drawPixmap(self.pixmap_offset[0], self.pixmap_offset[1], scaled_pixmap)

        # --- "TWO CIRCLES" BUG FIX ---
        # The circle is now drawn by feeder_core.py *before* this function.
        # We no longer draw a second circle here.

        painter.end()
        self.setPixmap(canvas_pixmap)

    def clear_frame(self):
        """Show a blank screen when disconnected."""
        self.img_size = (0, 0)
        blank_pixmap = QPixmap(self.size())
        blank_pixmap.fill(Qt.black)
        self.setPixmap(blank_pixmap)
        self.setText("Camera Disconnected")

    def _pixmap_to_image(self, p_pos):
        """Convert QPoint from pixmap coordinates to image coordinates."""
        if self.img_size == (0, 0) or self.pixmap_size == (0, 0) or self.pixmap_size[0] == 0 or self.pixmap_size[
            1] == 0:
            return None

        # Adjust for pixmap offset
        px = p_pos.x() - self.pixmap_offset[0]
        py = p_pos.y() - self.pixmap_offset[1]

        # Scale to image coordinates
        scale_x = self.img_size[0] / self.pixmap_size[0]
        scale_y = self.img_size[1] / self.pixmap_size[1]

        img_x = int(px * scale_x)
        img_y = int(py * scale_y)

        # Clamp to image boundaries
        img_x = max(0, min(self.img_size[0] - 1, img_x))
        img_y = max(0, min(self.img_size[1] - 1, img_y))

        return (img_x, img_y)

    def _image_to_pixmap(self, i_pos):
        """Convert image coordinates to pixmap coordinates."""
        if self.img_size == (0, 0) or self.pixmap_size == (0, 0) or self.img_size[0] == 0 or self.img_size[1] == 0:
            return (0, 0)

        scale_x = self.pixmap_size[0] / self.img_size[0]
        scale_y = self.pixmap_size[1] / self.img_size[1]

        px = int(i_pos[0] * scale_x) + self.pixmap_offset[0]
        py = int(i_pos[1] * scale_y) + self.pixmap_offset[1]

        return (px, py)

    def _is_inside_zone(self, img_pos):
        """Check if a point (in image coords) is inside the existing zone."""
        if not self.zone_center or self.zone_radius == 0:
            return False

        dist = np.sqrt((img_pos[0] - self.zone_center[0]) ** 2 + (img_pos[1] - self.zone_center[1]) ** 2)
        return dist < self.zone_radius

    def mousePressEvent(self, ev: QMouseEvent):
        if ev.button() != Qt.LeftButton or self.img_size == (0, 0):
            return

        img_pos = self._pixmap_to_image(ev.position().toPoint())
        if not img_pos:
            return

        if self._is_inside_zone(img_pos):
            # Start dragging the existing circle
            self.is_dragging = True
            self.drag_offset = (img_pos[0] - self.zone_center[0], img_pos[1] - self.zone_center[1])
        else:
            # Start drawing a new circle
            self.is_drawing = True
            self.start_point = img_pos
            self.zone_center = img_pos
            self.zone_radius = 0

        self.update()  # Force repaint

    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.img_size == (0, 0):
            return

        img_pos = self._pixmap_to_image(ev.position().toPoint())
        if not img_pos:
            return

        if self.is_dragging:
            # Move the circle center
            self.zone_center = (img_pos[0] - self.drag_offset[0], img_pos[1] - self.drag_offset[1])
            self.zone_selected.emit(self.zone_center[0], self.zone_center[1], self.zone_radius)

        elif self.is_drawing and self.start_point:
            # Calculate new radius
            r = np.sqrt((img_pos[0] - self.start_point[0]) ** 2 + (img_pos[1] - self.start_point[1]) ** 2)
            self.zone_radius = int(r)
            self.zone_center = self.start_point
            self.zone_selected.emit(self.zone_center[0], self.zone_center[1], self.zone_radius)

    def mouseReleaseEvent(self, ev: QMouseEvent):
        if ev.button() != Qt.LeftButton:
            return

        if self.is_dragging:
            self.is_dragging = False
            self.drag_offset = None

        elif self.is_drawing:
            self.is_drawing = False
            self.start_point = None

    def wheelEvent(self, ev: QWheelEvent):
        """Handle resizing the circle with the scroll wheel."""
        if self.img_size == (0, 0) or not self.zone_center:
            return

        img_pos = self._pixmap_to_image(ev.position().toPoint())
        if not img_pos or not self._is_inside_zone(img_pos):
            return  # Only resize if mouse is inside the circle

        # Determine scroll direction
        delta = ev.angleDelta().y()
        if delta > 0:
            self.zone_radius = max(5, self.zone_radius + 5)  # Increase radius
        elif delta < 0:
            self.zone_radius = max(5, self.zone_radius - 5)  # Decrease radius

        self.zone_selected.emit(self.zone_center[0], self.zone_center[1], self.zone_radius)

    def keyPressEvent(self, ev: QKeyEvent):
        """Handle deleting the circle with the 'Del' key."""
        if ev.key() == Qt.Key_Delete:
            self.zone_center = None
            self.zone_radius = 0
            self.zone_selected.emit(0, 0, 0)  # Emit a cleared zone

    @Slot(int, int, int)
    def update_zone_visual(self, x, y, r):
        """Slot to update the visual from the core (e.g., if set by another user)."""
        self.zone_center = (x, y)
        self.zone_radius = r


# --- Main GUI Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feeder Control Panel")
        self.setGeometry(100, 100, 450, 900)  # x, y, w, h

        # Create the core controller
        self.controller = FeederController()

        # Create the video canvas window
        self.canvas_window = VideoCanvas()
        self.canvas_window.setWindowTitle("Live Canvas (Draw Zone Here)")
        self.canvas_window.setGeometry(600, 100, 800, 600)
        self.canvas_window.show()

        # --- Connect Signals ---
        self.controller.frame_ready.connect(self.canvas_window.set_frame)
        self.controller.status_updated.connect(self.log_status)
        self.controller.recording_toggled.connect(self.update_record_button_style)
        self.controller.zone_updated.connect(self.canvas_window.update_zone_visual)
        self.canvas_window.zone_selected.connect(self.controller.set_feeding_zone)

        # Worker thread for blocking tasks
        self.worker = None

        # --- Main Layout ---
        # This widget will hold all the controls
        self.scroll_content = QWidget()
        main_layout = QVBoxLayout(self.scroll_content)  # Add layout to the content widget
        main_layout.setSpacing(10)

        # --- Connection Group ---
        connection_group = QGroupBox("Connection")
        connection_layout = QGridLayout()

        # Camera
        self.cam_index_input = QLineEdit("0")
        self.cam_index_input.setFixedWidth(40)
        self.btn_connect_cam = QPushButton("Connect Camera")

        # Arduino
        self.port_combo = QComboBox()
        self.btn_refresh_ports = QPushButton("Refresh")
        self.btn_connect_arduino = QPushButton("Connect Arduino")

        connection_layout.addWidget(QLabel("Camera Index:"), 0, 0)
        connection_layout.addWidget(self.cam_index_input, 0, 1)
        connection_layout.addWidget(self.btn_connect_cam, 0, 2, 1, 2)
        connection_layout.addWidget(QLabel("Serial Port:"), 1, 0)
        connection_layout.addWidget(self.port_combo, 1, 1, 1, 2)
        connection_layout.addWidget(self.btn_refresh_ports, 1, 3)
        connection_layout.addWidget(self.btn_connect_arduino, 2, 0, 1, 4)

        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)

        # --- Camera Control Group ---
        camera_control_group = QGroupBox("Camera Controls")
        camera_control_layout = QGridLayout()
        self.btn_exposure_lock = QPushButton("Exposure Lock: OFF")
        self.btn_autofocus = QPushButton("Autofocus: ON")
        self.slider_focus = QSlider(Qt.Horizontal)
        self.slider_focus.setRange(0, 255)
        self.slider_focus.setEnabled(False)  # Disabled by default

        camera_control_layout.addWidget(self.btn_exposure_lock, 0, 0)
        camera_control_layout.addWidget(self.btn_autofocus, 0, 1)
        camera_control_layout.addWidget(QLabel("Manual Focus:"), 1, 0)
        camera_control_layout.addWidget(self.slider_focus, 1, 1)

        camera_control_group.setLayout(camera_control_layout)
        main_layout.addWidget(camera_control_group)

        # --- Electrode Settings Group ---
        electrode_group = QGroupBox("Electrode Settings (Analog 0-1023)")
        electrode_layout = QGridLayout()
        self.spin_thresh1 = QSpinBox()
        self.spin_thresh1.setRange(0, 1023)
        self.spin_thresh1.setValue(600)
        self.spin_thresh2 = QSpinBox()
        self.spin_thresh2.setRange(0, 1023)
        self.spin_thresh2.setValue(580)
        self.spin_thresh3 = QSpinBox()
        self.spin_thresh3.setRange(0, 1023)
        self.spin_thresh3.setValue(530)
        self.btn_apply_electrodes = QPushButton("Apply Electrode Settings")

        electrode_layout.addWidget(QLabel("Threshold 1 (Back):"), 0, 0)
        electrode_layout.addWidget(self.spin_thresh1, 0, 1)
        electrode_layout.addWidget(QLabel("Threshold 2 (Middle):"), 1, 0)
        electrode_layout.addWidget(self.spin_thresh2, 1, 1)
        electrode_layout.addWidget(QLabel("Threshold 3 (Front):"), 2, 0)
        electrode_layout.addWidget(self.spin_thresh3, 2, 1)
        electrode_layout.addWidget(self.btn_apply_electrodes, 3, 0, 1, 2)

        electrode_group.setLayout(electrode_layout)
        main_layout.addWidget(electrode_group)

        # --- Detection Settings Group ---
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QGridLayout()
        self.ids_input = QLineEdit("1, 2, 5-10")
        self.min_size_input = QLineEdit("50")
        self.max_size_input = QLineEdit("2000")
        self.timeout_input = QLineEdit("10.0")

        # NEW Frame Inputs
        self.entry_frames_input = QLineEdit("15")
        self.exit_frames_input = QLineEdit("30")

        self.btn_apply_detection = QPushButton("Apply Detection Settings")

        detection_layout.addWidget(QLabel("Allowed IDs:"), 0, 0)
        detection_layout.addWidget(self.ids_input, 0, 1)
        detection_layout.addWidget(QLabel("Min Size (Perim):"), 1, 0)
        detection_layout.addWidget(self.min_size_input, 1, 1)
        detection_layout.addWidget(QLabel("Max Size (Perim):"), 2, 0)
        detection_layout.addWidget(self.max_size_input, 2, 1)
        detection_layout.addWidget(QLabel("Feeding Duration (s):"), 3, 0)  # Renamed
        detection_layout.addWidget(self.timeout_input, 3, 1)
        detection_layout.addWidget(QLabel("Entry Confirmation (frames):"), 4, 0)
        detection_layout.addWidget(self.entry_frames_input, 4, 1)
        detection_layout.addWidget(QLabel("Exit Confirmation (frames):"), 5, 0)
        detection_layout.addWidget(self.exit_frames_input, 5, 1)
        detection_layout.addWidget(self.btn_apply_detection, 6, 0, 1, 2)

        detection_group.setLayout(detection_layout)
        main_layout.addWidget(detection_group)

        # --- Motor Settings Group ---
        motor_group = QGroupBox("Motor Settings")
        motor_layout = QGridLayout()
        self.speed_input = QLineEdit("1000")
        self.accel_input = QLineEdit("500")
        self.btn_apply_motor = QPushButton("Apply Motor Settings")

        # --- NEW Start/Stop Button ---
        self.btn_start_waiting = QPushButton("Start Motor (Waiting Mode)")
        self.btn_start_waiting.setCheckable(True)  # Make it a toggle
        self.btn_start_waiting.setStyleSheet("background-color: #4CAF50; color: white;")  # Green

        # Manual buttons
        self.btn_manual_eject = QPushButton("Manual Eject")
        self.btn_manual_retract = QPushButton("Manual Retract")
        self.btn_stop_motor = QPushButton("STOP MOTOR (Hard Stop)")
        self.btn_stop_motor.setStyleSheet("background-color: #D32F2F; color: white;")

        motor_layout.addWidget(QLabel("Motor Speed (steps/s):"), 0, 0)
        motor_layout.addWidget(self.speed_input, 0, 1)
        motor_layout.addWidget(QLabel("Motor Accel (steps/s^2):"), 1, 0)
        motor_layout.addWidget(self.accel_input, 1, 1)
        motor_layout.addWidget(self.btn_apply_motor, 2, 0, 1, 2)
        motor_layout.addWidget(self.btn_start_waiting, 3, 0, 1, 2)  # NEW
        motor_layout.addWidget(self.btn_manual_eject, 4, 0)
        motor_layout.addWidget(self.btn_manual_retract, 4, 1)
        motor_layout.addWidget(self.btn_stop_motor, 5, 0, 1, 2)

        motor_group.setLayout(motor_layout)
        main_layout.addWidget(motor_group)

        # --- Recording Group ---
        recording_group = QGroupBox("Recording")
        recording_layout = QGridLayout()
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "Camera Resolution (default)",
            "1920x1080 (1080p)",
            "1280x720 (720p)",
            "640x480 (480p)"
        ])
        self.btn_record = QPushButton("START RECORDING")
        self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")

        recording_layout.addWidget(QLabel("Output Resolution:"), 0, 0)
        recording_layout.addWidget(self.resolution_combo, 0, 1)
        recording_layout.addWidget(self.btn_record, 1, 0, 1, 2)

        recording_group.setLayout(recording_layout)
        main_layout.addWidget(recording_group)

        # --- Status Log ---
        status_group = QGroupBox("Status Log")
        status_layout = QVBoxLayout()
        self.status_log = QLabel("Welcome. Connect camera and Arduino.")
        self.status_log.setWordWrap(True)
        self.status_log.setAlignment(Qt.AlignTop)
        self.status_log.setMinimumHeight(100)
        self.status_log.setStyleSheet("background-color: #222; border: 1px solid #555; padding: 5px;")
        status_layout.addWidget(self.status_log)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        main_layout.addStretch()  # Push everything up

        # --- Set Central Widget ---
        # Create the scroll area and set the content widget
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setCentralWidget(self.scroll_area)  # Set scroll area as central widget

        # --- Connect Button Clicks ---
        self.btn_connect_cam.clicked.connect(self.toggle_camera)
        self.btn_connect_arduino.clicked.connect(self.toggle_arduino)
        self.btn_refresh_ports.clicked.connect(self.refresh_ports)

        self.btn_apply_detection.clicked.connect(self.apply_detection_settings)
        self.btn_apply_motor.clicked.connect(self.apply_motor_settings)
        self.btn_apply_electrodes.clicked.connect(self.apply_electrode_settings)

        # NEW motor button logic
        self.btn_start_waiting.toggled.connect(self.toggle_waiting_mode)
        self.btn_manual_eject.clicked.connect(self.controller.prime_pump)
        self.btn_manual_retract.clicked.connect(self.controller.retract_pump)
        self.btn_stop_motor.clicked.connect(self.on_hard_stop_clicked)

        self.btn_record.clicked.connect(self.on_record_button_pressed)

        self.btn_exposure_lock.clicked.connect(self.toggle_exposure)
        self.btn_autofocus.clicked.connect(self.toggle_autofocus)
        self.slider_focus.valueChanged.connect(self.on_focus_slider_change)

        # --- Finalize ---
        self.refresh_ports()
        self.set_dark_theme()

    def set_dark_theme(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #3c3c3c;
                color: #f0f0f0;
                font-size: 11px;
            }
            QMainWindow {
                background-color: #2b2b2b;
            }
            QScrollArea {
                border: none;
            }
            QGroupBox {
                background-color: #454545;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                background-color: #454545;
                color: #f0f0f0;
            }
            QPushButton {
                background-color: #5a5a5a;
                border: 1px solid #666;
                padding: 5px 10px;
                border-radius: 3px;
                min-height: 20px; /* Ensure buttons have height */
            }
            QPushButton:hover {
                background-color: #6a6a6a;
                border: 1px solid #777;
            }
            QPushButton:pressed {
                background-color: #505050;
            }
            QPushButton:checked { /* For toggle buttons */
                background-color: #D32F2F;
                border: 1px solid #C00000;
                color: white;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #2c2c2c;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
                color: #f0f0f0;
                min-height: 20px; /* Ensure inputs have height */
            }
            QComboBox::drop-down {
                border: none;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 8px;
                background: #2c2c2c;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #8a8a8a;
                border: 1px solid #999;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                border: 1px solid #2b2b2b;
                background: #2b2b2b;
                width: 15px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #5a5a5a;
                min-height: 20px;
                border-radius: 7px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            QLabel {
                padding: 2px;
            }
        """)

    @Slot(str)
    def log_status(self, message):
        """Append a message to the status log."""
        if not message:
            return
        current_text = self.status_log.text()
        if "Welcome." in current_text:
            current_text = ""

        lines = current_text.split('\n')
        if len(lines) > 20:
            lines = lines[-20:]  # Keep only the last 20 lines

        #lines.append(message)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        lines.append(f"[{timestamp}] {message}")
        self.status_log.setText("\n".join(lines))

    @Slot(bool)
    def update_record_button_style(self, is_recording):
        """Update the record button text and style."""
        if is_recording:
            self.btn_record.setText("STOP RECORDING")
            self.btn_record.setStyleSheet("background-color: #D32F2F; color: white; padding: 10px;")
        else:
            self.btn_record.setText("START RECORDING")
            self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")

    # --- Connection Slots ---

    def refresh_ports(self):
        """Scan for and update the list of serial ports."""
        self.port_combo.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        if not ports:
            self.port_combo.addItem("No ports found")
        else:
            self.port_combo.addItems(ports)

    def toggle_camera(self):
        """Connect or disconnect the camera."""
        if not self.controller.is_processing:
            try:
                cam_index = int(self.cam_index_input.text())
            except ValueError:
                self.log_status("Error: Camera Index must be an integer.")
                return

            if self.controller.connect_camera(cam_index):
                self.btn_connect_cam.setText("Disconnect Camera")
                self.btn_connect_cam.setStyleSheet("background-color: #D32F2F;")  # Red
            else:
                self.log_status("Failed to connect to camera.")
        else:
            # --- Disconnect Camera ---
            self.controller.stop_camera()
            self.btn_connect_cam.setText("Connect Camera")
            self.btn_connect_cam.setStyleSheet("")  # Reset style
            self.canvas_window.clear_frame()

    def toggle_arduino(self):
        """Connect or disconnect the Arduino in a worker thread."""
        if not self.controller.arduino:
            port = self.port_combo.currentText()
            if not port or "No ports found" in port:
                self.log_status("Error: No serial port selected.")
                return

            self.btn_connect_arduino.setText("Connecting...")
            self.btn_connect_arduino.setEnabled(False)

            # Run the blocking connect() call in a worker thread
            self.worker = Worker(self.controller.connect_arduino, port)
            self.worker.finished.connect(self.on_arduino_connected)
            self.worker.start()
        else:
            # --- Disconnect Arduino ---
            # Also stop the waiting mode if it's on
            if self.btn_start_waiting.isChecked():
                self.btn_start_waiting.setChecked(False)  # This will call toggle_waiting_mode

            self.controller.stop_arduino()
            self.btn_connect_arduino.setText("Connect Arduino")
            self.btn_connect_arduino.setStyleSheet("")  # Reset style
            self.btn_connect_arduino.setEnabled(True)

    @Slot(bool, str)
    def on_arduino_connected(self, success, message):
        """Callback for when the Arduino worker thread finishes."""
        self.btn_connect_arduino.setEnabled(True)
        if success:
            self.btn_connect_arduino.setText("Disconnect Arduino")
            self.btn_connect_arduino.setStyleSheet("background-color: #D32F2F;")  # Red
        else:
            self.log_status(f"Arduino connection failed: {message}")
            self.btn_connect_arduino.setText("Connect Arduino")
            self.btn_connect_arduino.setStyleSheet("")  # Reset style

        self.worker = None  # Clear the worker

    # --- Settings Slots ---

    def apply_detection_settings(self):
        """Send detection settings to the core."""
        settings = {
            "allowed_ids": self.ids_input.text(),
            "min_size": self.min_size_input.text(),
            "max_size": self.max_size_input.text(),
            "timeout": self.timeout_input.text(),
            "entry_frames": self.entry_frames_input.text(),  # NEW
            "exit_frames": self.exit_frames_input.text()  # NEW
        }
        self.controller.update_detection_settings(settings)

    def apply_motor_settings(self):
        """Send motor settings to the core."""
        settings = {
            "motor_speed": self.speed_input.text(),
            "motor_accel": self.accel_input.text()
            
        }
        self.controller.update_motor_settings(settings)

    def apply_electrode_settings(self):
        """Send electrode settings to the core."""
        settings = {
            "threshold_1": self.spin_thresh1.value(),
            "threshold_2": self.spin_thresh2.value(),
            "threshold_3": self.spin_thresh3.value()
        }
        self.controller.update_electrode_settings(settings)

    def on_record_button_pressed(self):
        """Handle the record button click, includes resolution setting."""
        # 1. Update the resolution setting in the core
        res_text = self.resolution_combo.currentText()
        self.controller.update_recording_settings(res_text)

        # 2. Toggle the recording state
        self.controller.toggle_recording()

    def on_hard_stop_clicked(self):
        """Handle the hard stop button, which also stops waiting mode."""
        # Un-toggle the waiting button if it's on
        if self.btn_start_waiting.isChecked():
            self.btn_start_waiting.setChecked(False)

        self.controller.stop_pump()  # Sends 'S'

    def toggle_waiting_mode(self, checked):
        """Handle the 'Start/Stop Motor (Waiting Mode)' toggle."""
        if checked:
            # --- Start Waiting Mode ---
            if not self.controller.arduino:
                self.log_status("Cannot start motor: Arduino not connected.")
                self.btn_start_waiting.setChecked(False)  # Revert toggle
                return

            self.controller.start_motor_waiting_mode()  # Sends 'W'
            self.btn_start_waiting.setText("Stop Motor (Waiting Mode)")
            # Style is set by :checked in stylesheet

        else:
            # --- Stop Waiting Mode ---
            self.controller.stop_pump()  # Sends 'S'
            self.btn_start_waiting.setText("Start Motor (Waiting Mode)")
            # Style is set by :unchecked

    # --- Camera Control Slots ---

    def toggle_exposure(self):
        """Toggle exposure lock on/off."""
        if not self.controller.cap:
            self.log_status("Camera not connected.")
            return
        # Get the *opposite* of the current state
        lock = not self.controller.settings['exposure_lock']
        self.controller.set_exposure_lock(lock)
        self.btn_exposure_lock.setText(f"Exposure Lock: {'ON' if lock else 'OFF'}")

    def toggle_autofocus(self):
        """Toggle autofocus on/off."""
        if not self.controller.cap:
            self.log_status("Camera not connected.")
            return
        # Get the *opposite* of the current state
        enable = not self.controller.settings['autofocus']
        self.controller.set_autofocus(enable)
        self.btn_autofocus.setText(f"Autofocus: {'ON' if enable else 'OFF'}")
        # Enable/disable slider
        self.slider_focus.setEnabled(not enable)
        if enable:
            self.log_status("Autofocus ON. Slider disabled.")
        else:
            self.log_status("Autofocus OFF. Slider enabled.")
            # Set focus to current slider value
            self.on_focus_slider_change(self.slider_focus.value())

    def on_focus_slider_change(self, value):
        """Called when the focus slider is moved."""
        if not self.controller.settings['autofocus']:
            self.controller.set_focus(value)

    def closeEvent(self, event):
        """Ensure all threads and hardware are stopped on exit."""
        self.controller.status_updated.emit("Closing application...")
        self.controller.stop_all()
        self.canvas_window.close()
        event.accept()


# --- Application Entry Point ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

