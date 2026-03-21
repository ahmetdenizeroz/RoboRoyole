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

from feeder_core import FeederController
from VideoCanvas import VideoCanvas

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

# --- Main GUI Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feeder Control Panel")
        self.setGeometry(0, 0, 1920/2, 1000)  # x, y, w, h

        # Create the core controller
        self.controller = FeederController()

        # Create the video canvas window
        self.canvas_window = VideoCanvas()
        self.canvas_window.setWindowTitle("Live Canvas (Draw Zone Here)")
        self.canvas_window.setGeometry(1000, 100, 900, 900)
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
        self.scroll_content = QWidget()
        main_layout = QHBoxLayout(self.scroll_content)  # Add layout to the content widget
        
        # Left and Right Column Layouts
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()

        # Add the Columns to the Master Layout
        main_layout.addLayout(left_column, 1)
        main_layout.addLayout(right_column, 2)



#############################################################################
#                           Connection Group                                #           
#############################################################################
        connection_group = QGroupBox("Connection")                          #
        connection_layout = QGridLayout()                                   #
        connection_layout.setColumnStretch(0, 1)
        connection_layout.setColumnStretch(1, 1)
        connection_layout.setColumnStretch(2, 1)
        connection_layout.setColumnStretch(3, 1)                            #                           
                                                                            #
        # Camera                                                            #
        self.cam_index_input = QLineEdit("0")                               #                           #
        self.btn_connect_cam = QPushButton("Connect Camera")                #
                                                                            #
        # Arduino                                                           #
        self.port_combo = QComboBox()                                       #
        self.btn_refresh_ports = QPushButton("Refresh")                     #
        self.btn_connect_arduino = QPushButton("Connect Arduino")           #
                                                                            #
        connection_layout.addWidget(QLabel("Camera Index:"), 0, 0, 1, 1)    #
        connection_layout.addWidget(self.cam_index_input, 0, 1, 1, 1)       #
        connection_layout.addWidget(self.btn_connect_cam, 0, 2, 1, 2)       #

        connection_layout.addWidget(QLabel("Serial Port:"), 1, 0, 1, 1)     #
        connection_layout.addWidget(self.port_combo, 1, 1, 1, 1)            #
        connection_layout.addWidget(self.btn_refresh_ports, 1, 2, 1, 2)     #
        connection_layout.addWidget(self.btn_connect_arduino, 2, 0, 1, 4)   #
                                                                            #
        connection_group.setLayout(connection_layout)                       #
        left_column.addWidget(connection_group)                             #
#############################################################################
        
#############################################################################
#                           Camera Control Group                            #
#############################################################################   
        camera_control_group = QGroupBox("Camera Controls")                 #
        camera_control_layout = QGridLayout()
        camera_control_layout.setColumnStretch(0, 1)
        camera_control_layout.setColumnStretch(1, 1)                        #
        self.btn_exposure_lock = QPushButton("Exposure Lock: OFF")          #
        self.btn_autofocus = QPushButton("Autofocus: ON")                   #
        self.slider_focus = QSlider(Qt.Horizontal)                          #
        self.slider_focus.setRange(0, 255)                                  #
        self.slider_focus.setEnabled(False)  # Disabled by default          #
                                                                            #
        camera_control_layout.addWidget(self.btn_exposure_lock, 0, 0)       #
        camera_control_layout.addWidget(self.btn_autofocus, 0, 1)           #
        camera_control_layout.addWidget(QLabel("Manual Focus:"), 1, 0, 1, 2)#
        camera_control_layout.addWidget(self.slider_focus, 2, 0, 1, 2)      #
                                                                            #
        camera_control_group.setLayout(camera_control_layout)               #
        right_column.addWidget(camera_control_group)                         #  
#############################################################################
    
#############################################################################
#                           Electrode Settings Group                        #
#############################################################################
        electrode_group = QGroupBox("Electrode Settings (Analog 0-1023)")   #
        electrode_layout = QGridLayout()                                    #
        electrode_layout.setColumnStretch(0, 1)
        electrode_layout.setColumnStretch(1, 1) 
                                                                            #
        self.spin_thresh1 = QSpinBox()                                      #
        self.spin_thresh1.setRange(0, 1023)                                 #
        self.spin_thresh1.setValue(600)                                     #
                                                                            #
        self.spin_hysterisis1 = QSpinBox()                                  #
        self.spin_hysterisis1.setRange(0, 1023)                             #
        self.spin_hysterisis1.setValue(600)                                 #
                                                                            #
        self.spin_thresh2 = QSpinBox()                                      #
        self.spin_thresh2.setRange(0, 1023)                                 #
        self.spin_thresh2.setValue(580)                                     #
                                                                            #
        self.spin_hysterisis2 = QSpinBox()                                  #
        self.spin_hysterisis2.setRange(0, 1023)                             #
        self.spin_hysterisis2.setValue(600)                                 #
                                                                            #
        self.spin_thresh3 = QSpinBox()                                      #
        self.spin_thresh3.setRange(0, 1023)                                 #
        self.spin_thresh3.setValue(530)                                     #
                                                                            #
        self.spin_hysterisis3 = QSpinBox()                                  #
        self.spin_hysterisis3.setRange(0, 1023)                             #
        self.spin_hysterisis3.setValue(600)                                 #
                                                                            #
        self.btn_apply_electrodes = QPushButton("Apply Electrode Settings") #
                                                                            #
        electrode_layout.addWidget(QLabel("Threshold 1 (Back):"), 0, 0)     #
        electrode_layout.addWidget(self.spin_thresh1, 0, 1)                 #
        electrode_layout.addWidget(QLabel("Threshold 2 (Middle):"), 1, 0)   #
        electrode_layout.addWidget(self.spin_thresh2, 1, 1)                 #
        electrode_layout.addWidget(QLabel("Threshold 3 (Front):"), 2, 0)    #
        electrode_layout.addWidget(self.spin_thresh3, 2, 1)                 #
                                                                            #
        electrode_layout.addWidget(QLabel("Hysterisis 1 (Back):"), 0, 3)    #
        electrode_layout.addWidget(self.spin_hysterisis1, 0, 4)             #
        electrode_layout.addWidget(QLabel("Hysterisis 2 (Middle):"), 1, 3)  #
        electrode_layout.addWidget(self.spin_hysterisis2, 1, 4)             #
        electrode_layout.addWidget(QLabel("Hysterisis 3 (Front):"), 2, 3)   #
        electrode_layout.addWidget(self.spin_hysterisis3, 2, 4)             #
                                                                            #
        electrode_layout.addWidget(self.btn_apply_electrodes, 3, 0, 1, 5)   #
                                                                            #
        electrode_group.setLayout(electrode_layout)                         #
        left_column.addWidget(electrode_group)                              #
#############################################################################

#############################################################################
#                           Detection Settings Group                        #
#############################################################################
        detection_group = QGroupBox("Detection Settings")                   #   
        detection_layout = QGridLayout()                                    #
        detection_layout.setColumnStretch(0, 1)
        detection_layout.setColumnStretch(1, 1) 
        self.ids_input = QLineEdit("1, 2, 5-10")                            #
        self.min_size_input = QLineEdit("50")                               #
        self.max_size_input = QLineEdit("2000")                             #   
        self.timeout_input = QLineEdit("10.0")                              #   
                                                                            #
        # NEW Frame Inputs                                                  #
        self.entry_frames_input = QLineEdit("15")                           #   
        self.exit_frames_input = QLineEdit("30")                            #       
                                                                            #
        self.btn_apply_detection = QPushButton("Apply Detection Settings")  #
                                                                            #
        detection_layout.addWidget(QLabel("Allowed IDs:"), 0, 0)            #
        detection_layout.addWidget(self.ids_input, 0, 1)                    #
        detection_layout.addWidget(QLabel("Min Size (Perim):"), 1, 0)       #
        detection_layout.addWidget(self.min_size_input, 1, 1)               #
        detection_layout.addWidget(QLabel("Max Size (Perim):"), 2, 0)       #
        detection_layout.addWidget(self.max_size_input, 2, 1)               #
        detection_layout.addWidget(QLabel("Feeding Duration (s):"), 3, 0)   #
        detection_layout.addWidget(self.timeout_input, 3, 1)                #
        detection_layout.addWidget(QLabel("Entry Confirmation (frames):"), 4, 0)
        detection_layout.addWidget(self.entry_frames_input, 4, 1)           #
        detection_layout.addWidget(QLabel("Exit Confirmation (frames):"), 5, 0)
        detection_layout.addWidget(self.exit_frames_input, 5, 1)            #
        detection_layout.addWidget(self.btn_apply_detection, 6, 0, 1, 2)    #
                                                                            #
        detection_group.setLayout(detection_layout)                         #   
        left_column.addWidget(detection_group)                              #   
#############################################################################

#############################################################################
#                           Motor Settings Group                            #
#############################################################################
        motor_group = QGroupBox("Motor Settings")                           #
        motor_layout = QGridLayout()                                        #
        motor_layout.setColumnStretch(0, 1)                                 #
        motor_layout.setColumnStretch(1, 1)                                 #
        self.speed_input = QLineEdit("1000")                                #   
        self.accel_input = QLineEdit("500")                                 #   
        self.btn_apply_motor = QPushButton("Apply Motor Settings")          #
                                                                            #
        # --- NEW Start/Stop Button ---                                     #
        self.btn_start_waiting = QPushButton("Start Motor (Waiting Mode)")  #
        self.btn_start_waiting.setCheckable(True)  # Make it a toggle       #
        self.btn_start_waiting.setStyleSheet("background-color: #4CAF50; color: white;")  # Green
                                                                            #
        # Manual buttons                                                    #
        self.btn_manual_eject = QPushButton("Manual Eject")                 #
        self.btn_manual_retract = QPushButton("Manual Retract")             #
        self.btn_stop_motor = QPushButton("STOP MOTOR (Hard Stop)")         #
        self.btn_stop_motor.setStyleSheet("background-color: #D32F2F; color: white;")
                                                                            #
        motor_layout.addWidget(QLabel("Motor Speed (steps/s):"), 0, 0)      #
        motor_layout.addWidget(self.speed_input, 0, 1)                      #
        motor_layout.addWidget(QLabel("Motor Accel (steps/s^2):"), 1, 0)    #
        motor_layout.addWidget(self.accel_input, 1, 1)                      #
        motor_layout.addWidget(self.btn_apply_motor, 2, 0, 1, 2)            #
        motor_layout.addWidget(self.btn_start_waiting, 3, 0, 1, 2)  # NEW   #
        motor_layout.addWidget(self.btn_manual_eject, 4, 0)                 #
        motor_layout.addWidget(self.btn_manual_retract, 4, 1)               #
        motor_layout.addWidget(self.btn_stop_motor, 5, 0, 1, 2)             #
                                                                            #
        motor_group.setLayout(motor_layout)                                 #
        right_column.addWidget(motor_group)                                  #
#############################################################################

#############################################################################
#                           Recording Group                                 #
#############################################################################
        recording_group = QGroupBox("Recording")                            #
        recording_layout = QGridLayout()                                    #       
        self.resolution_combo = QComboBox()                                 #   
        self.resolution_combo.addItems([                                    #
            "Camera Resolution (default)",                                  #
            "1920x1080 (1080p)",                                            #
            "1280x720 (720p)",                                              #
            "640x480 (480p)"                                                #
        ])                                                                  #
        self.btn_record = QPushButton("START RECORDING")                    #
        self.btn_record.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
                                                                            #
        recording_layout.addWidget(QLabel("Output Resolution:"), 0, 0)      #
        recording_layout.addWidget(self.resolution_combo, 0, 1)             #
        recording_layout.addWidget(self.btn_record, 1, 0, 1, 2)             #
                                                                            #
        recording_group.setLayout(recording_layout)                         #
        right_column.addWidget(recording_group)                              #
#############################################################################

#############################################################################
#                           Status Log                                      #
#############################################################################
        status_group = QGroupBox("Status Log")                              #
        status_layout = QVBoxLayout()                                       #
        self.status_log = QLabel("Welcome. Connect camera and Arduino.")    #
        self.status_log.setObjectName("statusLogLabel")
        self.status_log.setWordWrap(True)                                   #
        self.status_log.setAlignment(Qt.AlignTop)                           #
        self.status_log.setMinimumHeight(100)                               #    
        #self.status_log.setStyleSheet("background-color: #222; border: 1px solid #555; padding: 5px;")
        status_layout.addWidget(self.status_log)                            #   
        status_group.setLayout(status_layout)                               #
        right_column.addWidget(status_group)                                #
#############################################################################

#############################################################################
#                           Image Processing Controls                       #
#############################################################################
        image_process_group = QGroupBox("Image Processing Variables")
        image_process_layout = QGridLayout()
        slider_configs = [
            ("Blur Size", "slider_blur", 7, 51),
            ("Threshold", "slider_thresh", 15, 255),
            ("Open Kernel", "slider_open_k", 2, 15),
            ("Open Power", "slider_open_p", 1, 10),
            ("Close Kernel", "slider_close_k", 2, 15),
            ("Close Power", "slider_close_p", 2, 10),
            ("Box Width", "slider_width", 200, 300),
            ("Box Height", "slider_height", 200, 300),
            ("Min Bee Area", "slider_area", 50, 1000)
        ]
        self.sliders = {}
        for i, (name, attr, default, maximum) in enumerate(slider_configs):
            # 1. Label
            label = QLabel(name + ":")
            
            # 2. The Slider
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, maximum)
            slider.setValue(default)
            setattr(self, attr, slider) # Saves it as self.slider_blur, etc.

            # 3. The Value Readout (so you can see the exact number)
            val_label = QLabel(str(default))
            val_label.setFixedWidth(30)
            self.sliders[attr] = val_label # Store reference to update later

            # Add to grid
            image_process_layout.addWidget(label, i, 0)
            image_process_layout.addWidget(slider, i, 1)
            image_process_layout.addWidget(val_label, i, 2)

            # Connect signal to update the readout and the logic
            slider.valueChanged.connect(lambda val, l=val_label: l.setText(str(val)))
            #slider.valueChanged.connect(self.apply_image_processing_settings)

        self.btn_apply_image = QPushButton("Apply Image Settings")
        image_process_layout.addWidget(self.btn_apply_image, 9, 0, 1, 3) # Span across all 3 columns
        self.btn_recapture_bg = QPushButton("Recapture Background")
        image_process_layout.addWidget(self.btn_recapture_bg, 10, 0, 1, 3) # Span across all 3 columns

        image_process_group.setLayout(image_process_layout)
        left_column.addWidget(image_process_group)       
#############################################################################
        
        
        
        # Push everything to the top in both columns
        left_column.addStretch()
        right_column.addStretch()


        # --- Set Central Widget ---
        # Create the scroll area and set the content widget
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
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

        self.btn_apply_image.clicked.connect(self.apply_image_processing_settings)
        self.btn_recapture_bg.clicked.connect(self.recapture_bg)

        self.btn_record.clicked.connect(self.on_record_button_pressed)

        self.btn_exposure_lock.clicked.connect(self.toggle_exposure)
        self.btn_autofocus.clicked.connect(self.toggle_autofocus)
        self.slider_focus.valueChanged.connect(self.on_focus_slider_change)

        # --- Finalize ---
        self.refresh_ports()
        #self.set_dark_theme()
        self.set_bee_theme()

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

    def set_bee_theme(self):
        """Applies the custom Bee Theme with a Dark Grey Status Log and White Text."""
        self.setStyleSheet("""
            /* --- Main Hive Background --- */
            QWidget {
                background-color: #1A1A1A; /* Deep Black */
                color: #f9c901;           /* Main Gold Text */
                font-size: 11px;
            }
            QMainWindow {
                background-color: #1A1A1A;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }

            /* --- Honey-Colored Group Boxes --- */
            QGroupBox {
                background-color: #896800; /* Medium Amber */
                color: #1A1A1A;           /* Black text for contrast on Amber */
                border: 2px solid #6b4701; /* Dark Brown Border */
                border-radius: 8px;
                margin-top: 15px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 10px;
                background-color: #f6e000; /* Brightest Yellow */
                color: #1A1A1A;           /* Dark text on title */
                border-radius: 4px;
            }

            /* --- Worker Bee Buttons --- */
            QPushButton {
                background-color: #6b4701; /* Dark Brown */
                color: #f6e000;           /* Bright Yellow text */
                border: 1px solid #f9c901;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #985b10; /* Lighter Brown */
                color: #f6e000;
            }
            QPushButton:pressed {
                background-color: #f9c901; /* Gold Flash */
                color: #1A1A1A;
            }
            QPushButton:checked {
                background-color: #f6e000;
                color: #1A1A1A;
            }

            /* --- Input Cells --- */
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #1A1A1A; /* Black inside */
                border: 1px solid #985b10; /* Amber border */
                padding: 5px;
                border-radius: 3px;
                color: #f6e000;           /* Bright text */
            }

            /* --- The Dark Grey Status Log (White Text) --- */
            QLabel#statusLogLabel {
                background-color: #222222; /* Dark Grey background */
                color: #FFFFFF;           /* Pure White text */
                border: 1px solid #6b4701; /* Brown border to stay in theme */
                padding: 8px;
                border-radius: 4px;
            }

            /* --- Sliders (Pollen Path) --- */
            QSlider::groove:horizontal {
                border: 1px solid #6b4701;
                height: 6px;
                background: #1A1A1A;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #f9c901;
                border: 1px solid #6b4701;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }

            /* --- Scrollbars --- */
            QScrollBar:vertical {
                background: #1A1A1A;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background: #896800;
                min-height: 20px;
                border-radius: 6px;
            }

            /* --- Label Contrast Logic --- */
            /* Labels inside Amber GroupBoxes must be Black */
            QGroupBox QLabel {
                color: #1A1A1A;
                background: transparent;
            }
            /* General Labels outside GroupBoxes remain Gold */
            QLabel {
                color: #f9c901;
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
            "threshold_3": self.spin_thresh3.value(),
            "hysterisis_1": self.spin_hysterisis1.value(),
            "hysterisis_2": self.spin_hysterisis2.value(),
            "hysterisis_3": self.spin_hysterisis3.value()
        }
        self.controller.update_electrode_settings(settings)

    def apply_image_processing_settings(self):
        """Send image processing (OpenCV) variables to the core."""
        settings = {
            "blur_size": self.slider_blur.value(),
            "threshold": self.slider_thresh.value(),
            "open_kernel": self.slider_open_k.value(),
            "open_power": self.slider_open_p.value(),
            "close_kernel": self.slider_close_k.value(),
            "close_power": self.slider_close_p.value(),
            "box_width": self.slider_width.value(),
            "box_height": self.slider_height.value(),
            "min_area": self.slider_area.value()
        }
        # This sends the dictionary to your controller
        self.controller.update_image_processing(settings)
    def recapture_bg(self):
        """ Send command to recapture bg for bg substruction"""
        self.controller.should_recapture_bg = True

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