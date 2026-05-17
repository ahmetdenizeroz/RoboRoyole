#TODO: learn why the binary image crated by bee_detector is so different than 3x3_detector.py

import cv2
import numpy as np
import time
import serial
import threading
from PySide6.QtCore import QObject, Signal, Slot
import os
import psutil

from workers.bee_detector import BeeDetector
from workers.camera_init import CameraInitThread
from workers.arduino_reader import ArduinoReader
from workers.video_recorder import VideoRecorder
from workers.bee_drawer import BeeDrawer

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"
os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"] = "video_codec;h264_nvenc"

# To priotarize the python code in OS level
try:
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
except Exception as e:
    print(f"Priority boost failed: {e}")

class FeederController(QObject):
    # Signals to communicate with the GUI
    frame_ready = Signal(np.ndarray)  # Emits the annotated video frame
    status_updated = Signal(str)  # Emits a status message
    recording_toggled = Signal(bool)  # Emits the recording state
    zone_updated = Signal(int, int, int)
    arduino_send_failed = Signal(str)

    def __init__(self):
        super().__init__()

        self.log_lock = threading.Lock()
        self.serial_lock = threading.Lock()
        self.unique_frames_captured = 0
        self.fps = 60
        # --- Hardware & State ---
        self.log_file = None
        self.cap = None
        self.arduino = None
        self.is_processing = False
        self.is_recording = False
        self.processing_thread = None  # Renamed for clarity
        self.settings_lock = threading.Lock()  # Protects self.settings
        self.arduino_send_failed.connect(self._handle_arduino_send_failed)

        # --- ArUco Setup ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        marker_txt_path = os.path.join(base_dir, "3x3_1.txt")

        self.bee_detector = BeeDetector(marker_txt_path)
        self.bee_detector.set_debug(show_debug=False, debug_marker_id=0)
        self.bee_drawer = BeeDrawer()

        self.should_recapture_bg = False
        # --- Default Settings ---
        self.settings = {
            # Camera
            "processing_target_fps": 30.0,
            "exposure_lock": False,
            "autofocus": True,
            "focus": 0,
            # Electrodes
            "threshold_1": 512,
            "threshold_2": 512,
            "threshold_3": 512,
            "hysterisis_1": 60,
            "hysterisis_2": 60,
            "hysterisis_3": 60,
            "margin": 50,
            # Detection
            "allowed_ids": set([1, 2, 5, 6, 7, 8, 9, 10]),
            "min_candidate_area": 100,
            "max_candidate_area": 500,
            "timeout_ms": 10000,  # Arduino timeout
            "entry_frames": 15,  # Python: consecutive frames to confirm entry
            "exit_frames": 30,  # Python: consecutive frames to confirm exit
            "trigger_mode": "both",
            "bg_mode": "With BG",
            # Motor
            "motor_speed": 1000,
            "motor_accel": 500,
            # Recording
            "output_resolution": None,  # (w, h) tuple or None
            # Zone
            "zone_center": (0, 0),
            "zone_radius": 0,
            # Image Processing
            "blur_size": 7,
            "threshold": 15,
            "open_kernel": 2,
            "open_power": 1,
            "close_kernel": 2,
            "close_power": 2,
            "box_width": 200,
            "box_height": 200,
            "min_area" : 2000,
            "roi_x": 0,
            "roi_y": 0,
            "roi_w": 640,
            "roi_h": 480,
            "sec_zone_center": (0, 0),
            "sec_zone_radius": 0,
            "diff_intensity": 20,
        }
        self._apply_custom_detector_settings(self.settings.copy())
        # --- New Frame-based Zone Logic ---
        self.is_feeding = False  # This is our new, simple state
        self.frames_in_zone_counter = 0
        self.frames_out_of_zone_counter = 0

        self.arduino_reader = None  # Thread for reading Arduino messages
        self.cam_init_worker = None
        
    # ===================================
    #  Hardware Connection
    # ===================================

    def connect_camera(self, index=0):
        """(GUI Thread) Now non-blocking. Launches a background thread to open the camera."""
        if self.cam_init_worker and self.cam_init_worker.isRunning():
            self.status_updated.emit("Camera is already opening, please wait...")
            return False

        self.status_updated.emit(f"Opening camera index {index} in background...")
        self._log_to_file(f"Starting background camera init for index {index}...")

        #self.cam_init_worker = CameraInitThread(index, self.fps)
        self.cam_init_worker = CameraInitThread(index, self.fps)
        self.cam_init_worker.finished.connect(self._on_camera_init_finished)
        self.cam_init_worker.start()
        return True

    def _on_camera_init_finished(self, cap, success, message):
        """Callback when the background camera thread finishes."""
        if success and cap:
            self.cap = cap
            self.status_updated.emit(message)
            self._log_to_file(message)

            # 3. Trigger the settings dialog ONLY IF requested 
            # (Be careful: this will pop up a window and might block the thread briefly)
            # self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

            # Apply initial camera state from stored settings
            self.set_exposure_lock(self.settings['exposure_lock'])
            self.set_autofocus(self.settings['autofocus'])

            # Start the actual heavy-lifting processing loop
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.processing_thread.start()
            
            self.status_updated.emit("Camera Ready. Processing loop started.")
            self._log_to_file("Camera Ready. Processing loop started.")
        else:
            self.status_updated.emit(f"Connection Failed: {message}")
            self._log_to_file(f"Connection Failed: {message}")
            self.cap = None

    def stop_camera(self):
        """(GUI Thread) Stops the camera and processing thread."""
        if self.is_processing:
            self.is_processing = False
            # Wait for thread to finish
            if self.processing_thread:
                self.processing_thread.join(timeout=1.0)

            if self.cap:
                self.cap.release()
                self.cap = None

            if self.is_recording:
                self.toggle_recording()  # This will stop and release the writer

            self.status_updated.emit("Camera disconnected.")
            self._log_to_file("Camera disconnected.")

    def update_secondary_zone(self, settings: dict):
        """(GUI Thread) Updates the secondary detection zone."""
        with self.settings_lock:
            try:
                self.settings["sec_zone_center"] = (int(settings.get("x", 0)), int(settings.get("y", 0)))
                self.settings["sec_zone_radius"] = int(settings.get("r", 0))
                self.settings["diff_intensity"] = int(settings.get("intensity", 20))
                self.status_updated.emit("Secondary zone settings updated.")
            except Exception as e:
                self.status_updated.emit(f"Error secondary zone: {e}")

    def connect_arduino(self, port):
        """(Worker Thread) Connects to the Arduino."""
        try:
            self.status_updated.emit(f"Connecting to Arduino on {port}...")
            self._log_to_file(f"Connecting to Arduino on {port}...")

            # 2-second timeout for connection
            self.arduino = serial.Serial(
                port=port,
                baudrate=115200,
                timeout=0.5,
                write_timeout=0.5
            )
            time.sleep(2)  # Wait for Arduino to reset

            if not self.arduino.is_open:
                raise Exception("Serial port not open.")

            self.arduino.flushInput()
            self.arduino_reader = ArduinoReader(self.arduino)
            self.arduino_reader.message_received.connect(self.on_arduino_message)
            self.arduino_reader.error_received.connect(self.on_arduino_reader_error)
            self.arduino_reader.disconnected.connect(self.on_arduino_reader_disconnected)
            self.arduino_reader.start()
            self.status_updated.emit("Arduino reader thread started.")
            self._log_to_file("Arduino reader thread started.")
            self.status_updated.emit("Arduino connected. Applying all settings...")
            self._log_to_file("Arduino connected. Applying all settings...")
            self.apply_all_settings_to_arduino()
            return True
        except Exception as e:
            self.status_updated.emit(f"Arduino Error: {e}")
            self._log_to_file(f"Arduino Error: {e}")

            if self.arduino:
                self._close_serial_in_background(self.arduino)
            self.arduino = None
            return False
    def _close_serial_in_background(self, arduino):
        def closer():
            try:
                arduino.close()
                self.status_updated.emit("Arduino serial closed.")
                self._log_to_file("Arduino serial closed.")
            except Exception as e:
                self.status_updated.emit(f"Error closing Arduino serial: {e}")
                self._log_to_file(f"Error closing Arduino serial: {e}")

        threading.Thread(target=closer, daemon=True).start()    
    def stop_arduino(self):
        """Disconnect Arduino safely without racing serial writes."""

        # First detach shared objects under lock.
        with self.serial_lock:
            arduino = self.arduino
            reader = self.arduino_reader
            self.arduino = None
            self.arduino_reader = None

        # Stop reader outside the serial lock.
        if reader:
            try:
                reader.stop()
                self.status_updated.emit("Arduino reader stopped.")
                self._log_to_file("Arduino reader stopped.")
            except Exception as e:
                self.status_updated.emit(f"Error stopping Arduino reader: {e}")
                self._log_to_file(f"Error stopping Arduino reader: {e}")

        # Close serial outside the serial lock.
        if arduino:
            self._close_serial_in_background(arduino)
            self.status_updated.emit("Arduino disconnected.")
            self._log_to_file("Arduino disconnected.")


    def stop_all(self):
        """(GUI Thread) Stops all hardware and threads."""
        self.stop_camera()
        self.stop_arduino()
        self.status_updated.emit("All systems stopped.")
        self._log_to_file("All systems stopped.")

    def _send_to_arduino(self, command: str):
        """Send a command to Arduino without allowing serial failure to freeze GUI."""
        error_msg = None

        with self.serial_lock:
            if not self.arduino or not self.arduino.is_open:
                return

            try:
                self.arduino.write(command.encode("ascii"))
            except Exception as e:
                error_msg = f"Arduino Send Error while sending {repr(command)}: {repr(e)}"

        # Important: handle failure AFTER releasing serial_lock.
        if error_msg:
            self.status_updated.emit(error_msg)
            self._log_to_file(error_msg)
            self.arduino_send_failed.emit(error_msg)

    # ===================================
    #  Settings Updates
    # ===================================

    def apply_all_settings_to_arduino(self):
        """Sends all current settings to the Arduino."""
        with self.settings_lock:
            settings = self.settings.copy()  # Make a snapshot
        self._send_to_arduino(f"V{settings['motor_speed']}\n")
        self._send_to_arduino(f"A{settings['motor_accel']}\n")
        self._send_to_arduino(f"T{settings['threshold_1']}\n")
        self._send_to_arduino(f"U{settings['threshold_2']}\n")
        self._send_to_arduino(f"I{settings['threshold_3']}\n")
        self._send_to_arduino(f"G{settings['hysterisis_1']}\n")
        self._send_to_arduino(f"H{settings['hysterisis_2']}\n")
        self._send_to_arduino(f"J{settings['hysterisis_3']}\n")
        self._send_to_arduino(f"M{settings['timeout_ms']}\n")
        self._send_to_arduino(f"X{settings['margin']}\n")
        self.status_updated.emit("All settings sent to Arduino.")
        self._log_to_file("All settings sent to Arduino.")


    def update_detection_settings(self, settings: dict):
        """(GUI Thread) Updates detection-related settings."""
        try:
            with self.settings_lock:
                ids_str = settings.get("allowed_ids", "")
                self.settings["allowed_ids"] = self._parse_ids(ids_str)

                self.settings["min_candidate_area"] = int(settings.get("min_candidate_area", 100))
                self.settings["max_candidate_area"] = int(settings.get("max_candidate_area", 500))

                self.settings["entry_frames"] = int(settings.get("entry_frames", 15))
                self.settings["exit_frames"] = int(settings.get("exit_frames", 30))

                self.settings["trigger_mode"] = settings.get("trigger_mode", "both")

                timeout_s = float(settings.get("timeout", 10.0))
                self.settings["timeout_ms"] = int(timeout_s * 1000)

                settings_snapshot = self.settings.copy()

            # Apply GUI detection settings to the custom detector too
            self._apply_custom_detector_settings(settings_snapshot)

            self.status_updated.emit(
                f"Detection settings updated. IDs: {settings_snapshot['allowed_ids']}"
            )
            self._log_to_file(
                f"Detection settings updated. IDs: {settings_snapshot['allowed_ids']}"
            )

            # Send the one setting the Arduino cares about
            self._send_to_arduino(f"M{settings_snapshot['timeout_ms']}\n")

        except Exception as e:
            self.status_updated.emit(f"Error parsing detection settings: {e}")
            self._log_to_file(f"Error parsing detection settings: {e}")

    def update_motor_settings(self, settings: dict):
        """(GUI Thread) Updates motor-related settings."""
        with self.settings_lock:
            try:
                self.settings["motor_speed"] = int(settings.get("motor_speed", 1000))
                self.settings["motor_accel"] = int(settings.get("motor_accel", 500))

                self.status_updated.emit(
                    f"Motor settings updated: Speed={self.settings['motor_speed']}, Accel={self.settings['motor_accel']}")
                self._log_to_file(f"Motor settings updated: Speed={self.settings['motor_speed']}, Accel={self.settings['motor_accel']}")

                # Send to Arduino
                self._send_to_arduino(f"V{self.settings['motor_speed']}\n")
                self._send_to_arduino(f"A{self.settings['motor_accel']}\n")
            except Exception as e:
                self.status_updated.emit(f"Error parsing motor settings: {e}")
                self._log_to_file(f"Error parsing motor settings: {e}")

    def update_electrode_settings(self, settings: dict):
        """(GUI Thread) Updates electrode threshold settings."""
        with self.settings_lock:
            try:
                self.settings["threshold_1"] = int(settings.get("threshold_1", 512))
                self.settings["threshold_2"] = int(settings.get("threshold_2", 512))
                self.settings["threshold_3"] = int(settings.get("threshold_3", 512))
                self.settings["hysterisis_1"] = int(settings.get("hysterisis_1", 512))
                self.settings["hysterisis_2"] = int(settings.get("hysterisis_2", 512))
                self.settings["hysterisis_3"] = int(settings.get("hysterisis_3", 512))
                self.settings["margin"] = int(settings.get("margin", 50))

                self.status_updated.emit(
                    f"Electrode thresholds updated: T1={self.settings['threshold_1']}, T2={self.settings['threshold_2']}, T3={self.settings['threshold_3']}, H1={self.settings['hysterisis_1']}, H2={self.settings['hysterisis_2']}, H3={self.settings['hysterisis_3']}, Margin = {self.settings['margin']}")
                self._log_to_file(f"Electrode thresholds updated: T1={self.settings['threshold_1']}, T2={self.settings['threshold_2']}, T3={self.settings['threshold_3']}, H1={self.settings['hysterisis_1']}, H2={self.settings['hysterisis_2']}, H3={self.settings['hysterisis_3']}, Margin = {self.settings['margin']}")

                # Send to Arduino
                self._send_to_arduino(f"T{self.settings['threshold_1']}\n")
                self._send_to_arduino(f"U{self.settings['threshold_2']}\n")
                self._send_to_arduino(f"I{self.settings['threshold_3']}\n")
                self._send_to_arduino(f"G{self.settings['hysterisis_1']}\n")
                self._send_to_arduino(f"H{self.settings['hysterisis_2']}\n")
                self._send_to_arduino(f"J{self.settings['hysterisis_3']}\n")
                self._send_to_arduino(f"X{self.settings['margin']}\n")
            except Exception as e:
                self.status_updated.emit(f"Error parsing electrode settings: {e}")
                self._log_to_file(f"Error parsing electrode settings: {e}")

    def update_image_processing(self, settings: dict):
        with self.settings_lock:
            try:
                self.settings["blur_size"] = int(settings.get("blur_size", 7))
                self.settings["threshold"] = int(settings.get("threshold", 15))
                self.settings["open_kernel"] = int(settings.get("open_kernel", 2))
                self.settings["open_power"] = int(settings.get("open_power", 1))
                self.settings["close_kernel"] = int(settings.get("close_kernel", 2))
                self.settings["close_power"] = int(settings.get("close_power", 2))
                self.settings["box_width"] = int(settings.get("box_width", 200))
                self.settings["box_height"] = int(settings.get("box_height", 200))
                self.settings["min_area"] = int(settings.get("min_area", 200))
                self.settings["roi_x"] = int(settings.get("roi_x", 0))
                self.settings["roi_y"] = int(settings.get("roi_y", 0))
                self.settings["roi_w"] = int(settings.get("roi_w", 640))
                self.settings["roi_h"] = int(settings.get("roi_h", 480))
                self.settings["bg_mode"] = settings.get("bg_mode", "With BG")
                self.status_updated.emit("Image processing settings updated.")
                self._log_to_file("Image processing settings updated.")  
            except Exception as e:
                self.status_updated.emit(f"Error parsing image processing settings: {e}")
                self._log_to_file(f"Error parsing image processing settings: {e}")

    def update_recording_settings(self, resolution_text: str):
        """(GUI Thread) Updates the output resolution setting."""
        with self.settings_lock:
            if "Camera Resolution" in resolution_text or not resolution_text:
                self.settings["output_resolution"] = None
                self.status_updated.emit("Recording res set to: Camera Resolution")
                self._log_to_file("Recording res set to: Camera Resolution")

            else:
                try:
                    w, h = map(int, resolution_text.split('(')[0].strip().split('x'))
                    self.settings["output_resolution"] = (w, h)
                    self.status_updated.emit(f"Recording res set to: {w}x{h}")
                    self._log_to_file(f"Recording res set to: {w}x{h}")

                except Exception as e:
                    self.status_updated.emit(f"Error parsing resolution: {e}")
                    self._log_to_file(f"Error parsing resolution: {e}")

                    self.settings["output_resolution"] = None

    def _apply_custom_detector_settings(self, settings_snapshot: dict):
        self.bee_detector.set_candidate_settings(
            min_area=float(settings_snapshot.get("min_candidate_area", 100)),
            max_contour_area=float(settings_snapshot.get("max_candidate_area", 500)),
        )

    def _parse_ids(self, ids_str: str):
        """Parses the ID string (e.g., "1, 2, 5-10") into a set."""
        ids = set()
        if not ids_str:
            return ids
        try:
            parts = ids_str.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    ids.update(range(start, end + 1))
                else:
                    ids.add(int(part))
        except Exception as e:
            self.status_updated.emit(f"Invalid ID format: {e}")
            self._log_to_file(f"Invalid ID format: {e}")

        return ids

    @Slot(int, int, int)
    def set_feeding_zone(self, x, y, r):
        """(GUI Thread) Updates the feeding zone coordinates."""
        with self.settings_lock:
            self.settings["zone_center"] = (x, y)
            self.settings["zone_radius"] = r

    @Slot(str)
    def _handle_arduino_send_failed(self, msg: str):
        self.status_updated.emit("Arduino communication marked unhealthy. Disconnecting safely.")
        self._log_to_file("Arduino communication marked unhealthy. Disconnecting safely.")
        self.stop_arduino()

    @Slot(str)
    def on_arduino_reader_error(self, msg):
        self.status_updated.emit(msg)
        self._log_to_file(msg)
        self.status_updated.emit("Arduino reader error detected. Disconnecting safely.")
        self._log_to_file("Arduino reader error detected. Disconnecting safely.")
        self.stop_arduino()


    @Slot(str)
    def on_arduino_reader_disconnected(self, msg):
        self.status_updated.emit(msg)
        self._log_to_file(msg)

    # ===================================
    #  Pump & Recording Controls
    # ===================================

    def log_pump_event(self, event_type: str, tag_id: str):
        """(Any Thread) Logs a pump event to the console/status."""
        # This could be expanded to write to a file
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.status_updated.emit(f"PUMP_LOG: {timestamp} | {event_type} | ID: {tag_id}")
        self._log_to_file(f"PUMP_LOG: {timestamp} | {event_type} | ID: {tag_id}")

    def _log_to_file(self, text: str):
        """Write log line to text file if recording is active."""
        if self.log_file:
            with self.log_lock:
                try:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    self.log_file.write(f"[{timestamp}] {text}\n")
                    self.log_file.flush()
                except Exception:
                    pass  # Prevent crashes if log file fails


    @Slot()
    def start_motor_waiting_mode(self):
        """(GUI Thread) Tell Arduino to enter 'Waiting' mode."""
        self.status_updated.emit("Starting Motor (Waiting Mode)...")
        self._log_to_file("Starting Motor (Waiting Mode)...")

        self._send_to_arduino('W\n')  # New 'Waiting' command

    @Slot()
    def prime_pump(self):
        """(GUI Thread) Manually run the eject/prime command."""
        self.status_updated.emit("Manual Prime/Eject command sent.")
        self._log_to_file("Manual Prime/Eject command sent.")

        self.log_pump_event("MANUAL_EJECT_PRIME", "MANUAL")
        self._send_to_arduino('P\n')  # Prime/Manual Eject

    @Slot()
    def retract_pump(self):
        """(GUI Thread) Manually run the retract command."""
        self.status_updated.emit("Manual Retract command sent.")
        self._log_to_file("Manual Retract command sent.")

        self.log_pump_event("MANUAL_RETRACT", "MANUAL")
        self._send_to_arduino('L\n')  # Manual Retract

    @Slot()
    def stop_pump(self):
        """(GUI Thread) Manually send a STOP command."""
        self.status_updated.emit("Manual STOP command sent.")
        self._log_to_file("Manual STOP command sent.")

        self.log_pump_event("MANUAL_STOP", "MANUAL")
        self._send_to_arduino('S\n')  # Stop

    @Slot()
    def reset_calibration(self):
        """(GUI Thread) Resets the soft stop step calibration on Arduino."""
        self.status_updated.emit("Resetting step calibration...")
        self._log_to_file("Resetting step calibration...")
        self._send_to_arduino('K\n')

##################################################################################################
    @Slot()
    def toggle_recording(self):
        """(GUI Thread) Starts or stops video recording."""
        if not self.is_recording:
            # --- Start Recording ---
            if not self.cap:
                self.status_updated.emit("Cannot record: Camera not connected.")
                self._log_to_file("Cannot record: Camera not connected.")
                return
            
            # Get frame size from camera
            frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.recording_fps = 30


            # Determine output size
            with self.settings_lock:
                output_res = self.settings.get("output_resolution")

            if output_res:
                out_w, out_h = output_res
            else:
                out_w, out_h = frame_w, frame_h

            # Create filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"feed_output_{timestamp}_{out_w}x{out_h}.mp4"

            self.video_recorder = VideoRecorder(filename, self.recording_fps, (out_w, out_h))
            self.video_recorder.start()

            self.rec_start_time = None
            self.total_frames_sent = 0
            self.unique_frames_captured = 0
            self.duplicated_frames_count = 0
            self.dropped_frames_count = 0
            self.last_sync_log_time = time.perf_counter()

            log_filename = filename.replace(".mp4", ".txt")
            self.log_file = open(log_filename, "a", encoding="utf-8")
            self.status_updated.emit(f"Log recording started: {log_filename}")
            self._log_to_file(f"Log recording started: {log_filename}")

            self.is_recording = True
            self.status_updated.emit(f"Recording started: {filename}")
            self._log_to_file(f"Recording started: {filename}")
            self.recording_toggled.emit(True)
        else:
            # --- Stop Recording ---
            self.is_recording = False
            if self.video_recorder:
                self.video_recorder.stop()
                self.video_recorder = None
            if self.log_file:
                self.log_file.close()
                self.log_file = None

            self.status_updated.emit("Log recording stopped.")
            self._log_to_file("Log recording stopped.")
            self.status_updated.emit("Recording stopped.")
            self._log_to_file("Recording stopped.")
            self.recording_toggled.emit(False)
#############################################################################################


    @Slot(str)
    def on_arduino_message(self, msg):
        """Display Arduino messages in GUI status log."""
        self.status_updated.emit(msg)
        self._log_to_file(msg)

    # ===================================
    #  Camera & Detection
    # ===================================

    def set_exposure_lock(self, lock: bool):
        """(Any Thread) Sets the camera's auto exposure lock and manual shutter speed."""
        if self.cap:
            if lock:
                # 1. First, set to Manual Mode (Value 1 = Manual, 3 = Auto in MSMF)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
                
                # 2. Set Exposure. MUST be -6 or faster (more negative) to allow 60 FPS.
                # -6 = 15.6ms, -7 = 7.8ms, -8 = 3.9ms
                self.cap.set(cv2.CAP_PROP_EXPOSURE, -6) 
                
                # 3. Set Gain to 0 to eliminate electronic noise
                self.cap.set(cv2.CAP_PROP_GAIN, 0)
                
                msg = "Exposure LOCKED (Manual, Exp: -6, Gain: 0)"
            else:
                # Switch back to fully automatic mode
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) 
                msg = "Exposure UNLOCKED (Auto Mode)"

            self.status_updated.emit(msg)
            self._log_to_file(msg)

            with self.settings_lock:
                self.settings['exposure_lock'] = lock

    def set_autofocus(self, enable: bool):
        """(Any Thread) Sets the camera's autofocus."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if enable else 0)
            self.status_updated.emit(f"Autofocus set to: {enable}")
            self._log_to_file(f"Autofocus set to: {enable}")

            with self.settings_lock:
                self.settings['autofocus'] = enable

    def set_focus(self, value: int):
        """(Any Thread) Sets the camera's manual focus."""
        if self.cap and not self.settings['autofocus']:
            self.cap.set(cv2.CAP_PROP_FOCUS, value)
            # Don't log this, it's too noisy
            # self.status_updated.emit(f"Manual Focus set to: {value}")
            with self.settings_lock:
                self.settings['focus'] = value

    def _is_tag_in_zone(self, tag_center: tuple, zone_center: tuple, zone_radius: int):
        """Helper to check if a tag's center is inside the circle."""
        if zone_radius <= 0:
            return False
        dist = np.sqrt((tag_center[0] - zone_center[0]) ** 2 + (tag_center[1] - zone_center[1]) ** 2)
        return dist < zone_radius

    # ===================================
    #  Processing Helpers
    # ===================================

    def _read_camera_frame(self):
        if not self.cap or not self.cap.isOpened():
            self.status_updated.emit("Processing loop: Camera not available.")
            self._log_to_file("Processing loop: Camera not available.")
            time.sleep(0.01)
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.status_updated.emit("Processing loop: Failed to read frame.")
            self._log_to_file("Processing loop: Failed to read frame.")
            time.sleep(0.01)
            return None

        return frame

    def _compute_fps(self, prev_time: float):
        curr_time = time.perf_counter()
        delta_time = curr_time - prev_time
        fps = 1 / delta_time if delta_time > 0 else 0
        return fps, curr_time

    def _prepare_frame_artifacts(self, frame: np.ndarray):
        timestamp_str = time.strftime("%d %H:%M:%S")
        h, w = frame.shape[:2]

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.0
        thickness = 2

        (label_width, label_height), baseline = cv2.getTextSize(
            timestamp_str, font, font_scale, thickness
        )
        x_pos = w - label_width - 15
        y_pos = h - 15

        original_frame = frame.copy()
        annotated_frame = frame.copy()

        return {
            "timestamp_str": timestamp_str,
            "h": h,
            "w": w,
            "font": font,
            "font_scale": font_scale,
            "thickness": thickness,
            "x_pos": x_pos,
            "y_pos": y_pos,
            "original_frame": original_frame,
            "annotated_frame": annotated_frame,
        }

    def _get_processing_settings_snapshot(self):
        with self.settings_lock:
            return {
                "allowed_ids": self.settings["allowed_ids"],
                "zone_center": self.settings["zone_center"],
                "zone_radius": self.settings["zone_radius"],
                "entry_frames_target": self.settings["entry_frames"],
                "exit_frames_target": self.settings["exit_frames"],
                "output_res": self.settings["output_resolution"],
                "b_size": self.settings.get("blur_size", 7) | 1,
                "t_val": self.settings.get("threshold", 15),
                "op_k": max(1, self.settings.get("open_kernel", 2)),
                "op_p": self.settings.get("open_power", 1),
                "cl_k": max(1, self.settings.get("close_kernel", 2)),
                "cl_p": self.settings.get("close_power", 2),
                "box_w": self.settings.get("box_width", 200),
                "box_h": self.settings.get("box_height", 200),
                "min_area": self.settings.get("min_area", 500),
                "trigger_mode": self.settings.get("trigger_mode", "both"),
                "bg_mode": self.settings.get("bg_mode", "With BG"),
                "rx": self.settings["roi_x"],
                "ry": self.settings["roi_y"],
                "rw": self.settings["roi_w"],
                "rh": self.settings["roi_h"],
                "sec_center": self.settings["sec_zone_center"],
                "sec_radius": self.settings["sec_zone_radius"],
                "diff_limit": self.settings["diff_intensity"],
            }

    def _prepare_gray_and_blur(self, frame: np.ndarray, blur_size: int):
        gray = frame[:,:,2] #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        return gray, blur

    def _update_background_references(self, blur, bg_mode, sec_radius, first_blur, secondary_blur):
        if self.should_recapture_bg:
            if bg_mode == "With BG":
                first_blur = blur.copy()
            if sec_radius > 0:
                secondary_blur = blur.copy()

            self.should_recapture_bg = False
            self.status_updated.emit("Background updated.")
            self._log_to_file("Background reference frame recaptured.")
            return first_blur, secondary_blur, True

        if bg_mode == "With BG" and first_blur is None:
            first_blur = blur.copy()
            self.status_updated.emit("Background updated.")
            self._log_to_file("Background reference frame recaptured.")
            return first_blur, secondary_blur, True

        if sec_radius > 0 and secondary_blur is None:
            secondary_blur = blur.copy()

        return first_blur, secondary_blur, False

    def _process_secondary_zone(
        self,
        blur,
        gray_shape,
        annotated_frame,
        sec_center,
        sec_radius,
        diff_limit,
        secondary_blur,
        last_sec_log_time,
    ):
        if sec_radius <= 0:
            return last_sec_log_time

        if secondary_blur is not None:
            sec_diff = cv2.absdiff(secondary_blur, blur)

            sec_mask = np.zeros(gray_shape, dtype=np.uint8)
            cv2.circle(sec_mask, sec_center, sec_radius, 255, -1)

            mean_val = cv2.mean(sec_diff, mask=sec_mask)[0]

            current_time = time.perf_counter()
            if mean_val > diff_limit and (current_time - last_sec_log_time) >= 0.5:
                self.status_updated.emit(f"Possible feeding (Intensity: {mean_val:.1f})")
                self._log_to_file(f"Possible feeding detected at secondary zone: {mean_val:.1f}")
                last_sec_log_time = current_time

            sec_color = (255, 255, 0)
            if mean_val > diff_limit:
                sec_color = (255, 0, 0)

            cv2.circle(annotated_frame, sec_center, sec_radius, sec_color, 1)
        else:
            cv2.circle(annotated_frame, sec_center, sec_radius, (255, 255, 0), 1)

        return last_sec_log_time

    def _run_detector_on_crop(self, crop_gray, scale_factor):
        if crop_gray is None or crop_gray.size == 0:
            return None
        
        zoom = cv2.resize(
            crop_gray,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LANCZOS4,
        )
        return self.bee_detector.detect(zoom)

    def _map_candidate_quads_to_global(self, candidates, x_offset, y_offset, scale_factor):
        offset = np.array([x_offset, y_offset], dtype=np.float32)
        candidate_global_quads = []

        for quad in candidates:
            global_quad = (np.asarray(quad, dtype=np.float32) / scale_factor) + offset
            candidate_global_quads.append(global_quad)

        return candidate_global_quads

    def _map_detections_to_global(
        self,
        detections,
        x_offset,
        y_offset,
        scale_factor,
        allowed_ids,
    ):
        offset = np.array([x_offset, y_offset], dtype=np.float32)
        accepted = []

        for det in detections:
            tag_id = int(det["id"])
            if tag_id not in allowed_ids:
                continue

            local_corners = np.asarray(det["marker_corners"], dtype=np.float32) / scale_factor

            global_det = dict(det)
            global_det["quad"] = (np.asarray(det["quad"], dtype=np.float32) / scale_factor) + offset
            global_det["marker_corners"] = local_corners + offset
            accepted.append(global_det)

        return accepted

    def _draw_crop_detection_overlays(self, annotated_frame, candidate_global_quads, accepted):
        if candidate_global_quads:
            drawn = self.bee_drawer.draw_quads(
                annotated_frame,
                candidate_global_quads,
                colors=[(0, 255, 255)] * len(candidate_global_quads),
                draw_center=False,
                draw_first_corner=False,
                draw_orientation=False,
                copy=False,
            )
            if drawn is not annotated_frame:
                annotated_frame[:, :] = drawn

        if accepted:
            drawn = self.bee_drawer.draw_detected_markers(
                annotated_frame,
                accepted,
                draw_ids=True,
                draw_angles=True,
                draw_center=False,
                draw_first_corner=True,
                draw_orientation=False,
                copy=False,
                border_color_overrides={1: (0, 255, 0)},
                text_color_overrides={1: (0, 255, 0)},
            )
            if drawn is not annotated_frame:
                annotated_frame[:, :] = drawn

    def _detect_custom_markers_in_crop(
        self,
        crop_gray,
        x_offset,
        y_offset,
        scale_factor,
        allowed_ids,
        annotated_frame,
    ):
        """
        Detect custom markers in a crop, convert detections back to global coordinates,
        draw accepted detections, and return the accepted global detections.
        """
        result = self._run_detector_on_crop(crop_gray, scale_factor)
        if result is None:
            return []

        candidate_global_quads = self._map_candidate_quads_to_global(
            result.get("candidates", []),
            x_offset,
            y_offset,
            scale_factor,
        )

        accepted = self._map_detections_to_global(
            result.get("detections", []),
            x_offset,
            y_offset,
            scale_factor,
            allowed_ids,
        )

        self._draw_crop_detection_overlays(
            annotated_frame,
            candidate_global_quads,
            accepted,
        )

        return accepted

    def _evaluate_trigger(self, trigger_mode, motion_in_zone, aruco_in_zone):
        if trigger_mode == "motion":
            return motion_in_zone
        if trigger_mode == "aruco":
            return aruco_in_zone
        if trigger_mode == "both":
            return motion_in_zone and aruco_in_zone
        return False

    def _run_with_bg_detection(
        self,
        gray,
        blur,
        first_blur,
        annotated_frame,
        settings,
        frame_info,
    ):
        h = frame_info["h"]
        w = frame_info["w"]

        allowed_ids = settings["allowed_ids"]
        zone_center = settings["zone_center"]
        zone_radius = settings["zone_radius"]
        box_w = settings["box_w"]
        box_h = settings["box_h"]
        min_area = settings["min_area"]
        trigger_mode = settings["trigger_mode"]
        rx = settings["rx"]
        ry = settings["ry"]
        rw = settings["rw"]
        rh = settings["rh"]
        t_val = settings["t_val"]
        op_k = settings["op_k"]
        op_p = settings["op_p"]
        cl_k = settings["cl_k"]
        cl_p = settings["cl_p"]

        is_frame_triggered = False
        current_tag_id = "None"

        cv2.rectangle(annotated_frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[ry:ry + rh, rx:rx + rw] = 255

        diff = cv2.absdiff(first_blur, blur)
        diff = cv2.bitwise_and(diff, diff, mask=mask)

        _, thresh = cv2.threshold(diff, t_val, 255, cv2.THRESH_BINARY)

        kernel_open = np.ones((op_k, op_k), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=op_p)

        kernel_close = np.ones((cl_k, cl_k), np.uint8)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=cl_p)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            motion_in_zone = self._is_tag_in_zone((cX, cY), zone_center, zone_radius)

            mu11 = M["mu11"]
            mu20 = M["mu20"]
            mu02 = M["mu02"]
            angle = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))

            cv2.circle(annotated_frame, (cX, cY), 5, (255, 0, 0), -1)
            line_len = 40
            p2 = (int(cX + line_len * np.cos(angle)), int(cY + line_len * np.sin(angle)))
            cv2.line(annotated_frame, (cX, cY), p2, (255, 255, 0), 2)
            cv2.drawContours(annotated_frame, [cnt], -1, (255, 255, 0), 1)

            aruco_in_zone = False
            y1, y2 = np.clip([cY - box_h // 2, cY + box_h // 2], 0, h)
            x1, x2 = np.clip([cX - box_w // 2, cX + box_w // 2], 0, w)

            if y2 > y1 and x2 > x1:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                bee_crop_gray = gray[y1:y2, x1:x2]

                detections = self._detect_custom_markers_in_crop(
                    bee_crop_gray,
                    x1,
                    y1,
                    scale_factor=1.0,
                    allowed_ids=allowed_ids,
                    annotated_frame=annotated_frame,
                )

                for det in detections:
                    current_tag_id = str(det["id"])
                    global_corners = det["marker_corners"]

                    t_cX = int(global_corners[:, 0].mean())
                    t_cY = int(global_corners[:, 1].mean())

                    if self._is_tag_in_zone((t_cX, t_cY), zone_center, zone_radius):
                        aruco_in_zone = True

            if self._evaluate_trigger(trigger_mode, motion_in_zone, aruco_in_zone):
                is_frame_triggered = True

        return is_frame_triggered, current_tag_id

    def _run_without_bg_detection(
        self,
        gray,
        annotated_frame,
        settings,
        frame_info,
    ):
        h = frame_info["h"]
        w = frame_info["w"]

        allowed_ids = settings["allowed_ids"]
        zone_center = settings["zone_center"]
        zone_radius = settings["zone_radius"]

        is_frame_triggered = False
        current_tag_id = "None"

        if zone_radius > 0:
            zX, zY = zone_center
            x1, y1 = max(0, zX - zone_radius), max(0, zY - zone_radius)
            x2, y2 = min(w, zX + zone_radius), min(h, zY + zone_radius)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            zone_crop = gray[y1:y2, x1:x2]

            if zone_crop.size > 0:
                detections = self._detect_custom_markers_in_crop(
                    zone_crop,
                    x1,
                    y1,
                    scale_factor=1.0,
                    allowed_ids=allowed_ids,
                    annotated_frame=annotated_frame,
                )

                if detections:
                    is_frame_triggered = True
                    #current_tag_id = str(detections[-1]["id"])
                    current_tag_id = str(detections[0]["id"])

        return is_frame_triggered, current_tag_id

    def _update_zone_counters(self, is_frame_triggered, entry_frames_target, exit_frames_target):
        if is_frame_triggered:
            self.frames_out_of_zone_counter = 0
            self.frames_in_zone_counter = min(
                self.frames_in_zone_counter + 1,
                entry_frames_target,
            )
        else:
            self.frames_in_zone_counter = 0
            self.frames_out_of_zone_counter = min(
                self.frames_out_of_zone_counter + 1,
                exit_frames_target,
            )

    def _handle_feed_state_transition(self, current_tag_id, entry_frames_target, exit_frames_target):
        if not self.is_feeding and (self.frames_in_zone_counter >= entry_frames_target):
            self.is_feeding = True
            self.status_updated.emit(f"Tag {current_tag_id} confirmed. Sending FEED command.")
            self._log_to_file(f"Tag {current_tag_id} confirmed. Sending FEED command.")
            self._send_to_arduino('F\n')
            self.log_pump_event("AUTO_FEED", current_tag_id)
            self.frames_in_zone_counter = 0

        elif self.is_feeding and (self.frames_out_of_zone_counter >= exit_frames_target):
            self.is_feeding = False
            self.status_updated.emit("Tag lost. Re-arming trigger.")
            self._log_to_file("Tag lost. Re-arming trigger.")
            self.log_pump_event("TAG_LOST_REARM", "N/A")
            self.frames_out_of_zone_counter = 0

    def _draw_main_overlays(self, annotated_frame, original_frame, frame_info, settings, fps):
        zone_center = settings["zone_center"]
        zone_radius = settings["zone_radius"]

        font = frame_info["font"]
        font_scale = frame_info["font_scale"]
        thickness = frame_info["thickness"]
        timestamp_str = frame_info["timestamp_str"]
        x_pos = frame_info["x_pos"]
        y_pos = frame_info["y_pos"]
        h = frame_info["h"]
        w = frame_info["w"]

        if zone_radius > 0:
            if self.is_feeding:
                zone_color = (0, 255, 0)
            else:
                zone_color = (0, 0, 255)

            if self.frames_in_zone_counter > 0 and not self.is_feeding:
                zone_color = (0, 255, 255)
            elif self.frames_out_of_zone_counter > 0 and self.is_feeding:
                zone_color = (0, 165, 255)

            cv2.circle(annotated_frame, zone_center, zone_radius, zone_color, 2)

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (w - 150, h - 50), font, 0.7, (0, 255, 0), 2)

        cv2.putText(original_frame, timestamp_str, (x_pos + 1, y_pos + 1), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(original_frame, timestamp_str, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness)

        cv2.putText(annotated_frame, timestamp_str, (x_pos + 1, y_pos + 1), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(annotated_frame, timestamp_str, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness)

    def _handle_recording_frame(self, original_frame, output_res):
        if not self.is_recording or not self.video_recorder:
            return

        try:
            now = time.perf_counter()
            if self.rec_start_time is None:
                self.rec_start_time = now

            self.unique_frames_captured += 1
            elapsed_time = now - self.rec_start_time

            frames_should_be_sent = round(elapsed_time * self.recording_fps) + 1
            gap = frames_should_be_sent - self.total_frames_sent

            if gap >= 1:
                if output_res:
                    record_frame = cv2.resize(original_frame, output_res, interpolation=cv2.INTER_AREA)
                else:
                    record_frame = original_frame

                for i in range(gap):
                    if self.video_recorder.add_frame(record_frame):
                        self.total_frames_sent += 1
                        if i > 0:
                            self.duplicated_frames_count += 1
                    else:
                        self.dropped_frames_count += 1

            if (now - self.last_sync_log_time) >= 3600:
                cpu_usage = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().percent
                q_load = self.video_recorder.frame_queue.qsize()
                drift = frames_should_be_sent - self.total_frames_sent

                goal_fps = self.settings.get("processing_target_fps", 30.0)
                frames_theoretically_required = max(1, elapsed_time * goal_fps)
                efficiency = (self.unique_frames_captured / frames_theoretically_required) * 100

                sync_msg = (
                    f"HEALTH_REPORT: VideoTime: {elapsed_time:.1f}s |"
                    f"Sync: {self.total_frames_sent}/{frames_should_be_sent} | "
                    f"Unique Frames: {self.unique_frames_captured} |"
                    f"Time: {elapsed_time/3600:.1f}h | "
                    f"Frames: {self.total_frames_sent}/{frames_should_be_sent} | "
                    f"CPU: {cpu_usage}% | RAM: {ram_usage}% | "
                    f"Drift: {drift}f | Queue: {q_load}/512 | "
                    f"Efficiency: {efficiency:.1f}% | "
                    f"Dropped: {self.dropped_frames_count}|"
                    f"Duplicated: {self.duplicated_frames_count}"
                )

                print(sync_msg)
                self._log_to_file(sync_msg)
                self.last_sync_log_time = time.perf_counter()

        except Exception as e:
            self.status_updated.emit(f"Video write error: {e}")
            self._log_to_file(f"Video write error: {e}")
            self.toggle_recording()

    def _emit_gui_frame(self, annotated_frame):
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        self.frame_ready.emit(rgb_frame)

    # ===================================
    #  Main Processing Loop
    # ===================================

    def _process_loop(self):
        prev_time = time.perf_counter()
        last_sec_log_time = 0
        first_blur = None
        secondary_blur = None

        while self.is_processing:
            frame = self._read_camera_frame()
            if frame is None:
                continue

            fps, prev_time = self._compute_fps(prev_time)
            frame_info = self._prepare_frame_artifacts(frame)
            settings = self._get_processing_settings_snapshot()

            gray, blur = self._prepare_gray_and_blur(frame, settings["b_size"])

            first_blur, secondary_blur, should_continue = self._update_background_references(
                blur=blur,
                bg_mode=settings["bg_mode"],
                sec_radius=settings["sec_radius"],
                first_blur=first_blur,
                secondary_blur=secondary_blur,
            )
            if should_continue:
                continue

            last_sec_log_time = self._process_secondary_zone(
                blur=blur,
                gray_shape=gray.shape,
                annotated_frame=frame_info["annotated_frame"],
                sec_center=settings["sec_center"],
                sec_radius=settings["sec_radius"],
                diff_limit=settings["diff_limit"],
                secondary_blur=secondary_blur,
                last_sec_log_time=last_sec_log_time,
            )

            if settings["bg_mode"] == "With BG":
                is_frame_triggered, current_tag_id = self._run_with_bg_detection(
                    gray=gray,
                    blur=blur,
                    first_blur=first_blur,
                    annotated_frame=frame_info["annotated_frame"],
                    settings=settings,
                    frame_info=frame_info,
                )
            else:
                is_frame_triggered, current_tag_id = self._run_without_bg_detection(
                    gray=gray,
                    annotated_frame=frame_info["annotated_frame"],
                    settings=settings,
                    frame_info=frame_info,
                )

            self._update_zone_counters(
                is_frame_triggered,
                settings["entry_frames_target"],
                settings["exit_frames_target"],
            )

            self._handle_feed_state_transition(
                current_tag_id,
                settings["entry_frames_target"],
                settings["exit_frames_target"],
            )

            self._draw_main_overlays(
                annotated_frame=frame_info["annotated_frame"],
                original_frame=frame_info["original_frame"],
                frame_info=frame_info,
                settings=settings,
                fps=fps,
            )

            self._handle_recording_frame(
                original_frame=frame_info["original_frame"],
                output_res=settings["output_res"],
            )

            self._emit_gui_frame(frame_info["annotated_frame"])

        self.status_updated.emit("Processing loop stopped.")
        self._log_to_file("Processing loop stopped.")