# feeder_core.py
# This file contains the core logic for the Feeder Controller.
# It runs in a separate thread and handles camera, detection, and Arduino comms.

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import serial
import threading
from PySide6.QtCore import QObject, QThread, Signal, Slot  # <-- FIX: Import Slot

# --- Arduino Read Thread ---
class ArduinoReader(QThread):
    message_received = Signal(str)  # Emits full lines from Arduino

    def __init__(self, arduino_serial):
        super().__init__()
        self.arduino = arduino_serial
        self.buffer = b""  # Accumulate partial lines

    def run(self):
        while True:
            if not self.arduino or not self.arduino.is_open:
                break  # Exit loop when serial is closed

            try:
                waiting = self.arduino.in_waiting
            except Exception:
                break  # Serial died → stop thread

            if waiting > 0:
                try:
                    data = self.arduino.read(waiting)
                except Exception:
                    break  # If read fails, exit gracefully

                self.buffer += data
                lines = self.buffer.split(b'\n')
                self.buffer = lines[-1]

                for line in lines[:-1]:
                    if line.strip():
                        msg = line.decode('ascii', errors='ignore').strip()
                        self.message_received.emit(f"[Arduino]: {msg}")

            self.msleep(50)

    def stop(self):
        self.quit()
        self.wait(500)

class FeederController(QObject):
    # Signals to communicate with the GUI
    frame_ready = Signal(np.ndarray)  # Emits the annotated video frame
    status_updated = Signal(str)  # Emits a status message
    recording_toggled = Signal(bool)  # Emits the recording state
    zone_updated = Signal(int, int, int)  # Emits (x, y, r) for visual sync

    def __init__(self):
        super().__init__()

        # --- Hardware & State ---
        self.log_file = None
        self.cap = None
        self.arduino = None
        self.is_processing = False
        self.is_recording = False
        self.video_writer = None
        self.processing_thread = None  # Renamed for clarity
        self.settings_lock = threading.Lock()  # Protects self.settings

        # --- ArUco Setup ---
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # --- Default Settings ---
        self.settings = {
            # Camera
            "exposure_lock": False,
            "autofocus": True,
            "focus": 0,
            # Electrodes
            "threshold_1": 512,
            "threshold_2": 512,
            "threshold_3": 512,
            # Detection
            "allowed_ids": set([1, 2, 5, 6, 7, 8, 9, 10]),
            "min_size": 50,
            "max_size": 2000,
            "timeout_ms": 10000,  # Arduino timeout
            "entry_frames": 15,  # Python: consecutive frames to confirm entry
            "exit_frames": 30,  # Python: consecutive frames to confirm exit
            # Motor
            "motor_speed": 1000,
            "motor_accel": 500,
            # Recording
            "output_resolution": None,  # (w, h) tuple or None
            # Zone
            "zone_center": (0, 0),
            "zone_radius": 0
        }

        # --- New Frame-based Zone Logic ---
        self.is_feeding = False  # This is our new, simple state
        self.frames_in_zone_counter = 0
        self.frames_out_of_zone_counter = 0

        self.arduino_reader = None  # Thread for reading Arduino messages

    # ===================================
    #  Hardware Connection
    # ===================================

    def connect_camera(self, index=0):
        """(GUI Thread) Connects to the camera and starts the processing thread."""
        try:
            self.status_updated.emit(f"Opening camera index {index}...")
            self._log_to_file(f"Opening camera index {index}...")
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {index}.")
            
            # 1. Force MJPEG Compression to unblock USB bandwidth
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # 2. Set Resolution (720p is often better for hitting 60fps on webcams)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1260)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # 3. Request High FPS
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # Apply initial camera settings
            self.set_exposure_lock(self.settings['exposure_lock'])
            self.set_autofocus(self.settings['autofocus'])
            if not self.settings['autofocus']:
                self.set_focus(self.settings['focus'])

            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.processing_thread.start()
            self.status_updated.emit("Camera connected. Processing loop started.")
            self._log_to_file("Camera connected. Processing loop started.")

            return True
        except Exception as e:
            self.status_updated.emit(f"Camera Error: {e}")
            self._log_to_file(f"Camera Error: {e}")

            self.cap = None
            return False

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


    def connect_arduino(self, port):
        """(Worker Thread) Connects to the Arduino."""
        try:
            self.status_updated.emit(f"Connecting to Arduino on {port}...")
            self._log_to_file(f"Connecting to Arduino on {port}...")

            # 2-second timeout for connection
            self.arduino = serial.Serial(port, 115200, timeout=2)
            time.sleep(2)  # Wait for Arduino to reset

            if not self.arduino.is_open:
                raise Exception("Serial port not open.")

            self.arduino.flushInput()
            self.arduino_reader = ArduinoReader(self.arduino)
            self.arduino_reader.message_received.connect(self.on_arduino_message)
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
                self.arduino.close()
            self.arduino = None
            return False

    def stop_arduino(self):
        """(GUI Thread) Disconnects from the Arduino."""
        if self.arduino:
            try:
                self.arduino.close()
            except Exception as e:
                self.status_updated.emit(f"Error disconnecting Arduino: {e}")
                self._log_to_file(f"Error disconnecting Arduino: {e}")

            self.arduino = None
            self.status_updated.emit("Arduino disconnected.")
            self._log_to_file("Arduino disconnected.")

        if self.arduino_reader:
            self.arduino_reader.stop()
            self.arduino_reader = None
        self.status_updated.emit("Arduino reader stopped.")
        self._log_to_file("Arduino reader stopped.")


    def stop_all(self):
        """(GUI Thread) Stops all hardware and threads."""
        self.stop_camera()
        self.stop_arduino()
        self.status_updated.emit("All systems stopped.")
        self._log_to_file("All systems stopped.")

    def _send_to_arduino(self, command: str):
        """(Any Thread) Sends a command to the Arduino, thread-safe."""
        if self.arduino and self.arduino.is_open:
            try:
                self.arduino.write(command.encode('ascii'))
            except Exception as e:
                self.status_updated.emit(f"Arduino Send Error: {e}")
                self._log_to_file(f"Arduino Send Error: {e}")
                # Try to stop and disconnect
                self.stop_arduino()
        else:
            # We don't log this one, it's too noisy if Arduino is just disconnected
            # self.status_updated.emit("Cannot send: Arduino not connected.")
            pass

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
        self._send_to_arduino(f"M{settings['timeout_ms']}\n")
        self.status_updated.emit("All settings sent to Arduino.")
        self._log_to_file("All settings sent to Arduino.")


    def update_detection_settings(self, settings: dict):
        """(GUI Thread) Updates detection-related settings."""
        with self.settings_lock:
            try:
                # 1. Parse Allowed IDs
                ids_str = settings.get("allowed_ids", "")
                self.settings["allowed_ids"] = self._parse_ids(ids_str)

                # 2. Parse Sizes
                self.settings["min_size"] = int(settings.get("min_size", 50))
                self.settings["max_size"] = int(settings.get("max_size", 2000))

                # 3. Parse Delays (NEW)
                self.settings["entry_frames"] = int(settings.get("entry_frames", 15))
                self.settings["exit_frames"] = int(settings.get("exit_frames", 30))

                # 4. Parse Arduino Timeout (convert from s to ms)
                timeout_s = float(settings.get("timeout", 10.0))
                self.settings["timeout_ms"] = int(timeout_s * 1000)

                self.status_updated.emit(f"Detection settings updated. IDs: {self.settings['allowed_ids']}")
                self._log_to_file(f"Detection settings updated. IDs: {self.settings['allowed_ids']}")

                # Send the one setting the Arduino cares about
                self._send_to_arduino(f"M{self.settings['timeout_ms']}\n")

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

                self.status_updated.emit(
                    f"Electrode thresholds updated: T1={self.settings['threshold_1']}, T2={self.settings['threshold_2']}, T3={self.settings['threshold_3']}")
                self._log_to_file(f"Electrode thresholds updated: T1={self.settings['threshold_1']}, T2={self.settings['threshold_2']}, T3={self.settings['threshold_3']}")

                # Send to Arduino
                self._send_to_arduino(f"T{self.settings['threshold_1']}\n")
                self._send_to_arduino(f"U{self.settings['threshold_2']}\n")
                self._send_to_arduino(f"I{self.settings['threshold_3']}\n")
            except Exception as e:
                self.status_updated.emit(f"Error parsing electrode settings: {e}")
                self._log_to_file(f"Error parsing electrode settings: {e}")

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

        # Don't log this, it's too noisy
        # self.status_updated.emit(f"Zone updated: ({x}, {y}), r={r}")

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
    def toggle_recording(self):
        """(GUI Thread) Starts or stops video recording."""
        if not self.is_recording:
            # --- Start Recording ---
            if not self.cap:
                self.status_updated.emit("Cannot record: Camera not connected.")
                self._log_to_file("Cannot record: Camera not connected.")

                return

            try:
                # Get frame size from camera
                frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0: fps = 30  # Default fallback

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

                # Create VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (out_w, out_h))
                log_filename = filename.replace(".mp4", ".txt")
                self.log_file = open(log_filename, "a", encoding="utf-8")
                self.status_updated.emit(f"Log recording started: {log_filename}")
                self._log_to_file(f"Log recording started: {log_filename}")


                if not self.video_writer.isOpened():
                    raise Exception(f"Could not open video writer for {filename}")

                self.is_recording = True
                self.status_updated.emit(f"Recording started: {filename}")
                self._log_to_file(f"Recording started: {filename}")

                self.recording_toggled.emit(True)

            except Exception as e:
                self.status_updated.emit(f"Recording Error: {e}")
                self._log_to_file(f"Recording Error: {e}")

                if self.video_writer:
                    self.video_writer.release()
                self.video_writer = None
        else:
            # --- Stop Recording ---
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            self.status_updated.emit("Log recording stopped.")
            self._log_to_file("Log recording stopped.")

            self.status_updated.emit("Recording stopped.")
            self._log_to_file("Recording stopped.")

            self.recording_toggled.emit(False)
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
            # 0.25 = Manual Exposure, 0.75 = Auto Exposure
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if lock else 0.75)
            
            # NEW: If locked, force a very fast shutter speed to stop motion blur
            if lock:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, -7) # Tweak this between -4 and -8
                
            self.status_updated.emit(f"Exposure Lock set to: {lock}")
            self._log_to_file(f"Exposure Lock set to: {lock}")

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
    #  MAIN PROCESSING LOOP
    # ===================================

    def _process_loop(self):
        """
        (Processing Thread) This is the main loop that reads frames,
        performs detection, and sends signals.
        """
        while self.is_processing:
            if not self.cap or not self.cap.isOpened():
                self.status_updated.emit("Processing loop: Camera not available.")
                self._log_to_file("Processing loop: Camera not available.")

                time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.status_updated.emit("Processing loop: Failed to read frame.")
                self._log_to_file("Processing loop: Failed to read frame.")

                time.sleep(0.1)
                continue

            # --- 1. Setup ---
            # Zaman bilgisini hazırla
            timestamp_str = time.strftime("%H:%M:%S")
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Yazı boyutunu hesapla
            (label_width, label_height), baseline = cv2.getTextSize(timestamp_str, font, font_scale, thickness)
            h, w = frame.shape[:2]
            x_pos = w - label_width - 15
            y_pos = h - 15

            # Yazıyı ana frame üzerine yaz (Böylece her yere yansır)
            # Siyah gölge
            cv2.putText(frame, timestamp_str, (x_pos+1, y_pos+1), font, font_scale, (0, 0, 0), thickness)
            # Beyaz ana metin
            cv2.putText(frame, timestamp_str, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness)

            # Şimdi kopyaları al
            original_frame = frame.copy()  # Üzerinde saat olan frame kayıt için hazır
            annotated_frame = frame        # Üzerinde saat olan frame analiz ve GUI için hazır
            original_frame = frame.copy()  # For recording
            annotated_frame = frame
           
            # Upscale for small marker detection
            UPSCALE = 1.0   # try: 1.3 → 2.0
            proc = cv2.resize(frame, None, fx=UPSCALE, fy=UPSCALE,
                                            interpolation=cv2.INTER_CUBIC)


            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

            red = frame[:,:,2]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            red = clahe.apply(red)

            th = cv2.adaptiveThreshold(
                red, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                15, 4
            )   



            # Get current settings in a thread-safe way
            with self.settings_lock:
                allowed_ids = self.settings["allowed_ids"]
                min_size = self.settings["min_size"]
                max_size = self.settings["max_size"]
                zone_center = self.settings["zone_center"]
                zone_radius = self.settings["zone_radius"]
                entry_frames_target = self.settings["entry_frames"]
                exit_frames_target = self.settings["exit_frames"]
                output_res = self.settings["output_resolution"]

            # --- 2. ArUco Detection ---
            #corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            corners, ids, _ = self.aruco_detector.detectMarkers(proc)
            if corners is not None and len(corners) > 0:
                scaled_corners = []
                for c in corners:
                    scaled_corners.append(c / UPSCALE)
                corners = scaled_corners


            is_tag_in_zone = False
            current_tag_id = "None"

            filtered_corners_for_drawing = []
            filtered_ids_for_drawing = []

            if ids is not None:
                for i, tag_id in enumerate(ids.flatten()):
                    if tag_id in allowed_ids:
                        # Calculate perimeter (size)
                        c = corners[i][0]
                        perimeter = cv2.arcLength(c, True)

                        if min_size < perimeter < max_size:
                            # This is a valid tag
                            filtered_corners_for_drawing.append(corners[i])
                            filtered_ids_for_drawing.append(np.array([tag_id]))

                            # Get tag center
                            M = cv2.moments(c)
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])  # <-- FIX: Was "m1DRAFT"

                            # Check if this tag is in the zone
                            if self._is_tag_in_zone((cX, cY), zone_center, zone_radius):
                                is_tag_in_zone = True
                                current_tag_id = str(tag_id)
                                break  # Found a valid tag in the zone

            # --- 3. NEW Frame-based State Logic ---

            if is_tag_in_zone:
                self.frames_out_of_zone_counter = 0  # Reset miss counter
                self.frames_in_zone_counter = min(self.frames_in_zone_counter + 1,
                                                  entry_frames_target)  # Increment hit counter, cap at target
            else:
                self.frames_in_zone_counter = 0  # Reset hit counter
                self.frames_out_of_zone_counter = min(self.frames_out_of_zone_counter + 1,
                                                      exit_frames_target)  # Increment miss counter, cap at target

            # --- State Transition ---

            # Check for FEED condition
            # Has not been fed AND has been in zone for X frames
            if not self.is_feeding and (self.frames_in_zone_counter >= entry_frames_target):
                self.is_feeding = True
                self.status_updated.emit(f"Tag {current_tag_id} confirmed. Sending FEED command.")
                self._log_to_file(f"Tag {current_tag_id} confirmed. Sending FEED command.")

                self._send_to_arduino('F\n')  # NEW 'Feed' command
                self.log_pump_event("AUTO_FEED", current_tag_id)
                self.frames_in_zone_counter = 0  # Reset counter

            # Check for RE-ARM condition
            # Is in feeding state AND has been out of zone for X frames
            elif self.is_feeding and (self.frames_out_of_zone_counter >= exit_frames_target):
                self.is_feeding = False  # Re-arm the trigger
                self.status_updated.emit("Tag lost. Re-arming trigger.")
                self._log_to_file("Tag lost. Re-arming trigger.")

                # We no longer send 'R' - Arduino handles its own return
                self.log_pump_event("TAG_LOST_REARM", "N/A")
                self.frames_out_of_zone_counter = 0  # Reset counter

            # --- 4. Drawing & Annotation ---

            # Draw the feeding zone
            if zone_radius > 0:
                # Set color based on state
                if self.is_feeding:
                    zone_color = (0, 255, 0)  # Green
                else:
                    zone_color = (0, 0, 255)  # Red

                # Show entry/exit progress
                if self.frames_in_zone_counter > 0 and not self.is_feeding:
                    zone_color = (0, 255, 255)  # Yellow (Entering)
                elif self.frames_out_of_zone_counter > 0 and self.is_feeding:
                    zone_color = (0, 165, 255)  # Orange (Leaving)

                cv2.circle(annotated_frame, zone_center, zone_radius, zone_color, 2)

            # Draw only the filtered, valid markers
            if filtered_ids_for_drawing:
                # Fix for cv2.error: 'ids' is not a numpy array
                # We convert our list of arrays into a single Nx1 numpy array
                ids_to_draw = np.array(filtered_ids_for_drawing, dtype=np.int32).reshape(-1, 1)
                aruco.drawDetectedMarkers(annotated_frame, filtered_corners_for_drawing, ids_to_draw)

            # --- 5. Recording ---
            if self.is_recording and self.video_writer:
                try:
                    # Resize if necessary
                    if output_res:
                        record_frame = cv2.resize(original_frame, output_res, interpolation=cv2.INTER_AREA)
                    else:
                        record_frame = original_frame
        
                    self.video_writer.write(record_frame)
                except Exception as e:
                    self.status_updated.emit(f"Video write error: {e}")
                    self._log_to_file(f"Video write error: {e}")

                    self.toggle_recording()  # Stop recording if it fails

            # --- 6. Emit Frame for GUI ---
            # Convert from BGR (OpenCV) to RGB (Qt)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(rgb_frame)

            # Modest delay to prevent 100% CPU and allow GUI to respond
            time.sleep(0.01)

        # --- End of Loop ---
        self.status_updated.emit("Processing loop stopped.")
        self._log_to_file("Processing loop stopped.")

