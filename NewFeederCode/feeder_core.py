# feeder_core.py
# This file contains the core logic for the Feeder Controller.
# It runs in a separate thread and handles camera, detection, and Arduino comms.

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import serial
import threading
from PySide6.QtCore import QObject, QThread, Signal, Slot  # <-- FIX: Import Slot,
import queue
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"
os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"] = "video_codec;h264_nvenc"

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

class VideoRecorder(QThread):
    def __init__(self, filename, fps, resolution):
        super().__init__()
        # 1. We use 'mp4v' or 'H264' as a placeholder; 
        # The environment variable above forces it to nvenc
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 2. Hardware Acceleration parameters
        # 0 = Your RTX 3060
        params = [
            cv2.VIDEOWRITER_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY,
            cv2.VIDEOWRITER_PROP_HW_DEVICE, 0 
        ]
        self.writer = cv2.VideoWriter(filename, fourcc, fps, resolution, params)
        if not self.writer.isOpened():
            self.writer = cv2.VideoWriter(filename, fourcc, fps, resolution)
            print("Failed to record with GPU")
        # Final safety check
        if not self.writer.isOpened():
            print("CRITICAL: VideoWriter could not be opened even on CPU fallback.")
        
        self.frame_queue = queue.Queue(maxsize=128)
        self.running = True

    def add_frame(self, frame):
        if self.running:
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass 

    def run(self):
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                self.writer.write(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
        self.writer.release()

    def stop(self):
        self.running = False
        self.wait()

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
        self.should_recapture_bg = False
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
            "hysterisis_1": 60,
            "hysterisis_2": 60,
            "hysterisis_3": 60,
            "margin": 50,
            # Detection
            "allowed_ids": set([1, 2, 5, 6, 7, 8, 9, 10]),
            "min_size": 50,
            "max_size": 2000,
            "timeout_ms": 10000,  # Arduino timeout
            "entry_frames": 15,  # Python: consecutive frames to confirm entry
            "exit_frames": 30,  # Python: consecutive frames to confirm exit
            "trigger_mode": "both",
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
            self.cap = cv2.VideoCapture("test.mp4")#, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera {index}.")
            
            # 1. Force MJPEG Compression to unblock USB bandwidth
            #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            # 2. Set Resolution (720p is often better for hitting 60fps on webcams)
            #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            #self.cap.set(cv2.CAP_PROP_BACKLIGHT, 0)
            #self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

            # 3. Request High FPS
            self.cap.set(cv2.CAP_PROP_FPS, 60)

            # Apply initial camera settings
            #self.set_exposure_lock(self.settings['exposure_lock'])
            #self.set_autofocus(self.settings['autofocus'])
            #if not self.settings['autofocus']:
            #    self.set_focus(self.settings['focus'])

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
        self._send_to_arduino(f"G{settings['hysterisis_1']}\n")
        self._send_to_arduino(f"H{settings['hysterisis_2']}\n")
        self._send_to_arduino(f"J{settings['hysterisis_3']}\n")
        self._send_to_arduino(f"M{settings['timeout_ms']}\n")
        self._send_to_arduino(f"X{settings['margin']}\n")
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

                #Detection type
                self.settings["trigger_mode"] = settings.get("trigger_mode", "both")

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
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30

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

            self.video_recorder = VideoRecorder(filename, fps, (out_w, out_h))
            self.video_recorder.start()

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
            # 0.25 = Manual Exposure, 0.75 = Auto Exposure
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
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


#########################################################################################################################################
#                                                               MAIN PROCESSING LOOP                                                    #
#########################################################################################################################################
    def _process_loop(self):
        prev_time = time.perf_counter()
        last_sec_log_time = 0
        """
        (Processing Thread) This is the main loop that reads frames,
        performs detection, and sends signals.
        """
        first_blur = None

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
                time.sleep(0.01)
                continue

            # --- 1. Calculate FPS ---
            curr_time = time.perf_counter()
            delta_time = curr_time - prev_time
            fps = 1 / delta_time if delta_time > 0 else 0
            prev_time = curr_time

            # --- 2. Setup & Global Timestamp ---
            timestamp_str = time.strftime("%H:%M:%S")
            font = cv2.FONT_HERSHEY_SIMPLEX
            h, w = frame.shape[:2]

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.0
            thickness = 2

            # Calculate text size
            (label_width, label_height), baseline = cv2.getTextSize(timestamp_str, font, font_scale, thickness)
            x_pos = w - label_width - 15
            y_pos = h - 15

            original_frame = frame.copy()   
            annotated_frame = frame.copy()
            # Get current settings in a thread-safe way
            with self.settings_lock:
                # ArUco/Zone settings (Untouched)
                allowed_ids = self.settings["allowed_ids"]
                min_size = self.settings["min_size"]
                max_size = self.settings["max_size"]
                zone_center = self.settings["zone_center"]
                zone_radius = self.settings["zone_radius"]
                entry_frames_target = self.settings["entry_frames"]
                exit_frames_target = self.settings["exit_frames"]
                output_res = self.settings["output_resolution"]
                
                b_size = self.settings.get("blur_size", 7) | 1
                t_val = self.settings.get("threshold", 15)
                op_k = max(1, self.settings.get("open_kernel", 2))
                op_p = self.settings.get("open_power", 1)
                cl_k = max(1, self.settings.get("close_kernel", 2))
                cl_p = self.settings.get("close_power", 2)
                box_w = self.settings.get("box_width", 200)
                box_h = self.settings.get("box_height", 200)
                min_area = self.settings.get("min_area", 500)
                trigger_mode = self.settings.get("trigger_mode", "both")
                rx, ry, rw, rh = self.settings["roi_x"], self.settings["roi_y"], self.settings["roi_w"], self.settings["roi_h"]
                sec_center = self.settings["sec_zone_center"]
                sec_radius = self.settings["sec_zone_radius"]
                diff_limit = self.settings["diff_intensity"]

            cv2.rectangle(annotated_frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (b_size, b_size), 0)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            mask[ry:ry+rh, rx:rx+rw] = 255

            if first_blur is None or self.should_recapture_bg:
                first_blur = blur.copy()
                self.should_recapture_bg = False  # Reset the flag
                self.status_updated.emit("Background updated.")
                self._log_to_file("Background reference frame recaptured.")
                continue

            diff = cv2.absdiff(first_blur, blur)
            diff = cv2.bitwise_and(diff, diff, mask=mask) # Only look at ROI

            if sec_radius > 0:
                # Create a mask for the secondary circle
                sec_mask = np.zeros(diff.shape, dtype=np.uint8)
                cv2.circle(sec_mask, sec_center, sec_radius, 255, -1)       
                # Calculate mean difference intensity within that circle
                mean_val = cv2.mean(diff, mask=sec_mask)[0]
                # Draw the secondary zone (Blue for idle, Cyan for "Possible Feeding")
                current_time = time.perf_counter()
                if mean_val > diff_limit and (current_time - last_sec_log_time) >= 0.5:
                    self.status_updated.emit(f"Possible feeding (Intensity: {mean_val:.1f})")
                    self._log_to_file(f"Possible feeding detected at secondary zone: {mean_val:.1f}")
                    last_sec_log_time = current_time

                sec_color = (255, 255, 0) # Cyan
                if mean_val > diff_limit:
                    sec_color = (255, 0, 0) # Blue (Alert/Feeding)
                cv2.circle(annotated_frame, sec_center, sec_radius, sec_color, 1)


            _, thresh = cv2.threshold(diff, t_val, 255, cv2.THRESH_BINARY)

            kernel_open = np.ones((op_k, op_k), np.uint8)
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=op_p)
            kernel_close = np.ones((cl_k, cl_k), np.uint8)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=cl_p)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # --- 3. Bee Detection & Tracking Logic ---
            is_frame_triggered = False
            current_tag_id = "None"
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue

                # A. Calculate Moments for Centroid and Orientation
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                motion_in_zone = self._is_tag_in_zone((cX, cY), zone_center, zone_radius)
                
                # Second moment of inertia for direction
                mu11 = M['mu11']
                mu20 = M['mu20']
                mu02 = M['mu02']
                angle = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))
                
                # Draw Centroid and Direction Line
                cv2.circle(annotated_frame, (cX, cY), 5, (255, 0, 0), -1)
                line_len = 40
                p2 = (int(cX + line_len * np.cos(angle)), int(cY + line_len * np.sin(angle)))
                cv2.line(annotated_frame, (cX, cY), p2, (255, 255, 0), 2)
                cv2.drawContours(annotated_frame, [cnt], -1, (255, 255, 0), 1)

                # B. Targeted ArUco Detection in Crop
                aruco_in_zone = False
                y1, y2 = np.clip([cY - box_h//2, cY + box_h//2], 0, h)
                x1, x2 = np.clip([cX - box_w//2, cX + box_w//2], 0, w)
                
                if y2 > y1 and x2 > x1:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    bee_crop_gray = gray[y1:y2, x1:x2]
                    
                    bee_crop_norm = cv2.normalize(bee_crop_gray, None, 0, 255, cv2.NORM_MINMAX)
                    scale_factor = 1.5
                    zoom_for_aruco = cv2.resize(bee_crop_norm, None, fx=scale_factor, fy=scale_factor, 
                                                interpolation=cv2.INTER_LANCZOS4)                 
                    # 3. ArUco Parameters with proportional Rates
                    try:
                        params = cv2.aruco.DetectorParameters()
                    except AttributeError:
                        params = cv2.aruco.DetectorParameters_create()

                    # Using the zoom_for_aruco dimensions for the rate calculation
                    z_h, z_w = zoom_for_aruco.shape[:2]
                    max_dim = max(z_h, z_w)
                    
                    params.minMarkerPerimeterRate = min_size / max_dim if max_dim > 0 else 0.03
                    params.maxMarkerPerimeterRate = max_size / max_dim if max_dim > 0 else 4.0

                    c_crop, ids, _ = cv2.aruco.detectMarkers(zoom_for_aruco, self.aruco_dict, parameters=params)


                    if ids is not None:
                        for i, tag_id in enumerate(ids.flatten()):
                            if tag_id in allowed_ids:
                                perimeter = cv2.arcLength(c_crop[i], True)
                                if min_size < perimeter < max_size:
                                    # Offset corners back to global frame
                                    global_corners = (c_crop[i]/scale_factor) + np.array([x1, y1])
                                    current_tag_id = str(tag_id)
                                    # Calculate Tag Center for ArUco-mode triggering
                                    t_cX = int(global_corners[0][:, 0].mean())
                                    t_cY = int(global_corners[0][:, 1].mean())
                                    # Check Zone logic using the specific tag in this crop
                                    if self._is_tag_in_zone((t_cX, t_cY), zone_center, zone_radius):
                                        aruco_in_zone = True
                                    cv2.aruco.drawDetectedMarkers(annotated_frame, [global_corners], np.array([[tag_id]]))    
                # --- EVALUATE TRIGGER BASED ON MODE ---
                if trigger_mode == "motion":
                    if motion_in_zone: is_frame_triggered = True
                elif trigger_mode == "aruco":
                    if aruco_in_zone: is_frame_triggered = True
                elif trigger_mode == "both":
                    if motion_in_zone and aruco_in_zone: is_frame_triggered = True
            if is_frame_triggered:
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

            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (w-150, h-50), font, 0.7, (0, 255, 0), 2)

            # --- 5. Recording ---
            if self.is_recording and self.video_recorder:
                try:
                    # Resize if necessary
                    if output_res:
                        record_frame = cv2.resize(original_frame, output_res, interpolation=cv2.INTER_AREA)
                    else:
                        record_frame = original_frame
        
                    self.video_recorder.add_frame(record_frame)
                except Exception as e:
                    self.status_updated.emit(f"Video write error: {e}")
                    self._log_to_file(f"Video write error: {e}")
                    self.toggle_recording()  # Stop recording if it fails

            # --- 6. Emit Frame for GUI ---
            # Convert from BGR (OpenCV) to RGB (Qt)

            cv2.putText(annotated_frame, timestamp_str, (x_pos+1, y_pos+1), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(annotated_frame, timestamp_str, (x_pos, y_pos), font, font_scale, (255, 255, 255), thickness)
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(rgb_frame)

            # Modest delay to prevent 100% CPU and allow GUI to respond
            time.sleep(0.001)

        # --- End of Loop ---
        self.status_updated.emit("Processing loop stopped.")
        self._log_to_file("Processing loop stopped.")