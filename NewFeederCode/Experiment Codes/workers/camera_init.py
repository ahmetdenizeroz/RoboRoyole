import os
import cv2
from PySide6.QtCore import QThread, Signal


class CameraInitThread(QThread):
    """Opens either a camera (int index) or a video file path (str)."""
    finished = Signal(object, bool, str)

    def __init__(self, source, fps):
        super().__init__()
        self.source = source   # int camera index OR str video path
        self.fps = fps

    def run(self):
        try:
            # -----------------------------
            # 1) Open source
            # -----------------------------
            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                source_type = "camera"

            elif isinstance(self.source, str):
                if not os.path.exists(self.source):
                    self.finished.emit(None, False, f"Video file not found: {self.source}")
                    return

                cap = cv2.VideoCapture(self.source)
                source_type = "video"

            else:
                self.finished.emit(
                    None,
                    False,
                    "Source must be either an integer camera index or a video file path string.",
                )
                return

            if not cap.isOpened():
                self.finished.emit(None, False, f"Failed to open {source_type}: {self.source}")
                return

            # -----------------------------
            # 2) Apply settings only if camera
            # -----------------------------
            if source_type == "camera":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                cap.set(cv2.CAP_PROP_BACKLIGHT, 0)

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                msg = f"Camera Linked: {w}x{h}"

            else:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                file_fps = cap.get(cv2.CAP_PROP_FPS)
                msg = f"Video Linked: {os.path.basename(self.source)} | {w}x{h} | {file_fps:.2f} FPS"

            self.finished.emit(cap, True, msg)

        except Exception as e:
            self.finished.emit(None, False, f"Thread Error: {str(e)}")