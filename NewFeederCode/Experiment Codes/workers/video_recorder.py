import queue
import subprocess
from PySide6.QtCore import QThread


class VideoRecorder(QThread):
    def __init__(self, filename, fps, resolution):
        super().__init__()
        self.filename = filename
        self.fps = fps
        self.resolution = resolution
        self.frame_queue = queue.Queue(maxsize=512)
        self.running = True
        self.proc = None

    def add_frame(self, frame):
        if self.running:
            try:
                self.frame_queue.put_nowait(frame)
                return True
            except queue.Full:
                print("A frame is dropped")
                return False
        return False

    def run(self):
        w, h = self.resolution

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-b:v", "3M",
            "-maxrate", "3M",
            "-bufsize", "6M",
            self.filename,
        ]

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"Failed to start ffmpeg writer: {e}")
            self.running = False
            return

        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)

                if self.proc is not None and self.proc.stdin is not None and frame is not None:
                    self.proc.stdin.write(frame.tobytes())

                self.frame_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"FFmpeg write error: {e}")
                break

        try:
            if self.proc and self.proc.stdin:
                self.proc.stdin.close()
            if self.proc:
                self.proc.wait(timeout=5)
        except Exception:
            try:
                if self.proc:
                    self.proc.kill()
            except Exception:
                pass

        print("Video file released.")

    def stop(self):
        self.running = False
        self.wait()