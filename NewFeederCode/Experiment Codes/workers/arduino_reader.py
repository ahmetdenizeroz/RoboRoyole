from PySide6.QtCore import QThread, Signal

class ArduinoReader(QThread):
    message_received = Signal(str)

    def __init__(self, arduino_serial):
        super().__init__()
        self.arduino = arduino_serial
        self.buffer = b""

    def run(self):
        while True:
            if not self.arduino or not self.arduino.is_open:
                break

            try:
                waiting = self.arduino.in_waiting
            except Exception:
                break

            if waiting > 0:
                try:
                    data = self.arduino.read(waiting)
                except Exception:
                    break

                self.buffer += data
                lines = self.buffer.split(b"\n")
                self.buffer = lines[-1]

                for line in lines[:-1]:
                    if line.strip():
                        msg = line.decode("ascii", errors="ignore").strip()
                        self.message_received.emit(f"[Arduino]: {msg}")

            self.msleep(50)

    def stop(self):
        self.quit()
        self.wait(500)