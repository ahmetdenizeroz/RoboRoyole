from PySide6.QtCore import QThread, Signal


class HardwareSerialReader(QThread):
    message_received = Signal(str)
    error_received = Signal(str)
    disconnected = Signal(str)

    def __init__(self, hardware_serial):
        super().__init__()
        self.hardware = hardware_serial
        self.buffer = b""
        self.running = True

    def run(self):
        reason = "SerialReader exited."

        while self.running:
            if not self.hardware or not self.hardware.is_open:
                reason = "SerialReader stopped: serial port is closed."
                break

            try:
                waiting = self.hardware.in_waiting
            except Exception as e:
                reason = f"SerialReader in_waiting error: {repr(e)}"
                self.error_received.emit(reason)
                break

            if waiting > 0:
                try:
                    data = self.hardware.read(waiting)
                except Exception as e:
                    reason = f"SerialReader read error: {repr(e)}"
                    self.error_received.emit(reason)
                    break

                self.buffer += data
                lines = self.buffer.split(b"\n")
                self.buffer = lines[-1]

                for line in lines[:-1]:
                    if line.strip():
                        msg = line.decode("ascii", errors="ignore").strip()
                        self.message_received.emit(f"[Hardware]: {msg}")

            self.msleep(50)

        self.disconnected.emit(reason)

    def stop(self):
        self.running = False
        self.wait(500)