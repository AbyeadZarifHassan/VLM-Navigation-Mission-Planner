"""QThread for polling MAVLink telemetry."""

from PyQt5.QtCore import QThread, pyqtSignal


class DroneThread(QThread):
    """Polls MAVLink connection for telemetry data."""

    telemetry_updated = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)  # connected, message

    def __init__(self, mavlink_comm, parent=None):
        super().__init__(parent)
        self.mavlink = mavlink_comm
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            if not self.mavlink.connected:
                self.msleep(500)
                continue
            try:
                data = self.mavlink.get_telemetry()
                if data:
                    self.telemetry_updated.emit(data)
            except Exception as e:
                self.connection_status.emit(False, str(e))
            self.msleep(50)  # ~20 Hz

    def stop(self):
        self._running = False
        self.wait(2000)
