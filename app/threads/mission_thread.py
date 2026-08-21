"""QThread for mission upload/download/start via MAVLink."""

import time
from PyQt5.QtCore import QThread, pyqtSignal

from app.drone.mission import Waypoint, COMMAND_ID_TO_NAME


class MissionThread(QThread):
    """Handles long-running mission protocol operations off the UI thread."""

    upload_progress = pyqtSignal(int, int)  # current, total
    upload_complete = pyqtSignal()
    upload_error = pyqtSignal(str)

    download_complete = pyqtSignal(list)  # list[Waypoint]
    download_error = pyqtSignal(str)

    mission_status = pyqtSignal(str)  # free-form log text
    clear_complete = pyqtSignal()
    clear_error = pyqtSignal(str)

    def __init__(self, mavlink_comm, parent=None):
        super().__init__(parent)
        self.mavlink = mavlink_comm
        self._running = False
        self._pending_op = None  # ("upload", waypoints) | ("download",) | ("clear",)

    # ── Public API (called from UI thread) ────────────────

    def request_upload(self, waypoints):
        self._pending_op = ("upload", waypoints)

    def request_download(self):
        self._pending_op = ("download",)

    def request_clear(self):
        self._pending_op = ("clear",)

    # ── Thread loop ───────────────────────────────────────

    def run(self):
        self._running = True
        while self._running:
            op = self._pending_op
            if op is None:
                self.msleep(100)
                continue
            self._pending_op = None

            try:
                if op[0] == "upload":
                    self._do_upload(op[1])
                elif op[0] == "download":
                    self._do_download()
                elif op[0] == "clear":
                    self._do_clear()
            except Exception as e:
                self.mission_status.emit(f"ERROR: {e}")

    def stop(self):
        self._running = False
        self.wait(2000)

    # ── Upload handshake ──────────────────────────────────

    def _do_upload(self, waypoints):
        if not self.mavlink.connected:
            self.upload_error.emit("Not connected to drone")
            return

        n = len(waypoints)
        if n == 0:
            self.upload_error.emit("No waypoints to upload")
            return

        self.mission_status.emit(f"[MISSION] Starting upload of {n} waypoints...")
        print(f"[MISSION] Uploading {n} waypoints")

        # Step 1: announce count
        self.mavlink.send_mission_count(n)

        # Step 2: answer MISSION_REQUEST_INT for each item
        sent_count = 0
        start_time = time.time()
        timeout_total = 30.0  # overall timeout

        while sent_count < n:
            if time.time() - start_time > timeout_total:
                self.upload_error.emit(
                    f"Upload timeout after {sent_count}/{n} items"
                )
                return

            msg = self.mavlink.recv_mission_message(timeout=5.0)
            if msg is None:
                self.upload_error.emit(f"No response at item {sent_count}/{n}")
                return

            msg_type = msg.get_type()

            if msg_type in ("MISSION_REQUEST_INT", "MISSION_REQUEST"):
                seq = msg.seq
                if seq >= n:
                    self.upload_error.emit(f"Drone requested out-of-range seq={seq}")
                    return
                wp = waypoints[seq]
                self.mavlink.send_mission_item_int(
                    seq=seq,
                    command_id=wp.command_id,
                    lat=wp.lat, lon=wp.lon, alt=wp.alt,
                    p1=wp.p1, p2=wp.p2, p3=wp.p3, p4=wp.p4,
                    current=(1 if seq == 0 else 0),
                )
                sent_count = seq + 1
                self.upload_progress.emit(sent_count, n)
                print(f"[MISSION] Sent wp {seq}: {wp.command} lat={wp.lat} lon={wp.lon} alt={wp.alt}")

            elif msg_type == "MISSION_ACK":
                if msg.type == 0:  # MAV_MISSION_ACCEPTED
                    self.mission_status.emit(
                        f"[MISSION] Upload complete ({sent_count}/{n} items accepted)"
                    )
                    self.upload_complete.emit()
                    return
                else:
                    self.upload_error.emit(f"Drone rejected mission (ACK type={msg.type})")
                    return

        # Wait for final ACK
        msg = self.mavlink.recv_mission_message(timeout=5.0)
        if msg and msg.get_type() == "MISSION_ACK" and msg.type == 0:
            self.mission_status.emit("[MISSION] Upload complete")
            self.upload_complete.emit()
        else:
            self.upload_error.emit("No MISSION_ACK received after final item")

    # ── Download handshake ────────────────────────────────

    def _do_download(self):
        if not self.mavlink.connected:
            self.download_error.emit("Not connected to drone")
            return

        self.mission_status.emit("[MISSION] Requesting mission from drone...")
        self.mavlink.request_mission_list()

        # Receive MISSION_COUNT
        msg = self.mavlink.recv_mission_message(timeout=5.0)
        if msg is None or msg.get_type() != "MISSION_COUNT":
            self.download_error.emit("No MISSION_COUNT received")
            return
        total = msg.count

        if total == 0:
            self.mission_status.emit("[MISSION] Drone has no waypoints")
            self.download_complete.emit([])
            return

        # Request each item
        waypoints = []
        for seq in range(total):
            self.mavlink.request_mission_item(seq)
            msg = self.mavlink.recv_mission_message(timeout=5.0)
            if msg is None or msg.get_type() != "MISSION_ITEM_INT":
                self.download_error.emit(f"Missing item {seq}/{total}")
                return
            cmd_name = COMMAND_ID_TO_NAME.get(msg.command, "NAV_WAYPOINT")
            wp = Waypoint(
                seq=msg.seq,
                command=cmd_name,
                lat=msg.x / 1e7,
                lon=msg.y / 1e7,
                alt=msg.z,
                p1=msg.param1, p2=msg.param2, p3=msg.param3, p4=msg.param4,
            )
            waypoints.append(wp)

        # Send final ACK
        self.mavlink.send_mission_ack(0)
        self.mission_status.emit(f"[MISSION] Downloaded {total} waypoints")
        self.download_complete.emit(waypoints)

    # ── Clear mission ─────────────────────────────────────

    def _do_clear(self):
        if not self.mavlink.connected:
            self.clear_error.emit("Not connected to drone")
            return
        self.mission_status.emit("[MISSION] Clearing drone mission memory...")
        self.mavlink.clear_all_missions()

        msg = self.mavlink.recv_mission_message(timeout=3.0)
        if msg and msg.get_type() == "MISSION_ACK" and msg.type == 0:
            self.mission_status.emit("[MISSION] Mission cleared")
            self.clear_complete.emit()
        else:
            self.clear_error.emit("No ACK received for clear")
