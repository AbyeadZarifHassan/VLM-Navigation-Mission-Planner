"""MAVLink communication module for ArduPilot drones."""

import time
import threading
from pymavlink import mavutil
import math 


class MAVLinkComm:
    """Handles MAVLink connection, telemetry, and command sending."""

    def __init__(self):
        self.connection = None
        self.connected = False
        self._lock = threading.Lock()
        self.target_system = 1
        self.target_component = 1
        self._telemetry = {
            "altitude": 0.0,
            "groundspeed": 0.0,
            "airspeed": 0.0,
            "heading": 0,
            "battery_voltage": 0.0,
            "battery_remaining": -1,
            "gps_fix": 0,
            "gps_satellites": 0,
            "lat": 0.0,
            "lon": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "mode": "UNKNOWN",
            "armed": False,
            "motor_pwm": (0, 0, 0, 0),  # M1-M4 PWM values
            "mission_current_wp": 0,
            "mission_reached_wp": -1,
        }

    def connect(self, connection_string="udp:127.0.0.1:14550"):
        """Establish MAVLink connection."""
        try:
            self.connection = mavutil.mavlink_connection(connection_string)
            self.connection.wait_heartbeat(timeout=10)
            self.target_system = self.connection.target_system
            self.target_component = self.connection.target_component
            self._request_data_streams()
            self.connected = True
            return True
        except Exception as e:
            self.connected = False
            raise ConnectionError(f"Failed to connect: {e}")

    def _request_data_streams(self):
        """Ask ArduPilot to send all telemetry streams."""
        streams = [
            (mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS, 2),
            (mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 2),
            (mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 2),
            (mavutil.mavlink.MAV_DATA_STREAM_POSITION, 5),
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 10),
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA2, 5),
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA3, 2),
        ]
        for stream_id, rate in streams:
            self.connection.mav.request_data_stream_send(
                self.target_system,
                self.target_component,
                stream_id,
                rate,
                1  # start
            )

    def disconnect(self):
        """Close the MAVLink connection."""
        if self.connection:
            self.connection.close()
        self.connected = False

    def get_telemetry(self):
        """Poll and return latest telemetry data."""
        if not self.connected:
            return None

        with self._lock:
            # Read all available messages, updating persistent state
            while True:
                msg = self.connection.recv_match(blocking=False)
                if msg is None:
                    break
                msg_type = msg.get_type()

                if msg_type == "HEARTBEAT":
                    self._telemetry["mode"] = mavutil.mode_string_v10(msg)
                    self._telemetry["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

                elif msg_type == "GLOBAL_POSITION_INT":
                    self._telemetry["altitude"] = msg.relative_alt / 1000.0
                    self._telemetry["lat"] = msg.lat / 1e7
                    self._telemetry["lon"] = msg.lon / 1e7

                elif msg_type == "VFR_HUD":
                    self._telemetry["groundspeed"] = msg.groundspeed
                    self._telemetry["airspeed"] = msg.airspeed
                    self._telemetry["heading"] = msg.heading
                    self._telemetry["altitude"] = msg.alt

                elif msg_type == "SYS_STATUS":
                    self._telemetry["battery_voltage"] = msg.voltage_battery / 1000.0
                    self._telemetry["battery_remaining"] = msg.battery_remaining

                elif msg_type == "ATTITUDE":
                    self._telemetry["roll"] = msg.roll
                    self._telemetry["pitch"] = msg.pitch
                    self._telemetry["yaw"] = msg.yaw

                elif msg_type == "GPS_RAW_INT":
                    self._telemetry["gps_fix"] = msg.fix_type
                    self._telemetry["gps_satellites"] = msg.satellites_visible

                elif msg_type == "SERVO_OUTPUT_RAW":
                    # Motor PWM values for quad (servo1-4)
                    self._telemetry["motor_pwm"] = (
                        msg.servo1_raw, msg.servo2_raw,
                        msg.servo3_raw, msg.servo4_raw,
                    )

                elif msg_type == "MISSION_CURRENT":
                    self._telemetry["mission_current_wp"] = msg.seq

                elif msg_type == "MISSION_ITEM_REACHED":
                    self._telemetry["mission_reached_wp"] = msg.seq

        return dict(self._telemetry)

    def send_manual_control(self, x=0, y=0, z=500, r=0):
        """Send MANUAL_CONTROL message.

        Args:
            x: forward/backward (-1000 to 1000), positive = forward
            y: left/right (-1000 to 1000), positive = right
            z: throttle (0 to 1000), 500 = mid
            r: yaw (-1000 to 1000), positive = clockwise
        """
        if not self.connected:
            return
        with self._lock:
            self.connection.mav.manual_control_send(
                self.target_system,
                int(x), int(y), int(z), int(r),
                0  # buttons
            )

    def send_velocity(self, vx=0, vy=0, vz=0):
        if not self.connected:
            return

        with self._lock:
            self.connection.mav.set_position_target_local_ned_send(
                0,
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # ← ArduPilot নিজেই rotate করবে
                0b0000110111000111,
                0, 0, 0,
                vx, vy, vz,   # ← raw body frame, conversion নেই
                0, 0, 0,
                0, 0
            )

    def set_mode(self, mode_name):
        """Set flight mode by name (e.g., 'GUIDED', 'LOITER', 'STABILIZE')."""
        if not self.connected:
            return False
        mode_id = self.connection.mode_mapping().get(mode_name)
        if mode_id is None:
            return False
        with self._lock:
            self.connection.set_mode(mode_id)
        return True

    def arm(self):
        """Arm the drone."""
        if not self.connected:
            return
        with self._lock:
            self.connection.arducopter_arm()

    def disarm(self):
        """Disarm the drone."""
        if not self.connected:
            return
        with self._lock:
            self.connection.arducopter_disarm()

    def hover(self):
        """Command the drone to hover in place (zero velocity)."""
        self.send_velocity(0, 0, 0)

    def takeoff(self, altitude_m=3.0):
        """Command the drone to takeoff to target altitude (GUIDED mode only)."""
        if not self.connected:
            return False
        with self._lock:
            self.connection.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,  # confirmation
                0, 0, 0, 0,  # params 1-4 (unused)
                0, 0,  # lat/lon (0 = current)
                altitude_m  # target altitude
            )
        return True

    def is_airborne(self, min_altitude=0.5):
        """Return True if drone is above min_altitude (meters)."""
        return self._telemetry.get("altitude", 0.0) >= min_altitude

    # ── Mission Protocol ──────────────────────────────────

    def send_mission_count(self, count):
        """Initiate mission upload by announcing the total count."""
        if not self.connected:
            return
        with self._lock:
            self.connection.mav.mission_count_send(
                self.target_system,
                self.target_component,
                count,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )

    def send_mission_item_int(self, seq, command_id, lat, lon, alt,
                               p1=0, p2=0, p3=0, p4=0, current=0, autocontinue=1):
        """Send a single mission item (INT format, lat/lon scaled to 1e7)."""
        if not self.connected:
            return
        with self._lock:
            self.connection.mav.mission_item_int_send(
                self.target_system,
                self.target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                command_id,
                current,
                autocontinue,
                p1, p2, p3, p4,
                int(lat * 1e7),
                int(lon * 1e7),
                alt,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )

    def clear_all_missions(self):
        """Erase all waypoints from drone memory."""
        if not self.connected:
            return
        with self._lock:
            self.connection.mav.mission_clear_all_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )

    def mission_start(self):
        """Send MAV_CMD_MISSION_START to begin the uploaded mission."""
        if not self.connected:
            return
        with self._lock:
            self.connection.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_MISSION_START,
                0,  # confirmation
                0, 0, 0, 0, 0, 0, 0,  # params 1-7 unused
            )

    def request_mission_list(self):
        """Request the current mission from the drone (starts download)."""
        if not self.connected:
            return
        with self._lock:
            self.connection.mav.mission_request_list_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )

    def request_mission_item(self, seq):
        """Request a specific mission item by sequence number."""
        if not self.connected:
            return
        with self._lock:
            self.connection.mav.mission_request_int_send(
                self.target_system,
                self.target_component,
                seq,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )

    def send_mission_ack(self, ack_type=0):
        """Send MISSION_ACK (0 = MAV_MISSION_ACCEPTED)."""
        if not self.connected:
            return
        with self._lock:
            self.connection.mav.mission_ack_send(
                self.target_system,
                self.target_component,
                ack_type,
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
            )

    def set_mission_current(self, seq):
        """Jump to a specific waypoint (used to resume after avoidance)."""
        if not self.connected:
            return
        with self._lock:
            self.connection.mav.mission_set_current_send(
                self.target_system,
                self.target_component,
                seq,
            )

    def recv_mission_message(self, timeout=3.0):
        """Wait briefly for any mission-protocol message.

        Returns the message or None on timeout. Used by MissionThread during
        upload/download handshakes. Does not hold the lock while blocking.
        """
        if not self.connected:
            return None
        return self.connection.recv_match(
            type=[
                "MISSION_REQUEST_INT", "MISSION_REQUEST",
                "MISSION_ACK", "MISSION_COUNT", "MISSION_ITEM_INT",
            ],
            blocking=True,
            timeout=timeout,
        )
