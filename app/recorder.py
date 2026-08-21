"""Session recorder — saves RGB video + depth video + decision log synchronized.

Directory layout for each session:
    recordings/
      session_YYYYMMDD_HHMMSS/
        rgb.mp4              (or .avi if MP4 codec unavailable)
        depth.mp4
        telemetry.csv        (100ms samples of all drone state)
        events.csv           (avoidance state transitions + VLM decisions)
        session_info.json    (metadata — start time, advisor mode, camera etc.)

Usage:
    rec = SessionRecorder(base_dir="recordings")
    rec.start(metadata={"advisor_mode": "gpt", ...})
    rec.write_rgb(rgb_frame)
    rec.write_depth(depth_colored)
    rec.log_telemetry(telemetry_dict)
    rec.log_event("avoidance_state", {"state": "HOVERING", ...})
    rec.log_event("vlm_decision", {"direction": "upper_right", ...})
    rec.stop()
"""

import os
import json
import csv
import time
import threading
from datetime import datetime

import cv2
import numpy as np


class SessionRecorder:
    """Thread-safe session recorder for RGB video, depth video, and events."""

    DEFAULT_FPS = 15.0  # target FPS for saved videos (approx UI display rate)

    def __init__(self, base_dir="recordings"):
        self.base_dir = base_dir
        self._session_dir = None
        self._start_time = None

        # Video writers
        self._rgb_writer = None
        self._depth_writer = None
        self._rgb_size = None
        self._depth_size = None
        self._fps = self.DEFAULT_FPS

        # CSV handles
        self._telemetry_file = None
        self._telemetry_writer = None
        self._events_file = None
        self._events_writer = None

        self._lock = threading.Lock()
        self._active = False

    @property
    def is_recording(self):
        return self._active

    @property
    def session_dir(self):
        return self._session_dir

    # ── Start / Stop ───────────────────────────────────────

    def start(self, metadata=None, fps=None):
        """Create session directory and open all log files + video writers."""
        with self._lock:
            if self._active:
                print("[RECORDER] Already recording — ignoring start request")
                return self._session_dir

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._session_dir = os.path.join(self.base_dir, f"session_{ts}")
            os.makedirs(self._session_dir, exist_ok=True)
            self._start_time = time.time()
            self._fps = fps or self.DEFAULT_FPS

            # Videos opened lazily on first write (need frame size)
            self._rgb_writer = None
            self._depth_writer = None
            self._rgb_size = None
            self._depth_size = None

            # Telemetry CSV — sampled drone state
            telem_path = os.path.join(self._session_dir, "telemetry.csv")
            self._telemetry_file = open(telem_path, "w", newline="")
            self._telemetry_writer = csv.DictWriter(
                self._telemetry_file,
                fieldnames=[
                    "timestamp", "t_elapsed",
                    "mode", "armed",
                    "altitude", "altitude_abs",
                    "lat", "lon",
                    "roll_deg", "pitch_deg", "yaw_deg",
                    "heading", "groundspeed", "airspeed",
                    "battery_v", "battery_pct",
                    "gps_fix", "gps_sats",
                    "mission_current_wp",
                    "fwd_dist_m", "left_dist_m", "right_dist_m",
                    "up_dist_m", "down_dist_m",
                    "upper_left_m", "upper_right_m",
                    "lower_left_m", "lower_right_m",
                    "avoidance_state", "avoidance_action",
                ],
            )
            self._telemetry_writer.writeheader()

            # Events CSV — state transitions and VLM decisions
            evt_path = os.path.join(self._session_dir, "events.csv")
            self._events_file = open(evt_path, "w", newline="")
            self._events_writer = csv.DictWriter(
                self._events_file,
                fieldnames=[
                    "timestamp", "t_elapsed", "event_type",
                    "state", "action", "direction",
                    "vx", "vy", "vz", "duration_s",
                    "fwd_dist_m", "left_dist_m", "right_dist_m",
                    "altitude_m", "advisor_mode",
                    "reasoning", "raw_json",
                ],
            )
            self._events_writer.writeheader()

            # Metadata
            meta = {
                "session_id": f"session_{ts}",
                "start_time": ts,
                "start_unix": self._start_time,
                "target_fps": self._fps,
            }
            if metadata:
                meta.update(metadata)
            with open(os.path.join(self._session_dir, "session_info.json"), "w") as f:
                json.dump(meta, f, indent=2)

            self._active = True
            print(f"[RECORDER] Started session: {self._session_dir}")
            return self._session_dir

    def stop(self):
        """Close all writers and finalize the session."""
        with self._lock:
            if not self._active:
                return
            if self._rgb_writer is not None:
                self._rgb_writer.release()
                self._rgb_writer = None
            if self._depth_writer is not None:
                self._depth_writer.release()
                self._depth_writer = None
            if self._telemetry_file is not None:
                self._telemetry_file.close()
                self._telemetry_file = None
                self._telemetry_writer = None
            if self._events_file is not None:
                self._events_file.close()
                self._events_file = None
                self._events_writer = None

            # Append end_time to session_info.json
            info_path = os.path.join(self._session_dir, "session_info.json")
            try:
                with open(info_path, "r") as f:
                    meta = json.load(f)
                meta["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
                meta["duration_s"] = round(time.time() - self._start_time, 2)
                with open(info_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception as e:
                print(f"[RECORDER] Warning: could not finalize metadata: {e}")

            print(f"[RECORDER] Stopped. Saved to: {self._session_dir}")
            self._active = False

    # ── Video writers ──────────────────────────────────────

    def _open_video(self, path, size):
        """Open a VideoWriter — try MP4 first, fall back to AVI."""
        w, h = size
        # Try MP4 / H264 first
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self._fps, (w, h))
        if not writer.isOpened():
            # Fall back to AVI / XVID
            path = path.replace(".mp4", ".avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(path, fourcc, self._fps, (w, h))
        if not writer.isOpened():
            print(f"[RECORDER] Failed to open video writer for {path}")
            return None
        print(f"[RECORDER] Opened {path} ({w}x{h} @ {self._fps}fps)")
        return writer

    def write_rgb(self, rgb_frame):
        """Write an RGB frame (numpy H,W,3 uint8 RGB)."""
        if not self._active or rgb_frame is None:
            return
        with self._lock:
            if self._rgb_writer is None:
                h, w = rgb_frame.shape[:2]
                self._rgb_size = (w, h)
                path = os.path.join(self._session_dir, "rgb.mp4")
                self._rgb_writer = self._open_video(path, self._rgb_size)
            if self._rgb_writer is not None:
                # OpenCV expects BGR
                bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                self._rgb_writer.write(bgr)

    def write_depth(self, depth_colored):
        """Write a colorized depth frame (numpy H,W,3 uint8 RGB)."""
        if not self._active or depth_colored is None:
            return
        with self._lock:
            if self._depth_writer is None:
                h, w = depth_colored.shape[:2]
                self._depth_size = (w, h)
                path = os.path.join(self._session_dir, "depth.mp4")
                self._depth_writer = self._open_video(path, self._depth_size)
            if self._depth_writer is not None:
                bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
                self._depth_writer.write(bgr)

    # ── Logging ────────────────────────────────────────────

    def log_telemetry(self, telemetry, distances_9seg=None,
                      avoidance_state=None, avoidance_action=None):
        """Log one row of drone + depth state to telemetry.csv."""
        if not self._active or self._telemetry_writer is None:
            return
        import math
        d = distances_9seg or {}
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "t_elapsed": round(time.time() - self._start_time, 3),
            "mode": telemetry.get("mode", ""),
            "armed": telemetry.get("armed", False),
            "altitude": round(telemetry.get("altitude", 0.0), 3),
            "altitude_abs": round(telemetry.get("altitude_abs", 0.0), 3),
            "lat": telemetry.get("lat", 0.0),
            "lon": telemetry.get("lon", 0.0),
            "roll_deg": round(math.degrees(telemetry.get("roll", 0.0)), 2),
            "pitch_deg": round(math.degrees(telemetry.get("pitch", 0.0)), 2),
            "yaw_deg": round(math.degrees(telemetry.get("yaw", 0.0)), 2),
            "heading": telemetry.get("heading", 0),
            "groundspeed": round(telemetry.get("groundspeed", 0.0), 2),
            "airspeed": round(telemetry.get("airspeed", 0.0), 2),
            "battery_v": round(telemetry.get("battery_voltage", 0.0), 2),
            "battery_pct": telemetry.get("battery_remaining", -1),
            "gps_fix": telemetry.get("gps_fix", 0),
            "gps_sats": telemetry.get("gps_satellites", 0),
            "mission_current_wp": telemetry.get("mission_current_wp", 0),
            "fwd_dist_m": _safe_round(d.get("center")),
            "left_dist_m": _safe_round(d.get("left")),
            "right_dist_m": _safe_round(d.get("right")),
            "up_dist_m": _safe_round(d.get("up")),
            "down_dist_m": _safe_round(d.get("down")),
            "upper_left_m": _safe_round(d.get("upper_left")),
            "upper_right_m": _safe_round(d.get("upper_right")),
            "lower_left_m": _safe_round(d.get("lower_left")),
            "lower_right_m": _safe_round(d.get("lower_right")),
            "avoidance_state": avoidance_state or "",
            "avoidance_action": avoidance_action or "",
        }
        with self._lock:
            if self._telemetry_writer is not None:
                self._telemetry_writer.writerow(row)
                self._telemetry_file.flush()

    def log_event(self, event_type, data):
        """Log a discrete event (state transition, VLM decision, error, etc.).

        event_type examples:
            "avoidance_state"   - state machine transition
            "vlm_request"       - VLM advisor queried
            "vlm_decision"      - VLM returned a direction
            "vlm_error"         - VLM call failed
            "manual_command"    - user pressed a manual control
            "mode_change"       - drone flight mode changed
            "recording_note"    - arbitrary annotation
        """
        if not self._active or self._events_writer is None:
            return
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "t_elapsed": round(time.time() - self._start_time, 3),
            "event_type": event_type,
            "state": data.get("state", ""),
            "action": data.get("action", ""),
            "direction": data.get("direction", ""),
            "vx": data.get("vx", ""),
            "vy": data.get("vy", ""),
            "vz": data.get("vz", ""),
            "duration_s": data.get("duration", ""),
            "fwd_dist_m": data.get("fwd_dist_m", ""),
            "left_dist_m": data.get("left_dist_m", ""),
            "right_dist_m": data.get("right_dist_m", ""),
            "altitude_m": data.get("altitude_m", ""),
            "advisor_mode": data.get("advisor_mode", ""),
            "reasoning": data.get("reasoning", ""),
            "raw_json": json.dumps(data.get("raw", ""))[:500] if data.get("raw") else "",
        }
        with self._lock:
            if self._events_writer is not None:
                self._events_writer.writerow(row)
                self._events_file.flush()


def _safe_round(v, ndigits=3):
    """Round a number if it's a number, else return empty string."""
    if v is None:
        return ""
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return ""
