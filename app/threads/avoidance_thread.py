"""QThread for obstacle avoidance state machine with vision-enabled VLM advisor.

Comprehensive logging for thesis analysis:
- vlm_request / vlm_decision / vlm_error with latency
- command_sent — actual MAVLink velocity pushed to drone (verifies VLM decision → execution)
- maneuver_outcome — did obstacle clear after each EXECUTING phase
- session_summary — per-episode aggregate stats (total maneuvers, time, attempts)
"""

import time
import threading
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from app.drone.avoidance import AvoidanceState, AvoidanceAction
from app.vision.gpt_advisor import GPTAdvisor
from app.vision.gemini_advisor import GeminiAdvisor


class AvoidanceThread(QThread):
    """Runs the obstacle avoidance state machine in a loop."""

    state_changed = pyqtSignal(str, str)  # state_name, action_name
    decision_made = pyqtSignal(str)  # action description for logging

    # GPT log signals
    gpt_sent = pyqtSignal(str)
    gpt_received = pyqtSignal(str)
    gpt_error = pyqtSignal(str)

    # Command display on video feed
    command_display = pyqtSignal(str)  # direction + velocity text for overlay

    def __init__(self, avoidance, mavlink_comm, parent=None):
        super().__init__(parent)
        self.avoidance = avoidance
        self.mavlink = mavlink_comm
        self._running = False
        self._enabled = False
        self._forward_distance = None
        self._last_state = None
        self._pre_avoidance_mode = None  # mode before avoidance took over

        # VLM advisors
        self._gpt_advisor = GPTAdvisor()
        self._gemini_advisor = GeminiAdvisor()
        self._gpt_pending = False  # True while VLM call is in-flight

        # Vision inputs (latest from camera_thread)
        self._latest_frame = None
        self._latest_depth_colored = None
        self._distances = None

        self._avoidance_mode = "auto"
        self._avoidance_in_mission = True
        self._current_flight_mode = "UNKNOWN"
        self._state_is_idle = True

        # Recorder (set externally from main_window)
        self._recorder = None

        # Logging state — tracks episode stats for summary
        self._episode_start_time = None       # when OBSTACLE_DETECTED fired
        self._episode_maneuver_count = 0      # how many EXECUTING phases this episode
        self._episode_vlm_calls = 0           # how many VLM queries this episode
        self._vlm_request_time = None         # for latency calculation
        self._last_executing_velocity = None  # track what was sent for outcome
        self._last_executing_start = None     # for duration measurement

    def set_recorder(self, recorder):
        """Attach a SessionRecorder for logging events."""
        self._recorder = recorder

    def set_enabled(self, enabled):
        self._enabled = enabled
        if not enabled:
            self._restore_mode()
            self.avoidance.reset()
            self._gpt_pending = False
            print("[AVOIDANCE] Disabled — state reset to IDLE")
        else:
            print(f"[AVOIDANCE] Enabled — mode={self._avoidance_mode}")

    def set_avoidance_mode(self, mode):
        self._avoidance_mode = mode
        print(f"[AVOIDANCE] Mode set to: {mode}")

    def set_avoidance_during_mission(self, enabled):
        self._avoidance_in_mission = enabled
        print(f"[AVOIDANCE] Avoidance during mission: {'ON' if enabled else 'OFF'}")

    def update_current_mode(self, mode):
        self._current_flight_mode = mode

    def _switch_to_guided(self):
        if not self.mavlink.connected:
            return
        current_mode = self._current_flight_mode or "UNKNOWN"
        if current_mode != "GUIDED":
            self._pre_avoidance_mode = current_mode
        print(f"[AVOIDANCE] Mode: {current_mode} -> GUIDED (forcing switch...)")

        for attempt in range(5):
            self.mavlink.set_mode("GUIDED")
            self.mavlink.hover()
            time.sleep(0.2)
            telem = self.mavlink.get_telemetry() or {}
            if telem.get("mode") == "GUIDED":
                print(f"[AVOIDANCE] GUIDED confirmed after {attempt+1} attempt(s)")
                return

        print(f"[AVOIDANCE] WARNING: GUIDED mode not confirmed after 5 attempts")

    def _restore_mode(self):
        if not self._pre_avoidance_mode or not self.mavlink.connected:
            return
        prev_mode = self._pre_avoidance_mode
        print(f"[AVOIDANCE] Mode: GUIDED -> {prev_mode} (restoring...)")

        for attempt in range(5):
            self.mavlink.set_mode(prev_mode)
            time.sleep(0.2)
            telem = self.mavlink.get_telemetry() or {}
            if telem.get("mode") == prev_mode:
                print(f"[AVOIDANCE] {prev_mode} confirmed after {attempt+1} attempt(s)")
                break

        if prev_mode == "AUTO":
            time.sleep(0.3)
            self.mavlink.mission_start()
            print(f"[AVOIDANCE] Mission resumed (MISSION_START sent)")
        self._pre_avoidance_mode = None

    # ── Inputs from main_window ───────────────────────────

    def update_distance(self, distance):
        self._forward_distance = distance

    def update_altitude(self, altitude):
        self.avoidance.update_altitude(altitude)

    def update_frame(self, rgb_frame):
        if rgb_frame is not None:
            self._latest_frame = rgb_frame.copy()

    def update_depth_colored(self, depth_colored):
        if depth_colored is not None:
            self._latest_depth_colored = depth_colored.copy()

    def update_distances(self, distances):
        self._distances = distances

    # ── Logging helpers ───────────────────────────────────

    def _log(self, event_type, **kwargs):
        """Shortcut for logging an event with distance + altitude context."""
        if not self._recorder:
            return
        d = self._distances or {}
        defaults = {
            "fwd_dist_m": d.get("center") or 0,
            "left_dist_m": d.get("left") or 0,
            "right_dist_m": d.get("right") or 0,
            "altitude_m": self.avoidance._current_altitude,
            "advisor_mode": self._avoidance_mode,
        }
        defaults.update(kwargs)
        self._recorder.log_event(event_type, defaults)

    # ── VLM Workers (vision-enabled) ──────────────────────

    def _call_gpt(self, telemetry, frame, depth_colored, distances_9seg):
        """Worker — runs GPT-4o Vision call in a background thread."""
        request_start = time.time()
        try:
            center = (distances_9seg or {}).get("center") or 0
            L = (distances_9seg or {}).get("left") or 0
            R = (distances_9seg or {}).get("right") or 0
            prompt_summary = (
                f"Vision+Telemetry -> GPT-4o  |  "
                f"fwd={center:.2f}m L={L:.2f}m R={R:.2f}m  "
                f"alt={telemetry.get('altitude', 0):.1f}m  "
                f"hdg={telemetry.get('heading', 0)} deg"
            )
            self.gpt_sent.emit(prompt_summary)

            self._log("vlm_request", advisor_mode="gpt",
                      attempt_num=self._episode_vlm_calls + 1)

            cmd = self._gpt_advisor.ask(
                telemetry=telemetry,
                frame=frame,
                depth_colored=depth_colored,
                distances_9seg=distances_9seg,
                threshold=self.avoidance.DISTANCE_THRESHOLD,
            )

            latency_ms = round((time.time() - request_start) * 1000, 1)

            if cmd:
                vx, vy, vz = cmd["vx"], cmd["vy"], cmd["vz"]
                dur = cmd["duration"]
                direction = cmd["direction"]
                reasoning = cmd["reasoning"]

                self.avoidance.set_gpt_command(vx, vy, vz, dur)
                self._episode_vlm_calls += 1

                recv_msg = (
                    f"{direction} for {dur:.1f}s  "
                    f"vel=({vx:.2f}, {vy:.2f}, {vz:.2f})  "
                    f"| {reasoning} [{latency_ms:.0f}ms]"
                )
                self.gpt_received.emit(recv_msg)

                self._log("vlm_decision", advisor_mode="gpt",
                          direction=direction, vx=vx, vy=vy, vz=vz, duration=dur,
                          latency_ms=latency_ms, reasoning=reasoning,
                          attempt_num=self._episode_vlm_calls, raw=cmd)

                print(f"[GPT] OK {direction} vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} "
                      f"dur={dur:.1f}s | {reasoning} [{latency_ms:.0f}ms]")
            else:
                self.gpt_error.emit("GPT returned empty response — using fallback")
                print("[GPT] Empty response — fallback to smart auto")
                self._log("vlm_error", advisor_mode="gpt",
                          reasoning="empty response", latency_ms=latency_ms)

        except Exception as e:
            latency_ms = round((time.time() - request_start) * 1000, 1)
            error_msg = str(e)[:120]
            self.gpt_error.emit(error_msg)
            print(f"[GPT] ERROR: {error_msg} — fallback to smart auto")
            self._log("vlm_error", advisor_mode="gpt",
                      reasoning=str(e)[:200], latency_ms=latency_ms)

        finally:
            self._gpt_pending = False

    def _call_gemini(self, telemetry, frame, depth_colored, distances_9seg):
        """Worker — runs Gemini Vision call in a background thread."""
        request_start = time.time()
        try:
            center = (distances_9seg or {}).get("center") or 0
            L = (distances_9seg or {}).get("left") or 0
            R = (distances_9seg or {}).get("right") or 0
            prompt_summary = (
                f"Vision+Telemetry -> Gemini  |  "
                f"fwd={center:.2f}m L={L:.2f}m R={R:.2f}m  "
                f"alt={telemetry.get('altitude', 0):.1f}m  "
                f"hdg={telemetry.get('heading', 0)} deg"
            )
            self.gpt_sent.emit(prompt_summary)

            self._log("vlm_request", advisor_mode="gemini",
                      attempt_num=self._episode_vlm_calls + 1)

            cmd = self._gemini_advisor.ask(
                telemetry=telemetry,
                frame=frame,
                depth_colored=depth_colored,
                distances_9seg=distances_9seg,
                threshold=self.avoidance.DISTANCE_THRESHOLD,
            )

            latency_ms = round((time.time() - request_start) * 1000, 1)

            if cmd:
                vx, vy, vz = cmd["vx"], cmd["vy"], cmd["vz"]
                dur = cmd["duration"]
                direction = cmd["direction"]
                reasoning = cmd["reasoning"]

                self.avoidance.set_gpt_command(vx, vy, vz, dur)
                self._episode_vlm_calls += 1

                recv_msg = (
                    f"{direction} for {dur:.1f}s  "
                    f"vel=({vx:.2f}, {vy:.2f}, {vz:.2f})  "
                    f"| {reasoning} [{latency_ms:.0f}ms]"
                )
                self.gpt_received.emit(recv_msg)

                self._log("vlm_decision", advisor_mode="gemini",
                          direction=direction, vx=vx, vy=vy, vz=vz, duration=dur,
                          latency_ms=latency_ms, reasoning=reasoning,
                          attempt_num=self._episode_vlm_calls, raw=cmd)

                print(f"[GEMINI] OK {direction} vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} "
                      f"dur={dur:.1f}s | {reasoning} [{latency_ms:.0f}ms]")
            else:
                self.gpt_error.emit("Gemini returned empty response — using fallback")
                print("[GEMINI] Empty response — fallback to smart auto")
                self._log("vlm_error", advisor_mode="gemini",
                          reasoning="empty response", latency_ms=latency_ms)

        except Exception as e:
            latency_ms = round((time.time() - request_start) * 1000, 1)
            error_msg = str(e)[:120]
            self.gpt_error.emit(error_msg)
            print(f"[GEMINI] ERROR: {error_msg} — fallback to smart auto")
            self._log("vlm_error", advisor_mode="gemini",
                      reasoning=str(e)[:200], latency_ms=latency_ms)

        finally:
            self._gpt_pending = False

    def _trigger_smart_auto(self, label="AUTO"):
        dist = self._forward_distance or 0
        alt = self.avoidance._current_altitude
        attempt = self.avoidance._attempt_count
        dists = self._distances or {}
        action, vx, vy, vz, dur = self.avoidance.decide_action_smart(dist, dists)
        self.avoidance.current_action = action
        self.avoidance.set_gpt_command(vx, vy, vz, dur)

        # Log auto decision too (for comparison with VLM decisions in analysis)
        self._log("vlm_decision", advisor_mode="auto",
                  direction=action.value, vx=vx, vy=vy, vz=vz, duration=dur,
                  latency_ms=0, reasoning="deterministic smart-auto fallback",
                  attempt_num=attempt + 1)

        L = dists.get("left") or 0
        R = dists.get("right") or 0
        print(
            f"[AVOIDANCE] {label}: {action.value} "
            f"vel=({vx:.2f},{vy:.2f},{vz:.2f}) dur={dur:.1f}s "
            f"| fwd={dist:.2f}m L={L:.2f}m R={R:.2f}m alt={alt:.1f}m attempt={attempt}"
        )

    # ── Main Loop ─────────────────────────────────────────

    def run(self):
        self._running = True
        last_mode_check = 0.0
        while self._running:
            if not self._enabled or not self.mavlink.connected:
                self.msleep(100)
                continue

            if (self._current_flight_mode == "AUTO"
                    and not self._avoidance_in_mission
                    and self.avoidance.state == AvoidanceState.IDLE):
                self.msleep(100)
                continue

            state, action, velocity = self.avoidance.update(self._forward_distance)

            # Periodically verify GUIDED mode during avoidance
            if state not in (AvoidanceState.IDLE,) and self._pre_avoidance_mode is not None:
                now_ts = time.time()
                if now_ts - last_mode_check > 0.5:
                    last_mode_check = now_ts
                    if self._current_flight_mode != "GUIDED":
                        self.mavlink.set_mode("GUIDED")
                        self.mavlink.hover()
                        print(f"[AVOIDANCE] Mode drift: {self._current_flight_mode} -> re-forcing GUIDED + hover")

            state_str = state.value
            action_str = action.value
            self._state_is_idle = (state == AvoidanceState.IDLE)

            if state != self._last_state:
                self.state_changed.emit(state_str, action_str)
                dist_str = f"{self._forward_distance:.2f}m" if self._forward_distance else "N/A"
                timestamp = time.strftime("%H:%M:%S")

                # Always log state transition
                self._log("avoidance_state", state=state_str, action=action_str)

                if state == AvoidanceState.OBSTACLE_DETECTED:
                    # Start of a new avoidance episode — reset episode counters
                    self._episode_start_time = time.time()
                    self._episode_maneuver_count = 0
                    self._episode_vlm_calls = 0

                    self._switch_to_guided()
                    msg = f"[{timestamp}] Obstacle detected at {dist_str} — stopping!"
                    self.decision_made.emit(msg)
                    print(f"[AVOIDANCE] *** OBSTACLE DETECTED *** dist={dist_str} threshold={self.avoidance.DISTANCE_THRESHOLD}m")

                elif state == AvoidanceState.HOVERING:
                    print(f"[AVOIDANCE] HOVERING -- stabilizing for {self.avoidance.HOVER_STABILIZE_TIME}s, drone must stop completely")
                    if self._avoidance_mode in ("gpt", "gemini"):
                        advisor_name = "GPT-4o" if self._avoidance_mode == "gpt" else "Gemini"

                        if self._latest_frame is None or self._latest_depth_colored is None:
                            print(f"[AVOIDANCE] No frame/depth yet — falling back to smart auto for {advisor_name}")
                            self._trigger_smart_auto(label="AUTO (no-vision fallback)")
                        elif not self._gpt_pending:
                            print(f"[AVOIDANCE] -> querying {advisor_name} with vision (waiting for response)...")
                            self._gpt_pending = True
                            telemetry = self.mavlink.get_telemetry() or {}
                            target = self._call_gpt if self._avoidance_mode == "gpt" else self._call_gemini
                            frame_snap = self._latest_frame
                            depth_snap = self._latest_depth_colored
                            distances_snap = self._distances
                            gpt_thread = threading.Thread(
                                target=target,
                                args=(telemetry, frame_snap, depth_snap, distances_snap),
                                daemon=True,
                            )
                            gpt_thread.start()
                    else:
                        self._trigger_smart_auto(label="AUTO")

                elif state == AvoidanceState.EXECUTING:
                    vx, vy, vz = velocity
                    gpt_dur = self.avoidance._gpt_duration
                    dur = gpt_dur if gpt_dur else self.avoidance.MANEUVER_DURATION
                    action_str = self.avoidance.current_action.value

                    dirs = []
                    if vx < -0.05: dirs.append("BACKWARD")
                    elif vx > 0.05: dirs.append("FORWARD")
                    if vy < -0.05: dirs.append("LEFT")
                    elif vy > 0.05: dirs.append("RIGHT")
                    if vz < -0.05: dirs.append("UP")
                    elif vz > 0.05: dirs.append("DOWN")
                    direction = " + ".join(dirs) if dirs else "HOVER"

                    # Episode tracking
                    self._episode_maneuver_count += 1
                    self._last_executing_velocity = (vx, vy, vz, dur, direction)
                    self._last_executing_start = time.time()

                    # KEY EVENT: command_sent — this is what's ACTUALLY pushed to the drone.
                    # Compare this with the prior vlm_decision to verify execution fidelity.
                    self._log("command_sent", direction=direction,
                              vx=vx, vy=vy, vz=vz, duration=dur,
                              attempt_num=self._episode_maneuver_count,
                              reasoning="velocity command dispatched to MAVLink")

                    msg = f"[{timestamp}] {direction}: vel=({vx:.2f},{vy:.2f},{vz:.2f}) {dur:.1f}s (dist={dist_str})"
                    self.decision_made.emit(msg)
                    self.command_display.emit(
                        f">> {direction}\nvel=({vx:.2f}, {vy:.2f}, {vz:.2f})  {dur:.1f}s"
                    )
                    print(f"[AVOIDANCE] EXECUTING: {direction} vel=({vx:.2f}, {vy:.2f}, {vz:.2f}) | duration={dur:.1f}s")

                elif state == AvoidanceState.COMPLETED:
                    self.command_display.emit("")

                    # Maneuver outcome: did obstacle clear?
                    fwd = self._forward_distance
                    cleared = (fwd is None or fwd >= self.avoidance.DISTANCE_THRESHOLD)
                    outcome = "cleared" if cleared else "retry"
                    duration = (time.time() - self._last_executing_start) if self._last_executing_start else 0

                    last_vel = self._last_executing_velocity or (0, 0, 0, 0, "")
                    self._log("maneuver_outcome",
                              direction=last_vel[4],
                              vx=last_vel[0], vy=last_vel[1], vz=last_vel[2],
                              duration=round(duration, 2),
                              outcome=outcome,
                              attempt_num=self._episode_maneuver_count)

                    print(f"[AVOIDANCE] MANEUVER COMPLETE -- outcome={outcome} (dist={dist_str})")

                elif state == AvoidanceState.CLEARANCE:
                    vx, vy, vz = velocity
                    self.command_display.emit(
                        f">> CLEARANCE (flying past obstacle)\nvel=({vx:.2f}, {vy:.2f}, {vz:.2f})  {self.avoidance.CLEARANCE_DURATION:.0f}s"
                    )
                    msg = f"[{timestamp}] CLEARANCE: flying forward to pass obstacle ({self.avoidance.CLEARANCE_DURATION:.0f}s)"
                    self.decision_made.emit(msg)

                    # Log clearance command too
                    self._log("command_sent", direction="CLEARANCE (forward)",
                              vx=vx, vy=vy, vz=vz,
                              duration=self.avoidance.CLEARANCE_DURATION,
                              reasoning="clearance phase — flying past obstacle")

                    print(f"[AVOIDANCE] CLEARANCE: flying forward vel=({vx:.2f}, {vy:.2f}, {vz:.2f}) for {self.avoidance.CLEARANCE_DURATION}s")

                elif state == AvoidanceState.IDLE:
                    self._restore_mode()
                    self.command_display.emit("")
                    print(f"[AVOIDANCE] IDLE -- obstacle cleared + passed, resuming normal flight")

                    # Session summary — if this is the end of an avoidance episode
                    if self._episode_start_time is not None:
                        total_time = round(time.time() - self._episode_start_time, 2)
                        self._log("session_summary",
                                  duration=total_time,
                                  attempt_num=self._episode_maneuver_count,
                                  outcome="success",
                                  reasoning=(
                                      f"episode complete: "
                                      f"{self._episode_maneuver_count} maneuvers, "
                                      f"{self._episode_vlm_calls} VLM calls, "
                                      f"{total_time}s total"
                                  ))
                        print(f"[AVOIDANCE] Episode summary: {self._episode_maneuver_count} maneuvers, "
                              f"{self._episode_vlm_calls} VLM calls, {total_time}s")
                        self._episode_start_time = None

                self._last_state = state

            # Send velocity commands when avoiding
            vx, vy, vz = velocity
            if state in (AvoidanceState.OBSTACLE_DETECTED, AvoidanceState.HOVERING,
                         AvoidanceState.COMPLETED):
                self.mavlink.hover()
            elif state in (AvoidanceState.EXECUTING, AvoidanceState.CLEARANCE):
                telem = self.mavlink.get_telemetry() or {}
                heading = telem.get("heading", 0)
                self.mavlink.send_velocity(vx, vy, vz)
                d = f"{self._forward_distance:.2f}m" if self._forward_distance else "N/A"
                phase = "CLEAR" if state == AvoidanceState.CLEARANCE else "AVOID"
                print(
                    f"[AVOIDANCE] >> {phase} vel=({vx:.2f}, {vy:.2f}, {vz:.2f}) dist={d}", end="\r"
                )

            self.msleep(50)  # 20 Hz

    def stop(self):
        self._running = False
        self.wait(2000)