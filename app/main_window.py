"""Main application window — orchestrates all components."""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QMessageBox, QStatusBar, QApplication, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QColor


class ModelLoaderThread(QThread):
    """Loads the depth model in background without blocking the UI."""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, depth_estimator, parent=None):
        super().__init__(parent)
        self.depth_estimator = depth_estimator

    def run(self):
        try:
            self.depth_estimator.load()
            self.finished.emit(self.depth_estimator.device)
        except Exception as e:
            self.error.emit(str(e))

from app.drone.mavlink_comm import MAVLinkComm
from app.drone.avoidance import ObstacleAvoidance, AvoidanceState
from app.vision.camera import Camera
from app.vision.depth_model import DepthEstimator
from app.threads.drone_thread import DroneThread
from app.threads.camera_thread import CameraThread
from app.threads.avoidance_thread import AvoidanceThread
from app.widgets.video_widget import VideoWidget
from app.widgets.depth_widget import DepthWidget
from app.widgets.status_panel import StatusPanel
from app.widgets.control_panel import ControlPanel
from app.widgets.avoidance_log import AvoidanceLog
from app.widgets.gpt_log import GPTLog
from app.widgets.mission_planner import MissionPlannerWidget
from app.threads.mission_thread import MissionThread
from app.recorder import SessionRecorder


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VLM-Drone Control")
        self.setMinimumSize(960, 600)
        self.setStyleSheet("background-color: #1e1e1e; color: #eee;")

        # Core components
        self._mavlink = MAVLinkComm()
        self._avoidance = ObstacleAvoidance()
        self._camera = Camera(device_id=0)
        self._depth_estimator = DepthEstimator()

        # Threads
        self._drone_thread = DroneThread(self._mavlink)
        self._camera_thread = CameraThread(self._camera, self._depth_estimator)
        self._avoidance_thread = AvoidanceThread(self._avoidance, self._mavlink)
        self._mission_thread = MissionThread(self._mavlink)

        # Recorder
        self._recorder = SessionRecorder(base_dir="recordings")
        self._telem_log_counter = 0
        self._last_distances = None

        # Latest telemetry
        self._last_telemetry = {}

        self._build_ui()
        self._connect_signals()
        self._setup_manual_control_timer()

        # Pass recorder to avoidance thread (for VLM event logging)
        self._avoidance_thread.set_recorder(self._recorder)

        self._load_depth_model()
        self._start_camera()

    # ── UI Layout ──────────────────────────────────────────

    def _build_ui(self):
        # Tab widget as central
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #333; background: #1e1e1e; }"
            "QTabBar::tab { background: #2a2a2a; color: #ccc; padding: 8px 18px; "
            "border: 1px solid #333; border-bottom: none; }"
            "QTabBar::tab:selected { background: #1e1e1e; color: #0af; "
            "border-top: 2px solid #0af; }"
        )
        self.setCentralWidget(self._tabs)

        # ── Tab 1: Control ───────────────────────────────
        control_tab = QWidget()
        main_layout = QHBoxLayout(control_tab)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Left column: video + controls
        left = QVBoxLayout()
        self._video_widget = VideoWidget("RGB Video")
        left.addWidget(self._video_widget, stretch=3)

        self._control_panel = ControlPanel()
        left.addWidget(self._control_panel, stretch=2)

        # Right column: depth + status + avoidance log
        right = QVBoxLayout()
        self._depth_widget = DepthWidget()
        right.addWidget(self._depth_widget, stretch=2)

        self._status_panel = StatusPanel()
        right.addWidget(self._status_panel, stretch=2)

        self._avoidance_log = AvoidanceLog()
        right.addWidget(self._avoidance_log, stretch=1)

        self._gpt_log = GPTLog()
        right.addWidget(self._gpt_log, stretch=1)

        main_layout.addLayout(left, stretch=3)
        main_layout.addLayout(right, stretch=2)

        self._tabs.addTab(control_tab, "Control")

        # ── Tab 2: Mission Planner ───────────────────────
        self._mission_planner = MissionPlannerWidget()
        self._tabs.addTab(self._mission_planner, "Mission Planner")

        # Status bar
        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet("color: #888; font-size: 11px;")
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready — not connected")

    # ── Signal Wiring ──────────────────────────────────────

    def _connect_signals(self):
        # Control panel
        self._control_panel.connect_requested.connect(self._on_connect)
        self._control_panel.disconnect_requested.connect(self._on_disconnect)
        self._control_panel.arm_requested.connect(self._mavlink.arm)
        self._control_panel.disarm_requested.connect(self._mavlink.disarm)
        self._control_panel.mode_change_requested.connect(self._mavlink.set_mode)
        self._control_panel.avoidance_toggled.connect(self._on_avoidance_toggle)
        self._control_panel.avoidance_mode_changed.connect(self._avoidance_thread.set_avoidance_mode)
        self._control_panel.camera_changed.connect(self._on_camera_changed)
        self._control_panel.recording_toggled.connect(self._on_recording_toggle)

        # Drone thread
        self._drone_thread.telemetry_updated.connect(self._on_telemetry)
        self._drone_thread.connection_status.connect(self._on_connection_status)

        # Camera thread
        self._camera_thread.frame_ready.connect(self._on_frame_ready)
        self._camera_thread.error_occurred.connect(
            lambda e: self._status_bar.showMessage(f"Camera error: {e}")
        )

        # Avoidance thread
        self._avoidance_thread.state_changed.connect(self._on_avoidance_state)
        self._avoidance_thread.decision_made.connect(self._avoidance_log.add_log_entry)

        # GPT advisor signals
        self._avoidance_thread.gpt_sent.connect(self._gpt_log.add_sent)
        self._avoidance_thread.gpt_received.connect(self._gpt_log.add_received)
        self._avoidance_thread.gpt_error.connect(self._gpt_log.add_error)

        # Command overlay on video
        self._avoidance_thread.command_display.connect(self._video_widget.set_command_text)

        # Mission Planner signals
        self._mission_planner.upload_requested.connect(self._on_mission_upload)
        self._mission_planner.download_requested.connect(self._on_mission_download)
        self._mission_planner.start_requested.connect(self._on_mission_start)
        self._mission_planner.clear_drone_requested.connect(self._on_mission_clear)
        self._mission_planner.avoidance_during_mission_toggled.connect(
            self._avoidance_thread.set_avoidance_during_mission
        )

        # Mission thread signals
        self._mission_thread.upload_progress.connect(self._mission_planner.set_upload_progress)
        self._mission_thread.upload_complete.connect(
            lambda: self._mission_planner.add_status("Upload complete")
        )
        self._mission_thread.upload_error.connect(
            lambda err: self._mission_planner.add_status(f"Upload error: {err}")
        )
        self._mission_thread.download_complete.connect(self._on_mission_downloaded)
        self._mission_thread.download_error.connect(
            lambda err: self._mission_planner.add_status(f"Download error: {err}")
        )
        self._mission_thread.mission_status.connect(self._mission_planner.add_status)
        self._mission_thread.clear_complete.connect(
            lambda: self._mission_planner.add_status("Drone mission cleared")
        )
        self._mission_thread.clear_error.connect(
            lambda err: self._mission_planner.add_status(f"Clear error: {err}")
        )

    # ── Initialization ─────────────────────────────────────

    def _load_depth_model(self):
        self._status_bar.showMessage("Loading Depth Anything V2 Small model in background...")
        self._model_loader = ModelLoaderThread(self._depth_estimator)
        self._model_loader.finished.connect(
            lambda: self._status_bar.showMessage(f"Depth model loaded on {self._depth_estimator.device.upper()} — depth estimation active")
        )
        self._model_loader.error.connect(
            lambda e: self._status_bar.showMessage(f"Depth model failed: {e}")
        )
        self._model_loader.start()

    def _start_camera(self):
        try:
            self._camera.open()
            self._camera_thread.start()
            self._status_bar.showMessage("Camera started")
        except Exception as e:
            self._status_bar.showMessage(f"Camera error: {e}")

    def _on_camera_changed(self, device_id):
        """Switch to a different camera device."""
        self._camera_thread.stop()
        self._camera.release()
        self._camera.device_id = device_id
        try:
            self._camera.open()
            self._camera_thread = CameraThread(self._camera, self._depth_estimator)
            self._camera_thread.frame_ready.connect(self._on_frame_ready)
            self._camera_thread.error_occurred.connect(
                lambda e: self._status_bar.showMessage(f"Camera error: {e}")
            )
            self._camera_thread.start()
            self._status_bar.showMessage(f"Switched to Camera {device_id}")
        except Exception as e:
            self._status_bar.showMessage(f"Camera {device_id} failed: {e}")

    # ── Connection ─────────────────────────────────────────

    def _on_connect(self, connection_string):
        try:
            self._mavlink.connect(connection_string)
            self._control_panel.set_connected(True)
            self._drone_thread.start()
            self._avoidance_thread.start()
            self._mission_thread.start()
            self._status_bar.showMessage(f"Connected to {connection_string}")
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))
            self._status_bar.showMessage(f"Connection failed: {e}")

    def _on_disconnect(self):
        self._mission_thread.stop()
        self._avoidance_thread.stop()
        self._drone_thread.stop()
        self._mavlink.disconnect()
        self._control_panel.set_connected(False)
        self._status_bar.showMessage("Disconnected")

    # ── Mission Planner handlers ───────────────────────────

    def _on_mission_upload(self, waypoints):
        if not self._mavlink.connected:
            QMessageBox.warning(self, "Not Connected", "Connect to drone first.")
            return
        self._mission_planner.add_status(f"Uploading {len(waypoints)} waypoints...")
        self._mission_thread.request_upload(waypoints)

    def _on_mission_download(self):
        if not self._mavlink.connected:
            QMessageBox.warning(self, "Not Connected", "Connect to drone first.")
            return
        self._mission_planner.add_status("Downloading mission from drone...")
        self._mission_thread.request_download()

    def _on_mission_downloaded(self, waypoints):
        self._mission_planner._clear_all()
        for wp in waypoints:
            self._mission_planner._add_waypoint_row(wp)
        self._mission_planner.add_status(f"Downloaded {len(waypoints)} waypoints")

    def _on_mission_start(self):
        if not self._mavlink.connected:
            QMessageBox.warning(self, "Not Connected", "Connect to drone first.")
            return
        # Arm, set AUTO, start mission
        self._mission_planner.add_status("Starting mission: arm + AUTO + MISSION_START")
        self._mavlink.arm()
        self._mavlink.set_mode("AUTO")
        self._mavlink.mission_start()

    def _on_mission_clear(self):
        if not self._mavlink.connected:
            QMessageBox.warning(self, "Not Connected", "Connect to drone first.")
            return
        self._mission_planner.add_status("Clearing drone mission memory...")
        self._mission_thread.request_clear()

    def _on_connection_status(self, connected, message):
        if not connected:
            self._status_bar.showMessage(f"Connection issue: {message}")

    # ── Recording ──────────────────────────────────────────

    def _on_recording_toggle(self, start):
        """Handle Start/Stop Recording button from control panel."""
        if start:
            metadata = {
                "advisor_mode": self._avoidance_thread._avoidance_mode,
                "camera_id": self._camera.device_id,
                "depth_model": getattr(self._depth_estimator, "MODEL_ID", "unknown"),
                "depth_device": self._depth_estimator.device,
            }
            session_dir = self._recorder.start(metadata=metadata)
            self._control_panel.set_recording_path(session_dir)
            self._status_bar.showMessage(f"Recording started: {session_dir}")
        else:
            self._recorder.stop()
            self._status_bar.showMessage("Recording stopped")

    # ── Telemetry ──────────────────────────────────────────

    def _on_telemetry(self, data):
        self._last_telemetry = data
        self._status_panel.update_telemetry(data)
        self._avoidance_thread.update_altitude(data.get("altitude", 0))
        # Feed current position + mission progress to Mission Planner
        self._mission_planner.set_drone_position(
            data.get("lat", 0.0), data.get("lon", 0.0), data.get("altitude", 0.0)
        )
        current_wp = data.get("mission_current_wp", 0)
        mode = data.get("mode", "UNKNOWN")
        # Only show mission progress when in AUTO mode
        if mode == "AUTO":
            self._mission_planner.set_mission_progress(current_wp, 0)
        # Tell avoidance thread about current flight mode for AUTO-resume logic
        self._avoidance_thread.update_current_mode(mode)

        # Log telemetry row every 10th update (~500ms at 20Hz)
        if self._recorder.is_recording:
            self._telem_log_counter += 1
            if self._telem_log_counter >= 10:
                self._telem_log_counter = 0
                self._recorder.log_telemetry(
                    telemetry=data,
                    distances_9seg=self._last_distances,
                    avoidance_state=self._avoidance.state.value,
                    avoidance_action=self._avoidance.current_action.value,
                )

    # ── Camera / Depth ─────────────────────────────────────

    def _on_frame_ready(self, rgb_frame, depth_colored, center_distance, distances):
        # Update video widget
        self._video_widget.update_frame(rgb_frame)
        self._depth_widget.update_depth(depth_colored)

        # Update distance overlay
        if center_distance is not None:
            L = (distances or {}).get("left", 0) or 0
            R = (distances or {}).get("right", 0) or 0
            self._video_widget.set_distance_text(
                f"Fwd: {center_distance:.2f}m  L: {L:.2f}m  R: {R:.2f}m"
            )
        else:
            self._video_widget.set_distance_text("")

        # Cache for recorder telemetry logging
        self._last_distances = distances

        # Feed distance, frame, depth_colored to avoidance thread (for VLM vision input)
        self._avoidance_thread.update_distance(center_distance)
        self._avoidance_thread.update_distances(distances)
        self._avoidance_thread.update_frame(rgb_frame)
        self._avoidance_thread.update_depth_colored(depth_colored)

        # Recording — write frames if active
        if self._recorder.is_recording:
            self._recorder.write_rgb(rgb_frame)
            self._recorder.write_depth(depth_colored)

    # ── Avoidance ──────────────────────────────────────────

    def _on_avoidance_toggle(self, enabled):
        self._avoidance_thread.set_enabled(enabled)
        if enabled:
            self._status_bar.showMessage("Obstacle avoidance ENABLED")
        else:
            self._status_bar.showMessage("Obstacle avoidance disabled")
            self._video_widget.clear_overlay()

    def _on_avoidance_state(self, state, action):
        self._avoidance_log.update_state(state, action)

        # Overlay on video
        if state == AvoidanceState.IDLE.value:
            self._video_widget.set_overlay("", QColor(0, 255, 0))
        elif state == AvoidanceState.OBSTACLE_DETECTED.value:
            self._video_widget.set_overlay("OBSTACLE DETECTED — STOPPING", QColor(255, 0, 0))
        elif state == AvoidanceState.HOVERING.value:
            mode = self._avoidance_thread._avoidance_mode
            if mode == "gpt":
                self._video_widget.set_overlay("HOVERING -- QUERYING GPT-4o...", QColor(255, 255, 0))
            elif mode == "gemini":
                self._video_widget.set_overlay("HOVERING -- QUERYING GEMINI...", QColor(255, 255, 0))
            else:
                self._video_widget.set_overlay("HOVERING -- DECIDING...", QColor(255, 255, 0))
        elif state == AvoidanceState.EXECUTING.value:
            # Direction is shown by command_display overlay, just show EXECUTING
            self._video_widget.set_overlay("EXECUTING AVOIDANCE", QColor(255, 128, 0))
        elif state == AvoidanceState.COMPLETED.value:
            self._video_widget.set_overlay("MANEUVER COMPLETE", QColor(0, 200, 255))
        elif state == AvoidanceState.CLEARANCE.value:
            self._video_widget.set_overlay("CLEARANCE -- FLYING PAST OBSTACLE", QColor(0, 200, 100))

    # ── Manual Control ─────────────────────────────────────

    def _setup_manual_control_timer(self):
        """Timer to send manual control commands at 10 Hz."""
        self._manual_timer = QTimer()
        self._manual_timer.setInterval(100)
        self._manual_timer.timeout.connect(self._send_manual_control)
        self._manual_timer.start()

    def _send_manual_control(self):
        if not self._mavlink.connected:
            return
        # Don't send manual controls while avoidance is active and not idle
        if (self._avoidance_thread._enabled and
                not self._avoidance_thread._state_is_idle):
            return
        x, y, z, r = self._control_panel.get_manual_control_values()
        if x != 0 or y != 0 or z != 500 or r != 0:
            self._mavlink.send_manual_control(x, y, z, r)

    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            self._control_panel.handle_key_press(event.key())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            self._control_panel.handle_key_release(event.key())
        super().keyReleaseEvent(event)

    # ── Cleanup ────────────────────────────────────────────

    def closeEvent(self, event):
        # Stop recorder first so final frames are flushed
        if self._recorder.is_recording:
            self._recorder.stop()
        self._manual_timer.stop()
        self._mission_thread.stop()
        self._avoidance_thread.stop()
        self._camera_thread.stop()
        self._drone_thread.stop()
        self._camera.release()
        self._mavlink.disconnect()
        super().closeEvent(event)