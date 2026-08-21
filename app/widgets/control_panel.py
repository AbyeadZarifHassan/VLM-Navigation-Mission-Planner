"""Manual control panel with keyboard controls, connection settings, and avoidance config."""

import os
import json
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QGroupBox, QSpinBox, QPushButton, QLineEdit, QComboBox,
    QRadioButton, QStackedWidget, QButtonGroup, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


# Config file path (project root)
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "app_config.json")


def _load_config():
    try:
        with open(_CONFIG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_config(data):
    try:
        existing = _load_config()
        existing.update(data)
        with open(_CONFIG_PATH, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"[CONFIG] Failed to save: {e}")


class ControlPanel(QWidget):
    """Manual control panel with connection settings and keyboard control."""

    connect_requested = pyqtSignal(str)  # connection string
    disconnect_requested = pyqtSignal()
    arm_requested = pyqtSignal()
    disarm_requested = pyqtSignal()
    mode_change_requested = pyqtSignal(str)
    avoidance_toggled = pyqtSignal(bool)
    avoidance_mode_changed = pyqtSignal(str)  # "auto" or "gpt"
    camera_changed = pyqtSignal(int)  # camera device id
    api_key_saved = pyqtSignal(str)  # API key value
    recording_toggled = pyqtSignal(bool)  # True = start, False = stop

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll)

        # Camera group
        cam_group = QGroupBox("Camera")
        cam_group.setStyleSheet(self._group_style())
        cam_layout = QHBoxLayout(cam_group)
        cam_layout.addWidget(QLabel("Device:"))
        self._cam_combo = QComboBox()
        self._cam_combo.setStyleSheet("background: #2a2a2a; color: #eee; padding: 4px;")
        for i in range(4):
            self._cam_combo.addItem(f"Camera {i}", i)
        self._cam_combo.currentIndexChanged.connect(
            lambda: self.camera_changed.emit(self._cam_combo.currentData())
        )
        cam_layout.addWidget(self._cam_combo)
        layout.addWidget(cam_group)

        # ── Connection group ──────────────────────────────────
        conn_group = QGroupBox("Connection")
        conn_group.setStyleSheet(self._group_style())
        conn_main = QVBoxLayout(conn_group)

        radio_row = QHBoxLayout()
        self._serial_radio = QRadioButton("Serial")
        self._udp_radio = QRadioButton("UDP")
        self._serial_radio.setStyleSheet("color: #eee;")
        self._udp_radio.setStyleSheet("color: #eee;")
        self._udp_radio.setChecked(True)
        radio_group = QButtonGroup(self)
        radio_group.addButton(self._serial_radio)
        radio_group.addButton(self._udp_radio)
        radio_row.addWidget(self._serial_radio)
        radio_row.addWidget(self._udp_radio)
        radio_row.addStretch()
        conn_main.addLayout(radio_row)

        self._conn_stack = QStackedWidget()

        serial_page = QWidget()
        serial_layout = QHBoxLayout(serial_page)
        serial_layout.setContentsMargins(0, 0, 0, 0)
        serial_layout.addWidget(QLabel("COM:"))
        self._com_combo = QComboBox()
        self._com_combo.setStyleSheet("background: #2a2a2a; color: #eee; padding: 4px;")
        self._com_combo.setMinimumWidth(100)
        serial_layout.addWidget(self._com_combo)
        self._refresh_btn = QPushButton("↻")
        self._refresh_btn.setFixedWidth(30)
        self._refresh_btn.setStyleSheet(self._btn_style("#555"))
        self._refresh_btn.setToolTip("Refresh COM ports")
        self._refresh_btn.clicked.connect(self._scan_ports)
        serial_layout.addWidget(self._refresh_btn)
        serial_layout.addWidget(QLabel("Baud:"))
        self._baud_combo = QComboBox()
        self._baud_combo.setStyleSheet("background: #2a2a2a; color: #eee; padding: 4px;")
        for baud in ["9600", "19200", "38400", "57600", "115200"]:
            self._baud_combo.addItem(baud)
        self._baud_combo.setCurrentText("57600")
        serial_layout.addWidget(self._baud_combo)
        self._conn_stack.addWidget(serial_page)

        udp_page = QWidget()
        udp_layout = QHBoxLayout(udp_page)
        udp_layout.setContentsMargins(0, 0, 0, 0)
        self._udp_input = QLineEdit("udp:127.0.0.1:14550")
        self._udp_input.setStyleSheet("background: #2a2a2a; color: #eee; border: 1px solid #555; padding: 4px;")
        udp_layout.addWidget(self._udp_input)
        self._conn_stack.addWidget(udp_page)
        self._conn_stack.setCurrentIndex(1)
        conn_main.addWidget(self._conn_stack)

        self._serial_radio.toggled.connect(lambda checked: self._conn_stack.setCurrentIndex(0 if checked else 1))
        self._serial_radio.toggled.connect(lambda checked: self._scan_ports() if checked else None)

        btn_row = QHBoxLayout()
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setFixedHeight(32)
        self._connect_btn.setStyleSheet(self._btn_style("#2196F3"))
        self._connect_btn.clicked.connect(self._on_connect)
        btn_row.addWidget(self._connect_btn)
        self._disconnect_btn = QPushButton("Disconnect")
        self._disconnect_btn.setFixedHeight(32)
        self._disconnect_btn.setStyleSheet(self._btn_style("#666"))
        self._disconnect_btn.clicked.connect(self.disconnect_requested.emit)
        self._disconnect_btn.setEnabled(False)
        btn_row.addWidget(self._disconnect_btn)
        conn_main.addLayout(btn_row)

        layout.addWidget(conn_group)

        # ── Arm/Mode group ────────────────────────────────────
        cmd_group = QGroupBox("Commands")
        cmd_group.setStyleSheet(self._group_style())
        cmd_layout = QHBoxLayout(cmd_group)
        self._arm_btn = QPushButton("ARM")
        self._arm_btn.setStyleSheet(self._btn_style("#f44336"))
        self._arm_btn.clicked.connect(self.arm_requested.emit)
        cmd_layout.addWidget(self._arm_btn)
        self._disarm_btn = QPushButton("DISARM")
        self._disarm_btn.setStyleSheet(self._btn_style("#4CAF50"))
        self._disarm_btn.clicked.connect(self.disarm_requested.emit)
        cmd_layout.addWidget(self._disarm_btn)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["STABILIZE", "GUIDED", "AUTO", "LOITER", "ALT_HOLD", "LAND", "RTL"])
        self._mode_combo.setStyleSheet("background: #2a2a2a; color: #eee; padding: 4px;")
        cmd_layout.addWidget(self._mode_combo)
        self._mode_btn = QPushButton("Set Mode")
        self._mode_btn.setStyleSheet(self._btn_style("#FF9800"))
        self._mode_btn.clicked.connect(lambda: self.mode_change_requested.emit(self._mode_combo.currentText()))
        cmd_layout.addWidget(self._mode_btn)
        layout.addWidget(cmd_group)

        # ── Speed + Manual Control ────────────────────────────
        ctrl_group = QGroupBox("Manual Control (WASD + Q/E + R/F)")
        ctrl_group.setStyleSheet(self._group_style())
        ctrl_layout = QVBoxLayout(ctrl_group)
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed (%):"))
        self._speed_spin = QSpinBox()
        self._speed_spin.setRange(0, 100)
        self._speed_spin.setValue(30)
        self._speed_spin.setStyleSheet("background: #2a2a2a; color: #eee; padding: 4px;")
        speed_layout.addWidget(self._speed_spin)
        ctrl_layout.addLayout(speed_layout)
        self._key_label = QLabel("Keys: --")
        self._key_label.setFont(QFont("Consolas", 11))
        self._key_label.setAlignment(Qt.AlignCenter)
        self._key_label.setStyleSheet("color: #0f0; background: #111; padding: 8px; border: 1px solid #333;")
        ctrl_layout.addWidget(self._key_label)
        legend = QLabel(
            "W/S: Forward/Back  |  A/D: Left/Right\n"
            "Q/E: Yaw Left/Right  |  R/F: Up/Down"
        )
        legend.setStyleSheet("color: #888; font-size: 10px;")
        legend.setAlignment(Qt.AlignCenter)
        ctrl_layout.addWidget(legend)
        layout.addWidget(ctrl_group)

        # ── Avoidance toggle + mode selector ──────────────────
        avoid_group = QGroupBox("Obstacle Avoidance")
        avoid_group.setStyleSheet(self._group_style())
        avoid_layout = QVBoxLayout(avoid_group)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._avoid_mode_combo = QComboBox()
        self._avoid_mode_combo.addItem("Auto (Random)", "auto")
        self._avoid_mode_combo.addItem("GPT-4o Advisor", "gpt")
        self._avoid_mode_combo.addItem("Gemini Advisor", "gemini")
        self._avoid_mode_combo.setStyleSheet("background: #2a2a2a; color: #eee; padding: 4px;")
        self._avoid_mode_combo.currentIndexChanged.connect(
            lambda: self.avoidance_mode_changed.emit(self._avoid_mode_combo.currentData())
        )
        mode_row.addWidget(self._avoid_mode_combo)
        avoid_layout.addLayout(mode_row)
        self._avoid_btn = QPushButton("Enable Avoidance")
        self._avoid_btn.setCheckable(True)
        self._avoid_btn.setStyleSheet(self._btn_style("#9C27B0"))
        self._avoid_btn.toggled.connect(self._on_avoidance_toggle)
        avoid_layout.addWidget(self._avoid_btn)
        layout.addWidget(avoid_group)

        # ── Session Recording ─────────────────────────────────
        rec_group = QGroupBox("Session Recording")
        rec_group.setStyleSheet(self._group_style())
        rec_layout = QVBoxLayout(rec_group)
        rec_btn_row = QHBoxLayout()
        self._rec_btn = QPushButton("● Start Recording")
        self._rec_btn.setCheckable(True)
        self._rec_btn.setStyleSheet(self._btn_style("#f44336"))
        self._rec_btn.toggled.connect(self._on_rec_toggle)
        rec_btn_row.addWidget(self._rec_btn)
        rec_layout.addLayout(rec_btn_row)
        self._rec_status = QLabel("Not recording")
        self._rec_status.setStyleSheet("color: #888; font-size: 10px;")
        self._rec_status.setWordWrap(True)
        rec_layout.addWidget(self._rec_status)
        layout.addWidget(rec_group)

        # ── OpenAI API Key ────────────────────────────────────
        api_group = QGroupBox("OpenAI API Key")
        api_group.setStyleSheet(self._group_style())
        api_layout = QVBoxLayout(api_group)
        key_row = QHBoxLayout()
        self._api_key_input = QLineEdit()
        self._api_key_input.setEchoMode(QLineEdit.Password)
        self._api_key_input.setPlaceholderText("sk-...")
        self._api_key_input.setStyleSheet("background: #2a2a2a; color: #eee; border: 1px solid #555; padding: 4px;")
        key_row.addWidget(self._api_key_input)
        self._api_save_btn = QPushButton("Save")
        self._api_save_btn.setStyleSheet(self._btn_style("#4CAF50"))
        self._api_save_btn.clicked.connect(self._on_save_api_key)
        key_row.addWidget(self._api_save_btn)
        self._api_clear_btn = QPushButton("Clear")
        self._api_clear_btn.setStyleSheet(self._btn_style("#f44336"))
        self._api_clear_btn.clicked.connect(self._on_clear_api_key)
        key_row.addWidget(self._api_clear_btn)
        api_layout.addLayout(key_row)
        self._api_status = QLabel("")
        self._api_status.setStyleSheet("color: #888; font-size: 10px;")
        api_layout.addWidget(self._api_status)
        layout.addWidget(api_group)

        # ── Gemini API Key ────────────────────────────────────
        gemini_group = QGroupBox("Gemini API Key")
        gemini_group.setStyleSheet(self._group_style())
        gemini_layout = QVBoxLayout(gemini_group)
        gemini_key_row = QHBoxLayout()
        self._gemini_key_input = QLineEdit()
        self._gemini_key_input.setEchoMode(QLineEdit.Password)
        self._gemini_key_input.setPlaceholderText("AIza...")
        self._gemini_key_input.setStyleSheet("background: #2a2a2a; color: #eee; border: 1px solid #555; padding: 4px;")
        gemini_key_row.addWidget(self._gemini_key_input)
        self._gemini_save_btn = QPushButton("Save")
        self._gemini_save_btn.setStyleSheet(self._btn_style("#4CAF50"))
        self._gemini_save_btn.clicked.connect(self._on_save_gemini_key)
        gemini_key_row.addWidget(self._gemini_save_btn)
        self._gemini_clear_btn = QPushButton("Clear")
        self._gemini_clear_btn.setStyleSheet(self._btn_style("#f44336"))
        self._gemini_clear_btn.clicked.connect(self._on_clear_gemini_key)
        gemini_key_row.addWidget(self._gemini_clear_btn)
        gemini_layout.addLayout(gemini_key_row)
        self._gemini_status = QLabel("")
        self._gemini_status.setStyleSheet("color: #888; font-size: 10px;")
        gemini_layout.addWidget(self._gemini_status)
        layout.addWidget(gemini_group)

        layout.addStretch()

        self._pressed_keys = set()
        self._load_api_key()
        self._load_gemini_key()

    # ── Port scanning ─────────────────────────────────────

    def _scan_ports(self):
        """Scan for available COM ports."""
        self._com_combo.clear()
        try:
            from serial.tools.list_ports import comports
            ports = sorted(comports(), key=lambda p: p.device)
            for port in ports:
                self._com_combo.addItem(f"{port.device} — {port.description}", port.device)
            if not ports:
                self._com_combo.addItem("No COM ports found", "")
        except ImportError:
            self._com_combo.addItem("pyserial not installed", "")

    # ── API key persistence ───────────────────────────────

    def _load_api_key(self):
        """Load saved API key from config file."""
        cfg = _load_config()
        key = cfg.get("openai_api_key", "")
        if key:
            self._api_key_input.setText(key)
            os.environ["OPENAI_API_KEY"] = key
            masked = key[:7] + "..." + key[-4:] if len(key) > 11 else "***"
            self._api_status.setText(f"Loaded from config ({masked})")
            self._api_status.setStyleSheet("color: #4CAF50; font-size: 10px;")

    def _on_save_api_key(self):
        key = self._api_key_input.text().strip()
        if not key:
            return
        _save_config({"openai_api_key": key})
        os.environ["OPENAI_API_KEY"] = key
        self.api_key_saved.emit(key)
        masked = key[:7] + "..." + key[-4:] if len(key) > 11 else "***"
        self._api_status.setText(f"Saved ({masked})")
        self._api_status.setStyleSheet("color: #4CAF50; font-size: 10px;")
        print(f"[CONFIG] API key saved ({masked})")

    def _on_clear_api_key(self):
        self._api_key_input.clear()
        os.environ.pop("OPENAI_API_KEY", None)
        _save_config({"openai_api_key": ""})
        self._api_status.setText("Cleared")
        self._api_status.setStyleSheet("color: #f44; font-size: 10px;")
        print("[CONFIG] API key cleared")

    def _load_gemini_key(self):
        """Load saved Gemini API key from config file."""
        cfg = _load_config()
        key = cfg.get("gemini_api_key", "")
        if key:
            self._gemini_key_input.setText(key)
            os.environ["GEMINI_API_KEY"] = key
            masked = key[:7] + "..." + key[-4:] if len(key) > 11 else "***"
            self._gemini_status.setText(f"Loaded from config ({masked})")
            self._gemini_status.setStyleSheet("color: #4CAF50; font-size: 10px;")

    def _on_save_gemini_key(self):
        key = self._gemini_key_input.text().strip()
        if not key:
            return
        _save_config({"gemini_api_key": key})
        os.environ["GEMINI_API_KEY"] = key
        masked = key[:7] + "..." + key[-4:] if len(key) > 11 else "***"
        self._gemini_status.setText(f"Saved ({masked})")
        self._gemini_status.setStyleSheet("color: #4CAF50; font-size: 10px;")
        print(f"[CONFIG] Gemini API key saved ({masked})")

    def _on_clear_gemini_key(self):
        self._gemini_key_input.clear()
        os.environ.pop("GEMINI_API_KEY", None)
        _save_config({"gemini_api_key": ""})
        self._gemini_status.setText("Cleared")
        self._gemini_status.setStyleSheet("color: #f44; font-size: 10px;")
        print("[CONFIG] Gemini API key cleared")

    # ── Manual control ────────────────────────────────────

    def get_speed(self):
        return self._speed_spin.value() / 100.0

    def get_manual_control_values(self):
        speed = self._speed_spin.value() * 10
        x = y = r = 0
        z = 500
        if Qt.Key_W in self._pressed_keys: x = speed
        if Qt.Key_S in self._pressed_keys: x = -speed
        if Qt.Key_A in self._pressed_keys: y = -speed
        if Qt.Key_D in self._pressed_keys: y = speed
        if Qt.Key_Q in self._pressed_keys: r = -speed
        if Qt.Key_E in self._pressed_keys: r = speed
        if Qt.Key_R in self._pressed_keys: z = 500 - speed // 2  # R = UP (throttle > mid)
        if Qt.Key_F in self._pressed_keys: z = 500 + speed // 2  # F = DOWN (throttle < mid)
        return (x, y, z, r)

    def handle_key_press(self, key):
        self._pressed_keys.add(key)
        self._update_key_display()

    def handle_key_release(self, key):
        self._pressed_keys.discard(key)
        self._update_key_display()

    def _update_key_display(self):
        if not self._pressed_keys:
            self._key_label.setText("Keys: --")
            return
        key_names = {
            Qt.Key_W: "W↑", Qt.Key_S: "S↓", Qt.Key_A: "A←", Qt.Key_D: "D→",
            Qt.Key_Q: "Q⟲", Qt.Key_E: "E⟳", Qt.Key_R: "R▲", Qt.Key_F: "F▼"
        }
        active = [key_names.get(k, "") for k in self._pressed_keys if k in key_names]
        self._key_label.setText("Keys: " + " ".join(active) if active else "Keys: --")

    # ── Connection ────────────────────────────────────────

    def set_connected(self, connected):
        self._connect_btn.setEnabled(not connected)
        self._disconnect_btn.setEnabled(connected)
        if connected:
            self._connect_btn.setText("Connected")
            self._connect_btn.setStyleSheet(self._btn_style("#4CAF50"))
        else:
            self._connect_btn.setText("Connect")
            self._connect_btn.setStyleSheet(self._btn_style("#2196F3"))

    def _on_connect(self):
        if self._serial_radio.isChecked():
            port = self._com_combo.currentData()
            baud = self._baud_combo.currentText()
            if port:
                conn_str = f"{port},{baud}"
            else:
                return
        else:
            conn_str = self._udp_input.text().strip()
        self.connect_requested.emit(conn_str)

    def _on_avoidance_toggle(self, checked):
        self._avoid_btn.setText("Avoidance ON" if checked else "Enable Avoidance")
        self.avoidance_toggled.emit(checked)

    # ── Recording ─────────────────────────────────────────

    def _on_rec_toggle(self, checked):
        """Internal handler — emits recording_toggled so main_window can start/stop recorder."""
        if checked:
            self._rec_btn.setText("■ Stop Recording")
            self._rec_status.setText("Starting...")
            self._rec_status.setStyleSheet("color: #f44; font-size: 10px; font-weight: bold;")
        else:
            self._rec_btn.setText("● Start Recording")
            self._rec_status.setText("Not recording")
            self._rec_status.setStyleSheet("color: #888; font-size: 10px;")
        self.recording_toggled.emit(checked)

    def set_recording_path(self, path):
        """Called from main_window to display current recording session path."""
        self._rec_status.setText(f"Recording → {path}")
        self._rec_status.setStyleSheet("color: #f44; font-size: 10px; font-weight: bold;")

    # ── Styles ────────────────────────────────────────────

    @staticmethod
    def _group_style():
        return """
            QGroupBox {
                font-weight: bold; font-size: 11px; color: #0af;
                border: 1px solid #333; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        """

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{
                background-color: {color}; color: white; border: none;
                padding: 5px 10px; border-radius: 3px; font-weight: bold;
                font-size: 11px;
            }}
            QPushButton:hover {{ background-color: {color}; opacity: 0.8; }}
            QPushButton:pressed {{ background-color: #333; }}
            QPushButton:disabled {{ background-color: #555; color: #999; }}
        """