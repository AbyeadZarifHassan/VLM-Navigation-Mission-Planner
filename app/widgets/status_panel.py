"""Drone telemetry status panel."""

import math
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QLabel, QGroupBox
)
from PyQt5.QtCore import Qt


class StatusPanel(QWidget):
    """Displays drone telemetry data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        group = QGroupBox("Drone Status")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 13px; color: #0af;
                border: 1px solid #333; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        """)
        grid = QGridLayout(group)
        grid.setSpacing(4)

        self._labels = {}
        fields = [
            ("Mode", "mode"), ("Armed", "armed"),
            ("Altitude", "altitude"), ("Heading", "heading"),
            ("Ground Speed", "groundspeed"), ("Air Speed", "airspeed"),
            ("Battery", "battery"), ("GPS", "gps"),
            ("Roll", "roll"), ("Pitch", "pitch"),
            ("Yaw", "yaw_deg"), ("Lat/Lon", "latlon"),
            ("Motors PWM M1-M4", "motor_pwm"), ("Mission", "mission_wp"),
        ]
        label_style = "color: #888; font-size: 11px;"
        value_style = "color: #eee; font-size: 12px; font-family: Consolas;"

        for i, (display_name, key) in enumerate(fields):
            row, col = divmod(i, 2)
            name_label = QLabel(f"{display_name}:")
            name_label.setStyleSheet(label_style)
            val_label = QLabel("--")
            val_label.setStyleSheet(value_style)
            grid.addWidget(name_label, row, col * 2)
            grid.addWidget(val_label, row, col * 2 + 1)
            self._labels[key] = val_label

        layout.addWidget(group)
        layout.addStretch()

    def update_telemetry(self, data):
        """Update labels with telemetry dict."""
        self._labels["mode"].setText(data.get("mode", "--"))

        armed = data.get("armed", False)
        self._labels["armed"].setText("ARMED" if armed else "DISARMED")
        self._labels["armed"].setStyleSheet(
            f"color: {'#f44' if armed else '#4f4'}; font-size: 12px; font-family: Consolas; font-weight: bold;"
        )

        self._labels["altitude"].setText(f"{data.get('altitude', 0):.1f} m")
        self._labels["heading"].setText(f"{data.get('heading', 0)}°")
        self._labels["groundspeed"].setText(f"{data.get('groundspeed', 0):.1f} m/s")
        self._labels["airspeed"].setText(f"{data.get('airspeed', 0):.1f} m/s")

        batt_v = data.get("battery_voltage", 0)
        batt_pct = data.get("battery_remaining", -1)
        batt_text = f"{batt_v:.1f}V"
        if batt_pct >= 0:
            batt_text += f" ({batt_pct}%)"
        self._labels["battery"].setText(batt_text)

        gps_fix = data.get("gps_fix", 0)
        gps_sat = data.get("gps_satellites", 0)
        fix_names = {0: "No GPS", 1: "No Fix", 2: "2D", 3: "3D", 4: "DGPS", 5: "RTK Float", 6: "RTK Fixed"}
        self._labels["gps"].setText(f"{fix_names.get(gps_fix, '?')} ({gps_sat} sats)")

        self._labels["roll"].setText(f"{math.degrees(data.get('roll', 0)):.1f}°")
        self._labels["pitch"].setText(f"{math.degrees(data.get('pitch', 0)):.1f}°")
        self._labels["yaw_deg"].setText(f"{math.degrees(data.get('yaw', 0)):.1f}°")

        lat = data.get("lat", 0)
        lon = data.get("lon", 0)
        self._labels["latlon"].setText(f"{lat:.6f}, {lon:.6f}")

        # Motor PWM — shows if commands are actually affecting motors
        m1, m2, m3, m4 = data.get("motor_pwm", (0, 0, 0, 0))
        self._labels["motor_pwm"].setText(f"{m1} {m2} {m3} {m4}")
        motors_spinning = any(p > 1100 for p in (m1, m2, m3, m4))
        pwm_color = "#4f4" if motors_spinning else "#888"
        self._labels["motor_pwm"].setStyleSheet(
            f"color: {pwm_color}; font-size: 11px; font-family: Consolas;"
        )

        # Mission progress — only shows when in AUTO mode
        mode = data.get("mode", "")
        wp = data.get("mission_current_wp", 0)
        if mode == "AUTO":
            self._labels["mission_wp"].setText(f"WP {wp} (active)")
            self._labels["mission_wp"].setStyleSheet(
                "color: #0f0; font-size: 12px; font-family: Consolas; font-weight: bold;"
            )
        else:
            self._labels["mission_wp"].setText("--")
            self._labels["mission_wp"].setStyleSheet(
                "color: #888; font-size: 12px; font-family: Consolas;"
            )
