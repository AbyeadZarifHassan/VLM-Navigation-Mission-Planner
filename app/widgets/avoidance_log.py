"""Avoidance decision log widget."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QGroupBox
from PyQt5.QtCore import Qt


class AvoidanceLog(QWidget):
    """Displays a scrollable log of avoidance decisions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        group = QGroupBox("Avoidance Log")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 11px; color: #0af;
                border: 1px solid #333; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        """)
        group_layout = QVBoxLayout(group)

        self._state_label = QLabel("State: IDLE | Action: NONE")
        self._state_label.setStyleSheet(
            "color: #0f0; font-family: Consolas; font-size: 12px; "
            "background: #111; padding: 4px; border: 1px solid #333;"
        )
        group_layout.addWidget(self._state_label)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(120)
        self._log_text.setStyleSheet(
            "background: #111; color: #ccc; font-family: Consolas; "
            "font-size: 11px; border: 1px solid #333;"
        )
        group_layout.addWidget(self._log_text)

        layout.addWidget(group)

    def update_state(self, state, action):
        """Update the current state display."""
        color_map = {
            "IDLE": "#0f0",
            "OBSTACLE_DETECTED": "#f00",
            "HOVERING": "#ff0",
            "EXECUTING": "#f80",
            "COMPLETED": "#0af",
            "CLEARANCE": "#0c8",
        }
        color = color_map.get(state, "#ccc")
        self._state_label.setText(f"State: {state} | Action: {action}")
        self._state_label.setStyleSheet(
            f"color: {color}; font-family: Consolas; font-size: 12px; "
            f"background: #111; padding: 4px; border: 1px solid #333;"
        )

    def add_log_entry(self, text):
        """Append a log entry."""
        self._log_text.append(text)
        # Auto-scroll to bottom
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
