"""GPT-4o command log widget — shows sent prompts and received decisions."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QGroupBox
from PyQt5.QtCore import Qt


class GPTLog(QWidget):
    """Displays GPT-4o communication log: sent prompts, received commands, errors."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        group = QGroupBox("GPT-4o Advisor")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; font-size: 11px; color: #0af;
                border: 1px solid #333; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
        """)
        group_layout = QVBoxLayout(group)

        # Status label
        self._status_label = QLabel("Status: IDLE")
        self._status_label.setStyleSheet(
            "color: #888; font-family: Consolas; font-size: 12px; "
            "background: #111; padding: 4px; border: 1px solid #333;"
        )
        group_layout.addWidget(self._status_label)

        # Log text area
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(120)
        self._log_text.setStyleSheet(
            "background: #111; color: #ccc; font-family: Consolas; "
            "font-size: 11px; border: 1px solid #333;"
        )
        group_layout.addWidget(self._log_text)

        layout.addWidget(group)

    def add_sent(self, text):
        """Log a sent prompt (cyan)."""
        import time
        ts = time.strftime("%H:%M:%S")
        self._log_text.append(
            f'<span style="color: #0af;">[{ts}] &gt;&gt; {text}</span>'
        )
        self._status_label.setText("Status: WAITING for GPT-4o...")
        self._status_label.setStyleSheet(
            "color: #ff0; font-family: Consolas; font-size: 12px; "
            "background: #111; padding: 4px; border: 1px solid #333;"
        )
        self._auto_scroll()

    def add_received(self, text):
        """Log a received GPT command (green)."""
        import time
        ts = time.strftime("%H:%M:%S")
        self._log_text.append(
            f'<span style="color: #0f0;">[{ts}] &lt;&lt; {text}</span>'
        )
        self._status_label.setText("Status: COMMAND RECEIVED")
        self._status_label.setStyleSheet(
            "color: #0f0; font-family: Consolas; font-size: 12px; "
            "background: #111; padding: 4px; border: 1px solid #333;"
        )
        self._auto_scroll()

    def add_error(self, text):
        """Log an error (red)."""
        import time
        ts = time.strftime("%H:%M:%S")
        self._log_text.append(
            f'<span style="color: #f44;">[{ts}] ERROR: {text}</span>'
        )
        self._status_label.setText("Status: ERROR — using fallback")
        self._status_label.setStyleSheet(
            "color: #f44; font-family: Consolas; font-size: 12px; "
            "background: #111; padding: 4px; border: 1px solid #333;"
        )
        self._auto_scroll()

    def _auto_scroll(self):
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
