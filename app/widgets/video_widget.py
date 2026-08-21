"""RGB video display widget with decision overlay."""

import numpy as np
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor
from PyQt5.QtCore import Qt


class VideoWidget(QWidget):
    """Displays RGB video feed with avoidance decision overlay."""

    def __init__(self, title="RGB Video", parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(2, 2, 2, 2)

        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignCenter)
        self._title_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #ccc;")
        self._layout.addWidget(self._title_label)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(320, 200)
        self._image_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        self._layout.addWidget(self._image_label)

        self._overlay_text = ""
        self._overlay_color = QColor(0, 255, 0)
        self._distance_text = ""
        self._command_text = ""

    def update_frame(self, rgb_frame):
        """Update display with new RGB frame (numpy H,W,3 uint8)."""
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        if self._overlay_text or self._distance_text or self._command_text:
            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.Antialiasing)

            if self._distance_text:
                font = QFont("Consolas", 14, QFont.Bold)
                painter.setFont(font)
                painter.setPen(QColor(255, 255, 0))
                painter.drawText(
                    scaled.rect().adjusted(10, 10, -10, 0),
                    Qt.AlignHCenter | Qt.AlignTop,
                    self._distance_text
                )

            if self._command_text:
                font = QFont("Consolas", 12, QFont.Bold)
                painter.setFont(font)
                fm = painter.fontMetrics()

                lines = self._command_text.split("\n")
                y_offset = 35
                for line in lines:
                    text_rect = fm.boundingRect(line)
                    bg_rect = text_rect.adjusted(-6, -2, 6, 2)
                    bg_rect.moveTo(8, y_offset)
                    painter.fillRect(bg_rect, QColor(0, 0, 0, 180))

                    if "vel=" in line:
                        painter.setPen(QColor(0, 200, 255))
                    else:
                        painter.setPen(QColor(0, 255, 100))
                    painter.drawText(10, y_offset + fm.ascent(), line)
                    y_offset += fm.height() + 4

            if self._overlay_text:
                font = QFont("Consolas", 16, QFont.Bold)
                painter.setFont(font)

                fm = painter.fontMetrics()
                text_rect = fm.boundingRect(self._overlay_text)
                bg_rect = text_rect.adjusted(-10, -5, 10, 5)
                bg_rect.moveCenter(scaled.rect().center())
                bg_rect.moveBottom(scaled.height() - 20)

                painter.fillRect(bg_rect, QColor(0, 0, 0, 160))
                painter.setPen(self._overlay_color)
                painter.drawText(bg_rect, Qt.AlignCenter, self._overlay_text)

            painter.end()

        self._image_label.setPixmap(scaled)

    def set_overlay(self, text, color=None):
        """Set the decision overlay text."""
        self._overlay_text = text
        if color:
            self._overlay_color = color

    def set_distance_text(self, text):
        """Set the distance display text."""
        self._distance_text = text

    def set_command_text(self, text):
        """Set the velocity command text overlay (top-left)."""
        self._command_text = text

    def clear_overlay(self):
        self._overlay_text = ""
        self._distance_text = ""
        self._command_text = ""