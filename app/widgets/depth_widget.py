"""Depth map display widget."""

import numpy as np
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class DepthWidget(QWidget):
    """Displays the colorized depth map."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        title = QLabel("Depth Map")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 12px; color: #ccc;")
        layout.addWidget(title)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(320, 240)
        self._image_label.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")
        layout.addWidget(self._image_label)

    def update_depth(self, depth_colored):
        """Update with colorized depth map (H, W, 3) uint8 RGB."""
        h, w, ch = depth_colored.shape
        bytes_per_line = ch * w
        qimg = QImage(depth_colored.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self._image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(scaled)
