"""Camera capture using OpenCV."""

import cv2
import numpy as np


class Camera:
    """Manages USB webcam capture."""

    def __init__(self, device_id=0, width=None, height=None):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        """Open the camera device."""
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device_id}")
        if self.width and self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always get latest frame
        return True

    def read_frame(self):
        """Read a single frame. Returns RGB numpy array or None."""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()