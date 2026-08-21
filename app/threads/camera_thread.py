"""QThread for camera capture and depth estimation."""

import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class CameraThread(QThread):
    """Captures camera frames and runs depth estimation.

    Display runs at ~30 FPS. Depth inference runs at a lower rate
    (DEPTH_FPS) to keep the UI smooth on CPU-only machines.
    """

    frame_ready = pyqtSignal(np.ndarray, np.ndarray, object, object)
    # Signals: (rgb_frame, depth_colorized, center_distance, distances_dict)

    error_occurred = pyqtSignal(str)

    DISPLAY_FPS = 30
    DEPTH_FPS = 3  # depth inference rate — raise if you have a GPU

    def __init__(self, camera, depth_estimator, parent=None):
        super().__init__(parent)
        self.camera = camera
        self.depth_estimator = depth_estimator
        self._running = False

        # Cached depth output — reused between inference frames
        self._last_depth_colored = None
        self._last_center_dist = None
        self._last_distances = None

    def run(self):
        self._running = True
        display_interval = 1.0 / self.DISPLAY_FPS
        depth_interval = 1.0 / self.DEPTH_FPS

        last_display = 0.0
        last_depth = 0.0

        while self._running:
            now = time.monotonic()

            # Throttle display to DISPLAY_FPS
            if now - last_display < display_interval:
                self.msleep(5)
                continue

            try:
                frame = self.camera.read_frame()
                if frame is None:
                    self.msleep(30)
                    continue

                last_display = time.monotonic()

                # Run depth inference only at DEPTH_FPS
                if (self.depth_estimator.pipe is not None and
                        now - last_depth >= depth_interval):
                    depth_norm, distances_9seg = self.depth_estimator.estimate(frame)
                    self._last_depth_colored = self.depth_estimator.colorize_depth(
                        depth_norm, distances_9seg
                    )
                    self._last_center_dist = distances_9seg.get("center")
                    self._last_distances = distances_9seg
                    last_depth = time.monotonic()
                    if self._last_center_dist is not None:
                        L = distances_9seg.get("left") or 0
                        R = distances_9seg.get("right") or 0
                        print(f"[DEPTH] center={self._last_center_dist:.2f}m  "
                              f"left={L:.2f}m  right={R:.2f}m")

                depth_colored = (
                    self._last_depth_colored
                    if self._last_depth_colored is not None
                    else np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
                )

                self.frame_ready.emit(
                    frame, depth_colored,
                    self._last_center_dist, self._last_distances,
                )

            except Exception as e:
                self.error_occurred.emit(str(e))

    def stop(self):
        self._running = False
        self.wait(3000)