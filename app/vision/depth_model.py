"""Depth Anything V2 Small for monocular depth estimation.

Returns 9-segment metric distances (3x3 grid) + a colorized depth map
with grid overlay and distance labels — used both for UI display and
as a visual reasoning aid for the VLM advisors.

Distance estimation uses the 90th percentile of disparity (was p75)
so that obstacles filling even ~10% of a cell (thin objects, edges,
partial occlusion) are detected as close, instead of being averaged
out by the background.
"""

import numpy as np
import torch
from transformers import pipeline


# 3x3 grid layout — top-left to bottom-right, row-major
SEGMENT_GRID = [
    ["upper_left", "up",   "upper_right"],
    ["left",       "center", "right"],
    ["lower_left", "down", "lower_right"],
]
SEGMENT_NAMES = [name for row in SEGMENT_GRID for name in row]


class DepthEstimator:
    """Wraps Depth Anything V2 Small for depth map inference."""

    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
    INFERENCE_SIZE = (320, 240)  # smaller = faster, less accurate

    # Percentile of disparity used to estimate the closest object in a cell.
    # Higher = more sensitive to small/partial obstacles.
    #   p75: catches obstacles filling >25% of cell (background-biased)
    #   p90: catches obstacles filling >10% of cell (RECOMMENDED)
    #   p95: catches obstacles filling >5%  of cell (sensitive, more noise)
    OBSTACLE_PERCENTILE = 90

    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.pipe = None
        # metric_distance = depth_scale / raw_disparity
        # To recalibrate: place an object at a known distance, read disparity,
        # set depth_scale = known_distance * raw_disparity.
        self.depth_scale = 2.0

    def load(self):
        """Load the depth estimation pipeline."""
        self.pipe = pipeline(
            "depth-estimation",
            model=self.MODEL_ID,
            device=self.device,
        )
        print("[DEPTH] Model loaded on", self.device)

    # ── Inference ──────────────────────────────────────────

    def estimate(self, rgb_frame):
        """Run depth estimation on an RGB frame.

        Args:
            rgb_frame: numpy (H, W, 3) uint8 RGB.

        Returns:
            depth_normalized: (H, W) float32 in [0,1] (for visualization).
            distances_9seg: dict mapping all 9 segment names to metric
                distances in meters (or None if invalid). Includes
                "center", "left", "right" keys for backward compat.
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import cv2
        from PIL import Image

        small = cv2.resize(rgb_frame, self.INFERENCE_SIZE)
        image = Image.fromarray(small)
        result = self.pipe(image)

        raw_disparity = result["predicted_depth"].numpy().squeeze()
        if raw_disparity.shape[:2] != rgb_frame.shape[:2]:
            raw_disparity = cv2.resize(
                raw_disparity, (rgb_frame.shape[1], rgb_frame.shape[0])
            )

        # Normalize for visualization
        d_min, d_max = raw_disparity.min(), raw_disparity.max()
        if d_max - d_min > 0:
            depth_normalized = (raw_disparity - d_min) / (d_max - d_min)
        else:
            depth_normalized = np.zeros_like(raw_disparity)

        # 9-segment metric distances
        distances_9seg = {}
        for r in range(3):
            for c in range(3):
                name = SEGMENT_GRID[r][c]
                distances_9seg[name] = self._estimate_segment_distance(
                    raw_disparity, r, c
                )

        return depth_normalized, distances_9seg

    def _estimate_segment_distance(self, raw_disparity, row, col):
        """Distance to closest meaningful object in a 3x3 grid cell (meters).

        row, col ∈ {0, 1, 2} — top-left = (0,0), bottom-right = (2,2).
        Uses OBSTACLE_PERCENTILE of disparity so partial / small obstacles
        in the cell are detected, not averaged away by background.
        """
        h, w = raw_disparity.shape
        y0 = row * (h // 3)
        y1 = (row + 1) * (h // 3) if row < 2 else h
        x0 = col * (w // 3)
        x1 = (col + 1) * (w // 3) if col < 2 else w

        roi = raw_disparity[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        disp_pct = float(np.percentile(roi, self.OBSTACLE_PERCENTILE))
        if disp_pct < 0.01:
            return 20.0
        distance = self.depth_scale / disp_pct
        return float(np.clip(distance, 0.1, 20.0))

    # ── Visualization ──────────────────────────────────────

    def colorize_depth(self, depth_normalized, distances_9seg=None):
        """Convert normalized depth map to a colorized RGB image.

        If distances_9seg is provided, overlays a 3x3 grid with per-segment
        distance labels (color-coded: red < 1m, yellow < 2m, green ≥ 2m).
        This image is used both for UI display and as the visual reference
        sent to the VLM advisor.
        """
        import cv2
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        colorized = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

        if distances_9seg is not None:
            self._draw_grid_overlay(rgb, distances_9seg)

        return rgb

    def _draw_grid_overlay(self, image, distances_9seg):
        """Draw 3x3 grid lines + distance labels on the image (in-place)."""
        import cv2
        h, w = image.shape[:2]
        h3, w3 = h // 3, w // 3

        # Grid lines
        for i in (1, 2):
            cv2.line(image, (i * w3, 0), (i * w3, h), (255, 255, 255), 1)
            cv2.line(image, (0, i * h3), (w, i * h3), (255, 255, 255), 1)

        # Per-segment distance labels (top-left of each cell)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for r in range(3):
            for c in range(3):
                name = SEGMENT_GRID[r][c]
                d = distances_9seg.get(name)
                if d is None:
                    continue
                x = c * w3 + 6
                y = r * h3 + 18
                text = f"{d:.1f}m"
                (tw, th), _ = cv2.getTextSize(text, font, 0.45, 1)

                # Color by proximity
                if d < 1.0:
                    color = (255, 80, 80)        # red — close
                elif d < 2.0:
                    color = (255, 220, 80)       # yellow — caution
                else:
                    color = (120, 255, 120)      # green — clear

                # Background box for legibility
                cv2.rectangle(
                    image,
                    (x - 2, y - th - 2),
                    (x + tw + 2, y + 3),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(image, text, (x, y), font, 0.45, color, 1, cv2.LINE_AA)