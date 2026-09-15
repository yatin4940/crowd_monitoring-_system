import cv2
import numpy as np

from config import HEATMAP_DECAY, HEATMAP_RADIUS, HEATMAP_ALPHA


class CrowdHeatmap:
    def __init__(
        self,
        width,
        height,
        decay=HEATMAP_DECAY,
        radius=HEATMAP_RADIUS,
        alpha=HEATMAP_ALPHA
    ):
        self.width = width
        self.height = height
        self.decay = decay
        self.radius = radius
        self.alpha = alpha
        self._buffer = np.zeros((height, width), dtype=np.float32)

    def update(self, tracked_objects):
        self._buffer *= self.decay

        for obj in tracked_objects:
            cx, cy = obj.centroid
            cx = int(np.clip(cx, 0, self.width - 1))
            cy = int(np.clip(cy, 0, self.height - 1))
            self._buffer[cy, cx] += 1.0

        k = self.radius * 2 + 1
        self._buffer = cv2.GaussianBlur(self._buffer, (k, k), 0)

    def overlay(self, frame):
        if self._buffer.max() < 1e-6:
            return frame

        normalised = cv2.normalize(
            self._buffer,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_JET)

        mask = (normalised > 10).astype(np.uint8)
        mask_3ch = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

        blended = cv2.addWeighted(
            frame,
            1.0 - self.alpha,
            coloured,
            self.alpha,
            0
        )

        result = np.where(mask_3ch > 0, blended, frame)
        return result.astype(np.uint8)

    def reset(self):
        self._buffer[:] = 0.0