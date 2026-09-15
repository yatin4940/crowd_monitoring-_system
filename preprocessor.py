import cv2
import numpy as np

from config import (
    PREPROCESS_EQUALIZE_HIST,
    PREPROCESS_DENOISE,
    PREPROCESS_GAMMA,
    PREPROCESS_MORPH_OPEN,
    PREPROCESS_SHOW_EDGES,
)


def _apply_gamma(frame, gamma=1.2):
    inv_gamma = 1.0 / gamma
    lut = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(frame, lut)


def _apply_hist_equalization(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _apply_morph_open(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)


def _compute_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    return edges


class FramePreprocessor:
    def __init__(
        self,
        equalize=PREPROCESS_EQUALIZE_HIST,
        denoise=PREPROCESS_DENOISE,
        gamma=PREPROCESS_GAMMA,
        morph_open=PREPROCESS_MORPH_OPEN,
        show_edges=PREPROCESS_SHOW_EDGES,
    ):
        self.equalize = equalize
        self.denoise = denoise
        self.gamma = gamma
        self.morph_open = morph_open
        self.show_edges = show_edges

    def process(self, frame):
        if frame is None or frame.size == 0:
            return frame

        if self.denoise:
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

        if self.gamma and self.gamma != 1.0:
            frame = _apply_gamma(frame, self.gamma)

        if self.equalize:
            frame = _apply_hist_equalization(frame)

        if self.morph_open:
            frame = _apply_morph_open(frame)

        return frame

    def edge_overlay(self, frame):
        if not self.show_edges:
            return frame

        edges = _compute_edges(frame)
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_bgr[edges > 0] = (255, 255, 0)

        return cv2.addWeighted(
            frame,
            0.8,
            edge_bgr,
            0.2,
            0,
        )