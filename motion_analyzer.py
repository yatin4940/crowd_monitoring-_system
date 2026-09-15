import cv2
import numpy as np


class MotionAnalyzer:
    def __init__(self):
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=40,
            detectShadows=False,
        )

        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

    def update(self, frame):
        if frame is None or frame.size == 0:
            return None, 0, []

        fg_mask = self._bg_subtractor.apply(frame)

        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_OPEN,
            self._morph_kernel
        )

        fg_mask = cv2.morphologyEx(
            fg_mask,
            cv2.MORPH_CLOSE,
            self._morph_kernel
        )

        contours, _ = cv2.findContours(
            fg_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        motion_regions = sum(
            1 for c in contours if cv2.contourArea(c) > 400
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=80,
            qualityLevel=0.1,
            minDistance=10,
            mask=fg_mask,
        )

        corner_pts = []

        if corners is not None:
            corner_pts = [
                (int(pt[0][0]), int(pt[0][1]))
                for pt in corners
            ]

        return fg_mask, motion_regions, corner_pts

    def draw_motion_overlay(
        self,
        frame,
        fg_mask,
        corner_pts,
        show_corners=False
    ):
        if fg_mask is None:
            return frame

        red_layer = np.zeros_like(frame)
        red_layer[fg_mask > 0] = (0, 0, 180)

        result = cv2.addWeighted(
            frame,
            1.0,
            red_layer,
            0.3,
            0
        )

        if show_corners:
            for x, y in corner_pts:
                cv2.circle(
                    result,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )

        return result