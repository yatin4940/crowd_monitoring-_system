import time
import os
import cv2
import numpy as np

from config import (
    BOX_COLOR,
    TRACK_ID_COLOR,
    LINE_COLOR,
    ZONE_COLOR_NORMAL,
    ZONE_COLOR_OVER,
    TEXT_COLOR,
    ALERT_COLOR,
    CLUSTER_DOT_COLOR,
    SNAPSHOT_DIR,
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


class FPSCounter:
    def __init__(self):
        self._prev_time = time.time()
        self.fps = 0.0

    def update(self):
        current_time = time.time()
        elapsed = current_time - self._prev_time
        self._prev_time = current_time

        if elapsed > 0:
            self.fps = 1.0 / elapsed

        return self.fps


def draw_boxes_and_ids(frame, tracked_objects):
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox[:4]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            BOX_COLOR,
            2
        )

        label = f"ID {obj.object_id}"

        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 8)),
            FONT,
            0.5,
            TRACK_ID_COLOR,
            2
        )

        cv2.circle(
            frame,
            obj.centroid,
            3,
            TRACK_ID_COLOR,
            -1
        )

    return frame


def draw_counting_line(frame, line_y, x_start, x_end):
    cv2.line(
        frame,
        (x_start, line_y),
        (x_end, line_y),
        LINE_COLOR,
        2
    )

    return frame


def draw_zones(frame, zone_status, zones_config):
    for name, zone_box in zones_config.items():
        x1, y1, x2, y2, _ = zone_box

        info = zone_status.get(name, {})
        overcrowded = info.get("overcrowded", False)

        color = (
            ZONE_COLOR_OVER
            if overcrowded
            else ZONE_COLOR_NORMAL
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        count = info.get("count", 0)
        max_capacity = info.get("max", 0)

        label = f"{name}: {count}/{max_capacity}"

        if overcrowded:
            label += " - OVERCROWDED"

        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 8)),
            FONT,
            0.5,
            color,
            2
        )

    return frame


def draw_statistics(frame, inside, entered, exited, fps):
    lines = [
        f"TOTAL INSIDE: {inside}",
        f"ENTERED: {entered}",
        f"EXITED: {exited}",
        f"FPS: {fps:.1f}",
    ]

    y = 25

    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            FONT,
            0.6,
            TEXT_COLOR,
            2
        )
        y += 25

    return frame


def draw_alerts(frame, alerts):
    y = frame.shape[0] - 15

    for message in reversed(alerts):
        cv2.putText(
            frame,
            message,
            (10, y),
            FONT,
            0.6,
            ALERT_COLOR,
            2
        )
        y -= 25

    return frame


def draw_cluster_centres(frame, cluster_centres):
    for cx, cy in cluster_centres:
        cx, cy = int(cx), int(cy)

        cv2.circle(
            frame,
            (cx, cy),
            10,
            CLUSTER_DOT_COLOR,
            2
        )

        cv2.line(
            frame,
            (cx - 14, cy),
            (cx + 14, cy),
            CLUSTER_DOT_COLOR,
            2
        )

        cv2.line(
            frame,
            (cx, cy - 14),
            (cx, cy + 14),
            CLUSTER_DOT_COLOR,
            2
        )

    return frame


def draw_analytics_overlay(frame, analytics):
    if not analytics:
        return frame

    scene = analytics.get("scene_state", "")
    spread = analytics.get("spread_index", 0.0)
    dlabels = analytics.get("density_labels", {})

    state_color = {
        "Safe": (0, 200, 0),
        "Caution": (0, 180, 255),
        "Danger": (0, 0, 255),
    }.get(scene, TEXT_COLOR)

    x_offset = frame.shape[1] - 260
    y = 25

    cv2.putText(
        frame,
        f"Scene: {scene}",
        (x_offset, y),
        FONT,
        0.6,
        state_color,
        2
    )

    y += 25

    cv2.putText(
        frame,
        f"Spread: {spread:.2f}",
        (x_offset, y),
        FONT,
        0.55,
        TEXT_COLOR,
        1
    )

    y += 22

    for zone_name, label in dlabels.items():
        cv2.putText(
            frame,
            f"{zone_name}: {label}",
            (x_offset, y),
            FONT,
            0.5,
            TEXT_COLOR,
            1
        )
        y += 20

    return frame


def draw_motion_count(frame, motion_regions):
    text = f"Motion blobs: {motion_regions}"

    cv2.putText(
        frame,
        text,
        (10, frame.shape[0] - 45),
        FONT,
        0.55,
        (200, 200, 0),
        1
    )

    return frame


def save_snapshot(frame):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(
        SNAPSHOT_DIR,
        f"snapshot_{timestamp}.jpg"
    )

    cv2.imwrite(filename, frame)

    return filename