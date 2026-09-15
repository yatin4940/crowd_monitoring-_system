import sys
import cv2

import config
from camera import Camera
from detector import PersonDetector
from tracker import CentroidTracker
from counter import LineCounter
from zone_manager import ZoneManager
from alert_manager import AlertManager
from preprocessor import FramePreprocessor
from heatmap import CrowdHeatmap
from motion_analyzer import MotionAnalyzer
from crowd_analyzer import CrowdAnalyzer
from utils import (
    FPSCounter,
    draw_boxes_and_ids,
    draw_counting_line,
    draw_zones,
    draw_statistics,
    draw_alerts,
    draw_cluster_centres,
    draw_analytics_overlay,
    draw_motion_count,
    save_snapshot,
)


def main():
    print("Starting Smart Crowd Monitoring System...")
    print("Controls: Q=quit  R=reset  S=snapshot  H=heatmap  M=motion  A=analytics  E=edges")

    try:
        detector = PersonDetector()
    except RuntimeError as exc:
        print(f"Error initialising detector: {exc}")
        sys.exit(1)

    tracker = CentroidTracker()
    counter = LineCounter()
    zone_manager = ZoneManager()
    alert_manager = AlertManager()
    preprocessor = FramePreprocessor()
    motion_analyzer = MotionAnalyzer()
    crowd_analyzer = CrowdAnalyzer()
    fps_counter = FPSCounter()

    heatmap = None

    show_heatmap = config.HEATMAP_ENABLED
    show_motion = config.MOTION_ANALYSIS_ENABLED
    show_analytics = config.SHOW_ANALYTICS_PANEL
    show_edges = config.PREPROCESS_SHOW_EDGES

    try:
        camera = Camera(
            config.CAMERA_SOURCE,
            frame_width=config.FRAME_WIDTH,
            frame_height=config.FRAME_HEIGHT,
        ).open()
    except RuntimeError as exc:
        print(f"Error opening camera/video: {exc}")
        sys.exit(1)

    consecutive_read_failures = 0
    max_read_failures = 30

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                consecutive_read_failures += 1

                if consecutive_read_failures >= max_read_failures:
                    print("Too many failed frame reads in a row. Stopping.")
                    break

                continue

            consecutive_read_failures = 0

            if heatmap is None:
                h, w = frame.shape[:2]
                heatmap = CrowdHeatmap(w, h)

            preprocessor.show_edges = show_edges
            processed_frame = preprocessor.process(frame.copy())

            try:
                detections = detector.detect(processed_frame)
            except Exception as exc:
                print(f"Detection error on this frame, skipping: {exc}")
                detections = []

            tracked_objects = tracker.update(detections)

            counter.update(tracked_objects)

            zone_status = zone_manager.update(tracked_objects)

            alerts = alert_manager.update(zone_status)

            fg_mask, motion_regions, corner_pts = motion_analyzer.update(frame)

            heatmap.update(tracked_objects)

            analytics = crowd_analyzer.update(
                tracked_objects,
                zone_status,
                counter.inside
            )

            display = frame.copy()

            if show_heatmap:
                display = heatmap.overlay(display)

            if show_motion and fg_mask is not None:
                display = motion_analyzer.draw_motion_overlay(
                    display,
                    fg_mask,
                    corner_pts,
                    show_corners=config.MOTION_SHOW_CORNERS
                )

            display = preprocessor.edge_overlay(display)

            display = draw_boxes_and_ids(display, tracked_objects)

            display = draw_counting_line(
                display,
                config.COUNT_LINE_Y,
                config.COUNT_LINE_X_START,
                config.COUNT_LINE_X_END,
            )

            display = draw_zones(display, zone_status, config.ZONES)

            fps = fps_counter.update()

            display = draw_statistics(
                display,
                counter.inside,
                counter.entered,
                counter.exited,
                fps
            )

            display = draw_alerts(display, alerts)
            display = draw_motion_count(display, motion_regions)

            if show_analytics and analytics:
                display = draw_cluster_centres(
                    display,
                    analytics.get("clusters", [])
                )
                display = draw_analytics_overlay(display, analytics)

            cv2.imshow(config.WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quitting...")
                break

            elif key == ord("r"):
                counter.reset()

                if heatmap:
                    heatmap.reset()

                print("Counters and heatmap reset.")

            elif key == ord("s"):
                filename = save_snapshot(display)
                print(f"Snapshot saved: {filename}")

            elif key == ord("h"):
                show_heatmap = not show_heatmap
                print(f"Heatmap overlay: {'ON' if show_heatmap else 'OFF'}")

            elif key == ord("m"):
                show_motion = not show_motion
                print(f"Motion overlay: {'ON' if show_motion else 'OFF'}")

            elif key == ord("a"):
                show_analytics = not show_analytics
                print(f"Analytics overlay: {'ON' if show_analytics else 'OFF'}")

            elif key == ord("e"):
                show_edges = not show_edges
                print(f"Edge overlay: {'ON' if show_edges else 'OFF'}")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()