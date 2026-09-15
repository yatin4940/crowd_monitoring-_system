# Smart Crowd Monitoring System

A real-time computer vision system that detects, tracks, and analyses
crowd density from a webcam or video file.  Built with Python, OpenCV,
and YOLOv8, it covers all five course modules of the Computer Vision
curriculum.

---

## Technologies Used

- Python 3.9+
- OpenCV (`opencv-python`)
- Ultralytics YOLOv8 (`ultralytics`)
- NumPy

---

## Installation & Running

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/smart-crowd-monitoring.git
cd smart-crowd-monitoring

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (default: webcam index 0)
python main.py

# 4. To use a video file instead, edit config.py:
#    CAMERA_SOURCE = "videos/sample.mp4"
```

The YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on first
run by the ultralytics library — an internet connection is required once.

---

## Keyboard Controls

| Key | Action |
|---|---|
| Q | Quit |
| R | Reset counters and heatmap |
| S | Save snapshot to `snapshots/` folder |
| H | Toggle heatmap overlay |
| M | Toggle motion overlay |
| A | Toggle analytics overlay |
| E | Toggle edge overlay (debug) |

---

## Project Structure

```
smart-crowd-monitoring/
├── main.py            # Entry point, pipeline orchestration
├── config.py          # All tunable parameters in one place
├── camera.py          # Camera/video source abstraction
├── preprocessor.py    # Image enhancement pipeline (M2, M3)
├── detector.py        # YOLOv8 person detection (M5)
├── tracker.py         # Centroid-based multi-object tracker (M5)
├── counter.py         # Entry/exit line crossing counter
├── zone_manager.py    # Rectangular zone occupancy monitoring
├── alert_manager.py   # Overcrowding alert with cooldown
├── heatmap.py         # Crowd density heatmap (M4)
├── motion_analyzer.py # Background subtraction + corner detection (M3)
├── crowd_analyzer.py  # K-Means, KNN, PCA, Naive Bayes analytics (M4, M5)
├── utils.py           # Drawing helpers, FPS counter, snapshot saver
├── requirements.txt
├── README.md
└── statement.md
```

---

## Testing

Run the system on the included sample or a webcam and verify:

1. Bounding boxes appear around detected people
2. Track IDs remain stable as people move
3. Entry/exit counters increment when someone crosses the line
4. Zone labels turn red when the zone exceeds its capacity
5. Heatmap brightens in areas where people linger
6. Analytics panel shows correct scene state

