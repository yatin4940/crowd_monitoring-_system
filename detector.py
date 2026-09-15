from config import DETECTION_MODEL, CONFIDENCE_THRESHOLD, PERSON_CLASS_ID


class PersonDetector:
    def __init__(self, model_path=DETECTION_MODEL, confidence=CONFIDENCE_THRESHOLD):
        self.confidence = confidence
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The 'ultralytics' package is required for detection. "
                "Install it with: pip install ultralytics"
            ) from exc

        try:
            model = YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load detection model '{model_path}'. "
                "Check your internet connection (needed once, to download "
                "the pretrained weights) and that the model name is valid."
            ) from exc

        return model

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        detections = []

        if not results:
            return detections

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            confidence = float(box.conf[0])

            if confidence < self.confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                (int(x1), int(y1), int(x2), int(y2), confidence)
            )

        return detections