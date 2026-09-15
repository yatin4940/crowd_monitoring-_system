import cv2


class Camera:
    def __init__(self, source, frame_width=None, frame_height=None):
        self.source = source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video source '{self.source}'. "
                "Check that the webcam index is correct and the camera is "
                "not in use by another application, or that the video "
                "file path exists."
            )

        return self

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")

        ok, frame = self.cap.read()

        if not ok or frame is None:
            return False, None

        if self.frame_width and self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()