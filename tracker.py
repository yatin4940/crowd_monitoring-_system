import numpy as np
from collections import OrderedDict

from config import MAX_DISAPPEARED, MAX_TRACK_DISTANCE


class TrackedObject:
    def __init__(self, object_id, centroid, bbox):
        self.object_id = object_id
        self.centroid = centroid
        self.bbox = bbox
        self.disappeared = 0


class CentroidTracker:
    def __init__(
        self,
        max_disappeared=MAX_DISAPPEARED,
        max_distance=MAX_TRACK_DISTANCE
    ):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid, bbox):
        self.objects[self.next_object_id] = TrackedObject(
            self.next_object_id,
            centroid,
            bbox
        )
        self.next_object_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]

    @staticmethod
    def _centroid_of(bbox):
        x1, y1, x2, y2 = bbox[:4]
        return (
            int((x1 + x2) / 2.0),
            int((y1 + y2) / 2.0)
        )

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id].disappeared += 1

                if self.objects[object_id].disappeared > self.max_disappeared:
                    self._deregister(object_id)

            return list(self.objects.values())

        input_centroids = [
            self._centroid_of(det)
            for det in detections
        ]

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self._register(centroid, detections[i])

            return list(self.objects.values())

        object_ids = list(self.objects.keys())
        object_centroids = [
            self.objects[object_id].centroid
            for object_id in object_ids
        ]

        distance_matrix = np.linalg.norm(
            np.array(object_centroids)[:, np.newaxis]
            - np.array(input_centroids)[np.newaxis, :],
            axis=2,
        )

        rows = distance_matrix.min(axis=1).argsort()
        cols = distance_matrix.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distance_matrix[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]

            self.objects[object_id].centroid = input_centroids[col]
            self.objects[object_id].bbox = detections[col]
            self.objects[object_id].disappeared = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(distance_matrix.shape[0])) - used_rows
        unused_cols = set(range(distance_matrix.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.objects[object_id].disappeared += 1

            if self.objects[object_id].disappeared > self.max_disappeared:
                self._deregister(object_id)

        for col in unused_cols:
            self._register(
                input_centroids[col],
                detections[col]
            )

        return list(self.objects.values())