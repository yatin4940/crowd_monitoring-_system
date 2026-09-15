from config import COUNT_LINE_Y, COUNT_MARGIN


class LineCounter:
    def __init__(self, line_y=COUNT_LINE_Y, margin=COUNT_MARGIN):
        self.line_y = line_y
        self.margin = margin
        self.entered = 0
        self.exited = 0
        self._last_y = {}
        self._counted_recently = set()

    def update(self, tracked_objects):
        active_ids = set()

        for obj in tracked_objects:
            object_id = obj.object_id
            _, current_y = obj.centroid
            active_ids.add(object_id)

            previous_y = self._last_y.get(object_id)
            self._last_y[object_id] = current_y

            if object_id in self._counted_recently:
                if abs(current_y - self.line_y) > self.margin:
                    self._counted_recently.discard(object_id)
                else:
                    continue

            if previous_y is None:
                continue

            crossed_downward = previous_y < self.line_y <= current_y
            crossed_upward = previous_y > self.line_y >= current_y

            if crossed_downward:
                self.entered += 1
                self._counted_recently.add(object_id)
            elif crossed_upward:
                self.exited += 1
                self._counted_recently.add(object_id)

        for object_id in list(self._last_y.keys()):
            if object_id not in active_ids:
                self._last_y.pop(object_id, None)
                self._counted_recently.discard(object_id)

    @property
    def inside(self):
        return max(0, self.entered - self.exited)

    def reset(self):
        self.entered = 0
        self.exited = 0
        self._last_y.clear()
        self._counted_recently.clear()