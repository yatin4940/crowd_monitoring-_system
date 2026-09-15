from config import ZONES


class ZoneManager:
    def __init__(self, zones=ZONES):
        self.zones = zones

    @staticmethod
    def _point_in_zone(point, zone_box):
        x, y = point
        x1, y1, x2, y2, _ = zone_box

        return x1 <= x <= x2 and y1 <= y <= y2

    def update(self, tracked_objects):
        status = {}

        for name, zone_box in self.zones.items():
            count = 0

            for obj in tracked_objects:
                if self._point_in_zone(obj.centroid, zone_box):
                    count += 1

            max_capacity = zone_box[4]

            status[name] = {
                "count": count,
                "max": max_capacity,
                "overcrowded": count > max_capacity,
            }

        return status