import time

from config import ALERT_COOLDOWN_SECONDS


class AlertManager:
    def __init__(self, cooldown_seconds=ALERT_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_time = {}
        self.active_alerts = []

    def update(self, zone_status):
        now = time.time()
        self.active_alerts = []

        for name, info in zone_status.items():
            if not info["overcrowded"]:
                continue

            message = f"WARNING: {name} overcrowded ({info['count']}/{info['max']})"
            self.active_alerts.append(message)

            last_time = self._last_alert_time.get(name, 0)
            if now - last_time >= self.cooldown_seconds:
                print(message)
                self._last_alert_time[name] = now

        return self.active_alerts