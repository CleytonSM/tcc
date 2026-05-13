"""Temporizador de alerta de posicao prona."""

import time


class ProneTimer:
    """Gerencia o tempo de posicao prona e dispara alertas."""

    def __init__(self, alert_threshold=1.0):
        self.alert_threshold = alert_threshold
        self.start_time = None

    def update(self, is_prone):
        if is_prone:
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            return elapsed >= self.alert_threshold, elapsed
        else:
            self.reset()
            return False, 0.0

    def reset(self):
        self.start_time = None
