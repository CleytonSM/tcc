"""Temporizador de alerta de ausencia do bebe."""

import time


class AbsenceTimer:
    """Gerencia o tempo de ausencia e dispara alertas."""

    def __init__(self, alert_threshold=5.0):
        self.alert_threshold = alert_threshold
        self.start_time = None
        self._elapsed = 0.0

    def update(self, is_absent):
        if is_absent:
            if self.start_time is None:
                self.start_time = time.time()
            self._elapsed = time.time() - self.start_time
            return self._elapsed >= self.alert_threshold, self._elapsed
        else:
            self.reset()
            return False, 0.0

    def elapsed(self):
        """Retorna o tempo decorrido atual."""
        if self.start_time is not None:
            return time.time() - self.start_time
        return 0.0

    def reset(self):
        self.start_time = None
        self._elapsed = 0.0
