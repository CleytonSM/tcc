"""Timer para confirmação de detecção de brinquedo."""

import time


class ToyAlertTimer:
    """Contador de confirmação para alertas de brinquedo.

    Evita falsos positivos exigindo que a detecção persista por N segundos
    antes de disparar o alerta.
    """

    def __init__(self, alert_threshold=5.0):
        self.alert_threshold = alert_threshold
        self.start_time = None

    def update(self, toy_detected):
        """Atualiza o timer e retorna (alert_active, elapsed_time)."""
        if toy_detected:
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            return elapsed >= self.alert_threshold, elapsed
        else:
            self.reset()
            return False, 0.0

    def reset(self):
        self.start_time = None
