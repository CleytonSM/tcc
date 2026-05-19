"""Serviço de persistência de alertas."""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AlertService:
    """Gerencia a persistência de alertas no banco de dados.

    Cada tipo de alerta (PRONE, TOY, ABSENCE) é registrado apenas uma vez
    quando o estado transiciona para 'alertado'.
    """

    def __init__(self, repository):
        """
        Args:
            repository: Instância de AlertRepository.
        """
        self._repo = repository
        # Estado interno para evitar duplicação
        self._active = {
            "PRONE": False,
            "TOY": False,
            "ABSENCE": False,
        }

    def record(self, alert_type, started_at, ended_at, duration_seconds):
        """Registra um alerta apenas se ainda não registrado.

        Args:
            alert_type: Tipo do alerta ('PRONE', 'TOY', 'ABSENCE').
            started_at: Quando o alerta começou.
            ended_at: Quando o alerta terminou.
            duration_seconds: Duração em segundos.
        """
        if self._active.get(alert_type, False):
            return

        if self._repo is None:
            return

        self._repo.save_alert(alert_type, started_at, ended_at, duration_seconds)
        self._active[alert_type] = True

    def reset(self, alert_type):
        """Reseta o estado de um tipo de alerta para permitir novo registro.

        Args:
            alert_type: Tipo do alerta a ser resetado.
        """
        self._active[alert_type] = False
