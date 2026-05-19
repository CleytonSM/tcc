"""Acesso ao banco de dados PostgreSQL para alertas."""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AlertRepository:
    """Persiste alertas no PostgreSQL."""

    def __init__(self, connection):
        """
        Args:
            connection: Conexão psycopg2 ou sqlite3.
        """
        self._conn = connection

    def save_alert(self, alert_type, started_at, ended_at, duration_seconds):
        """Salva um registro de alerta no banco."""
        try:
            query = (
                "INSERT INTO alerts (alert_type, started_at, ended_at, duration_seconds)"
                " VALUES (%s, %s, %s, %s)"
            )
            cursor = self._conn.cursor()
            cursor.execute(query, (alert_type, started_at, ended_at, duration_seconds))
            self._conn.commit()
            logger.info(
                "Alert saved: type=%s, duration=%ds",
                alert_type,
                duration_seconds,
            )
        except Exception:
            logger.exception("Failed to save alert")
