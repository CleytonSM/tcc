"""Configuracoes centrais do sistema de monitoramento."""

# Confianca minima para o MediaPipe detectar um rosto
FACE_DETECTION_CONFIDENCE = 0.4

# Deteccao tradicional de brinquedos
TOY_MIN_AREA = 600
TOY_MAX_AREA = 25000
TOY_CONF_TRADITIONAL = 0.0
TOY_CONFIRM_FRAMES = 5
TOY_IOU_THRESHOLD = 0.3

# Posicao prona
PRONE_CONFIRM_FRAMES = 20

# Temporizadores de alerta
PRONE_ALERT_THRESHOLD = 1.0  # segundos
ABSENCE_ALERT_THRESHOLD = 5.0  # segundos

# Video
MODEL_PATH = "best_12_5_26.pt"
VIDEO_PATH = "baby8.mp4"

# Janela
WINDOW_NAME = "Detection Window"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
