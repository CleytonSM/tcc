"""Configuracoes centrais do sistema de monitoramento."""

# Confianca minima para o MediaPipe detectar um rosto
FACE_DETECTION_CONFIDENCE = 0.4

# Deteccao tradicional de brinquedos
TOY_MIN_AREA = 600
TOY_MAX_AREA = 25000
TOY_CONF_TRADITIONAL = 0.0
TOY_CONFIRM_FRAMES = 5
TOY_IOU_THRESHOLD = 0.3

# Subtracao de fundo
BG_SUBTRACTION_THRESHOLD = 20       # Limiar absdiff contra fundo
BG_LEARNING_RATE = 0.05            # Taxa de atualizacao do modelo de fundo
BG_EDGE_MARGIN_PCT = 0.08          # Margem para excluir barras do berco
BG_BABY_DILATE_KERNEL = 15         # Dilatacao da mascara do bebe
BG_MIN_SHEET_PIXELS = 500          # Pixels minimos para estimativa valida

# Filtro de arestas
EDGE_SOBEL_THRESHOLD = 15.0        # Magnitude minima do gradiente nas bordas

# Consistencia de cor temporal
COLOR_CONSISTENCY_THRESHOLD = 40.0  # Distancia Euclidiana RGB max entre frames
COLOR_CONSISTENCY_FRAMES = 3        # Janela para media de cor

# Posicao prona
PRONE_CONFIRM_FRAMES = 20

# Temporizadores de alerta
PRONE_ALERT_THRESHOLD = 1.0  # segundos
TOY_ALERT_THRESHOLD = 5.0  # segundos
ABSENCE_ALERT_THRESHOLD = 5.0  # segundos

# Video
BABY_MODEL_PATH = "best_baby_15_05_26.pt"
TOY_MODEL_PATH = "best_toy_18_05_26.pt"
VIDEO_PATH = "baby8.mp4"

# Janela
WINDOW_NAME = "Detection Window"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
