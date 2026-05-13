from ultralytics import YOLO
import cv2
import mediapipe as mp
import warnings
import time
warnings.filterwarnings("ignore")

# --- PARÂMETROS AJUSTÁVEIS ---
# Confiança mínima para o MediaPipe detectar um rosto
FACE_DETECTION_CONFIDENCE = 0.4

# -----------------------------

# Número de frames consecutivos para confirmar posição prona (evita falsos positivos)
PRONE_CONFIRM_FRAMES = 20
# -----------------------------

# Inicializa o detector de rostos do MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    min_detection_confidence=FACE_DETECTION_CONFIDENCE,
    model_selection=1  # modelo para objetos distantes (melhor para top-down)
)

# TODO: Mudar a lógica futuramente. Pois, se o bebê não estiver no berço, o sistema vai detectar posição prona

def check_prone(frame, bbox):
    """
    Verifica posição do bebê usando MediaPipe Face Detection.

    Lógica baseada na distância entre os olhos:
    - Supino (barriga pra cima): rosto aponta para a câmera, mas em top-down
      o MediaPipe geralmente não detecta (ângulo incomum) → não é prone
    - Lateral/Prone: rosto de perfil → MediaPipe detecta, mas os olhos estão
      muito próximos entre si (distância X pequena) → É prone

    Retorna: eye_dist: float or None (None se não detectar rosto = Prone)
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Garantir que as coordenadas estejam dentro do frame
    h_frame, w_frame = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)

    # Se cair aqui, as coordenadas estão inválidas
    if x2 <= x1 or y2 <= y1:
        return None

    # Recortar a região do bebê e converter para RGB
    roi = frame[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    results = face_detector.process(roi_rgb)

    # Nenhum rosto detectado = posição PRONA (face voltada para baixo ou não visível)
    if not results.detections:
        return None

    # Rosto detectado: calcular distância entre os olhos no eixo X
    detection = results.detections[0]
    kps = detection.location_data.relative_keypoints
    # kps[0] = olho direito, kps[1] = olho esquerdo
    eye_dist = abs(kps[0].x - kps[1].x)
    print(eye_dist)
    return eye_dist


model = YOLO("best_12_5_26.pt")
video = cv2.VideoCapture("baby7.mp4")

# Janela redimensionável
cv2.namedWindow("Detection Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection Window", 800, 600)

BABY_CLASS_ID = None

class ProneTimer:
    def __init__(self, alert_threshold=4.0):
        self.alert_threshold = alert_threshold
        self.start_time = None

    def update(self, is_prone):
        """
        Atualiza o estado do temporizador de forma isolada (Single Responsibility).
        Retorna (alerta_ativo, tempo_decorrido).
        """
        if is_prone:
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            return elapsed >= self.alert_threshold, elapsed
        else:
            self.reset()
            return False, 0.0

    def reset(self):
        """Reseta o temporizador caso a posição prona falhe ou o bebê não esteja detectado."""
        self.start_time = None

class AbsenceTimer:
    def __init__(self, alert_threshold=5.0):
        self.alert_threshold = alert_threshold
        self.start_time = None

    def update(self, is_absent):
        """
        Atualiza o estado do temporizador de ausência (Single Responsibility).
        Retorna (alerta_ativo, tempo_decorrido).
        """
        if is_absent:
            if self.start_time is None:
                self.start_time = time.time()
            elapsed = time.time() - self.start_time
            return elapsed >= self.alert_threshold, elapsed
        else:
            self.reset()
            return False, 0.0

    def reset(self):
        """Reseta o temporizador caso o bebê seja detectado."""
        self.start_time = None

prone_timer = ProneTimer(alert_threshold=1.0)
absence_timer = AbsenceTimer(alert_threshold=5.0)

def draw_absence_alert(annotated_frame, elapsed_time, alert_active):
    """
    Desenha a interface visual de alerta de ausência no frame.
    """
    if not alert_active:
        return
        
    text = f"ALERTA CRITICO: Bebe nao detectado ({int(elapsed_time)}s)"
    font_scale = 1.0
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    h_frame, w_frame = annotated_frame.shape[:2]
    x1 = (w_frame - tw) // 2
    y1 = max(th + 10, int(h_frame * 0.1))

    # Fundo vermelho
    cv2.rectangle(
        annotated_frame,
        (x1 - 10, y1 - th - 10),
        (x1 + tw + 10, y1 + baseline + 5),
        (0, 0, 255), -1
    )
    cv2.putText(
        annotated_frame, text,
        (x1, y1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (255, 255, 255), thickness
    )

def draw_prone_alert(annotated_frame, bbox, elapsed_time, alert_active):
    """
    Desenha a interface visual de detecção e alerta no frame.
    Garante que a renderização fique separada da lógica temporal.
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Cor baseada no alerta: Vermelho para alerta (> 4s), Laranja apenas detectando
    box_color = (0, 0, 255) if alert_active else (0, 165, 255)

    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)

    if alert_active:
        text = f"ALERTA: Posicao Prona ({int(elapsed_time)}s)"
    else:
        text = f"Detectando Prona: {int(elapsed_time)}s"

    font_scale = max(0.6, (x2 - x1) / 300.0)
    thickness = max(2, int(font_scale * 2))
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(
        annotated_frame,
        (x1, max(0, y1 - th - baseline - 10)),
        (x1 + tw, max(0, y1)),
        box_color, -1
    )
    cv2.putText(
        annotated_frame, text,
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, (255, 255, 255), thickness
    )

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Inferência do YOLO
    results = model(frame, verbose=False)

    # Visualização padrão do YOLO (berço, brinquedos, bebê)
    annotated_frame = results[0].plot()

    # Buscar dinamicamente o ID da classe 'baby' na primeira iteração
    if BABY_CLASS_ID is None:
        for idx, name in results[0].names.items():
            if name.lower() == "baby":
                BABY_CLASS_ID = idx
                break
        if BABY_CLASS_ID is None:
            BABY_CLASS_ID = 0

    # Iterar sobre as detecções para aplicar a lógica de posição prona
    baby_detected = False
    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            bbox = box.xyxy[0].cpu().numpy()
            name = results[0].names[cls_id]

            if name == "baby":
                baby_detected = True
                baby_bbox = bbox

                # Verificar posição prona
                eye_dist = check_prone(frame, bbox)
                print(eye_dist)

                # Se eye_dist for None, significa que o media pipe não detectou o rosto
                # seguindo a lógica do usuário: não detectou = posição prona
                is_prone = eye_dist is None

                # Atualiza o temporizador isolado
                alert_active, elapsed_time = prone_timer.update(is_prone)

                if is_prone:
                    draw_prone_alert(annotated_frame, bbox, elapsed_time, alert_active)

    # Se bebê sumiu do frame, resetar contador
    if not baby_detected:
        prone_timer.reset()

    cv2.imshow("Detection Window", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
face_detector.close()