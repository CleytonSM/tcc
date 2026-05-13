from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")

# --- PARÂMETROS AJUSTÁVEIS ---
# Confiança mínima para o MediaPipe detectar um rosto
FACE_DETECTION_CONFIDENCE = 0.4

# Parâmetros para detecção tradicional de brinquedos
TOY_MIN_AREA = 600
TOY_MAX_AREA = 25000
TOY_CONF_TRADITIONAL = 0.0  # Mantido para consistência
TOY_CONFIRM_FRAMES = 5      # Número de frames para confirmar um brinquedo
TOY_IOU_THRESHOLD = 0.3     # Sobreposição mínima para considerar o mesmo objeto

# -----------------------------

# Número de frames consecutivos para confirmar posição prona (evita falsos positivos)
PRONE_CONFIRM_FRAMES = 20
# -----------------------------

class ToyTracker:
    """
    Rastreia candidatos a brinquedos através de múltiplos frames para reduzir falsos positivos.
    """
    def __init__(self):
        # Lista de candidatos: [{'bbox': (x, y, w, h), 'count': frames}]
        self.candidates = []

    def _calculate_iou(self, boxA, boxB):
        # box = (x, y, w, h)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        areaA = boxA[2] * boxA[3]
        areaB = boxB[2] * boxB[3]
        iou = interArea / float(areaA + areaB - interArea + 1e-6)
        return iou

    def update(self, current_detections):
        """
        Atualiza o estado dos candidatos com as novas detecções do frame.
        current_detections: Lista de bboxes globais (x, y, w, h)
        """
        new_candidates = []

        for det in current_detections:
            best_iou = 0
            best_idx = -1

            for i, cand in enumerate(self.candidates):
                iou = self._calculate_iou(det, cand['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= TOY_IOU_THRESHOLD:
                # Atualiza candidato existente
                self.candidates[best_idx]['count'] += 1
                self.candidates[best_idx]['bbox'] = det # Atualiza posição
                # Evita que o mesmo candidato seja usado por duas detecções no mesmo frame
                self.candidates[best_idx]['matched'] = True
            else:
                # Novo candidato
                new_candidates.append({'bbox': det, 'count': 1, 'matched': False})

        # Marcar quem não foi matchado neste frame
        for cand in self.candidates:
            if 'matched' not in cand: cand['matched'] = False # Init
            if not cand['matched']:
                cand['count'] -= 1 # Penaliza se sumiu
            else:
                cand['matched'] = False # Reset para o próximo frame

        # Filtrar candidatos que morreram (count <= 0) e mesclar novos
        self.candidates = [c for c in self.candidates if c['count'] > 0]
        self.candidates.extend(new_candidates)

    def get_confirmed_toys(self):
        """Retorna apenas brinquedos que passaram do threshold de frames."""
        return [c['bbox'] for c in self.candidates if c['count'] >= TOY_CONFIRM_FRAMES]

# Inicializa o detector de rostos do MediaPipe
mp_face_detection = mp.solutions.face_detection

def detect_toys_traditional(roi_berco):
    """
    Detecta objetos coloridos (brinquedos) dentro da ROI do berço usando Visão Computacional Tradicional.

    Abordagem:
    1. Converte para HSV para melhor segmentação de cores.
    2. Cria máscaras para cores saturadas (evitando tons neutros, pretos e brancos).
    3. Aplica filtros morfológicos para limpar ruídos.
    4. Encontra contornos e filtra por área e proporção (aspect ratio).

    Retorna: Lista de bounding boxes (x, y, w, h) relativas à ROI.
    """
    if roi_berco is None or roi_berco.size == 0:
        return []

    # 1. Conversão para HSV
    hsv = cv2.cvtColor(roi_berco, cv2.COLOR_BGR2HSV)

    # 2. Máscara de Saturação: Brinquedos tendem a ser cores vivas.
    # Aumentamos os limites para reduzir falsos positivos de sombras e dobras de roupa.
    # S > 70 e V > 70 para ignorar cores pálidas e áreas escuras.
    lower_saturated = np.array([0, 110, 110])
    upper_saturated = np.array([180, 200, 200])
    mask = cv2.inRange(hsv, lower_saturated, upper_saturated)

    # 3. Filtros Morfológicos
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove pequenos ruídos
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Preenche buracos

    # 4. Detecção de Contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    toys_bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if TOY_MIN_AREA < area < TOY_MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)

            # Filtro de Aspect Ratio: Evita detectar grades do berço (muito finas/longas)
            aspect_ratio = float(w) / h
            if 0.2 < aspect_ratio < 5.0:
                toys_bboxes.append((x, y, w, h))

    return toys_bboxes

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
video = cv2.VideoCapture("baby8.mp4")

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

# Instancia os temporizadores e rastreadores
prone_timer = ProneTimer(alert_threshold=1.0)
absence_timer = AbsenceTimer(alert_threshold=5.0)
toy_tracker = ToyTracker()

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

    # Iterar sobre as detecções para aplicar a lógica de posição prona e detecção de brinquedos
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

            elif name == "crib":
                # Recortar a ROI do berço
                cx1, cy1, cx2, cy2 = map(int, bbox)
                roi_berco = frame[max(0, cy1):cy2, max(0, cx1):cx2]

                # Detecção tradicional de brinquedos dentro do berço
                toys_relative = detect_toys_traditional(roi_berco)

                # Converter coordenadas relativas da ROI para coordenadas globais do frame
                toys_global = []
                for (tx, ty, tw, th) in toys_relative:
                    global_x = max(0, cx1) + tx
                    global_y = max(0, cy1) + ty
                    toys_global.append((global_x, global_y, tw, th))

                # Atualizar o rastreador temporal com as detecções globais
                toy_tracker.update(toys_global)

                # Desenhar apenas os brinquedos confirmados temporalmente
                confirmed_toys = toy_tracker.get_confirmed_toys()
                for (gx, gy, gw, gh) in confirmed_toys:
                    # Desenhar bounding box (Cor Ciano: (255, 255, 0))
                    cv2.rectangle(annotated_frame, (gx, gy),
                                  (gx + gw, gy + gh), (255, 255, 0), 2)

                    # Label "Toy-Trad"
                    cv2.putText(annotated_frame, "Toy-Trad", (gx, max(30, gy)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Se bebê sumiu do frame, resetar contador
    if not baby_detected:
        prone_timer.reset()

    cv2.imshow("Detection Window", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
face_detector.close()