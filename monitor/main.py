from ultralytics import YOLO
import cv2
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore")

# --- PARÂMETROS AJUSTÁVEIS ---
# Confiança mínima para o MediaPipe detectar um rosto
FACE_DETECTION_CONFIDENCE = 0.1

# Distância mínima entre os olhos (normalizada 0-1) para considerar rosto de FRENTE
# Abaixo desse valor = rosto de perfil = posição lateral/prona
# Valores observados: supino ~sem detecção | lateral/prone ~0.008 a 0.022
EYE_DIST_THRESHOLD = 0.05

# Número de frames consecutivos para confirmar posição prona (evita falsos positivos)
PRONE_CONFIRM_FRAMES = 8
# -----------------------------

# Inicializa o detector de rostos do MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    min_detection_confidence=FACE_DETECTION_CONFIDENCE,
    model_selection=1  # modelo para objetos distantes (melhor para top-down)
)

def check_prone(frame, bbox):
    """
    Verifica posição do bebê usando MediaPipe Face Detection.

    Lógica baseada na distância entre os olhos:
    - Supino (barriga pra cima): rosto aponta para a câmera, mas em top-down
      o MediaPipe geralmente não detecta (ângulo incomum) → não é prone
    - Lateral/Prone: rosto de perfil → MediaPipe detecta, mas os olhos estão
      muito próximos entre si (distância X pequena) → É prone

    Retorna: (is_prone: bool, eye_dist: float or None)
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Garantir que as coordenadas estejam dentro do frame
    h_frame, w_frame = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)

    if x2 <= x1 or y2 <= y1:
        return False, None

    # Recortar a região do bebê e converter para RGB
    roi = frame[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    results = face_detector.process(roi_rgb)

    # Nenhum rosto detectado = supino (olhando direto pra câmera, ângulo top-down)
    if not results.detections:
        return False, None

    # Rosto detectado: calcular distância entre os olhos no eixo X
    detection = results.detections[0]
    kps = detection.location_data.relative_keypoints
    # kps[0] = olho direito, kps[1] = olho esquerdo
    eye_dist = abs(kps[0].x - kps[1].x)

    # Distância pequena = rosto de perfil = lateral/prone
    is_prone = eye_dist < EYE_DIST_THRESHOLD
    return is_prone, eye_dist


model = YOLO("best copy.pt")
video = cv2.VideoCapture("baby7.mp4")

# Janela redimensionável
cv2.namedWindow("Detection Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection Window", 800, 600)

BABY_CLASS_ID = None

# Contador para confirmar posição prona por N frames consecutivos
prone_frame_counter = 0
confirmed_prone = False

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
            if cls_id == BABY_CLASS_ID:
                baby_detected = True
                bbox = box.xyxy[0].cpu().numpy()

                # Verificar posição prona
                is_prone, eye_dist = check_prone(frame, bbox)

                # Atualizar contador de confirmação
                if is_prone:
                    prone_frame_counter = min(prone_frame_counter + 1, PRONE_CONFIRM_FRAMES)
                else:
                    prone_frame_counter = max(prone_frame_counter - 1, 0)

                confirmed_prone = prone_frame_counter < PRONE_CONFIRM_FRAMES

                # Debug: mostrar eye_dist no canto da tela
                debug_text = f"eye_dist: {eye_dist:.3f}" if eye_dist is not None else "eye_dist: N/A"
                cv2.putText(annotated_frame, debug_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

                if confirmed_prone:
                    x1, y1, x2, y2 = map(int, bbox)

                    # Bbox vermelho
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # Texto de alerta com fundo vermelho
                    text = "[!] PRONE ALERT"
                    font_scale = max(0.6, (x2 - x1) / 300.0)
                    thickness = max(2, int(font_scale * 2))
                    (tw, th), baseline = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (x1, max(0, y1 - th - baseline - 10)),
                        (x1 + tw, max(0, y1)),
                        (0, 0, 255), -1
                    )
                    cv2.putText(
                        annotated_frame, text,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness
                    )

    # Se bebê sumiu do frame, resetar contador
    if not baby_detected:
        prone_frame_counter = 0
        confirmed_prone = False

    cv2.imshow("Detection Window", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
face_detector.close()