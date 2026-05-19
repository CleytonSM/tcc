"""Inicializacao e loop principal do sistema de monitoramento."""

from ultralytics import YOLO
import cv2
import warnings
warnings.filterwarnings("ignore")

import config
from prone_detector import check_prone, cleanup
from prone_timer import ProneTimer
from absence_timer import AbsenceTimer
from toy_alert_timer import ToyAlertTimer
from drawing import draw_prone_alert, draw_absence_alert, draw_toy_alert


def find_baby_class_id(results):
    """Busca o ID da classe 'baby' no modelo YOLO."""
    for idx, name in results[0].names.items():
        if name.lower() == "baby":
            return idx
    return 0


def process_frame(frame, model_baby, model_toy, baby_class_id, prone_timer, absence_timer, toy_timer):
    """Processa um frame e retorna o frame anotado com alertas."""
    results_baby = model_baby(frame, verbose=False)
    annotated_frame = results_baby[0].plot()

    baby_detected = False

    if hasattr(results_baby[0], 'boxes') and results_baby[0].boxes is not None:
        # Primeiro loop: coleta bboxes do bebe e berco
        baby_bbox_global = None
        crib_bbox_global = None

        for box in results_baby[0].boxes:
            cls_id = int(box.cls[0])
            bbox = box.xyxy[0].cpu().numpy()
            name = results_baby[0].names[cls_id]

            if name == "baby":
                baby_detected = True
                baby_bbox_global = bbox
                eye_dist = check_prone(frame, bbox)
                is_prone = eye_dist is None

                alert_active, elapsed_time = prone_timer.update(is_prone)

                if is_prone:
                    draw_prone_alert(annotated_frame, bbox, elapsed_time, alert_active)

            elif name == "crib":
                crib_bbox_global = bbox

        # Segundo loop: processa berco com contexto do bebe
        if crib_bbox_global is not None:
            cx1, cy1, cx2, cy2 = map(int, crib_bbox_global)

            # Converte bbox do bebe para coordenadas relativas ao berco
            baby_roi_bbox = None
            if baby_bbox_global is not None:
                bx, by, bw, bh = map(int, baby_bbox_global)
                baby_roi_bbox = (bx - cx1, by - cy1, bw, bh)

            results_toy = model_toy(frame, verbose=False)
        toys = []
        for r in results_toy:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    toys.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
                    

        # Desenha as bounding boxes dos brinquedos
        for (tx, ty, tw, th) in toys:
            cv2.rectangle(annotated_frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)

        # Atualiza timer de brinquedo
        toy_detected = len(toys) > 0
        toy_alert_active, toy_elapsed = toy_timer.update(toy_detected)
        if toy_alert_active:
            draw_toy_alert(annotated_frame, toy_elapsed, True)
    if not baby_detected:
        prone_timer.reset()

    return annotated_frame, baby_detected


def run():
    """Entrada principal do sistema de monitoramento."""
    model_baby = YOLO(config.BABY_MODEL_PATH)
    model_toy = YOLO(config.TOY_MODEL_PATH)
    video = cv2.VideoCapture(config.VIDEO_PATH)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

    baby_class_id = None
    prone_timer = ProneTimer()
    absence_timer = AbsenceTimer()
    toy_timer = ToyAlertTimer(alert_threshold=config.TOY_ALERT_THRESHOLD)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if baby_class_id is None:
            results = model_baby(frame, verbose=False)
            baby_class_id = find_baby_class_id(results)

        annotated_frame, baby_detected = process_frame(
            frame, model_baby, model_toy, baby_class_id, prone_timer, absence_timer, toy_timer
        )

        cv2.imshow(config.WINDOW_NAME, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    cleanup()


if __name__ == "__main__":
    run()
