"""Inicializacao e loop principal do sistema de monitoramento."""

from ultralytics import YOLO
import cv2
import warnings
warnings.filterwarnings("ignore")

import config
from toy_tracker import ToyTracker
from toy_detection import ToyDetector
from prone_detector import check_prone, cleanup
from prone_timer import ProneTimer
from absence_timer import AbsenceTimer
from drawing import draw_prone_alert, draw_absence_alert


def find_baby_class_id(results):
    """Busca o ID da classe 'baby' no modelo YOLO."""
    for idx, name in results[0].names.items():
        if name.lower() == "baby":
            return idx
    return 0


def process_frame(frame, model, baby_class_id, prone_timer, absence_timer, toy_detector, toy_tracker):
    """Processa um frame e retorna o frame anotado com alertas."""
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    baby_detected = False

    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        # Primeiro loop: coleta bboxes do bebe e berco
        baby_bbox_global = None
        crib_bbox_global = None

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            bbox = box.xyxy[0].cpu().numpy()
            name = results[0].names[cls_id]

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
            roi_berco = frame[max(0, cy1):cy2, max(0, cx1):cx2]

            # Converte bbox do bebe para coordenadas relativas ao berco
            baby_roi_bbox = None
            if baby_bbox_global is not None:
                bx, by, bw, bh = map(int, baby_bbox_global)
                baby_roi_bbox = (bx - cx1, by - cy1, bw, bh)

            toys_relative = toy_detector.detect(roi_berco, baby_roi_bbox)

            toys_global = []
            for (tx, ty, tw, th) in toys_relative:
                toys_global.append((max(0, cx1) + tx, max(0, cy1) + ty, tw, th))

            toy_tracker.update(toys_global, frame)

            for (gx, gy, gw, gh) in toy_tracker.get_confirmed_toys():
                cv2.rectangle(annotated_frame, (gx, gy),
                              (gx + gw, gy + gh), (255, 255, 0), 2)
                cv2.putText(annotated_frame, "Toy-Trad", (gx, max(30, gy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    if not baby_detected:
        prone_timer.reset()

    return annotated_frame, baby_detected


def run():
    """Entrada principal do sistema de monitoramento."""
    model = YOLO(config.MODEL_PATH)
    video = cv2.VideoCapture(config.VIDEO_PATH)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_NAME, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

    baby_class_id = None
    prone_timer = ProneTimer()
    absence_timer = AbsenceTimer()
    toy_detector = ToyDetector()
    toy_tracker = ToyTracker()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if baby_class_id is None:
            results = model(frame, verbose=False)
            baby_class_id = find_baby_class_id(results)

        annotated_frame, baby_detected = process_frame(
            frame, model, baby_class_id, prone_timer, absence_timer, toy_detector, toy_tracker
        )

        cv2.imshow(config.WINDOW_NAME, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    cleanup()


if __name__ == "__main__":
    run()
