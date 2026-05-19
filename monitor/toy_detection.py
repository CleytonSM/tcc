"""Toy detection using YOLO model inference."""

from ultralytics import YOLO
from config import TOY_MODEL_PATH, TOY_CONF_TRADITIONAL


class ToyDetector:
    """Detecta brinquedos no berco usando um modelo YOLO pré-treinado."""

    def __init__(self):
        self.model = YOLO(TOY_MODEL_PATH)

    def detect(self, roi_berco, baby_rel_bbox=None):
        """Retorna lista de bboxes de brinquedos detectados pelo YOLO.

        Args:
            roi_berco: ROI do berco (BGR)
            baby_rel_bbox: ignorado (mantido por compatibilidade de assinatura)
        """
        if roi_berco is None or roi_berco.size == 0:
            return []

        results = self.model.predict(
            source=roi_berco,
            conf=TOY_CONF_TRADITIONAL,
            verbose=False,
            imgsz=640,
        )

        bboxes = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                    bboxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

        return bboxes
