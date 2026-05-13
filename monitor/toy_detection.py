"""Deteccao de brinquedos usando visao computacional tradicional."""

import cv2
import numpy as np
from config import TOY_MIN_AREA, TOY_MAX_AREA


def detect_toys_traditional(roi_berco):
    """Detecta objetos coloridos dentro da ROI do berco.

    Abordagem:
    1. Converte para HSV para segmentacao de cores.
    2. Cria mascaras para cores saturadas.
    3. Aplica filtros morfologicos.
    4. Encontra contornos e filtra por area e aspect ratio.
    """
    if roi_berco is None or roi_berco.size == 0:
        return []

    hsv = cv2.cvtColor(roi_berco, cv2.COLOR_BGR2HSV)

    lower_saturated = np.array([0, 110, 110])
    upper_saturated = np.array([180, 200, 200])
    mask = cv2.inRange(hsv, lower_saturated, upper_saturated)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    toys_bboxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if TOY_MIN_AREA < area < TOY_MAX_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.2 < aspect_ratio < 5.0:
                toys_bboxes.append((x, y, w, h))

    return toys_bboxes
