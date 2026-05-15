"""Deteccao de brinquedos via subtracao de fundo com mascaramento."""

import cv2
import numpy as np
from config import (
    TOY_MIN_AREA, TOY_MAX_AREA,
    BG_SUBTRACTION_THRESHOLD,
    BG_EDGE_MARGIN_PCT,
    BG_BABY_DILATE_KERNEL,
    EDGE_SOBEL_THRESHOLD,
)


class ToyDetector:
    """Detecta brinquedos no berco usando subtracao de fundo acumulada.

    Mantem um modelo de fundo (running average) da area do colchao.
    Sombras transitorias se diluem no average; brinquedos persistentes
    sao confirmados pelo ToyTracker temporal.
    """

    def __init__(self):
        self.frame_count = 0
        self._bg_sum = None  # float64 accumulator

    # ---- background model ----

    def _exclusion_mask(self, h, w, baby_rel=None):
        """Mascara: 1 = area do colchao (centro), 0 = bordas + bebe."""
        mask = np.zeros((h, w), np.uint8)
        mx = int(w * BG_EDGE_MARGIN_PCT)
        my = int(h * BG_EDGE_MARGIN_PCT)
        mask[my:h - my, mx:w - mx] = 1

        if baby_rel is not None:
            bx, by, bw, bh = map(int, baby_rel)
            bm = np.zeros((h, w), np.uint8)
            bx1, by1 = max(0, bx), max(0, by)
            bx2, by2 = min(w, bx + bw), min(h, by + bh)
            if bx2 > bx1 and by2 > by1:
                bm[by1:by2, bx1:bx2] = 1
                bm = cv2.dilate(bm, np.ones((BG_BABY_DILATE_KERNEL, BG_BABY_DILATE_KERNEL), np.uint8), iterations=1)
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(bm))

        return mask

    def _update_background(self, roi, mask):
        """Acumula pixels do colchao para o running average."""
        sheet_pixels = roi[mask == 1]
        if len(sheet_pixels) == 0:
            return
        self.frame_count += 1
        if self._bg_sum is None:
            self._bg_sum = np.zeros(3, dtype=np.float64)
        self._bg_sum += sheet_pixels.mean(axis=0)

    @property
    def background(self):
        if self._bg_sum is None or self.frame_count == 0:
            return None
        return self._bg_sum / self.frame_count

    # ---- filters ----

    @staticmethod
    def _edge_sharpness(roi, contour):
        m = np.zeros(roi.shape[:2], np.uint8)
        cv2.drawContours(m, [contour], -1, 1, -1)
        gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        vals = mag[m == 1]
        return float(np.mean(vals)) if len(vals) else 0.0

    @staticmethod
    def _color_variance(roi, contour):
        m = np.zeros(roi.shape[:2], np.uint8)
        cv2.drawContours(m, [contour], -1, 1, -1)
        pixels = roi[m == 1]
        return float(np.std(pixels, axis=0).mean()) if len(pixels) else 0.0

    # ---- public API ----

    def detect(self, roi_berco, baby_rel_bbox=None):
        """Retorna lista de bboxes de brinquedos candidatos.

        Args:
            roi_berco: ROI do berco (BGR)
            baby_rel_bbox: (x, y, w, h) do bebe relativo ao berco, ou None
        """
        if roi_berco is None or roi_berco.size == 0:
            return []

        h, w = roi_berco.shape[:2]
        mask = self._exclusion_mask(h, w, baby_rel_bbox)
        self._update_background(roi_berco, mask)

        bg = self.background
        if bg is None:
            return []

        bg_img = np.full((h, w, 3), bg, dtype=roi_berco.dtype)
        diff = cv2.absdiff(roi_berco, bg_img)
        diff_gray = np.mean(diff, axis=2)
        binary = (diff_gray > BG_SUBTRACTION_THRESHOLD).astype(np.uint8) * 255
        binary = cv2.bitwise_and(binary, mask)

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (TOY_MIN_AREA < area < TOY_MAX_AREA):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if not (0.2 < bw / bh < 5.0):
                continue
            if self._edge_sharpness(roi_berco, cnt) < EDGE_SOBEL_THRESHOLD:
                continue
            if self._color_variance(roi_berco, cnt) < 5.0:
                continue
            results.append((x, y, bw, bh))

        return results
