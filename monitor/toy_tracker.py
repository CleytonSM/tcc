"""Rastreador temporal de brinquedos com filtro de consistencia de cor."""

import numpy as np
from config import TOY_IOU_THRESHOLD, TOY_CONFIRM_FRAMES, COLOR_CONSISTENCY_THRESHOLD, COLOR_CONSISTENCY_FRAMES


class ToyTracker:
    """Rastreia candidatos a brinquedos através de múltiplos frames."""

    def __init__(self):
        self.candidates = []

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        areaA = boxA[2] * boxA[3]
        areaB = boxB[2] * boxB[3]
        return interArea / float(areaA + areaB - interArea + 1e-6)

    def _mean_color(self, frame, bbox):
        """Calcula a cor media dentro de um bbox no frame."""
        x, y, w, h = map(int, bbox)
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return np.zeros(3, dtype=np.float32)
        return np.mean(roi, axis=(0, 1)).astype(np.float32)

    def _color_distance(self, colorA, colorB):
        """Distancia Euclidiana entre duas cores RGB."""
        return float(np.sqrt(np.sum((colorA - colorB) ** 2)))

    def update(self, current_detections, frame):
        """Atualiza candidatos com verificacao de consistencia de cor.

        Args:
            current_detections: Lista de bboxes (x, y, w, h)
            frame: Frame completo para extrair cores
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

            det_color = self._mean_color(frame, det)

            if best_iou >= TOY_IOU_THRESHOLD and best_idx >= 0:
                self.candidates[best_idx]['count'] += 1
                self.candidates[best_idx]['bbox'] = det

                # Verifica consistencia de cor
                prev_color = self.candidates[best_idx].get('color_history', [])
                if prev_color:
                    dist = self._color_distance(det_color, prev_color[-1])
                    if dist > COLOR_CONSISTENCY_THRESHOLD:
                        self.candidates[best_idx]['color_stable'] = False
                    else:
                        self.candidates[best_idx]['color_stable'] = True
                else:
                    self.candidates[best_idx]['color_stable'] = True

                # Atualiza historico de cor
                if 'color_history' not in self.candidates[best_idx]:
                    self.candidates[best_idx]['color_history'] = []
                self.candidates[best_idx]['color_history'].append(det_color)
                if len(self.candidates[best_idx]['color_history']) > COLOR_CONSISTENCY_FRAMES:
                    self.candidates[best_idx]['color_history'].pop(0)

                self.candidates[best_idx]['matched'] = True
            else:
                new_candidates.append({
                    'bbox': det,
                    'count': 1,
                    'matched': False,
                    'color_history': [det_color],
                    'color_stable': True,
                })

        for cand in self.candidates:
            if 'matched' not in cand:
                cand['matched'] = False
            if not cand['matched']:
                cand['count'] -= 1
            else:
                cand['matched'] = False

        self.candidates = [c for c in self.candidates if c['count'] > 0]
        self.candidates.extend(new_candidates)

    def get_confirmed_toys(self):
        """Retorna brinquedos confirmados com cor estavel."""
        return [
            c['bbox'] for c in self.candidates
            if c['count'] >= TOY_CONFIRM_FRAMES and c.get('color_stable', True)
        ]
