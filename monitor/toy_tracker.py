"""Rastreador temporal de brinquedos para reduzir falsos positivos."""

from config import TOY_IOU_THRESHOLD, TOY_CONFIRM_FRAMES


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

    def update(self, current_detections):
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
                self.candidates[best_idx]['count'] += 1
                self.candidates[best_idx]['bbox'] = det
                self.candidates[best_idx]['matched'] = True
            else:
                new_candidates.append({'bbox': det, 'count': 1, 'matched': False})

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
        return [c['bbox'] for c in self.candidates if c['count'] >= TOY_CONFIRM_FRAMES]
