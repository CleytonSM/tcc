"""Deteccao de posicao prona usando MediaPipe Face Detection."""

import cv2
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore")

from config import FACE_DETECTION_CONFIDENCE

mp_face_detection = mp.solutions.face_detection

face_detector = mp_face_detection.FaceDetection(
    min_detection_confidence=FACE_DETECTION_CONFIDENCE,
    model_selection=1
)


def check_prone(frame, bbox):
    """Verifica posicao do bebe usando MediaPipe.

    Retorna eye_dist (float) se rosto for detectado, ou None se nao for detectado (prone).
    """
    x1, y1, x2, y2 = map(int, bbox)

    h_frame, w_frame = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    results = face_detector.process(roi_rgb)

    if not results.detections:
        return None

    detection = results.detections[0]
    kps = detection.location_data.relative_keypoints
    eye_dist = abs(kps[0].x - kps[1].x)
    return eye_dist


def cleanup():
    """Fecha o detector de rostos."""
    face_detector.close()
