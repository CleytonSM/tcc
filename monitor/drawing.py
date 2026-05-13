"""Funcoes de desenho para alertas visuais nos frames."""

import cv2


def draw_absence_alert(annotated_frame, elapsed_time, alert_active):
    """Desenha a interface visual de alerta de ausencia."""
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
    """Desenha a interface visual de deteccao e alerta de posicao prona."""
    x1, y1, x2, y2 = map(int, bbox)

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
