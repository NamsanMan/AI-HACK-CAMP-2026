import cv2

from utils.score_smoothing import risk_color_bgr, risk_label


ROI_COLORS = {
    "left_cheek": (60, 220, 80),
    "right_cheek": (60, 220, 80),
    "forehead": (255, 180, 40),
}


def draw_overlays(frame, bbox=None, rois=None, scores=None, fps=0.0):
    scores = scores or {}
    state = scores.get("risk_state", "Low")
    state_color = risk_color_bgr(state)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, 2)
    if rois:
        for name, pts in rois.items():
            cv2.polylines(frame, [pts], True, ROI_COLORS.get(name, (0, 255, 0)), 2)

    lines = [
        (f"Risk Level: {risk_label(state)}", state_color),
        (f"Risk Score: {scores.get('fake_probability', 0.0):.3f}", state_color),
        (f"Liveness: {scores.get('liveness_score', 0.0):.3f}", (120, 240, 150)),
        (f"HR: {scores.get('estimated_hr', 0.0):.1f} bpm", (255, 255, 255)),
        (f"FPS: {fps:.1f}", (210, 220, 230)),
    ]
    y = 28
    for text, color in lines:
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 28
    return frame
