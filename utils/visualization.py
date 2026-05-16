import cv2


ROI_COLORS = {
    "left_cheek": (60, 220, 80),
    "right_cheek": (60, 220, 80),
    "forehead": (255, 180, 40),
}


def draw_overlays(frame, bbox=None, rois=None, scores=None, fps=0.0):
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 210, 255), 2)
    if rois:
        for name, pts in rois.items():
            cv2.polylines(frame, [pts], True, ROI_COLORS.get(name, (0, 255, 0)), 2)

    scores = scores or {}
    lines = [
        f"Risk(fake): {scores.get('fake_probability', 0.0):.3f}",
        f"Liveness: {scores.get('liveness_score', 0.0):.3f}",
        f"Confidence: {scores.get('confidence_score', 0.0):.3f}",
        f"HR: {scores.get('estimated_hr', 0.0):.1f} bpm",
        f"FPS: {fps:.1f}",
    ]
    y = 28
    for text in lines:
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28
    return frame

