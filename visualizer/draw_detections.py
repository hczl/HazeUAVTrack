# foggy_uav_system/visualizer/draw_detections.py

import cv2


def draw_detections(image, detections, color=(0, 255, 0)):
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = det.get('confidence', 0)
        cls_id = det.get('class_id', -1)
        label = f"{cls_id}:{conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image
