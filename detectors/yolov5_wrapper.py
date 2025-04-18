# foggy_uav_system/detectors/yolov5_wrapper.py

import torch
from pathlib import Path

class YOLOv5Detector:
    def __init__(self, model_path='yolov5s.pt', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.to(self.device)

    def predict(self, image):
        results = self.model(image)
        detections = results.xyxy[0].cpu().numpy()  # xyxy, conf, cls
        output = []
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            output.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'class_id': int(cls_id)
            })
        return output


def load_detector(name):
    if name == 'YOLOv5':
        return YOLOv5Detector()
    else:
        raise ValueError(f"Unsupported detector: {name}")
