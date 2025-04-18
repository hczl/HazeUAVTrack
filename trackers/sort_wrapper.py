# foggy_uav_system/trackers/sort_wrapper.py
import numpy as np
from sort import Sort

class SORTTracker:
    def __init__(self):
        self.tracker = Sort()

    def update(self, detections, image):
        # 将检测结果转换为 [x1, y1, x2, y2, score] 格式
        dets = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['confidence']
            dets.append([x1, y1, x2, y2, score])

        tracks = self.tracker.update(np.array(dets))
        output = []
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            output.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'track_id': int(track_id)
            })
        return output


def load_tracker(name):
    if name == 'SORT':
        return SORTTracker()
    else:
        raise ValueError(f"Unsupported tracker: {name}")
