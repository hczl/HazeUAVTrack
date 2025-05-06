# models/trackers/boxmot_tracker.py

from boxmot.tracker_zoo import create_tracker
import torch

class BoxMOTTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracker = create_tracker(cfg['boxmot']['tracker_name'], cfg['boxmot']['args'])
        self.frame_id = 0

    def update(self, detections, image_tensor, frame_id=None):
        self.frame_id += 1 if frame_id is None else frame_id

        # BoxMOT 要求 detections 是一个 Nx6 的 Tensor: xyxy + conf + cls
        if not isinstance(detections, torch.Tensor):
            detections = torch.tensor(detections)

        # image_tensor: [1, 3, H, W] -> [H, W, 3] (BoxMOT 接收 numpy 图像)
        img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        tracks = self.tracker.update(detections=detections, img=img_np)
        results = []
        for track in tracks:
            x1, y1, x2, y2 = track.tlbr
            track_id = track.track_id
            cls = track.cls
            results.append([x1, y1, x2, y2, track_id, cls])
        return results
