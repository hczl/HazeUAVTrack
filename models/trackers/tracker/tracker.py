# models/trackers/boxmot_tracker.py
import os
from pathlib import Path

import numpy as np
import torch
from boxmot.tracker_zoo import create_tracker
from torch import nn


class tracker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        tracker_type = cfg['method']['track_method']  # e.g. 'bytetrack'
        config_path = os.path.join(os.path.dirname(__file__), 'configs', f'{tracker_type}.yaml')
        self.tracker = create_tracker(
            tracker_type=tracker_type,
            tracker_config=config_path,
            reid_weights=Path('models/trackers/tracker/osnet_x0_25_msmt17.pt'),
            device=cfg.get('device', 'cuda')
        )

        self.frame_id = 0

    def update(self, detections, image_tensor, frame_id=None):

        self.frame_id += 1 if frame_id is None else frame_id

        # image_tensor: [1, 3, H, W] -> [H, W, 3] (BoxMOT 接收 numpy 图像)
        img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        detections_for_tracker = []
        for detection in detections:
            if len(detection) == 5:
                # 如果只有 5 个元素，补充一个默认的类信息
                detection_with_class = np.append(detection, 1)
            elif len(detection) == 6:
                # 如果已经有 6 个元素，则不做任何修改
                detection_with_class = detection
            else:
                # 如果检测数据不符合预期，可以选择打印警告或跳过
                print(f"Warning: Invalid detection format {detection}")
                continue
            detections_for_tracker.append(detection_with_class)
        detections_for_tracker = np.array(detections_for_tracker)
        tracks = self.tracker.update(detections_for_tracker, img_np)
        return tracks
