# models/trackers/boxmot_tracker.py
import os

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
            device=cfg.get('device', 'cuda')
        )

        self.frame_id = 0

    def update(self, detections, image_tensor, frame_id=None):

        self.frame_id += 1 if frame_id is None else frame_id

        # image_tensor: [1, 3, H, W] -> [H, W, 3] (BoxMOT 接收 numpy 图像)
        img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        tracks = self.tracker.update(detections, img_np)


        return tracks
