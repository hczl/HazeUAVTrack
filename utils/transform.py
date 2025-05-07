import math
import os

import torchvision.transforms.functional as F




def scale_ground_truth_boxes(gt_boxes_all, orig_size, resized_size):
    """
    将所有帧的GT框从原始图像尺寸映射到缩放后的图像尺寸。
    """
    orig_w, orig_h = orig_size
    new_w, new_h = resized_size
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    scaled_all = []
    for boxes in gt_boxes_all:
        scaled_frame = [
            [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y, obj_id]
            for x1, y1, x2, y2, obj_id in boxes
        ]
        scaled_all.append(scaled_frame)
    return scaled_all

def scale_ignore_regions(ignore_masks_all, orig_size, resized_size):
    """
    将所有帧的忽略区域从原始图像尺寸映射到缩放后的图像尺寸。
    """
    orig_w, orig_h = orig_size
    new_w, new_h = resized_size
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    scaled_all = []
    for ignore_boxes in ignore_masks_all:
        scaled_frame = [
            [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            for x1, y1, x2, y2 in ignore_boxes
        ]
        scaled_all.append(scaled_frame)
    return scaled_all

def load_annotations(label_dir, ignore_dir, num_frames):
    gts, ignores = [], []

    for idx in range(num_frames):
        frame_name = f"img{idx+1:06d}.txt"
        label_path = os.path.join(label_dir, frame_name)
        ignore_path = os.path.join(ignore_dir, frame_name)

        frame_gt = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        _, target_id, x, y, w, h = map(float, parts[:6])
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        frame_gt.append([x1, y1, x2, y2, int(target_id)])
        gts.append(frame_gt)

        frame_ignore = []
        if os.path.exists(ignore_path):
            with open(ignore_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        _, target_id, x, y, w, h = map(float, parts[:6])
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        frame_ignore.append([x1, y1, x2, y2])
        ignores.append(frame_ignore)

    return gts, ignores