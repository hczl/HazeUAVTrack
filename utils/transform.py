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

def load_annotations(label_dir,
                     ignore_dir,
                     num_frames,
                     wanted_cls=(1, 2, 3),  # car/truck/bus
                     occ_max=3,
                     oov_max=3):
    gts, ignores = [], []

    for idx in range(num_frames):
        fname = f"img{idx+1:06d}.txt"
        lab_p = os.path.join(label_dir, fname)
        ign_p = os.path.join(ignore_dir, fname)

        frame_gt, frame_ig = [], []

        # ---------- GT ------------
        if os.path.isfile(lab_p):
            with open(lab_p) as fh:
                for ln in fh:
                    parts = ln.strip().split(',')
                    if len(parts) < 9:
                        continue
                    (frame_i, tid, x, y, w, h,
                     outv, occ, cls) = map(float, parts[:9])

                    if int(frame_i) != idx+1:
                        continue                         # 只取这一帧
                    if int(tid) <= 0:
                        continue                         # -1 / 0 忽略
                    if int(cls) not in wanted_cls:
                        continue
                    if occ > occ_max or outv > oov_max:
                        # 高遮挡、出画 → 忽略框
                        frame_ig.append([x, y, x+w, y+h])
                        continue

                    x1, y1, x2, y2 = x, y, x+w, y+h
                    frame_gt.append([x1, y1, x2, y2, int(tid)])

        # ---------- Ignore (额外文件) ----------
        if os.path.isfile(ign_p):
            with open(ign_p) as fh:
                for ln in fh:
                    parts = ln.strip().split(',')
                    if len(parts) < 6:
                        continue
                    _, _, x, y, w, h = map(float, parts[:6])
                    frame_ig.append([x, y, x+w, y+h])

        gts.append(frame_gt)
        ignores.append(frame_ig)

    return gts, ignores
