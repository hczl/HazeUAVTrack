import math

import torch
from torchvision import transforms


def process_batch(batch):
    processed_images = []
    processed_targets = []
    processed_ignores = []
    images, targets, ignores = batch
    original_h = 540
    original_w = 1024
    new_h = 320
    new_w = 640
    scale_factor_w_final = new_w / original_w
    scale_factor_h_final = new_h / original_h
    # Use zip with a dummy ignore list for the 2-tuple case
    for i, (img, target, ignore) in enumerate(zip(images, targets, ignores)):

        img_resized = transforms.functional.resize(img, (new_h, new_w))
        processed_images.append(img_resized)  # Append processed image

        # --- Process Target Data (Same logic for both batch sizes) ---
        target_scaled = target.clone() if target.numel() > 0 else target # Clone only if not empty

        if target_scaled.numel() > 0:
            # Apply scaling to x, y, w, h (assuming cols 2-5 are bbox)
            # Adjust indices [2, 3, 4, 5] if your actual format is different.
            target_scaled[:, 2] *= scale_factor_w_final  # x coordinate scaled
            target_scaled[:, 3] *= scale_factor_h_final  # y coordinate scaled
            target_scaled[:, 4] *= scale_factor_w_final  # w coordinate scaled
            target_scaled[:, 5] *= scale_factor_h_final  # h coordinate scaled
        processed_targets.append(target_scaled)
        if ignore is not None:
            ignore_scaled = ignore.clone() if ignore.numel() > 0 else ignore

            if ignore_scaled.numel() > 0:
                # Apply scaling to x, y, w, h (assuming cols 2-5 are bbox)
                # Adjust indices [2, 3, 4, 5] if your actual format is different.
                ignore_scaled[:, 2] *= scale_factor_w_final  # x coordinate scaled
                ignore_scaled[:, 3] *= scale_factor_h_final  # y coordinate scaled
                ignore_scaled[:, 4] *= scale_factor_w_final  # w coordinate scaled
                ignore_scaled[:, 5] *= scale_factor_h_final  # h coordinate scaled
            processed_ignores.append(ignore_scaled)
        else: processed_ignores.append(torch.empty(0, len(target_scaled[0]), dtype=torch.float32))

    processed_images_tensor =torch.stack(processed_images, dim=0)
    return processed_images_tensor, processed_targets, processed_ignores

def build_detection_targets(bboxes, image_size, feature_size, ignore_regions=None):
    """
    bboxes: Tensor of shape [N, 6] (left, top, width, height, ...)
    image_size: (H, W)
    feature_size: (H', W')
    ignore_regions: Tensor of shape [M, 6] if provided
    """
    H, W = image_size
    Hf, Wf = feature_size
    scale_x = Wf / W
    scale_y = Hf / H

    gt_obj = torch.zeros((1, Hf, Wf))  # objectness
    gt_box = torch.zeros((4, Hf, Wf))  # dx, dy, w, h

    for box in bboxes:
        left, top, width, height = box[2:6]
        cx = (left + width / 2.0) * scale_x
        cy = (top + height / 2.0) * scale_y

        i = int(cy)
        j = int(cx)
        if i < 0 or i >= Hf or j < 0 or j >= Wf:
            continue

        dx = cx - j
        dy = cy - i
        gt_obj[0, i, j] = 1
        gt_box[:, i, j] = torch.tensor([dx, dy, width * scale_x, height * scale_y])

    # 构造 ignore mask
    ignore_mask = torch.zeros((1, Hf, Wf))
    if ignore_regions is not None:
        for ign in ignore_regions:
            l, t, w, h = ign[2:6]
            x0 = int((l) * scale_x)
            y0 = int((t) * scale_y)
            x1 = int((l + w) * scale_x)
            y1 = int((t + h) * scale_y)
            ignore_mask[0, y0:y1, x0:x1] = 1

    return gt_obj, gt_box, ignore_mask



