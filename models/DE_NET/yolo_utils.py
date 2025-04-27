import math

import torch
from torchvision import transforms
from torch.utils.data import DataLoader  # Import DataLoader for type hinting
from ultralytics.utils.tal import make_anchors
from typing import Generator, Tuple # 导入用于类型提示

def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def convert_targets_to_yolo(target_tensor, image_size):
    """
    将自定义格式的目标标签转换为YOLO格式:
    [frame_index, target_id, x, y, w, h, out-of-view, occlusion, class_id]
    转换为：
    [class_id, x_center, y_center, width, height]，并进行归一化
    这个函数处理的是单个图像的target tensor。
    """
    converted_targets = []
    img_h, img_w = image_size  # image_size is (height, width)
    # Handle empty target tensor
    if target_tensor.numel() == 0:
        # Return an empty tensor with the expected shape [0, 5]
        return torch.empty(0, 5, dtype=torch.float32)

    # Ensure target_tensor is treated as a list of rows
    if target_tensor.dim() == 1: # Handle case with a single box
         target_tensor = target_tensor.unsqueeze(0)

    for target in target_tensor:
        # Ensure target has enough dimensions (at least 9 elements)
        if len(target) < 9:
            # print(f"Warning: Target tensor row has unexpected length {len(target)}. Skipping.")
            continue

        class_id = int(target[8])  # object_category
        # Assuming x, y, w, h are already scaled to the new image size
        x = float(target[2])  # top-left x
        y = float(target[3])  # top-left y
        w = float(target[4])  # width
        h = float(target[5])  # height

        # 转为中心坐标并归一化 (using the new_w, new_h passed in image_size)
        # Ensure division by zero is handled if image_size dimensions are zero
        x_center = (x + w / 2.0) / img_w if img_w > 0 else 0.0
        y_center = (y + h / 2.0) / img_h if img_h > 0 else 0.0
        norm_w = w / img_w if img_w > 0 else 0.0
        norm_h = h / img_h if img_h > 0 else 0.0

        # Clamp normalized coordinates to be within [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))

        converted_targets.append([class_id, x_center, y_center, norm_w, norm_h])

    # Stack into a single tensor. If list is empty, this will be torch.Size([0, 5])
    # Use torch.stack for lists of tensors, or torch.tensor for list of lists.
    # torch.tensor is fine here.
    if not converted_targets: # Handle case where no valid targets were found
         return torch.empty(0, 5, dtype=torch.float32)

    return torch.tensor(converted_targets, dtype=torch.float32)


# Keep process_batch function internal or define it outside if preferred
def process_batch(batch):
    """Helper function to process a single batch from the original dataloader."""
    images, targets = batch  # images: [batch_size, C, H, W], targets: 列表 [target_tensor_for_img1, target_tensor_for_img2, ...]

    processed_images = []
    # Lists to collect data for the combined batch target dictionary
    all_batch_indices = []
    all_classes = []
    all_bboxes = []

    for i, (img, target) in enumerate(zip(images, targets)):  # 遍历每个图像和标签
        # target for a single image: tensor [num_boxes, 9]

        original_h, original_w = img.shape[1], img.shape[2]  # 获取原始高度和宽度

        max_size = 640

        # Calculate ratio for resizing while maintaining aspect ratio
        r = min(1.0, max_size / original_w)
        r = min(r, max_size / original_h)

        # Calculate new dimensions based on the ratio
        new_h = int(round(original_h * r))
        new_w = int(round(original_w * r))

        # Ensure new dimensions are multiples of 32
        new_h = max(32, math.floor(new_h / 32) * 32)  # Ensure minimum size is 32x32
        new_w = max(32, math.floor(new_w / 32) * 32)

        # Recalculate scale factors based on the *final* new_h, new_w
        # This is crucial for scaling the box coordinates correctly.
        scale_factor_w_final = new_w / original_w if original_w > 0 else 1.0
        scale_factor_h_final = new_h / original_h if original_h > 0 else 1.0

        # Resize the image
        # Note: transforms.functional.resize expects (H, W) tuple for size
        # Ensure image is float before resizing if necessary, or handle type appropriately
        img_resized = transforms.functional.resize(img, (new_h, new_w))
        processed_images.append(img_resized)  # 添加处理后的图像

        target_scaled = target.clone() if target.numel() > 0 else target # Clone only if not empty

        if target_scaled.numel() > 0:
            # Ensure we only scale if there are boxes
            # Apply scaling to x, y, w, h
            target_scaled[:, 2] *= scale_factor_w_final  # x coordinate scaled
            target_scaled[:, 3] *= scale_factor_h_final  # y coordinate scaled
            target_scaled[:, 4] *= scale_factor_w_final  # w coordinate scaled
            target_scaled[:, 5] *= scale_factor_h_final  # h coordinate scaled

        # Now convert the scaled target to YOLO format [class_id, x_center, y_center, w, h] (normalized)
        # Pass the final image size (new_h, new_w) for normalization
        yolo_boxes_for_image = convert_targets_to_yolo(target_scaled, (new_h, new_w))  # Pass tuple (height, width)

        # Collect the converted boxes and batch indices for this image
        if yolo_boxes_for_image.numel() > 0:  # Check if there are any boxes after conversion
            num_boxes = yolo_boxes_for_image.shape[0]
            # Create batch index tensor for these boxes
            # Ensure device consistency if needed (e.g., moving to GPU)
            batch_idx_tensor = torch.full((num_boxes,), i, dtype=torch.long, device=yolo_boxes_for_image.device if yolo_boxes_for_image.device else 'cpu')
            # Extract class IDs (first column)
            cls_tensor = yolo_boxes_for_image[:, 0].float()  # Ensure class ID is float (often required by loss functions)
            # Extract bounding boxes (last 4 columns)
            bboxes_tensor = yolo_boxes_for_image[:, 1:5]

            # Append to the batch lists
            all_batch_indices.append(batch_idx_tensor)
            all_classes.append(cls_tensor)
            all_bboxes.append(bboxes_tensor)

    # After processing all images in the batch, concatenate the collected data
    if len(all_batch_indices) > 0:
        final_batch_indices = torch.cat(all_batch_indices, dim=0)
        final_classes = torch.cat(all_classes, dim=0)
        final_bboxes = torch.cat(all_bboxes, dim=0)
    else:
        # Handle case where the entire batch has no objects
        final_batch_indices = torch.empty(0, dtype=torch.long)
        final_classes = torch.empty(0, dtype=torch.float32)
        final_bboxes = torch.empty(0, 4, dtype=torch.float32) # Ensure 4 columns even if empty

    # Create the target dictionary required by YOLO loss
    processed_targets_dict = {
        "batch_idx": final_batch_indices,
        "cls": final_classes,
        "bboxes": final_bboxes,
    }

    # Stack the list of images into a single tensor [B, C, H, W]
    # Ensure device consistency if needed
    processed_images_tensor = torch.stack(processed_images, dim=0)
    return processed_images_tensor, processed_targets_dict


def process_and_return_loaders(dataloader: DataLoader) -> Tuple[Generator, int]:
    """
    主函数：处理 DataLoader，动态缩放图像使其保持长宽比、最接近原图，且宽高被 32 整除。
    同时将标签转换为 YOLO loss 所需的字典格式。

    参数:
    - dataloader: 需要转换标签的 DataLoader（每个 batch 格式为 (images, targets)）

    返回:
    - 一个元组 (processed_dataloader_gen, total_batches):
        - processed_dataloader_gen: 一个生成器，迭代时产生处理后的 batch（images, targets），
                                其中 targets 是一个字典 {"batch_idx": ..., "cls": ..., "bboxes": ...}
        - total_batches: 原始 dataloader 中的总批次数。

    每个 batch 的输出格式（由 processed_dataloader_gen 产生）：
    - images: [batch_size, C, new_h, new_w] 张量，其中 new_h 和 new_w 是动态计算的
    - targets: 字典，包含：
        - "batch_idx": [N] 张量 (long)，表示每个 bounding box 属于批次中的哪个图像 (0 to batch_size-1)
        - "cls": [N] 张量 (float)，表示每个 bounding box 的类别 ID
        - "bboxes": [N, 4] 张量 (float)，表示每个 bounding box 的 [x_center, y_center, width, height] (已归一化)
        其中 N 是整个批次中所有图像中 bounding box 的总数。
    """

    # Calculate the total number of batches from the original dataloader
    # len(dataloader) gives the number of batches
    total_batches = len(dataloader)

    # Define the generator function
    def get_processed_loader(original_loader: DataLoader):
        # The original DataLoader handles batching the original data.
        # We iterate through its batches and process each one.
        # Note: This generator will produce `total_batches` items.
        for batch in original_loader:
            processed_batch = process_batch(batch)
            yield processed_batch  # Yield the processed batch

    # Create the generator
    processed_dataloader_gen = get_processed_loader(dataloader)

    # Return the generator and the total batch count
    return processed_dataloader_gen, total_batches


def changeed__call__(self, preds, batch):
    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
    loss = torch.zeros(3, device=self.device)  # box, cls, dfl
    feats = preds[1] if isinstance(preds, tuple) else preds
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
        (self.reg_max * 4, self.nc), 1
    )

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

    # Targets
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    # Pboxes
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
    # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
    # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
        # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)

    # Cls loss
    # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
    loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

    # Bbox loss
    if fg_mask.sum():
        target_bboxes /= stride_tensor
        loss[0], loss[2] = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
        )

    loss[0] *= self.hyp["box"]  # box gain
    loss[1] *= self.hyp['cls']  # cls gain
    loss[2] *= self.hyp['dfl'] # dfl gain

    return loss * batch_size, loss.detach()  # loss(box, cls, dfl)