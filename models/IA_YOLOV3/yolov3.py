import math

import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader  # Import DataLoader for type hinting
from ultralytics.utils.tal import make_anchors
from typing import Generator, Tuple, Any, Dict  # 导入用于类型提示
from ultralytics.utils.ops import xywh2xyxy, xywhn2xyxy, xyxy2xywh # 需要 xywhn2xyxy 转换忽略框，xyxy2xywh 计算中心点


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
    """
    Helper function to process a single batch from the original dataloader.
    Handles two cases: (images, targets) or (images, targets, ignore).
    Returns:
        - processed_images_tensor: Stacked batch of processed images [B, C, H, W]
        - processed_targets_dict: Dictionary for targets {"batch_idx", "cls", "bboxes"}
        - processed_ignore_list (optional, only if batch had ignore data):
          List of tensors, where each tensor is the YOLO format ignore boxes
          [num_boxes, 5] for the corresponding image in the batch.
    """

    # --- Common processing lists ---
    processed_images = []
    # Lists to collect data for the combined batch target dictionary
    all_batch_indices = []
    all_classes = []
    all_bboxes = []

    # --- Specific lists for ignore data (only used in 3-tuple case) ---
    all_yolo_ignore_boxes_list = []

    # --- Branching based on batch size ---
    if len(batch) == 2:
        images, targets = batch
        has_ignore = False
        # Create a dummy ignore list for zipping later, will be ignored in processing
        ignore_data_list = [None] * len(images)
    elif len(batch) == 3:
        images, targets, ignore_data_list = batch
        has_ignore = True
    else:
        raise ValueError(f"Unsupported batch format with {len(batch)} elements. Expected 2 or 3.")

    # --- Loop through each image and its corresponding data ---
    # Use zip with a dummy ignore list for the 2-tuple case
    for i, (img, target, ignore_data) in enumerate(zip(images, targets, ignore_data_list)):
        # img: [C, H, W] tensor
        # target: [num_target_boxes, 9] tensor
        # ignore_data: [num_ignore_boxes, 9] tensor (or None in 2-tuple case)

        original_h, original_w = img.shape[1], img.shape[2]  # Get original height and width

        # Define a max size, e.g., 640
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
        scale_factor_w_final = new_w / original_w if original_w > 0 else 1.0
        scale_factor_h_final = new_h / original_h if original_h > 0 else 1.0

        # Resize the image
        # Note: transforms.functional.resize expects (H, W) tuple for size
        # Ensure image is float before resizing if necessary, or handle type appropriately
        # Assuming input image is already a tensor suitable for resizing (e.g., float or uint8)
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

        # Convert the scaled target to YOLO format [class_id, x_center, y_center, w, h] (normalized)
        # Pass the final image size (new_h, new_w) for normalization
        # Assumes convert_targets_to_yolo takes (H, W) tuple
        yolo_boxes_for_image = convert_targets_to_yolo(target_scaled, (new_h, new_w))

        # Collect the converted target boxes and batch indices for this image
        if yolo_boxes_for_image.numel() > 0:  # Check if there are any boxes after conversion
            num_boxes = yolo_boxes_for_image.shape[0]
            # Create batch index tensor for these boxes
            batch_idx_tensor = torch.full((num_boxes,), i, dtype=torch.long, device=yolo_boxes_for_image.device)
            # Extract class IDs (first column)
            cls_tensor = yolo_boxes_for_image[:, 0].float()  # Ensure class ID is float
            # Extract bounding boxes (last 4 columns)
            bboxes_tensor = yolo_boxes_for_image[:, 1:5]

            # Append to the batch lists for targets
            all_batch_indices.append(batch_idx_tensor)
            all_classes.append(cls_tensor)
            all_bboxes.append(bboxes_tensor)

        # --- Process Ignore Data (Only in 3-tuple case) ---
        if has_ignore and ignore_data is not None:
            ignore_scaled = ignore_data.clone() if ignore_data.numel() > 0 else ignore_data # Clone only if not empty

            if ignore_scaled.numel() > 0:
                 # Apply scaling to x, y, w, h (assuming cols 2-5 are bbox)
                 # Adjust indices [2, 3, 4, 5] if your actual format is different.
                ignore_scaled[:, 2] *= scale_factor_w_final  # x coordinate scaled
                ignore_scaled[:, 3] *= scale_factor_h_final  # y coordinate scaled
                ignore_scaled[:, 4] *= scale_factor_w_final  # w coordinate scaled
                ignore_scaled[:, 5] *= scale_factor_h_final  # h coordinate scaled

            # Convert the scaled ignore data to YOLO format
            # Pass the final image size (new_h, new_w) for normalization
            # Assumes convert_targets_to_yolo takes (H, W) tuple
            yolo_ignore_boxes = convert_targets_to_yolo(ignore_scaled, (new_h, new_w))

            # Append the YOLO ignore boxes tensor for this image to the list
            all_yolo_ignore_boxes_list.append(yolo_ignore_boxes)
        elif has_ignore and ignore_data is None:
             # If ignore_data was None for this image (shouldn't happen with zip, but defensive)
             all_yolo_ignore_boxes_list.append(torch.empty(0, 5, dtype=torch.float32)) # Append empty tensor

    # --- After processing all images in the batch ---

    # Concatenate the collected TARGET data
    if len(all_batch_indices) > 0:
        final_batch_indices = torch.cat(all_batch_indices, dim=0)
        final_classes = torch.cat(all_classes, dim=0)
        final_bboxes = torch.cat(all_bboxes, dim=0)
    else:
        # Handle case where the entire batch has no objects (targets)
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
    processed_images_tensor = torch.stack(processed_images, dim=0)

    # --- Return based on original batch size ---
    if has_ignore:
        # Return processed images, targets dict, and list of ignore boxes
        return processed_images_tensor, processed_targets_dict, all_yolo_ignore_boxes_list
    else:
        # Return processed images and targets dict
        return processed_images_tensor, processed_targets_dict


def process_and_return_loaders(dataloader: DataLoader) -> Generator[Any, Any, Any]:
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
    return processed_dataloader_gen


def changeed__call__(self, preds, batch):
    """
    计算 box, cls 和 dfl 损失的总和，并乘以 batch size。
    包含处理忽略区域的逻辑，忽略落在指定忽略区域内的预测框。
    """
    # 初始化损失张量：box, cls, dfl
    loss = torch.zeros(3, device=self.device)

    # 从 preds 中获取特征图 (feats)。preds 可能是元组 (train_output, feats) 或直接是 feats
    feats = preds[1] if isinstance(preds, tuple) else preds

    # 将所有特征图展平并拼接，然后分割为分布预测 (pred_distri) 和类别分数 (pred_scores)
    # 假设 feats 是一个张量列表 [f1, f2, f3]，来自不同的尺度
    # 每个张量 fi 的形状是 [batch_size, num_features, height_i, width_i]
    # num_features = self.no = reg_max * 4 + nc
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
        (self.reg_max * 4, self.nc), 1
    )

    # 调整维度顺序为 [batch_size, num_anchors, num_features_per_anchor]
    pred_scores = pred_scores.permute(0, 2, 1).contiguous() # [b, h*w, nc]
    pred_distri = pred_distri.permute(0, 2, 1).contiguous() # [b, h*w, reg_max*4]

    # 获取数据类型和 batch size
    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    # 计算有效的输入图像尺寸 (高, 宽)，基于第一个特征图的尺寸和步长
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0] # 图像尺寸 (h,w)
    # 生成所有特征图上的锚点中心坐标和对应的步长张量
    # anchor_points: [h*w, 2] (步长缩放后的中心点), stride_tensor: [h*w, 1] (每个锚点的步长)
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
    num_anchors = anchor_points.shape[0] # 所有尺度上的总锚点数量

    # 处理真实标签 (Targets)
    # targets: [批次中所有 gt 的总数, 6] -> [batch_idx, cls, x, y, w, h] (归一化)
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    # preprocess 将归一化 xywh 目标转换为像素 xyxy，并按 batch_idx 分组
    # imgsz[[1, 0, 1, 0]] 是 [h, w, h, w]，用于缩放 xyxy
    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    # 分割为真实类别标签和真实边界框
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # gt_labels: [b, max_gt, 1], gt_bboxes: [b, max_gt, 4] (像素 xyxy)
    # 创建真实框掩码，如果 gt_bbox 不是全零则为 True
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0) # mask_gt: [b, max_gt, 1]

    # 处理预测边界框 (Pboxes)
    # 使用锚点和预测分布解码得到预测边界框 (像素 xyxy)
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # pred_bboxes: [b, h*w, 4] (xyxy, 步长缩放后的)

    # 任务对齐匹配器 (Task-Aligned Assigner)
    # 根据对齐度量将预测框与真实框进行匹配
    # 返回匹配的目标、目标分数和前景掩码 (fg_mask)
    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
        pred_scores.detach().sigmoid(), # 使用分离梯度的分数进行匹配
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), # 使用分离梯度的像素 xyxy 预测框进行匹配
        anchor_points * stride_tensor, # 像素坐标的锚点
        gt_labels, # 真实标签 [b, max_gt, 1]
        gt_bboxes, # 真实边界框 [b, max_gt, 4] (像素 xyxy)
        mask_gt, # 真实框掩码 [b, max_gt, 1]
    ) # target_bboxes: [b, h*w, 4] (匹配到的 gt 框像素 xyxy), target_scores: [b, h*w] (匹配分数), fg_mask: [b, h*w] (布尔掩码)

    # --- 开始处理忽略区域 ---
    # 初始化一个掩码，用于标记应该被忽略的预测框
    ignore_mask = torch.zeros_like(fg_mask, dtype=torch.bool) # 形状 [b, h*w]

    # 检查 batch 数据中是否存在忽略区域且不为空
    if "ignored_bboxes" in batch and batch["ignored_bboxes"].numel() > 0:
        # 将忽略区域数据移动到设备上
        ignored_bboxes_batch = batch["ignored_bboxes"].to(self.device)
        # ignored_bboxes_batch 的格式: [batch_idx, cls, x_c, y_c, w, h] (归一化)

        # 提取忽略区域的 batch 索引
        batch_idx_ign = ignored_bboxes_batch[:, 0].long()
        # 提取归一化 xywh 忽略区域坐标
        xywh_ign_norm = ignored_bboxes_batch[:, 2:6] # 形状 [所有忽略区域总数, 4]

        # 将归一化 xywh 忽略框转换为像素 xyxy 坐标
        # imgsz 是 [w, h]，xywhn2xyxy 需要 (归一化 xywh, 宽, 高)
        xyxy_ign_pixel = xywhn2xyxy(xywh_ign_norm, imgsz[1], imgsz[0]) # 形状 [所有忽略区域总数, 4] (像素 xyxy)

        # 将预测边界框 (已由步长缩放) 转换为像素 xyxy 坐标
        pred_bboxes_pixel = pred_bboxes * stride_tensor # 形状 [b, num_anchors, 4] (像素 xyxy)

        # 计算预测边界框的中心点（像素坐标）
        # 使用 xyxy2xywh 转换，然后取前两列 (x_c, y_c)
        pred_centers_pixel = xyxy2xywh(pred_bboxes_pixel)[..., :2] # 形状 [b, num_anchors, 2] (像素 xy)

        # 遍历 batch 中的每张图像
        for i in range(batch_size):
            # 选取当前图像的预测中心点
            centers_i = pred_centers_pixel[i] # 形状 [num_anchors, 2]

            # 选取当前图像的忽略区域框
            boxes_i_filter = (batch_idx_ign == i)
            boxes_i = xyxy_ign_pixel[boxes_i_filter] # 形状 [当前图像的忽略区域数 N_i, 4] (像素 xyxy)

            # 如果当前图像有忽略区域框，则进行中心点检查
            if boxes_i.numel() > 0:
                # 检查每个预测中心点是否落在当前图像的任何一个忽略框内
                # 调整 centers_i 形状以进行广播: (num_anchors, 1, 2)
                # 调整 boxes_i 形状以进行广播: (1, N_i, 4)
                centers_i_reshaped = centers_i.unsqueeze(1) # (num_anchors, 1, 2)
                boxes_i_reshaped = boxes_i.unsqueeze(0) # (1, N_i, 4)

                # 检查 x 坐标中心是否落在任何忽略框的 [x1, x2) 范围内
                x_inside = (centers_i_reshaped[..., 0] >= boxes_i_reshaped[..., 0]) & \
                           (centers_i_reshaped[..., 0] < boxes_i_reshaped[..., 2]) # (num_anchors, N_i)

                # 检查 y 坐标中心是否落在任何忽略框的 [y1, y2) 范围内
                y_inside = (centers_i_reshaped[..., 1] >= boxes_i_reshaped[..., 1]) & \
                           (centers_i_reshaped[..., 1] < boxes_i_reshaped[..., 3]) # (num_anchors, N_i)

                # 对于当前图像，如果一个预测框的中心点落入 *任何一个* 忽略框内，则标记为忽略
                is_ignored_i = (x_inside & y_inside).any(dim=1) # 形状 (num_anchors,)

                # 更新总的忽略掩码
                ignore_mask[i] = is_ignored_i

    # 将忽略掩码应用到前景掩码
    # 一个预测只有在被匹配器标记为前景 *并且* 不被忽略时，才被认为是真正的前景
    fg_mask = fg_mask & ~ignore_mask # 形状 [b, h*w]
    # --- 忽略区域处理结束 ---

    # 根据更新后的 fg_mask 重新计算 target_scores_sum
    # 只对仍然被认为是前景的预测计算分数总和
    target_scores_sum = max(target_scores[fg_mask].sum(), 1)

    # 计算类别损失 (Cls loss)
    # 计算所有预测的 BCE Loss，然后只对更新后的 fg_mask 为 True 的预测进行求和
    cls_loss_unreduced = self.bce(pred_scores, target_scores.to(dtype)) # 形状 [b, h*w]
    loss[1] = cls_loss_unreduced[fg_mask].sum() / target_scores_sum # BCE

    # 计算边界框损失 (Bbox loss)
    # 这部分已经依赖于 fg_mask。
    # bbox_loss 函数期望 target_bboxes 是经过步长缩放的。
    if fg_mask.sum():
        # 注意: target_bboxes 从 assigner 输出时已经是经过 stride_tensor 缩放的。
        # 原始代码在这里将其除以 stride_tensor 后再传给 bbox_loss。
        # 我们保持这个行为一致。
        target_bboxes_scaled = target_bboxes / stride_tensor # 形状 [b, h*w, 4] (匹配到的 gt 框经过步长缩放)

        loss[0], loss[2] = self.bbox_loss(
            pred_distri, # 形状 [b, h*w, reg_max*4]
            pred_bboxes, # 形状 [b, h*w, 4] (xyxy, 步长缩放后的)
            anchor_points, # 形状 [h*w, 2]
            target_bboxes_scaled, # 形状 [b, h*w, 4] (匹配到的 gt 框经过步长缩放)
            target_scores, # 形状 [b, h*w] (匹配分数) - bbox_loss 会根据 fg_mask 过滤它
            target_scores_sum, # 标量
            fg_mask # 形状 [b, h*w] (更新后的布尔掩码)
        )
    else:
        # 如果经过忽略处理后没有剩余的前景预测框，则 bbox 和 dfl 损失设为零
        loss[0] = torch.tensor(0.0, device=self.device)
        loss[2] = torch.tensor(0.0, device=self.device)

    # 应用损失增益系数
    loss[0] *= self.hyp["box"]  # box gain

    loss[1] *= self.hyp['cls']  # cls gain

    loss[2] *= self.hyp['dfl'] # dfl gain

    # 返回总损失（乘以 batch size）和分离梯度的损失分量
    return loss * batch_size, loss.detach() # loss(box, cls, dfl)