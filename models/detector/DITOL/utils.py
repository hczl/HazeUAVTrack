import math

import torch
from torch import nn
from torchvision import transforms
from torchvision.ops import generalized_box_iou


def process_batch(batch):
    """
        处理输入 batch 图像与目标框，执行：
        - 图像缩放；
        - 目标框与 ignore 框按比例缩放；
        - 转换为张量格式以供模型使用。

        输入：
            batch: 一个三元组 (images, targets, ignores)
                - images: List[Tensor], 原始图像
                - targets: List[Tensor], 每张图像对应的目标框，格式为 [N, 6]
                - ignores: List[Tensor], 可选的忽略区域框，格式为 [M, 6]

        输出：
            processed_images_tensor: Tensor, [B, 3, new_h, new_w]
            processed_targets: List[Tensor], 每张图像的缩放后目标框
            processed_ignores: List[Tensor], 每张图像的缩放后 ignore 区域
    """
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
    构建用于训练的目标标签张量，包括：
    - objectness mask；
    - 边框回归值；
    - ignore mask。

    输入：
        bboxes: Tensor [N, 6]，每行为一个真实框，格式中第3-6列为左上坐标及宽高
        image_size: Tuple[int, int] 原图大小 (H, W)
        feature_size: Tuple[int, int] 特征图大小 (Hf, Wf)
        ignore_regions: Tensor [M, 6], 可选的忽略区域

    输出：
        gt_obj: Tensor [1, Hf, Wf]，每个位置是否为中心点（硬标签）
        gt_box: Tensor [4, Hf, Wf]，每个位置的边框回归目标（dx, dy, w, h）
        ignore_mask: Tensor [1, Hf, Wf]，标记哪些位置属于 ignore 区域
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

class GIoULoss(nn.Module):
    """
       广义 IoU 损失函数（Generalized IoU Loss），用于计算预测框与真实框之间的空间重叠程度。

       使用：
           输入：
               pred_boxes: Tensor [N, 4]，预测框坐标，格式为 (x1, y1, x2, y2)
               target_boxes: Tensor [N, 4]，真实框坐标
           输出：
               Tensor: 标量，平均 GIoU 损失
    """
    def __init__(self):
        super(GIoULoss, self).__init__()

    def forward(self, pred_boxes, target_boxes):
        '''
        pred_boxes and target_boxes: [N, 4] in (x1, y1, x2, y2) format
        '''
        ious = generalized_box_iou(pred_boxes, target_boxes)  # [N, N]
        loss = 1 - torch.diag(ious)  # assume paired boxes
        return loss.mean()


def generate_soft_target(cx, cy, Hf, Wf, sigma=1.0, device='cpu'):
    """
    在特征图上以 (cx, cy) 为中心生成一个二维高斯热图，作为 objectness 的软标签。

    输入：
        cx, cy: 中心点坐标（float，特征图尺度）
        Hf, Wf: 特征图高度、宽度
        sigma: 高斯标准差，控制热图扩散范围
        device: 指定 tensor 的计算设备

    输出：
        heatmap: Tensor [1, Hf, Wf]，值在 (0,1) 间，中心值最大
    """
    x_grid = torch.arange(Wf, device=device).float()
    y_grid = torch.arange(Hf, device=device).float()
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
    dist_sq = (xx - cx)**2 + (yy - cy)**2
    heatmap = torch.exp(-dist_sq / (2 * sigma**2))
    return heatmap.unsqueeze(0)  # shape [1, Hf, Wf]


def center_sampling_mask(cx, cy, w, h, Hf, Wf, radius=0.25, device='cpu'):
    """
    生成一个矩形的二值掩码，表示以 (cx, cy) 为中心、以 (w, h) 为参考大小的采样区域，
    用于限制 soft target 的范围，提高训练稳定性。

    输入：
        cx, cy: 中心坐标（float，特征图尺度）
        w, h: 目标宽高（float，特征图尺度）
        Hf, Wf: 特征图大小
        radius: 采样区域相对目标宽高的比例（默认 0.25）
        device: tensor 所在设备

    输出：
        mask: Tensor [1, Hf, Wf]，值为 0 或 1，标记采样区域
    """
    x_grid = torch.arange(Wf, device=device).float()
    y_grid = torch.arange(Hf, device=device).float()
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
    dist_x = (xx - cx).abs()
    dist_y = (yy - cy).abs()
    mask = (dist_x < radius * w) & (dist_y < radius * h)
    return mask.float().unsqueeze(0)


