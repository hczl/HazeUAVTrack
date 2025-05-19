import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import non_max_suppression, xywhn2xyxy, xyxy2xywh, xywh2xyxy
from ultralytics.utils.tal import make_anchors
import types
def custom_v11_call(self, preds, batch):
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
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # [b, h*w, nc]
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # [b, h*w, reg_max*4]

    # 获取数据类型和 batch size
    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    # 计算有效的输入图像尺寸 (高, 宽)，基于第一个特征图的尺寸和步长
    imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # 图像尺寸 (h,w)
    # 生成所有特征图上的锚点中心坐标和对应的步长张量
    # anchor_points: [h*w, 2] (步长缩放后的中心点), stride_tensor: [h*w, 1] (每个锚点的步长)
    anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
    num_anchors = anchor_points.shape[0]  # 所有尺度上的总锚点数量

    # 处理真实标签 (Targets)
    # targets: [批次中所有 gt 的总数, 6] -> [batch_idx, cls, x, y, w, h] (归一化)
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    # preprocess 将归一化 xywh 目标转换为像素 xyxy，并按 batch_idx 分组
    # imgsz[[1, 0, 1, 0]] 是 [h, w, h, w]，用于缩放 xyxy
    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    # 分割为真实类别标签和真实边界框
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # gt_labels: [b, max_gt, 1], gt_bboxes: [b, max_gt, 4] (像素 xyxy)
    # 创建真实框掩码，如果 gt_bbox 不是全零则为 True
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)  # mask_gt: [b, max_gt, 1]

    # 处理预测边界框 (Pboxes)
    # 使用锚点和预测分布解码得到预测边界框 (像素 xyxy)
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # pred_bboxes: [b, h*w, 4] (xyxy, 步长缩放后的)

    # 任务对齐匹配器 (Task-Aligned Assigner)
    # 根据对齐度量将预测框与真实框进行匹配
    # 返回匹配的目标、目标分数和前景掩码 (fg_mask)
    _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
        pred_scores.detach().sigmoid(),  # 使用分离梯度的分数进行匹配
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),  # 使用分离梯度的像素 xyxy 预测框进行匹配
        anchor_points * stride_tensor,  # 像素坐标的锚点
        gt_labels,  # 真实标签 [b, max_gt, 1]
        gt_bboxes,  # 真实边界框 [b, max_gt, 4] (像素 xyxy)
        mask_gt,  # 真实框掩码 [b, max_gt, 1]
    )  # target_bboxes: [b, h*w, 4] (匹配到的 gt 框像素 xyxy), target_scores: [b, h*w] (匹配分数), fg_mask: [b, h*w] (布尔掩码)

    # --- 开始处理忽略区域 ---
    ignore_mask = torch.zeros_like(fg_mask, dtype=torch.bool)  # [b, h*w]

    if "ignored_bboxes" in batch and batch["ignored_bboxes"] is not None and batch["ignored_bboxes"].numel() > 0:
        ignored_bboxes_batch = batch["ignored_bboxes"].to(self.device)  # [N, 5] 或 [N, 6]

        # 判断是否包含 batch_idx
        has_batch_idx = ignored_bboxes_batch.size(1) == 6

        if has_batch_idx:
            # [batch_idx, cls_id, x_c, y_c, w, h]
            batch_idx_ign = ignored_bboxes_batch[:, 0].long()
            xywh_ign_norm = ignored_bboxes_batch[:, 2:6]
        else:
            # [cls_id, x_c, y_c, w, h] —— 没有 batch_idx，所有图共享 ignore 区域
            batch_idx_ign = None
            xywh_ign_norm = ignored_bboxes_batch[:, 1:5]

        # 转换为像素 xyxy
        xyxy_ign_pixel = xywhn2xyxy(xywh_ign_norm, imgsz[1], imgsz[0])  # [N, 4]

        # 预测框中心点
        pred_bboxes_pixel = pred_bboxes * stride_tensor
        pred_centers_pixel = xyxy2xywh(pred_bboxes_pixel)[..., :2]  # [b, h*w, 2]

        for i in range(batch_size):
            centers_i = pred_centers_pixel[i]  # [num_anchors, 2]

            if has_batch_idx:
                boxes_i = xyxy_ign_pixel[batch_idx_ign == i]
            else:
                boxes_i = xyxy_ign_pixel  # 所有图共享

            if boxes_i.numel() > 0:
                centers_i_reshaped = centers_i.unsqueeze(1)
                boxes_reshaped = boxes_i.unsqueeze(0)

                x_inside = (centers_i_reshaped[..., 0] >= boxes_reshaped[..., 0]) & \
                           (centers_i_reshaped[..., 0] < boxes_reshaped[..., 2])
                y_inside = (centers_i_reshaped[..., 1] >= boxes_reshaped[..., 1]) & \
                           (centers_i_reshaped[..., 1] < boxes_reshaped[..., 3])
                is_ignored = (x_inside & y_inside).any(dim=1)

                ignore_mask[i] = is_ignored

    # 应用 ignore 掩码
    fg_mask = fg_mask & ~ignore_mask
    # --- 忽略区域处理结束 ---

    # 根据更新后的 fg_mask 重新计算 target_scores_sum
    # 只对仍然被认为是前景的预测计算分数总和
    target_scores_sum = max(target_scores[fg_mask].sum(), 1)

    # 计算类别损失 (Cls loss)
    # 计算所有预测的 BCE Loss，然后只对更新后的 fg_mask 为 True 的预测进行求和
    cls_loss_unreduced = self.bce(pred_scores, target_scores.to(dtype))  # 形状 [b, h*w]
    loss[1] = cls_loss_unreduced[fg_mask].sum() / target_scores_sum  # BCE

    # 计算边界框损失 (Bbox loss)
    # 这部分已经依赖于 fg_mask。
    # bbox_loss 函数期望 target_bboxes 是经过步长缩放的。
    if fg_mask.sum():
        # 注意: target_bboxes 从 assigner 输出时已经是经过 stride_tensor 缩放的。
        # 原始代码在这里将其除以 stride_tensor 后再传给 bbox_loss。
        # 我们保持这个行为一致。
        target_bboxes_scaled = target_bboxes / stride_tensor  # 形状 [b, h*w, 4] (匹配到的 gt 框经过步长缩放)

        loss[0], loss[2] = self.bbox_loss(
            pred_distri,  # 形状 [b, h*w, reg_max*4]
            pred_bboxes,  # 形状 [b, h*w, 4] (xyxy, 步长缩放后的)
            anchor_points,  # 形状 [h*w, 2]
            target_bboxes_scaled,  # 形状 [b, h*w, 4] (匹配到的 gt 框经过步长缩放)
            target_scores,  # 形状 [b, h*w] (匹配分数) - bbox_loss 会根据 fg_mask 过滤它
            target_scores_sum,  # 标量
            fg_mask  # 形状 [b, h*w] (更新后的布尔掩码)
        )
    else:
        # 如果经过忽略处理后没有剩余的前景预测框，则 bbox 和 dfl 损失设为零
        loss[0] = torch.tensor(0.0, device=self.device)
        loss[2] = torch.tensor(0.0, device=self.device)
    # 应用损失增益系数
    loss[0] *= self.hyp['box']  # box gain

    loss[1] *= self.hyp['cls']  # cls gain

    loss[2] *= self.hyp['dfl']  # dfl gain

    # 返回总损失（乘以 batch size）和分离梯度的损失分量
    return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

def process_batch(batch):
    import torch
    from torch import empty, cat, stack, full, as_tensor

    if len(batch) == 2:
        images, targets = batch
        ignore_data_list = [None] * (len(images) if isinstance(images, list) else images.size(0))
    elif len(batch) == 3:
        images, targets, ignore_data_list = batch
    else:
        raise ValueError(f"Unsupported batch format with {len(batch)} elements. Expected 2 or 3.")

    all_batch_indices_targets, all_classes_targets, all_bboxes_targets = [], [], []
    all_ignore_rows = []

    if not isinstance(images, list):
        images = [img for img in images]

    for i, (img, target, ignore_data) in enumerate(zip(images, targets, ignore_data_list)):
        if img.dim() != 3:
             raise ValueError(f"Image {i} has unexpected dimensions: {img.dim()}. Expected 3 (C, H, W).")
        c, h, w = img.shape

        target_tensor = as_tensor(target, dtype=torch.float32) if target is not None else empty(0, 9, dtype=torch.float32)

        if target_tensor.numel() > 0:
            if target_tensor.dim() == 1:
                 target_tensor = target_tensor.unsqueeze(0)

            converted_targets = []
            for t in target_tensor:
                 if len(t) >= 6:
                    class_id = 0
                    x_tl, y_tl, box_w, box_h = float(t[2]), float(t[3]), float(t[4]), float(t[5])

                    x_c = (x_tl + box_w / 2.0) / w if w > 0 else 0.0
                    y_c = (y_tl + box_h / 2.0) / h if h > 0 else 0.0
                    norm_w = box_w / w if w > 0 else 0.0
                    norm_h = box_h / h if h > 0 else 0.0

                    x_c = max(0.0, min(1.0, x_c))
                    y_c = max(0.0, min(1.0, y_c))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))

                    converted_targets.append([class_id, x_c, y_c, norm_w, norm_h])

            yolo_boxes = torch.tensor(converted_targets, dtype=torch.float32) if converted_targets else empty(0, 5, dtype=torch.float32)

            if yolo_boxes.numel() > 0:
                num_boxes = yolo_boxes.size(0)
                all_batch_indices_targets.append(full((num_boxes,), i, dtype=torch.long))
                all_classes_targets.append(yolo_boxes[:, 0].float())
                all_bboxes_targets.append(yolo_boxes[:, 1:5])

        ignore_tensor = as_tensor(ignore_data, dtype=torch.float32) if ignore_data is not None else empty(0, 9, dtype=torch.float32)

        if ignore_tensor.numel() > 0:
             if ignore_tensor.dim() == 1:
                 ignore_tensor = ignore_tensor.unsqueeze(0)

             converted_ignores = []
             for ign in ignore_tensor:
                  if len(ign) >= 6:
                    class_id = 0
                    x_tl, y_tl, box_w, box_h = float(ign[2]), float(ign[3]), float(ign[4]), float(ign[5])

                    x_c = (x_tl + box_w / 2.0) / w if w > 0 else 0.0
                    y_c = (y_tl + box_h / 2.0) / h if h > 0 else 0.0
                    norm_w = box_w / w if w > 0 else 0.0
                    norm_h = box_h / h if h > 0 else 0.0

                    x_c = max(0.0, min(1.0, x_c))
                    y_c = max(0.0, min(1.0, y_c))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))

                    converted_ignores.append([float(i), float(class_id), x_c, y_c, norm_w, norm_h])

             if converted_ignores:
                 all_ignore_rows.extend(converted_ignores)

    processed_targets_dict = {
        "batch_idx": cat(all_batch_indices_targets) if all_batch_indices_targets else empty(0, dtype=torch.long),
        "cls": cat(all_classes_targets) if all_classes_targets else empty(0, dtype=torch.float32),
        "bboxes": cat(all_bboxes_targets) if all_bboxes_targets else empty((0, 4), dtype=torch.float32),
    }

    if all_ignore_rows:
         processed_targets_dict["ignored_bboxes"] = torch.tensor(all_ignore_rows, dtype=torch.float32)
    else:
         processed_targets_dict["ignored_bboxes"] = empty((0, 6), dtype=torch.float32)

    stacked_images = torch.stack(images)

    return {"img": stacked_images, **processed_targets_dict}


class YOLOV11(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.nc = int(1)
        YOLOV11_wrapper = YOLO('yolo11.yaml')
        YOLOV11_wrapper.load('models/detector/YOLOV11/yolo11n.pt')
        self.YOLOV11 = YOLOV11_wrapper.model
        self.YOLOV11.nc = self.nc
        self.loss_fn = YOLOV11_wrapper.loss
        v8DetectionLoss.__call__ = custom_v11_call
        self.device = cfg['device']

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self, x):
        out = self.YOLOV11(x)
        return out

    @torch.no_grad()
    def predict(self, high_res_images, conf_thresh=0.25, iou_thresh=0.45, max_det=300):
        self.YOLOV11.eval()
        raw_output = self.YOLOV11(high_res_images)

        detections_list_xyxy_conf_cls = non_max_suppression(
            raw_output,
            conf_thres=conf_thresh,
            iou_thres=iou_thresh,
            max_det=max_det,
            classes=None,
            agnostic=False,
        )

        detections_xyxy_conf = []
        for det_tensor in detections_list_xyxy_conf_cls:
             detections_xyxy_conf.append(det_tensor[:, :5])

        return detections_xyxy_conf

    def forward_loss(self, images, targets, ignore_list):
        self.YOLOV11.train()
        processed_batch_dict = process_batch((images, targets, ignore_list))

        processed_batch_dict = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                                for k, v in processed_batch_dict.items()}

        YOLOV11_output = self.YOLOV11(processed_batch_dict["img"])

        yolov3_loss_tuple = self.loss_fn(processed_batch_dict, YOLOV11_output)

        total_loss_for_backprop = yolov3_loss_tuple[0].sum()

        detached_component_losses = yolov3_loss_tuple[1]

        return {
            'total_loss': total_loss_for_backprop,
            'box_loss': detached_component_losses[0],  # 标量
            'cls_loss': detached_component_losses[1],  # 标量
            'dfl_loss': detached_component_losses[2],  # 标量

        }

