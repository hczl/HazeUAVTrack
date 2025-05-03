import torch
import yaml
from torch import nn
from torchvision.models import convnext_tiny
from torchvision.ops import nms

from models.detector.DITOL.utils import generate_soft_target, center_sampling_mask, compute_iou, GIoULoss, \
    compute_box_coordinates


class DITOL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        with open('models/detector/DITOL/ditol_config.yaml', 'r') as f:
            self.ditol_config = yaml.safe_load(f)

        # 主干网络用于特征提取
        self.backbone = convnext_tiny(pretrained=True).features

        # 本地化头部结构：用于目标检测
        self.localization_head = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.bbox_regression = nn.Conv2d(128, 4, kernel_size=1)  # 边界框回归
        self.objectness_score = nn.Conv2d(128, 1, kernel_size=1)  # 置信度评分
        self.bce_loss = nn.BCELoss()
        self.iou_loss = GIoULoss()

    def forward(self, x):
        features = self.backbone(x)
        loc_features = self.localization_head(features)
        bbox_pred = self.bbox_regression(loc_features)
        objectness = self.objectness_score(loc_features).sigmoid()
        return features, bbox_pred, objectness

    @torch.no_grad()
    def predict(self, dehaze_imgs, conf_thresh=0.5, iou_thresh=0.7):
        self.eval()
        inputs = dehaze_imgs.to(self.cfg['device'])
        _, bbox_pred, objectness = self(inputs)

        B, _, Hf, Wf = objectness.shape
        _, _, H, W = inputs.shape
        scale_x, scale_y = W / Wf, H / Hf

        results = []
        for b in range(B):
            # 提取当前样本的 objectness 和 bbox
            obj_map = objectness[b, 0]  # (Hf, Wf)
            keep_mask = obj_map > conf_thresh
            if keep_mask.sum() == 0:
                results.append([])
                continue

            # 提取满足条件的索引
            indices = keep_mask.nonzero(as_tuple=False)
            i_coords = indices[:, 0]
            j_coords = indices[:, 1]

            # 取出对应位置的 bbox
            boxes = bbox_pred[b][:, i_coords, j_coords]  # shape (4, N)
            boxes = boxes.permute(1, 0)  # (N, 4)

            # 还原到原图尺度
            boxes[:, [0, 2]] *= scale_x  # x1, x2
            boxes[:, [1, 3]] *= scale_y  # y1, y2

            scores = obj_map[i_coords, j_coords]  # shape (N,)

            # NMS
            keep = self.nms(boxes, scores, iou_thresh)
            boxes = boxes[keep]
            scores = scores[keep]

            batch_result = [[*boxes[i].tolist(), scores[i].item()] for i in range(len(boxes))]
            results.append(batch_result)

        return results

    def forward_loss(self, dehaze_imgs, targets, ignore_list):
        inputs = dehaze_imgs.to(self.cfg['device'])
        _, bbox_pred, objectness = self(inputs)
        B, _, Hf, Wf = objectness.shape
        _, _, H, W = dehaze_imgs.shape
        scale_x, scale_y = W / Wf, H / Hf
        det_loss = self.prepare_targets(targets, ignore_list, bbox_pred, objectness, B, Hf, Wf, H, W, scale_x, scale_y)
        return det_loss

    def nms(self, boxes, scores, iou_thresh):
        return nms(boxes, scores, iou_thresh)

    def prepare_targets(self, targets, ignore_list, bbox_pred, objectness, B, Hf, Wf, H, W, scale_x, scale_y):
        total_loss = torch.tensor(0.).to(self.cfg['device'])
        all_objectness_loss = torch.tensor(0.).to(self.cfg['device'])
        all_bbox_loss = torch.tensor(0.).to(self.cfg['device'])

        for i in range(B):  # Loop over batch
            gt_boxes = targets[i]
            obj_target = torch.zeros((1, Hf, Wf), device=self.cfg['device'])

            pred_boxes_list, target_boxes_list = [], []

            for box in gt_boxes:
                if len(box) < 6:
                    continue  # Skip invalid box

                left, top, width, height = box[2:6]
                cx_f = (left + width / 2.0) * Wf / W
                cy_f = (top + height / 2.0) * Hf / H

                j_int = int(cx_f)
                i_int = int(cy_f)

                if 0 <= i_int < Hf and 0 <= j_int < Wf:
                    soft_target = generate_soft_target(cx_f, cy_f, Hf, Wf, sigma=1.5, device=self.cfg['device'])
                    mask = center_sampling_mask(cx_f, cy_f, width * Wf / W, height * Hf / H, Hf, Wf, radius=0.3,
                                                device=self.cfg['device'])
                    obj_target = torch.max(obj_target, soft_target * mask)

                    # 获取预测框坐标
                    pred_box_coords = compute_box_coordinates(
                        bbox_pred[i, 0, i_int, j_int], bbox_pred[i, 1, i_int, j_int],
                        bbox_pred[i, 2, i_int, j_int], bbox_pred[i, 3, i_int, j_int],
                        j_int, i_int, scale_x, scale_y)

                    # 忽略掉与ignore区域有高IOU的预测框
                    ignore_boxes = ignore_list[i] if isinstance(ignore_list[i], list) else []
                    should_ignore = False
                    for ign in ignore_boxes:
                        if len(ign) < 6:
                            continue
                        left, top, width, height = ign[2:6]
                        x1 = left
                        y1 = top
                        x2 = left + width
                        y2 = top + height
                        ign_box_coords = [x1, y1, x2, y2]

                        iou = compute_iou(pred_box_coords, ign_box_coords)
                        if iou > self.cfg['iou_threshold']:
                            should_ignore = True
                            break

                    if should_ignore:
                        continue

                    # GT框转换为当前特征图位置的偏移表示
                    gt_dx = cx_f - j_int
                    gt_dy = cy_f - i_int
                    gt_w_f = width * Wf / W
                    gt_h_f = height * Hf / H
                    gt_box_coords = compute_box_coordinates(
                        gt_dx, gt_dy, gt_w_f, gt_h_f, j_int, i_int, scale_x, scale_y)

                    pred_boxes_list.append(pred_box_coords)
                    target_boxes_list.append(gt_box_coords)

            # Objectness loss
            obj_pred = objectness[i]
            objectness_loss = self.bce_loss(obj_pred, obj_target)
            all_objectness_loss += objectness_loss

            # BBox loss
            bbox_loss = torch.tensor(0.).to(self.cfg['device'])
            if pred_boxes_list:
                pred_boxes_tensor = torch.tensor(pred_boxes_list, device=self.cfg['device'])
                target_boxes_tensor = torch.tensor(target_boxes_list, device=self.cfg['device'])
                bbox_loss = self.iou_loss(pred_boxes_tensor, target_boxes_tensor)
            all_bbox_loss += bbox_loss

            # Weighted sum
            total_loss += (self.ditol_config['obj_weight'] * objectness_loss +
                           self.ditol_config['bbox_weight'] * bbox_loss)

        return {
            'objectness_loss': all_objectness_loss,
            'bbox_loss': all_bbox_loss,
            'total_loss': total_loss
        }

