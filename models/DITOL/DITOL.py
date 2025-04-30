import os
import sys

import torch
from torch import nn
from torchvision.models import convnext_tiny
from tqdm import tqdm
import torch.optim as optim
from .FALCON.FALCON import Falcon
from .FALCON.perceptual import PerceptualNet
from .FALCON.utils import make_dark_channel_tensor
from .utils import process_batch


class DITOL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.val_batch_nums = None
        self.train_batch_nums = None
        self.falcon = Falcon(config)
        self.backbone = convnext_tiny(pretrained=True).features  # 只保留特征提取层
        self.localization_head = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.bbox_regression = nn.Conv2d(128, 4, kernel_size=1)  # dx, dy, w, h
        self.objectness_score = nn.Conv2d(128, 1, kernel_size=1)  # objectness

        # loss
        self.mse = nn.MSELoss()
        self.perc_loss_network = PerceptualNet(net=self.falcon.falcon_config.train['perceptual']['net'],
                                          style_layers=self.falcon.falcon_config.train['perceptual']['style'],
                                          content_layers=self.falcon.falcon_config.train['perceptual']['content'], device=self.config['device'])
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, input):
        dehazed, t_map = self.falcon(input)  # 正确解包
        features = self.backbone(dehazed)  # 只将 dehazed 图像送入 ConvNeXt

        loc_features = self.localization_head(features)
        bbox_pred = self.bbox_regression(loc_features)
        objectness = self.objectness_score(loc_features).sigmoid()


        return dehazed, t_map, features, bbox_pred, objectness

    def train_step(self, tra_batch, clean_batch):
        self.optimizer.zero_grad()
        low_res_images, targets, ignore = process_batch(tra_batch)
        clean_image, _, _ = process_batch(clean_batch)
        low_res_images = low_res_images.to(self.config['device'])
        clean_image = clean_image.to(self.config['device'])

        t_gt = make_dark_channel_tensor(clean_image)

        # 网络前向传播
        dehazed, t_map, features, bbox_pred, objectness = self.forward(low_res_images)

        # 去雾损失
        loss_img = self.mse(dehazed, clean_image)
        loss_map = self.mse(t_map, t_gt)
        loss_perc = self.perc_loss_network(dehazed, clean_image) if self.falcon.falcon_config.train['perceptual'][
            'net'] else torch.tensor(0.).to(self.config['device'])

        # ----------- 目标检测损失 -----------
        det_loss = torch.tensor(0.).to(self.config['device'])

        B, _, Hf, Wf = objectness.shape
        _, _, H, W = low_res_images.shape
        scale_x = Wf / W
        scale_y = Hf / H

        for i in range(B):
            gt_boxes = targets[i]  # shape [N, 9]
            ign_boxes = ignore[i] if ignore is not None else []

            gt_obj = torch.zeros((1, Hf, Wf), device=self.config['device'])
            gt_box_map = torch.zeros((4, Hf, Wf), device=self.config['device'])
            ignore_mask = torch.zeros((1, Hf, Wf), device=self.config['device'])

            # 构造 GT 目标图
            for box in gt_boxes:
                left, top, width, height = box[2:6]
                cx = (left + width / 2.0) * scale_x
                cy = (top + height / 2.0) * scale_y
                j = int(cx)
                i_ = int(cy)
                if 0 <= j < Wf and 0 <= i_ < Hf:
                    dx = cx - j
                    dy = cy - i_
                    gt_obj[0, i_, j] = 1
                    gt_box_map[:, i_, j] = torch.tensor([dx, dy, width * scale_x, height * scale_y],
                                                        device=self.config['device'])

            # 构造 ignore 区域 mask
            for ign in ign_boxes:
                l, t, w, h = ign[2:6]
                x0 = int(l * scale_x)
                y0 = int(t * scale_y)
                x1 = int((l + w) * scale_x)
                y1 = int((t + h) * scale_y)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(Wf, x1), min(Hf, y1)
                ignore_mask[0, y0:y1, x0:x1] = 1

            # 应用 ignore mask：不计算损失
            obj_pred = objectness[i] * (1 - ignore_mask)
            box_pred = bbox_pred[i]

            objectness_loss = self.bce_loss(obj_pred, gt_obj)
            bbox_loss = self.l1_loss(box_pred, gt_box_map)

            det_loss += objectness_loss + bbox_loss

        # 总损失
        loss_final = (
                self.falcon.falcon_config.train['loss_ratio'][0] * loss_img +
                self.falcon.falcon_config.train['loss_ratio'][1] * loss_map +
                self.falcon.falcon_config.train['loss_ratio'][2] * loss_perc +
                0.1 * det_loss
        )

        loss_final.backward()
        self.optimizer.step()
        return loss_final

    def train_epoch(self, train_loader, clean_loader, epoch):
        epoch_loss = 0.0

        pbar = tqdm(zip(train_loader, clean_loader), total=self.train_batch_nums, desc=f"Epoch {epoch}")
        for batch_idx, (tra_batch, clean_batch) in enumerate(pbar):
            loss = self.train_step(tra_batch, clean_batch)

            epoch_loss += loss

            if batch_idx % self.config['train']['log_interval'] == 0:
                pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    'Batch': f'{batch_idx + 1}/{self.train_batch_nums}'
                })

        avg_loss = epoch_loss / self.train_batch_nums

        print(f"Epoch {epoch} 训练完成，平均 Loss: {avg_loss:.4f}")
        return avg_loss

    def train_model(self, train_loader, val_loader, train_clean_loader, val_clean_loader, start_epoch=0,
                    num_epochs=100, checkpoint_dir='./models/DITOL/checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_loss = float('inf')
        # 训练前预处理
        self.train_batch_nums = len(train_loader)
        self.val_batch_nums = len(val_loader)
        if self.config['train']['resume_training']:
            print("==> 尝试加载最近 checkpoint ...")
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                start_epoch = self.load_checkpoint(checkpoint_path)
            else:
                print("未找到 checkpoint，重新训练。")

        for epoch in range(start_epoch, num_epochs):
            self.train_epoch(train_loader, train_clean_loader, epoch)

            # 验证集
            if val_loader:
                self.eval()
                val_stats = self.evaluate(val_loader, val_clean_loader)

                # 保存最优模型
                if val_stats['loss'] < best_loss:
                    best_loss = val_stats['loss']
                    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
                    self.save_checkpoint(epoch, best_path)
                    print(f"==> Best model saved to {best_path}")

            # 定期保存 checkpoint
            if (epoch + 1) % self.config['train']['checkpoint_save_interval'] == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                self.save_checkpoint(epoch, checkpoint_path)
                print(f"==> Checkpoint saved to {checkpoint_path}")

        print("训练完成！")

    @torch.no_grad()
    def evaluate(self, val_loader, clean_loader):
        self.eval()
        total_loss = 0.0
        total_batches = len(val_loader)

        pbar = tqdm(zip(val_loader, clean_loader), total=total_batches, desc="Evaluating")
        for batch_idx, (val_batch, clean_batch) in enumerate(pbar):
            low_res_images, targets, ignore = process_batch(val_batch)
            clean_image, _, _ = process_batch(clean_batch)

            low_res_images = low_res_images.to(self.config['device'])
            clean_image = clean_image.to(self.config['device'])

            # 网络前向传播
            dehazed, t_map, features, bbox_pred, objectness = self.forward(low_res_images)
            t_gt = make_dark_channel_tensor(clean_image)
            # 去雾损失
            loss_img = self.mse(dehazed, clean_image)
            loss_map = self.mse(t_map, t_gt)
            loss_perc = self.perc_loss_network(dehazed, clean_image) if self.falcon.falcon_config.train['perceptual'][
                'net'] else torch.tensor(0.).to(self.config['device'])

            # 目标检测损失
            det_loss = torch.tensor(0.).to(self.config['device'])

            B, _, Hf, Wf = objectness.shape
            _, _, H, W = low_res_images.shape
            scale_x = Wf / W
            scale_y = Hf / H

            for i in range(B):
                gt_boxes = targets[i]
                ign_boxes = ignore[i] if ignore is not None else []

                gt_obj = torch.zeros((1, Hf, Wf), device=self.config['device'])
                gt_box_map = torch.zeros((4, Hf, Wf), device=self.config['device'])
                ignore_mask = torch.zeros((1, Hf, Wf), device=self.config['device'])

                for box in gt_boxes:
                    left, top, width, height = box[2:6]
                    cx = (left + width / 2.0) * scale_x
                    cy = (top + height / 2.0) * scale_y
                    j = int(cx)
                    i_ = int(cy)
                    if 0 <= j < Wf and 0 <= i_ < Hf:
                        dx = cx - j
                        dy = cy - i_
                        gt_obj[0, i_, j] = 1
                        gt_box_map[:, i_, j] = torch.tensor([dx, dy, width * scale_x, height * scale_y],
                                                            device=self.config['device'])

                for ign in ign_boxes:
                    l, t, w, h = ign[2:6]
                    x0 = int(l * scale_x)
                    y0 = int(t * scale_y)
                    x1 = int((l + w) * scale_x)
                    y1 = int((t + h) * scale_y)
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(Wf, x1), min(Hf, y1)
                    ignore_mask[0, y0:y1, x0:x1] = 1

                obj_pred = objectness[i] * (1 - ignore_mask)
                box_pred = bbox_pred[i]

                objectness_loss = self.bce_loss(obj_pred, gt_obj)
                bbox_loss = self.l1_loss(box_pred, gt_box_map)

                det_loss += objectness_loss + bbox_loss

            total = (
                    self.falcon.falcon_config.train['loss_ratio'][0] * loss_img +
                    self.falcon.falcon_config.train['loss_ratio'][1] * loss_map +
                    self.falcon.falcon_config.train['loss_ratio'][2] * loss_perc +
                    0.1 * det_loss
            )

            total_loss += total.item()

            pbar.set_postfix({
                'Loss': f'{total.item():.4f}',
                'Batch': f'{batch_idx + 1}/{total_batches}'
            })

        avg_loss = total_loss / total_batches
        print(f"验证完成，平均 Loss: {avg_loss:.4f}")

        return {
            'loss': avg_loss
        }

    def save_checkpoint(self, epoch, checkpoint_path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存到: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        self.config = checkpoint['config']
        print(f"检查点已加载，从 epoch {start_epoch} 继续训练 (如果需要)")
        return start_epoch

    @torch.no_grad()
    def predict(self, inputs, ignore=None, conf_thresh=0.5, iou_thresh=0.5):
        """
        inputs: Tensor [B, 3, H, W]
        ignore: optional list of ignore boxes per image
        return: list of list, each with [x1, y1, x2, y2, score]
        """
        self.eval()
        inputs = inputs.to(self.config['device'])

        _, _, _, bbox_pred, objectness = self.forward(inputs)
        B, _, Hf, Wf = objectness.shape
        _, _, H, W = inputs.shape
        scale_x = W / Wf
        scale_y = H / Hf

        results = []

        for b in range(B):
            obj = objectness[b, 0]  # [Hf, Wf]
            box = bbox_pred[b]  # [4, Hf, Wf]

            boxes = []
            scores = []

            for i in range(Hf):
                for j in range(Wf):
                    score = obj[i, j].item()
                    if score < conf_thresh:
                        continue
                    dx, dy, w, h = box[:, i, j].tolist()
                    cx = (j + dx) * scale_x
                    cy = (i + dy) * scale_y
                    bw = w * scale_x
                    bh = h * scale_y

                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)

            boxes = torch.tensor(boxes, device=self.config['device'])
            scores = torch.tensor(scores, device=self.config['device'])

            if boxes.shape[0] == 0:
                results.append([])
                continue

            # Apply NMS
            keep = self.nms(boxes, scores, iou_thresh)
            boxes = boxes[keep]
            scores = scores[keep]

            # Ignore 区域过滤
            if ignore is not None and ignore[b]:
                keep_final = []
                for idx in range(len(boxes)):
                    bx1, by1, bx2, by2 = boxes[idx].tolist()
                    overlaps_ignore = False
                    for ign in ignore[b]:
                        ix, iy, iw, ih = ign[2:6]
                        ix2, iy2 = ix + iw, iy + ih
                        iou = self.compute_iou([bx1, by1, bx2, by2], [ix, iy, ix2, iy2])
                        if iou > 0.5:
                            overlaps_ignore = True
                            break
                    if not overlaps_ignore:
                        keep_final.append(idx)
                boxes = boxes[keep_final]
                scores = scores[keep_final]

            result = [[*boxes[i].tolist(), scores[i].item()] for i in range(len(boxes))]
            results.append(result)

        return results

    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        xi1, yi1 = max(x1, x1g), max(y1, y1g)
        xi2, yi2 = min(x2, x2g), min(y2, y2g)
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.

    def nms(self, boxes, scores, iou_thresh):
        # Torchvision's built-in NMS
        from torchvision.ops import nms
        return nms(boxes, scores, iou_thresh)