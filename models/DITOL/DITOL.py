import os
import sys

import torch
from torch import nn
from torchvision.models import convnext_tiny
from torchvision.ops import nms
from tqdm import tqdm
import torch.optim as optim
from .FALCON.FALCON import Falcon
from .FALCON.perceptual import PerceptualNet
from .FALCON.utils import make_dark_channel_tensor
from .ditol_config import Settings
from .utils import process_batch, GIoULoss, generate_soft_target, center_sampling_mask


class DITOL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.val_batch_nums = None
        self.train_batch_nums = None

        # FALCON 模块用于去雾和 t 图生成
        self.falcon = Falcon(config)
        self.ditol_config = Settings()
        # 主干网络用于特征提取
        self.backbone = convnext_tiny(pretrained=True).features

        # 本地化头部结构：用于检测
        self.localization_head = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.bbox_regression = nn.Conv2d(128, 4, kernel_size=1)
        self.objectness_score = nn.Conv2d(128, 1, kernel_size=1)

        # 各类损失函数
        self.mse = nn.MSELoss()
        self.perc_loss_network = PerceptualNet(
            net=self.falcon.falcon_config.train['perceptual']['net'],
            style_layers=self.falcon.falcon_config.train['perceptual']['style'],
            content_layers=self.falcon.falcon_config.train['perceptual']['content'],
            device=self.config['device']
        )
        self.bce_loss = nn.BCELoss()
        self.iou_loss = GIoULoss()

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
        dehazed, t_map, features, bbox_pred, objectness = self.forward(low_res_images)

        loss_img = self.mse(dehazed, clean_image)
        loss_map = self.mse(t_map, t_gt)
        loss_perc = self.perc_loss_network(dehazed, clean_image) if self.falcon.falcon_config.train['perceptual']['net'] else torch.tensor(0.).to(self.config['device'])

        B, _, Hf, Wf = objectness.shape
        _, _, H, W = low_res_images.shape
        scale_x, scale_y = W / Wf, H / Hf
        det_loss = self.prepare_targets(targets, ignore, bbox_pred, objectness, B, Hf, Wf, H, W, scale_x, scale_y)

        loss_final = self.compute_total_loss(loss_img, loss_map, loss_perc, det_loss)
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
        for val_batch, clean_batch in pbar:
            low_res_images, targets, ignore = process_batch(val_batch)
            clean_image, _, _ = process_batch(clean_batch)
            low_res_images = low_res_images.to(self.config['device'])
            clean_image = clean_image.to(self.config['device'])

            dehazed, t_map, features, bbox_pred, objectness = self.forward(low_res_images)
            t_gt = make_dark_channel_tensor(clean_image)

            loss_img = self.mse(dehazed, clean_image)
            loss_map = self.mse(t_map, t_gt)
            loss_perc = self.perc_loss_network(dehazed, clean_image) if self.falcon.falcon_config.train['perceptual'][
                'net'] else torch.tensor(0.).to(self.config['device'])

            B, _, Hf, Wf = objectness.shape
            _, _, H, W = low_res_images.shape
            scale_x, scale_y = W / Wf, H / Hf
            det_loss = self.prepare_targets(targets, ignore, bbox_pred, objectness, B, Hf, Wf, H, W, scale_x, scale_y)

            total = self.compute_total_loss(loss_img, loss_map, loss_perc, det_loss)
            total_loss += total.item()
            pbar.set_postfix({'Loss': f'{total.item():.4f}'})

        avg_loss = total_loss / total_batches
        print(f"验证完成，平均 Loss: {avg_loss:.4f}")
        return {'loss': avg_loss}

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
    def predict(self, inputs, ignore=None, conf_thresh=0.5, iou_thresh=0.7):
        self.eval()
        inputs = inputs.to(self.config['device'])
        _, _, _, bbox_pred, objectness = self.forward(inputs)

        B, _, Hf, Wf = objectness.shape
        _, _, H, W = inputs.shape
        scale_x, scale_y = W / Wf, H / Hf

        results = []
        for b in range(B):
            obj = objectness[b, 0]
            box = bbox_pred[b]
            boxes, scores = [], []

            for i in range(Hf):
                for j in range(Wf):
                    score = obj[i, j].item()
                    if score < conf_thresh:
                        continue
                    dx, dy, w, h = box[:, i, j].tolist()
                    coords = self.compute_box_coordinates(dx, dy, w, h, j, i, scale_x, scale_y)
                    boxes.append(coords)
                    scores.append(score)

            if not boxes:
                results.append([])
                continue

            boxes_tensor = torch.tensor(boxes, device=self.config['device'])
            scores_tensor = torch.tensor(scores, device=self.config['device'])
            keep = self.nms(boxes_tensor, scores_tensor, iou_thresh)
            boxes_tensor = boxes_tensor[keep]
            scores_tensor = scores_tensor[keep]

            if ignore and ignore[b]:
                final_keep = []
                for idx in range(len(boxes_tensor)):
                    bx1, by1, bx2, by2 = boxes_tensor[idx].tolist()
                    overlap = any(self.compute_iou([bx1, by1, bx2, by2],
                                                   [ig[2], ig[3], ig[2] + ig[4], ig[3] + ig[5]]) > 0.5 for ig in
                                  ignore[b])
                    if not overlap:
                        final_keep.append(idx)
                boxes_tensor = boxes_tensor[final_keep]
                scores_tensor = scores_tensor[final_keep]

            batch_result = [[*boxes_tensor[i].tolist(), scores_tensor[i].item()] for i in range(len(boxes_tensor))]
            results.append(batch_result)

        return results

    def compute_box_coordinates(self, dx, dy, w, h, j, i_, scale_x, scale_y):
        cx = (j + dx) * scale_x
        cy = (i_ + dy) * scale_y
        bw = w * scale_x
        bh = h * scale_y
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        return [x1, y1, x2, y2]

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
        return nms(boxes, scores, iou_thresh)

    def prepare_targets(self, targets, ignore, bbox_pred, objectness, B, Hf, Wf, H, W, scale_x, scale_y):
        det_loss = torch.tensor(0.).to(self.config['device'])

        for i in range(B):  # Loop over batch
            gt_boxes = targets[i]
            ign_boxes = ignore[i] if ignore is not None and i < len(ignore) else []  # Ensure ignore list is long enough

            obj_target = torch.zeros((1, Hf, Wf), device=self.config['device'])
            # Initialize ignore mask for the current image
            ignore_mask = torch.zeros((1, Hf, Wf), device=self.config['device'])

            # --- Generate Ignore Mask ---
            if ign_boxes is not None:
                # Assuming ign_boxes format is [M, 6] with [class, score, x, y, w, h]
                # Scale ignore regions to feature map size and mark cells
                for ign in ign_boxes:
                    # Ensure ign has enough elements (at least 6)
                    if len(ign) < 6:
                        print(f"Warning: Invalid ignore box format for batch {i}: {ign}", file=sys.stderr)
                        continue  # Skip invalid ignore box

                    l, t, w, h = ign[2:6]
                    # Convert l, t, w, h (image scale) to feature map scale
                    # Note: Convert top-left coords and bottom-right coords to feature map scale
                    x0_f = (l * Wf / W)
                    y0_f = (t * Hf / H)
                    x1_f = ((l + w) * Wf / W)
                    y1_f = ((t + h) * Hf / H)

                    # Clamp coordinates to feature map bounds
                    x0_f = max(0., min(Wf - 1., x0_f))
                    y0_f = max(0., min(Hf - 1., y0_f))
                    x1_f = max(0., min(Wf - 1., x1_f))
                    y1_f = max(0., min(Hf - 1., y1_f))

                    # Mark integer cells within the scaled ignore region
                    x0_int = int(x0_f)
                    y0_int = int(y0_f)
                    x1_int = int(x1_f)
                    y1_int = int(y1_f)

                    # Ensure indices are within bounds [0, Hf-1] and [0, Wf-1]
                    x0_int = max(0, min(Wf, x0_int))
                    y0_int = max(0, min(Hf, y0_int))
                    x1_int = max(0, min(Wf, x1_int))
                    y1_int = max(0, min(Hf, y1_int))

                    # Mark the rectangular region in the mask
                    # Note: Slice indices are exclusive at the end, so use x1_int+1, y1_int+1
                    ignore_mask[0, y0_int:y1_int + 1, x0_int:x1_int + 1] = 1

            pred_boxes_list, target_boxes_list = [], []
            # --- Process Ground Truth Boxes for Positive Samples ---
            for box in gt_boxes:
                # Ensure box has enough elements (at least 6)
                if len(box) < 6:
                    print(f"Warning: Invalid target box format for batch {i}: {box}", file=sys.stderr)
                    continue  # Skip invalid target box

                left, top, width, height = box[2:6]

                # Calculate center in feature map scale
                cx_f = (left + width / 2.0) * Wf / W
                cy_f = (top + height / 2.0) * Hf / H

                # Integer grid cell coordinates
                j_int = int(cx_f)
                i_int = int(cy_f)

                # Check if center is within feature map bounds
                if 0 <= i_int < Hf and 0 <= j_int < Wf:
                    # --- Check if center is in an ignore region ---
                    if ignore_mask[0, i_int, j_int] == 1:
                        # If the center of the GT box is in an ignore region, skip it
                        continue

                    # --- Generate Soft Target and Mask ---
                    # Soft target for objectness at this location
                    soft_target = generate_soft_target(cx_f, cy_f, Hf, Wf, sigma=1.5, device=self.config['device'])
                    # Center sampling mask to limit soft target influence
                    mask = center_sampling_mask(cx_f, cy_f, width * Wf / W, height * Hf / H, Hf, Wf, radius=0.3,
                                                device=self.config['device'])
                    # Combine soft target, mask, and existing obj_target (take max in case of overlaps)
                    obj_target = torch.max(obj_target, soft_target * mask)

                    # --- Collect Bbox Regression Targets and Predictions ---
                    # Predicted box at the responsible grid cell (i_int, j_int)
                    # Convert predicted dx, dy, w, h relative to cell (j_int, i_int) back to image coords (x1, y1, x2, y2)
                    pred_box_coords = self.compute_box_coordinates(
                        bbox_pred[i, 0, i_int, j_int], bbox_pred[i, 1, i_int, j_int],
                        bbox_pred[i, 2, i_int, j_int], bbox_pred[i, 3, i_int, j_int],
                        j_int, i_int, scale_x,
                        scale_y)  # Note: scale_x, scale_y here are for image->feature map scaling used *in* compute_box_coordinates
                    # It seems compute_box_coordinates needs feature map cell coords (j_int, i_int) and image scales (scale_x, scale_y) to produce image-scale box coords.
                    # Let's check compute_box_coordinates definition...
                    # It takes j, i_, scale_x, scale_y. Yes, it uses feature map indices and image scales to convert cell-relative predictions back to image coordinates. This is correct.

                    # Ground truth box for this cell in image coordinates (x1, y1, x2, y2)
                    # This is the box centered at (cx_f, cy_f) with dimensions (width*Wf/W, height*Hf/H) in feature map scale,
                    # converted back to image scale. Or simpler: the original GT box scaled.
                    # Let's calculate the GT box relative offsets and sizes at the cell (j_int, i_int) and use compute_box_coordinates
                    gt_dx = cx_f - j_int
                    gt_dy = cy_f - i_int
                    gt_w_f = width * Wf / W
                    gt_h_f = height * Hf / H
                    # Use compute_box_coordinates with GT values to get GT box in image scale (x1, y1, x2, y2)
                    gt_box_coords = self.compute_box_coordinates(
                        gt_dx, gt_dy, gt_w_f, gt_h_f,
                        j_int, i_int, scale_x, scale_y)  # Again, uses feature map indices and image scales

                    pred_boxes_list.append(pred_box_coords)
                    target_boxes_list.append(gt_box_coords)

            # --- Calculate Losses for the current image ---
            obj_pred = objectness[i]  # Shape [1, Hf, Wf]

            # Objectness Loss: Apply ignore mask
            objectness_loss_per_pixel = self.bce_loss(obj_pred, obj_target)
            # Zero out loss in ignore regions
            objectness_loss_masked = objectness_loss_per_pixel * (1 - ignore_mask)
            # Calculate mean loss over non-ignored pixels
            non_ignored_pixels = (1 - ignore_mask).sum()
            if non_ignored_pixels > 0:
                objectness_loss = objectness_loss_masked.sum() / non_ignored_pixels
            else:
                objectness_loss = torch.tensor(0.).to(self.config['device'])  # Avoid division by zero

            # Bbox Loss: Only for collected positive samples (already filtered by ignore mask)
            bbox_loss = torch.tensor(0.).to(self.config['device'])  # Initialize bbox loss for this image
            if pred_boxes_list:  # Check if there are any positive samples
                pred_boxes_tensor = torch.tensor(pred_boxes_list, device=self.config['device'])
                target_boxes_tensor = torch.tensor(target_boxes_list, device=self.config['device'])
                # GIoU loss expects (x1, y1, x2, y2) format which compute_box_coordinates provides
                bbox_loss = self.iou_loss(pred_boxes_tensor, target_boxes_tensor)

            # Total detection loss for this image
            det_loss_image = (self.ditol_config.obj_weight * objectness_loss +
                              self.ditol_config.bbox_weight *bbox_loss)

            det_loss += det_loss_image

        # Total detection loss for the batch is the sum over images
        return det_loss

    def compute_total_loss(self, loss_img, loss_map, loss_perc, det_loss):
        # Ensure ditol_config is properly initialized and accessible
        # Assuming self.ditol_config.det_loss is the weight for detection loss
        # Assuming falcon_config.train['loss_ratio'] is [img_weight, map_weight, perc_weight]
        total_loss = (self.falcon.falcon_config.train['loss_ratio'][0] * loss_img +
                      self.falcon.falcon_config.train['loss_ratio'][1] * loss_map +
                      self.falcon.falcon_config.train['loss_ratio'][2] * loss_perc +
                      self.ditol_config.det_loss * det_loss)
        return total_loss

