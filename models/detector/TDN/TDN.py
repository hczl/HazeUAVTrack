from collections import deque

import numpy as np
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import ops
from torchvision.models import convnext_tiny, mobilenet_v3_large, resnet50
from torchvision.ops import nms
from torchvision.models import resnet18, ResNet18_Weights

import yaml
from torchvision.ops import box_iou
import torch.nn.functional as F

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def compute_iou(boxes1, boxes2):
    return box_iou(boxes1, boxes2)

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.where(targets == 1, inputs, 1 - inputs)
    loss = alpha * (1 - pt) ** gamma * BCE_loss
    return loss.sum() if reduction == 'sum' else loss.mean()

def decode_preds(preds, img_size):
    B, C, H, W = preds.shape
    assert C == 5, f"Expected input channels 5, but got {C}"

    stride_y = img_size[0] / H
    stride_x = img_size[1] / W

    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=preds.device),
                                    torch.arange(W, device=preds.device),
                                    indexing='ij')

    grid_x = grid_x.float().view(1, H, W).repeat(B, 1, 1)
    grid_y = grid_y.float().view(1, H, W).repeat(B, 1, 1)

    # These are already activated values from DetectionHead
    dx = preds[:, 0, :, :]
    dy = preds[:, 1, :, :]
    dw = preds[:, 2, :, :] # These are exp(raw_dw)
    dh = preds[:, 3, :, :] # These are exp(raw_dh)
    conf = preds[:, 4, :, :] # These are sigmoid(raw_conf)

    cx = (grid_x + dx) * stride_x
    cy = (grid_y + dy) * stride_y

    bw = dw * stride_x
    bh = dh * stride_y

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    boxes_conf = torch.stack([x1, y1, x2, y2, conf], dim=1)
    boxes_conf = boxes_conf.permute(0, 2, 3, 1).contiguous()
    boxes_conf = boxes_conf.view(B, H * W, 5)

    return [boxes_conf[i] for i in range(B)]


class LiteFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, num_levels=4):
        super(LiteFPN, self).__init__()
        self.out_channels = out_channels
        self.num_levels = num_levels

        self.reduce_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # learnable position encodings for each feature map
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(1, out_channels, 1, 1)) for _ in range(num_levels)
        ])

    def forward(self, features):
        reduced = [conv(f) for conv, f in zip(self.reduce_convs, features)]
        target_size = reduced[-1].shape[-2:]  # smallest scale (e.g., P5)

        # 统一下采样到最小尺寸后融合
        resized = [F.adaptive_avg_pool2d(f, target_size) for f in reduced]
        fused = torch.stack(resized, dim=0).sum(dim=0)

        # 逐层上采样回原始分辨率 + 加位置编码
        results = []
        for i in range(self.num_levels):
            upsampled = F.interpolate(fused, size=reduced[i].shape[-2:], mode='nearest')
            results.append(upsampled + self.pos_embeds[i])

        return results  # 返回 P2 ~ P5


class ResNet50LiteFPNBackbone(nn.Module):
    def __init__(self, out_channels=128, pretrained=True):
        super().__init__()
        resnet = resnet50(pretrained=pretrained)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool, resnet.layer1)  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5

        self.fpn = LiteFPN(in_channels_list=[256, 512, 1024, 2048], out_channels=out_channels)

    def forward(self, x):
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn([c2, c3, c4, c5])  # 返回 P2 ~ P5 + PosEncoding


class DeformableAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, num_points=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.in_channels = in_channels

        self.offset_proj = nn.Conv2d(in_channels, num_heads * num_points * 2, kernel_size=1)
        self.attn_weight_proj = nn.Conv2d(in_channels, num_heads * num_points, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()

        value = self.value_proj(x)

        offsets = self.offset_proj(x)
        attn_weights = self.attn_weight_proj(x)
        attn_weights = attn_weights.view(B, self.num_heads, self.num_points, H, W)
        attn_weights = F.softmax(attn_weights, dim=2)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack((grid_x, grid_y), dim=-1)
        base_grid = base_grid.view(1, 1, 1, H, W, 2)
        base_grid = base_grid.expand(B, self.num_heads, self.num_points, H, W, 2)

        offset = offsets.view(B, self.num_heads, self.num_points, 2, H, W)
        offset = offset.permute(0, 1, 2, 4, 5, 3)
        sampling_grid = base_grid + offset
        sampling_grid = sampling_grid.view(B * self.num_heads * self.num_points, H, W, 2)

        # Prepare value for grid_sample
        # Split value channels by head and repeat for points
        value_for_sample = value.view(B, self.num_heads, self.in_channels // self.num_heads, H, W)
        value_for_sample = value_for_sample.unsqueeze(2).expand(-1, -1, self.num_points, -1, -1, -1)
        value_for_sample = value_for_sample.reshape(B * self.num_heads * self.num_points, self.in_channels // self.num_heads, H, W)

        sampled = F.grid_sample(value_for_sample, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # Reshape sampled back to (B, num_heads, num_points, in_channels_per_head, H, W)
        sampled = sampled.view(B, self.num_heads, self.num_points, self.in_channels // self.num_heads, H, W)

        # Apply attention weights
        attn_weights = attn_weights.unsqueeze(3)
        weighted = (sampled * attn_weights).sum(dim=2)

        # Combine heads
        weighted = weighted.view(B, self.in_channels, H, W)

        return self.output_proj(weighted + x)

class ConvGRUCell(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.channels = channels
        self.conv_zr = nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        z_r = torch.sigmoid(self.conv_zr(combined))
        z, r = torch.split(z_r, self.channels, dim=1)
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_hat = torch.tanh(self.conv_h(combined_reset))
        h = (1 - z) * h_prev + z * h_hat
        return h

class MultiFPNConvGRU(nn.Module):
    def __init__(self, channels=256, kernel_size=3):
        super().__init__()
        self.fpn_count = 4
        self.gru_cells = nn.ModuleList([
            ConvGRUCell(channels, kernel_size) for _ in range(self.fpn_count)
        ])

    def forward(self, hist_states, current_tensor_list):
        # hist_states: List[Tensor] with shape [B, C, H, W] or None at t=0
        updated_hist_states = []
        for i in range(self.fpn_count):
            h_prev = hist_states[i] if hist_states[i] is not None else torch.zeros_like(current_tensor_list[i])
            h_new = self.gru_cells[i](current_tensor_list[i], h_prev)
            updated_hist_states.append(h_new)
        return updated_hist_states


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes * 5, kernel_size=1)
        )

        pi = 0.01
        bias_value = -torch.log(torch.tensor((1 - pi) / pi))

        with torch.no_grad():
            if num_classes > 0:
                 self.head[-1].bias[4::5].fill_(bias_value)

    def forward(self, feats):
        raw_preds_list = []
        activated_preds_list = []

        for x in feats:
            raw_preds = self.head(x)
            B, C, H, W = raw_preds.shape
            assert C == self.num_classes * 5, f"Expected {self.num_classes * 5}, got {C}"

            raw_preds_reshaped = raw_preds.view(B, self.num_classes, 5, H, W)

            dx_dy = torch.sigmoid(raw_preds_reshaped[:, :, 0:2, :, :])
            dw_dh = torch.exp(torch.clamp(raw_preds_reshaped[:, :, 2:4, :, :], max=4))
            conf = torch.sigmoid(raw_preds_reshaped[:, :, 4:5, :, :])

            activated_preds_reshaped = torch.cat([dx_dy, dw_dh, conf], dim=2)
            activated_preds = activated_preds_reshaped.view(B, self.num_classes * 5, H, W)


            raw_preds_list.append(raw_preds)
            activated_preds_list.append(activated_preds)

        return raw_preds_list, activated_preds_list

class TDN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg_dict = {}
        try:
             model_cfg_dict = load_config('models/detector/TDN/TDN_config.yaml')['model']
        except Exception as e:
             model_cfg_dict = {
                 'backbone_channel': 192,
                 'step': 4
             }
        self.step = model_cfg_dict.get('step', 4)

        self.backbone_channel = model_cfg_dict.get('backbone_channel', 192)

        self.backbone = ResNet50LiteFPNBackbone(out_channels=self.backbone_channel)

        self.attn = nn.ModuleList([DeformableAttention(in_channels=self.backbone_channel) for _ in range(4)])

        self.ConvGRU = MultiFPNConvGRU(channels=self.backbone_channel)

        self.detection = DetectionHead(in_channels=self.backbone_channel)
        self._hist_f = None

    def forward(self, x, training=False):
        f_t = self.backbone(x)

        attn_f = [self.attn[i](x_i) for i, x_i in enumerate(f_t)]
        F = 4
        B, C = attn_f[0].shape[:2]
        assert B % self.step == 0, "帧数必须是 step 的整数倍"
        if training:
            hist = [None] * F
            for i in range(self.step):
                indices = torch.arange(i, B, self.step)
                x_list = [attn_f[f][indices] for f in range(F)]
                hist = self.ConvGRU(hist, x_list)  # output: List of [B, C, H, W]
            output = self.detection(hist)
        else:
            self._hist_f = self.ConvGRU(self._hist_f, attn_f)
            output = self.detection(self._hist_f)

        return output

    def _compute_single_loss(self, raw_preds_flat, activated_preds_flat, targets, ignore_list, img_size, feature_map_size):

        H_f, W_f = feature_map_size
        stride_y = img_size[0] / H_f
        stride_x = img_size[1] / W_f

        grid_y, grid_x = torch.meshgrid(torch.arange(H_f, device=raw_preds_flat.device),
                                        torch.arange(W_f, device=raw_preds_flat.device),
                                        indexing='ij')
        grid_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1) # [H*W, 2]

        # Decode activated predictions for IoU calculation (matching and conf targets)
        pred_boxes_decoded = self.decode_activated_preds_to_boxes(activated_preds_flat, img_size, feature_map_size) # [H*W, 4]
        pred_confs_activated = activated_preds_flat[:, 4] # [H*W] These are sigmoid(raw_conf)

        gt_boxes = [[float(ann[2]), float(ann[3]),
                     float(ann[2]) + float(ann[4]),
                     float(ann[3]) + float(ann[5])] for ann in targets]
        gt_boxes_tensor = torch.tensor(gt_boxes, device=raw_preds_flat.device) if gt_boxes else None

        ignore_boxes = [[float(ann[2]), float(ann[3]),
                         float(ann[2]) + float(ann[4]),
                         float(ann[3]) + float(ann[5])] for ann in ignore_list]
        ignore_boxes_tensor = torch.tensor(ignore_boxes, device=raw_preds_flat.device) if ignore_boxes else None

        conf_loss = torch.tensor(0.0, device=raw_preds_flat.device)
        bbox_loss = torch.tensor(0.0, device=raw_preds_flat.device)
        num_matched = 0
        num_non_ignored = 0

        ignore_mask = None
        if ignore_boxes_tensor is not None and ignore_boxes_tensor.numel() > 0 and pred_boxes_decoded.numel() > 0:
             ious_ignore = compute_iou(pred_boxes_decoded, ignore_boxes_tensor)
             max_iou_ignore, _ = ious_ignore.max(dim=1)
             ignore_mask = max_iou_ignore > 0.5
        elif pred_boxes_decoded.numel() > 0:
             ignore_mask = torch.zeros_like(pred_confs_activated, dtype=torch.bool)

        if pred_confs_activated.numel() > 0:
             current_ignore_mask = ignore_mask if ignore_mask is not None else torch.zeros_like(pred_confs_activated, dtype=torch.bool)
             non_ignored_mask = ~current_ignore_mask
             non_ignored_preds_conf = pred_confs_activated[non_ignored_mask]

             if non_ignored_preds_conf.numel() > 0:
                  if gt_boxes_tensor is not None and gt_boxes_tensor.numel() > 0:
                       ious_gt = compute_iou(pred_boxes_decoded, gt_boxes_tensor)
                       best_iou_gt, _ = ious_gt.max(dim=1)
                       conf_target = (best_iou_gt > 0.1).float()
                       non_ignored_targets_conf = conf_target[non_ignored_mask]
                  else:
                       non_ignored_targets_conf = torch.zeros_like(non_ignored_preds_conf)

                  if getattr(self, "_use_mse", False): # Assuming a flag for MSE
                       conf_loss = F.mse_loss(non_ignored_preds_conf, non_ignored_targets_conf, reduction='mean')
                  else:
                       conf_loss = focal_loss(non_ignored_preds_conf, non_ignored_targets_conf, alpha=0.25, gamma=2.0, reduction='mean')

             num_non_ignored = non_ignored_mask.sum().item()


        if gt_boxes_tensor is not None and gt_boxes_tensor.numel() > 0 and pred_boxes_decoded.numel() > 0:
            ious_gt = compute_iou(pred_boxes_decoded, gt_boxes_tensor)
            best_iou_gt, best_gt_idx = ious_gt.max(dim=1)
            current_ignore_mask = ignore_mask if ignore_mask is not None else torch.zeros_like(pred_confs_activated, dtype=torch.bool)

            match_mask = (best_iou_gt > 0.1) & (~current_ignore_mask)

            matched_pred_raw = raw_preds_flat[match_mask] # Get RAW predictions for bbox loss
            matched_gt_boxes = gt_boxes_tensor[best_gt_idx[match_mask]]
            matched_grid_coords = grid_coords[match_mask]

            num_matched = len(matched_pred_raw)

            if num_matched > 0:
                gt_cx = (matched_gt_boxes[:, 0] + matched_gt_boxes[:, 2]) / 2
                gt_cy = (matched_gt_boxes[:, 1] + matched_gt_boxes[:, 3]) / 2
                gt_w = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
                gt_h = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]

                matched_grid_x = matched_grid_coords[:, 0]
                matched_grid_y = matched_grid_coords[:, 1]

                # Calculate target encoded values
                gt_dx_target = (gt_cx / stride_x) - matched_grid_x
                gt_dy_target = (gt_cy / stride_y) - matched_grid_y
                gt_dw_target = torch.log(gt_w / stride_x + 1e-6)
                gt_dh_target = torch.log(gt_h / stride_y + 1e-6)

                gt_encoded_targets = torch.stack([gt_dx_target, gt_dy_target, gt_dw_target, gt_dh_target], dim=1)

                # Get the predicted raw dx, dy, dw, dh
                pred_raw_bbox_vals = matched_pred_raw[:, :4]

                bbox_loss = F.smooth_l1_loss(pred_raw_bbox_vals, gt_encoded_targets, reduction='mean')


        return {
            'conf_loss': conf_loss,
            'bbox_loss': bbox_loss,
            'num_matched': num_matched,
            'num_non_ignored': num_non_ignored
        }

    def compute_batch_loss(self, raw_preds_list, activated_preds_list, targets, ignore_list, img_size):
        # raw_preds_list: List[Tensor], each of shape [B, 5, H_f, W_f]
        # activated_preds_list: List[Tensor], same as above

        B = raw_preds_list[0].shape[0]
        img_H, img_W = img_size

        batch_conf_loss = 0.0
        batch_bbox_loss = 0.0
        total_matched_in_batch = 0
        total_non_ignored_in_batch = 0

        num_images_in_batch = B

        for i in range(num_images_in_batch):
            single_image_targets = targets[i]
            single_image_ignore = ignore_list[i] if ignore_list else []

            # 遍历不同尺度的输出
            for raw_preds, activated_preds in zip(raw_preds_list, activated_preds_list):
                # Flatten [C, H, W] -> [H*W, C]
                raw_preds_flat = raw_preds[i].permute(1, 2, 0).contiguous().view(-1, 5)
                activated_preds_flat = activated_preds[i].permute(1, 2, 0).contiguous().view(-1, 5)
                H_f, W_f = raw_preds.shape[2:]

                loss_dict = self._compute_single_loss(
                    raw_preds_flat,
                    activated_preds_flat,
                    single_image_targets,
                    single_image_ignore,
                    img_size,
                    (H_f, W_f)
                )

                batch_conf_loss += loss_dict['conf_loss']
                batch_bbox_loss += loss_dict['bbox_loss']
                total_matched_in_batch += loss_dict['num_matched']
                total_non_ignored_in_batch += loss_dict['num_non_ignored']

        # 每张图片每个尺度都参与了计算，求平均需要除以 B * num_levels
        num_levels = len(raw_preds_list)
        denom = num_images_in_batch * num_levels
        final_avg_conf_loss = batch_conf_loss / denom
        final_avg_bbox_loss = batch_bbox_loss / denom
        final_avg_total_loss = final_avg_conf_loss + final_avg_bbox_loss

        log_dict = {
            'total_loss': final_avg_total_loss,
            'conf_loss': final_avg_conf_loss.detach(),
            'bbox_loss': final_avg_bbox_loss.detach(),
            'avg_matched_per_image': total_matched_in_batch / num_images_in_batch,
            'avg_non_ignored_per_image': total_non_ignored_in_batch / num_images_in_batch
        }

        return {'total_loss': final_avg_total_loss, **log_dict}

    def decode_activated_preds_to_boxes(self, activated_preds_flat, img_size, feature_map_size):
         N = activated_preds_flat.shape[0]
         H_img, W_img = img_size
         H_f, W_f = feature_map_size
         stride_y = H_img / H_f
         stride_x = W_img / W_f

         grid_y, grid_x = torch.meshgrid(torch.arange(H_f, device=activated_preds_flat.device),
                                         torch.arange(W_f, device=activated_preds_flat.device),
                                         indexing='ij')
         grid_x = grid_x.flatten()
         grid_y = grid_y.flatten()

         # These are already activated values
         dx = activated_preds_flat[:, 0]
         dy = activated_preds_flat[:, 1]
         dw = activated_preds_flat[:, 2]
         dh = activated_preds_flat[:, 3]

         cx = (grid_x + dx) * stride_x
         cy = (grid_y + dy) * stride_y
         bw = dw * stride_x
         bh = dh * stride_y

         x1 = cx - bw / 2
         y1 = cy - bh / 2
         x2 = cx + bw / 2
         y2 = cy + bh / 2

         return torch.stack([x1, y1, x2, y2], dim=1)

    def forward_loss(self, dehaze_imgs, targets, ignore_list=None):
        B, _, H_img, W_img = dehaze_imgs.shape
        indices = np.arange(self.step - 1, B, self.step).tolist()
        targets = [targets[i] for i in indices]

        if ignore_list is None:
            ignore_list = [[] for _ in range(B // self.step)]
        else:
            ignore_list = [ignore_list[i] for i in indices]

        raw_preds_list, activated_preds_list = self(dehaze_imgs, training=True)

        loss_dict = self.compute_batch_loss(
            raw_preds_list, activated_preds_list, targets, ignore_list, (H_img, W_img)
        )

        return loss_dict


    @torch.no_grad()
    def predict(self, high_res_images, conf_thresh=0.5, iou_thresh=0.45):
        self.eval()
        B, _, H_img, W_img = high_res_images.shape

        _, activated_preds_list = self(high_res_images)
        results = []

        for b in range(B):
            all_preds = []
            for preds in activated_preds_list:
                decoded = decode_preds(preds, (H_img, W_img))[b]  # 单张图像
                all_preds.append(decoded)
            preds_i = torch.cat(all_preds, dim=0)  # 多尺度合并

            if preds_i.numel() == 0:
                results.append(torch.empty((0, 5), device=high_res_images.device))
                continue

            boxes = preds_i[:, :4]
            scores = preds_i[:, 4]
            keep = scores > conf_thresh
            boxes, scores = boxes[keep], scores[keep]

            if boxes.numel() == 0:
                results.append(torch.empty((0, 5), device=high_res_images.device))
                continue

            nms_indices = nms(boxes, scores, iou_thresh)
            filtered_preds = torch.cat([boxes[nms_indices], scores[nms_indices].unsqueeze(1)], dim=1)
            results.append(filtered_preds)

        return results[0]

    def reset_memory(self):
        self._hist_f = None

