from collections import deque

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import ops
from torchvision.models import convnext_tiny, mobilenet_v3_large

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


class SwinTransformerFPNBackbone(nn.Module):
    def __init__(self, out_channels=192, model_name='swin_tiny_patch4_window7_224'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, img_size=(320, 640))

        self.stage1_idx = 1
        self.stage2_idx = 2
        self.stage3_idx = 3

        self.out_dims = [self.backbone.feature_info[i]['num_chs'] for i in [self.stage1_idx, self.stage2_idx, self.stage3_idx]]

        self.lateral1 = nn.Conv2d(self.out_dims[0], out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(self.out_dims[1], out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(self.out_dims[2], out_channels, kernel_size=1)

        self.out_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.fusion_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        features = self.backbone(x)

        c1 = features[self.stage1_idx]
        c2 = features[self.stage2_idx]
        c3 = features[self.stage3_idx]

        c1 = c1.permute(0, 3, 1, 2)
        c2 = c2.permute(0, 3, 1, 2)
        c3 = c3.permute(0, 3, 1, 2)

        p3 = self.out_conv3(self.lateral3(c3))
        p2 = self.out_conv2(self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest'))
        p1 = self.out_conv1(self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest'))

        p1_up = F.interpolate(p1, size=p3.shape[2:], mode='nearest')
        p2_up = F.interpolate(p2, size=p3.shape[2:], mode='nearest')
        fused = torch.cat([p1_up, p2_up, p3], dim=1)

        fused_out = self.fusion_conv(fused)

        return fused_out

class ResNet18FPNBackbone(nn.Module):
    def __init__(self, out_channels=192):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.stage1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1
        )
        self.stage2 = model.layer2
        self.stage3 = model.layer3

        self.lateral1 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, out_channels, kernel_size=1)

        self.out_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)

        p3 = self.lateral3(c3)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest')

        p3 = self.out_conv3(p3)
        p2 = self.out_conv2(p2)
        p1 = self.out_conv1(p1)

        return p3


class AttentionGuidedStem(nn.Module):
    def __init__(self, in_channels, out_channels, strides=None):
        super().__init__()
        if strides is None:
            strides = [2, 2, 2]

        if len(strides) != 3:
             raise ValueError("Strides list must have 3 elements")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=strides[0], padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=strides[1], padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, stride=strides[2], padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.stem(x)


class MobileNetV3FPNBackbone(nn.Module):
    def __init__(self, out_channels=192):
        super().__init__()
        model = mobilenet_v3_large(weights=None)

        self.stage1 = nn.Sequential(*model.features[:4])
        self.stage2 = nn.Sequential(*model.features[4:7])
        self.stage3 = nn.Sequential(*model.features[7:13])

        self.lateral1 = nn.Conv2d(24, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(40, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(112, out_channels, kernel_size=1)

        self.out_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)

        p3 = self.lateral3(c3)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest')

        p3 = self.out_conv3(p3)
        p2 = self.out_conv2(p2)
        p1 = self.out_conv1(p1)

        return p3

class KernelAttentionBlur(nn.Module):
    def __init__(self, in_channels, num_kernels=3):
        super().__init__()
        self.num_kernels = num_kernels
        kernel_sizes = [3, 5, 7][:num_kernels]

        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels, bias=False)
            for k in kernel_sizes
        ])

        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_kernels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_kernels, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        stacked = torch.stack(branch_outputs, dim=1)

        B, N, C, H, W = stacked.shape
        combined = stacked.view(B, N * C, H, W)

        attn_weights = self.attention_conv(combined)
        attn_weights = attn_weights.unsqueeze(2)

        weighted = (stacked * attn_weights).sum(dim=1)
        return weighted

class PositionalEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        embed_dim = hidden_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.encoder(x)

class ConvNeXtBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        model = convnext_tiny(weights=None)
        self.stem = nn.Sequential(
            model.features[0],
            model.features[1],
            model.features[2],
        )

    def forward(self, x):
        return self.stem(x)


class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, delta_p_processed, delta_f_hist):
        x = torch.cat([delta_p_processed, delta_f_hist], dim=1)
        B, C_in, H, W = x.size()
        C_out = self.query_conv.out_channels

        q = self.query_conv(x).view(B, C_out, H * W).permute(0, 2, 1)
        k = self.key_conv(x).view(B, C_out, H * W)
        v = self.value_conv(x).view(B, C_out, H * W).permute(0, 2, 1)

        attention = torch.bmm(q, k) / (C_out ** 0.5)
        attention = F.softmax(attention, dim=-1)

        out_attn = torch.bmm(attention, v)
        out_attn = out_attn.permute(0, 2, 1).view(B, C_out, H, W)

        gate = self.gate_conv(out_attn)
        residual = x[:, :C_out, :, :]
        gated_out = self.gamma * gate * out_attn + (1 - gate) * residual
        return gated_out


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
             self.head[-1].bias[4::5].fill_(bias_value)

    def forward(self, x):
        raw_preds = self.head(x)
        B, C, H, W = raw_preds.shape
        assert C == self.num_classes * 5, f"Expected output channels {self.num_classes * 5}, but got {C}"

        raw_preds = raw_preds.view(B, self.num_classes, 5, H, W)

        dx_dy = torch.sigmoid(raw_preds[:, :, 0:2, :, :])
        dw_dh = torch.exp(torch.clamp(raw_preds[:, :, 2:4, :, :], max=4))
        conf = torch.sigmoid(raw_preds[:, :, 4:5, :, :])

        activated_preds = torch.cat([dx_dy, dw_dh, conf], dim=2)
        return raw_preds.view(B, self.num_classes * 5, H, W), activated_preds.view(B, self.num_classes * 5, H, W)


class TDN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg_dict = {}
        try:
             model_cfg_dict = load_config('models/detector/TDN/TDN_config.yaml')['model']
        except Exception as e:
             model_cfg_dict = {
                 'history_length': 4,
                 'nums_p_channels': 48,
                 'hidden_channels': 128
             }

        self.hist_len = model_cfg_dict.get('history_length', 4)

        nums_f_channels = 192
        nums_p_channels = model_cfg_dict.get('nums_p_channels', 48)

        nums_deque_f = nums_f_channels + (self.hist_len - 1) * (nums_f_channels // self.hist_len)
        nums_deque_p = nums_p_channels + (self.hist_len - 1) * (nums_p_channels // self.hist_len)

        hidden_channels = model_cfg_dict.get('hidden_channels', 128)
        self.blur_attn = KernelAttentionBlur(in_channels=192)
        self.pos_enc = PositionalEncoder(3, nums_p_channels)
        # self.backbone = MobileNetV3FPNBackbone(out_channels=192)
        self.backbone = ResNet18FPNBackbone(out_channels=192)
        # self.backbone = SwinTransformerFPNBackbone(out_channels=192)

        self.hist_compressor_f = nn.Conv2d(nums_f_channels, nums_f_channels // self.hist_len, kernel_size=1)
        self.hist_compressor_p = nn.Conv2d(nums_p_channels, nums_p_channels // self.hist_len, kernel_size=1)

        self.mask_create = AttentionGuidedStem(nums_deque_p, nums_deque_f, strides=[1, 1, 1])

        self.attn_gen = FusionModule(nums_deque_f * 2, hidden_channels)

        self.head = DetectionHead(hidden_channels)

        self._hist_f = None
        self._hist_p = None

    def forward(self, x, training=False):
        B, _, H_img, W_img = x.shape

        f_t = self.backbone(x)
        f_t = self.blur_attn(f_t)

        _, C_f, H_f, W_f = f_t.shape

        p_t_full_res = self.pos_enc(x)
        p_t = F.interpolate(p_t_full_res, size=(H_f, W_f), mode='bilinear', align_corners=False)
        _, C_p, _, _ = p_t.shape

        f_embed = self.hist_compressor_f(f_t)
        p_embed = self.hist_compressor_p(p_t)

        if training:
            hist_f_list = []
            hist_p_list = []

            for b in range(B):
                frame_f = f_embed[b]
                frame_p = p_embed[b]

                fake_hist_f = []
                fake_hist_p = []

                for i in range(self.hist_len - 1):
                    if b - i - 1 >= 0:
                        fake_hist_f.insert(0, f_embed[b - i - 1])
                        fake_hist_p.insert(0, p_embed[b - i - 1])
                    else:
                        fake_hist_f.insert(0, frame_f)
                        fake_hist_p.insert(0, frame_p)

                hist_f_list.append(torch.stack(fake_hist_f, dim=0))
                hist_p_list.append(torch.stack(fake_hist_p, dim=0))

            hist_f_tensor = torch.stack(hist_f_list, dim=0)
            hist_p_tensor = torch.stack(hist_p_list, dim=0)
        else:
            if self._hist_f is None or self._hist_f.shape[0] != B:
                self._hist_f = f_embed.unsqueeze(1).repeat(1, self.hist_len, 1, 1, 1)
                self._hist_p = p_embed.unsqueeze(1).repeat(1, self.hist_len, 1, 1, 1)
            else:
                self._hist_f = torch.roll(self._hist_f, shifts=-1, dims=1)
                self._hist_f[:, -1, :, :, :] = f_embed

                self._hist_p = torch.roll(self._hist_p, shifts=-1, dims=1)
                self._hist_p[:, -1, :, :, :] = p_embed

            hist_f_tensor = self._hist_f[:, :-1, :, :, :]
            hist_p_tensor = self._hist_p[:, :-1, :, :, :]

        weights = self.get_decay_weights(self.hist_len - 1, gamma=0.8, device=x.device)

        fused_f_hist = self.fuse_history_batch(hist_f_tensor, weights)
        fused_p_hist = self.fuse_history_batch(hist_p_tensor, weights)

        hist_embed_f = torch.cat([f_t, fused_f_hist], dim=1)
        hist_embed_p = torch.cat([p_t, fused_p_hist], dim=1)

        processed_p = self.mask_create(hist_embed_p)
        attn_output = self.attn_gen(processed_p, hist_embed_f)

        raw_preds, activated_preds = self.head(attn_output)
        return raw_preds, activated_preds

    def fuse_history_batch(self, hist_tensor, weights):
        B, H_len_minus_1, C, H, W = hist_tensor.shape
        weighted_hist = hist_tensor * weights.view(1, H_len_minus_1, 1, 1, 1).to(hist_tensor.device)
        fused_hist = weighted_hist.view(B, H_len_minus_1 * C, H, W)
        return fused_hist

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

    def compute_batch_loss(self, raw_preds, activated_preds, targets, ignore_list, img_size):

        B, C, H_f, W_f = raw_preds.shape
        img_H, img_W = img_size

        batch_conf_loss = 0.0
        batch_bbox_loss = 0.0
        total_matched_in_batch = 0
        total_non_ignored_in_batch = 0

        num_images_in_batch = B

        for i in range(num_images_in_batch):
            single_image_raw_preds_flat = raw_preds[i].permute(1, 2, 0).contiguous().view(-1, 5)
            single_image_activated_preds_flat = activated_preds[i].permute(1, 2, 0).contiguous().view(-1, 5)
            single_image_targets = targets[i]
            single_image_ignore = ignore_list[i] if ignore_list else []

            loss_dict = self._compute_single_loss(
                single_image_raw_preds_flat,
                single_image_activated_preds_flat,
                single_image_targets,
                single_image_ignore,
                img_size,
                (H_f, W_f)
            )
            batch_conf_loss += loss_dict['conf_loss']
            batch_bbox_loss += loss_dict['bbox_loss']
            total_matched_in_batch += loss_dict['num_matched']
            total_non_ignored_in_batch += loss_dict['num_non_ignored']

        total_batch_loss = batch_conf_loss + batch_bbox_loss

        final_avg_conf_loss = batch_conf_loss / num_images_in_batch
        final_avg_bbox_loss = batch_bbox_loss / num_images_in_batch
        final_avg_total_loss = total_batch_loss / num_images_in_batch

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

        if ignore_list is None:
            ignore_list = [[] for _ in range(B)]

        raw_preds, activated_preds = self(dehaze_imgs, training=True)

        loss_dict = self.compute_batch_loss(
            raw_preds,
            activated_preds,
            targets,
            ignore_list,
            (H_img, W_img)
        )

        self.reset_memory()

        return loss_dict

    @torch.no_grad()
    def predict(self, high_res_images, conf_thresh=0.5, iou_thresh=0.45):
        self.eval()
        B, _, H_img, W_img = high_res_images.shape

        raw_preds, activated_preds = self(high_res_images) # Get activated preds for decode_preds
        decoded_preds_list = decode_preds(activated_preds, (H_img, W_img)) # Use activated preds

        results = []
        for preds_i in decoded_preds_list:
            if preds_i.numel() == 0:
                results.append(torch.empty((0, 5), device=high_res_images.device))
                continue

            if preds_i.ndim == 1:
                 preds_i = preds_i.unsqueeze(0)

            boxes = preds_i[:, :4]
            scores = preds_i[:, 4]

            keep = scores > conf_thresh
            boxes = boxes[keep]
            scores = scores[keep]

            if boxes.numel() == 0:
                results.append(torch.empty((0, 5), device=high_res_images.device))
                continue

            nms_indices = ops.nms(boxes, scores, iou_thresh)
            boxes = boxes[nms_indices]
            scores = scores[nms_indices]

            filtered_preds = torch.cat([boxes, scores.unsqueeze(1)], dim=1)
            results.append(filtered_preds)

        return results

    def reset_memory(self):
        self._hist_f = None
        self._hist_p = None

    def get_decay_weights(self, length, gamma=0.8, device='cpu'):
        weights = torch.tensor([gamma ** i for i in reversed(range(length))], dtype=torch.float32)
        return weights.to(device)
