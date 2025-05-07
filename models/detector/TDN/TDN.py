from collections import deque

import timm

from models.detector.TDN.utils import decode_preds, compute_iou, load_config, focal_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import ops
from torchvision.models import convnext_tiny, mobilenet_v3_large

from torchvision.models import resnet18, ResNet18_Weights

class SwinTransformerFPNBackbone(nn.Module):
    def __init__(self, out_channels=192, model_name='swin_tiny_patch4_window7_224'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, img_size=(320, 640))

        # 取中后三个阶段的通道维度
        self.stage1_idx = 1  # 对应 H/8
        self.stage2_idx = 2  # 对应 H/16
        self.stage3_idx = 3  # 对应 H/32

        self.out_dims = [self.backbone.feature_info[i]['num_chs'] for i in [self.stage1_idx, self.stage2_idx, self.stage3_idx]]

        # lateral 1x1 convs
        self.lateral1 = nn.Conv2d(self.out_dims[0], out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(self.out_dims[1], out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(self.out_dims[2], out_channels, kernel_size=1)

        # 3x3 convs after lateral
        self.out_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.fusion_conv = nn.Conv2d(192 * 3, 192, kernel_size=3, padding=1)
    def forward(self, x):
        features = self.backbone(x)

        c1 = features[self.stage1_idx]  # H/8
        c2 = features[self.stage2_idx]  # H/16
        c3 = features[self.stage3_idx]  # H/32

        # 确保通道顺序为 [B, C, H, W]
        c1 = c1.permute(0, 3, 1, 2)
        c2 = c2.permute(0, 3, 1, 2)
        c3 = c3.permute(0, 3, 1, 2)

        # 构建 FPN 路径
        p3 = self.out_conv3(self.lateral3(c3))
        p2 = self.out_conv2(self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest'))
        p1 = self.out_conv1(self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest'))

        # 可选：最终融合所有尺度特征（若后续模块需要单一输出）
        p1_up = F.interpolate(p1, size=p3.shape[2:], mode='nearest')
        p2_up = F.interpolate(p2, size=p3.shape[2:], mode='nearest')
        fused = torch.cat([p1_up, p2_up, p3], dim=1)  # [B, 192*3, H/32, W/32]

        fused_out = self.fusion_conv(fused)

        return fused_out

class ResNet18FPNBackbone(nn.Module):
    def __init__(self, out_channels=192):
        super().__init__()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.stage1 = nn.Sequential(  # H/4
            model.conv1,  # 64
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1  # [B, 64, H/4, W/4]
        )
        self.stage2 = model.layer2  # [B, 128, H/8, W/8]
        self.stage3 = model.layer3  # [B, 256, H/16, W/16]

        # Lateral projections to 192 channels
        self.lateral1 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, out_channels, kernel_size=1)

        self.out_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        c1 = self.stage1(x)  # [B, 64, H/4, W/4]
        c2 = self.stage2(c1)  # [B, 128, H/8, W/8]
        c3 = self.stage3(c2)  # [B, 256, H/16, W/16]

        # FPN lateral connections
        p3 = self.lateral3(c3)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest')

        p3 = self.out_conv3(p3)
        p2 = self.out_conv2(p2)
        p1 = self.out_conv1(p1)

        return p3  # Return highest level feature map


class AttentionGuidedStem(nn.Module):
    def __init__(self, in_channels, out_channels, strides=None):
        super().__init__()
        if strides is None:
            strides = [2, 2, 2] # Default to downsampling H/8

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

        # MobileNetV3 layers (feature map downsample ratio ≈ 4x, 8x, 16x)
        self.stage1 = nn.Sequential(*model.features[:4])  # out: [B, 24, H/4, W/4]
        self.stage2 = nn.Sequential(*model.features[4:7])  # out: [B, 40, H/8, W/8]
        self.stage3 = nn.Sequential(*model.features[7:13])  # out: [B, 112, H/16, W/16]

        # FPN lateral projections to unify channel dims
        self.lateral1 = nn.Conv2d(24, out_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(40, out_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(112, out_channels, kernel_size=1)

        self.out_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        c1 = self.stage1(x)  # low level, H/4
        c2 = self.stage2(c1)  # mid level, H/8
        c3 = self.stage3(c2)  # high level, H/16

        # Lateral projections
        p3 = self.lateral3(c3)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest')

        # Output feature maps
        p3 = self.out_conv3(p3)  # [B, 192, H/16, W/16]
        p2 = self.out_conv2(p2)  # [B, 192, H/8, W/8]
        p1 = self.out_conv1(p1)  # [B, 192, H/4, W/4]

        return p3  # Use the top-level output (H/16) for detection
class KernelAttentionBlur(nn.Module):
    def __init__(self, in_channels, num_kernels=3):
        super().__init__()
        self.num_kernels = num_kernels
        kernel_sizes = [3, 5, 7][:num_kernels]

        # 多尺度模糊卷积（depthwise）
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels, bias=False)
            for k in kernel_sizes
        ])

        # 通道注意力融合
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels * num_kernels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_kernels, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 分支卷积输出堆叠: (B, num_kernels, C, H, W)
        branch_outputs = [branch(x) for branch in self.branches]
        stacked = torch.stack(branch_outputs, dim=1)

        B, N, C, H, W = stacked.shape
        combined = stacked.view(B, N * C, H, W)

        # 注意力权重生成: (B, N, H, W)
        attn_weights = self.attention_conv(combined)
        attn_weights = attn_weights.unsqueeze(2)  # (B, N, 1, H, W)

        # 加权融合模糊特征
        weighted = (stacked * attn_weights).sum(dim=1)  # (B, C, H, W)
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

        with torch.no_grad(): # 确保不记录梯度
             self.head[-1].bias[4::5].fill_(bias_value)

    def forward(self, x):
        out = self.head(x)
        B, C, H, W = out.shape
        assert C == self.num_classes * 5, f"Expected output channels {self.num_classes * 5}, but got {C}"

        out = out.view(B, self.num_classes, 5, H, W)

        dx_dy = torch.sigmoid(out[:, :, 0:2, :, :]) # Keep sigmoid here based on decode_preds
        dw_dh = torch.exp(torch.clamp(out[:, :, 2:4, :, :], max=4)) # Keep exp here
        conf = torch.sigmoid(out[:, :, 4:5, :, :]) # Keep sigmoid here

        out = torch.cat([dx_dy, dw_dh, conf], dim=2)  # [B, num_classes, 5, H, W]
        return out.view(B, self.num_classes * 5, H, W)




class TDN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg_dict = {}
        try:
             model_cfg_dict = load_config('models/detector/TDN/TDN_config,yaml')['model']
        except Exception as e:
             print(f"Warning: Could not load config file models/detector/TDN/TDN_config.yaml. Using default values. Error: {e}")
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
            # 构造滑动窗口式伪历史
            hist_f_list = []
            hist_p_list = []

            for b in range(B):
                frame_f = f_embed[b]
                frame_p = p_embed[b]

                fake_hist_f = []
                fake_hist_p = []

                # 计算当前帧在batch中的位置 b
                for i in range(self.hist_len - 1):
                    if b - i - 1 >= 0:
                        fake_hist_f.insert(0, f_embed[b - i - 1])  # 插入历史帧
                        fake_hist_p.insert(0, p_embed[b - i - 1])
                    else:
                        fake_hist_f.insert(0, frame_f)  # 不足时填充当前帧
                        fake_hist_p.insert(0, frame_p)

                hist_f_list.append(torch.stack(fake_hist_f, dim=0))
                hist_p_list.append(torch.stack(fake_hist_p, dim=0))

            hist_f_tensor = torch.stack(hist_f_list, dim=0)  # [B, hist_len-1, C, H, W]
            hist_p_tensor = torch.stack(hist_p_list, dim=0)
        else:
            # 推理时使用真实滑动历史
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
        # 加权融合历史信息
        weights = self.get_decay_weights(self.hist_len - 1, gamma=0.8, device=x.device)

        fused_f_hist = self.fuse_history_batch(hist_f_tensor, weights)
        fused_p_hist = self.fuse_history_batch(hist_p_tensor, weights)

        # 拼接历史与当前帧
        hist_embed_f = torch.cat([f_t, fused_f_hist], dim=1)
        hist_embed_p = torch.cat([p_t, fused_p_hist], dim=1)

        processed_p = self.mask_create(hist_embed_p)
        attn_output = self.attn_gen(processed_p, hist_embed_f)

        out = self.head(attn_output)
        return out

    def fuse_history_batch(self, hist_tensor, weights):
        B, H_len_minus_1, C, H, W = hist_tensor.shape
        weighted_hist = hist_tensor * weights.view(1, H_len_minus_1, 1, 1, 1).to(hist_tensor.device)
        fused_hist = weighted_hist.view(B, H_len_minus_1 * C, H, W)
        return fused_hist

    def _compute_single_loss(self, pred_boxes_conf, targets, ignore_list):
        # 将目标框和忽略区域框转换为 tensor
        gt_boxes = [[float(ann[2]), float(ann[3]),
                     float(ann[2]) + float(ann[4]),
                     float(ann[3]) + float(ann[5])] for ann in targets]
        gt_boxes_tensor = torch.tensor(gt_boxes, device=pred_boxes_conf.device) if gt_boxes else None

        ignore_boxes = [[float(ann[2]), float(ann[3]),
                         float(ann[2]) + float(ann[4]),
                         float(ann[3]) + float(ann[5])] for ann in ignore_list]
        ignore_boxes_tensor = torch.tensor(ignore_boxes, device=pred_boxes_conf.device) if ignore_boxes else None

        pred_boxes = pred_boxes_conf[:, :4]
        pred_confs = pred_boxes_conf[:, 4]

        # 初始化损失和计数
        conf_loss = torch.tensor(0.0, device=pred_boxes_conf.device)
        bbox_loss = torch.tensor(0.0, device=pred_boxes_conf.device)
        num_matched = 0
        num_non_ignored = 0

        # 计算忽略区域掩码 (如果存在预测框和忽略框)
        ignore_mask = None
        if ignore_boxes_tensor is not None and ignore_boxes_tensor.numel() > 0 and pred_boxes.numel() > 0:
             ious_ignore = compute_iou(pred_boxes, ignore_boxes_tensor)
             max_iou_ignore, _ = ious_ignore.max(dim=1)
             ignore_mask = max_iou_ignore > 0.5
        elif pred_boxes.numel() > 0: # 只有预测框，没有忽略框
             ignore_mask = torch.zeros_like(pred_confs, dtype=torch.bool)
        # else: pred_boxes 为空，ignore_mask 不用于索引 pred_confs，无需定义

        # 计算非忽略的预测网格细胞数量
        if pred_confs.numel() > 0:
             num_non_ignored = (~ignore_mask).sum().item() if ignore_mask is not None else pred_confs.numel()


        # --- 计算置信度损失 ---
        if pred_confs.numel() > 0:
             # 获取非忽略的预测置信度和对应的目标（如果存在 GT 框）
             current_ignore_mask = ignore_mask if ignore_mask is not None else torch.zeros_like(pred_confs, dtype=torch.bool)
             non_ignored_mask = ~current_ignore_mask
             non_ignored_preds = pred_confs[non_ignored_mask]

             if non_ignored_preds.numel() > 0:
                  if gt_boxes_tensor is not None and gt_boxes_tensor.numel() > 0:
                       # 如果有 GT 框，计算 IoU 并确定目标
                       ious_gt = compute_iou(pred_boxes, gt_boxes_tensor)
                       best_iou_gt, _ = ious_gt.max(dim=1)
                       conf_target = (best_iou_gt > 0.1).float()
                       non_ignored_targets = conf_target[non_ignored_mask]
                  else:
                       # 如果没有 GT 框，所有非忽略的目标都是背景（置信度为 0）
                       non_ignored_targets = torch.zeros_like(non_ignored_preds)

                  # 计算置信度损失
                  if getattr(self, "_use_mse", False):
                       conf_loss = F.mse_loss(non_ignored_preds, non_ignored_targets, reduction='mean')
                  else:
                       # 使用 focal_loss，并确保 reduction='mean'
                       conf_loss = focal_loss(non_ignored_preds, non_ignored_targets, alpha=0.25, gamma=2.0, reduction='mean')
             # else: non_ignored_preds 为空，conf_loss 保持 0.0

        # --- 计算边界框损失 ---
        if gt_boxes_tensor is not None and gt_boxes_tensor.numel() > 0 and pred_boxes.numel() > 0:
            # 只有在有 GT 框和预测框时才计算边界框损失
            # 需要重新计算 ious_gt 和 best_gt_idx，因为上面计算置信度时可能没有计算
            # 或者在上面计算 conf_loss 前就计算好并保存
            # 为了简洁，这里重新计算或者确保上面计算了
            ious_gt = compute_iou(pred_boxes, gt_boxes_tensor)
            best_iou_gt, best_gt_idx = ious_gt.max(dim=1)
            current_ignore_mask = ignore_mask if ignore_mask is not None else torch.zeros_like(pred_confs, dtype=torch.bool)

            # 匹配条件：与 GT 的 IoU > 0.1 且不在忽略区域
            match_mask = (best_iou_gt > 0.1) & (~current_ignore_mask)
            matched_pred = pred_boxes[match_mask]
            matched_gt = gt_boxes_tensor[best_gt_idx[match_mask]]

            num_matched = len(matched_pred)

            if num_matched > 0:
                 # 使用 SmoothL1Loss，并确保 reduction='mean'
                 bbox_loss = F.smooth_l1_loss(matched_pred, matched_gt, reduction='mean')
            # else: bbox_loss 保持 0.0


        # 返回损失和计数
        return {
            'conf_loss': conf_loss,
            'bbox_loss': bbox_loss,
            'num_matched': num_matched,
            'num_non_ignored': num_non_ignored
        }

    def compute_batch_loss(self, preds, targets, ignore_list, img_size):
        pred_boxes_confs_list = decode_preds(preds, img_size)

        batch_conf_loss = 0.0
        batch_bbox_loss = 0.0
        total_matched_in_batch = 0
        total_non_ignored_in_batch = 0

        num_images_in_batch = len(pred_boxes_confs_list)

        for i in range(num_images_in_batch):
            single_image_preds = pred_boxes_confs_list[i]
            single_image_targets = targets[i]
            single_image_ignore = ignore_list[i] if ignore_list else []

            loss_dict = self._compute_single_loss(
                single_image_preds,
                single_image_targets,
                single_image_ignore
            )
            # 直接累加归一化后的损失
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
            'avg_matched_per_image': total_matched_in_batch / num_images_in_batch,  # Add stats for debugging
            'avg_non_ignored_per_image': total_non_ignored_in_batch / num_images_in_batch
        }

        return {'total_loss': final_avg_total_loss, **log_dict}


    def forward_loss(self, dehaze_imgs, targets, ignore_list=None):
        B, _, H_img, W_img = dehaze_imgs.shape

        if ignore_list is None:
            ignore_list = [[] for _ in range(B)]

        preds = self(dehaze_imgs, training=True)

        loss_dict = self.compute_batch_loss(
            preds,
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

        output = self(high_res_images)

        decoded_preds_list = decode_preds(output, (H_img, W_img))

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