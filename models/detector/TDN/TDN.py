from collections import deque
from models.detector.TDN.utils import decode_preds, compute_iou, load_config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import ops
from torchvision.models import convnext_tiny

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
    def __init__(self, in_channels, num_classes = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes * 5, kernel_size=1)
        )

    def forward(self, x):
        return self.head(x)

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
        self.backbone = ConvNeXtBackbone()

        self.hist_compressor_f = nn.Conv2d(nums_f_channels, nums_f_channels // self.hist_len, kernel_size=1)
        self.hist_compressor_p = nn.Conv2d(nums_p_channels, nums_p_channels // self.hist_len, kernel_size=1)

        self.mask_create = AttentionGuidedStem(nums_deque_p, nums_deque_f, strides=[1, 1, 1])

        self.attn_gen = FusionModule(nums_deque_f * 2, hidden_channels)

        self.head = DetectionHead(hidden_channels)

        self._hist_f = None
        self._hist_p = None

    def forward(self, x):
        B, _, H_img, W_img = x.shape

        f_t = self.backbone(x)
        f_t = self.blur_attn(f_t)

        _, C_f, H_f, W_f = f_t.shape

        p_t_full_res = self.pos_enc(x)
        p_t = F.interpolate(p_t_full_res, size=(H_f, W_f), mode='bilinear', align_corners=False)
        _, C_p, _, _ = p_t.shape

        f_embed = self.hist_compressor_f(f_t)
        p_embed = self.hist_compressor_p(p_t)

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

        out = self.head(attn_output)

        return out

    def fuse_history_batch(self, hist_tensor, weights):
        B, H_len_minus_1, C, H, W = hist_tensor.shape
        weighted_hist = hist_tensor * weights.view(1, H_len_minus_1, 1, 1, 1).to(hist_tensor.device)
        fused_hist = weighted_hist.view(B, H_len_minus_1 * C, H, W)
        return fused_hist

    def _compute_single_loss(self, pred_boxes_conf, targets, ignore_list):
        gt_boxes = [[float(ann[2]), float(ann[3]),
                     float(ann[2]) + float(ann[4]),
                     float(ann[3]) + float(ann[5])] for ann in targets]

        ignore_boxes = [[float(ann[2]), float(ann[3]),
                         float(ann[2]) + float(ann[4]),
                         float(ann[3]) + float(ann[5])] for ann in ignore_list]

        if not gt_boxes:
            if pred_boxes_conf.numel() > 0:
                 conf_target = torch.zeros_like(pred_boxes_conf[:, 4])
                 conf_loss = F.mse_loss(pred_boxes_conf[:, 4], conf_target, reduction='sum')
                 bbox_loss = torch.tensor(0.0, device=pred_boxes_conf.device)
            else:
                 zero = torch.tensor(0.0, device=pred_boxes_conf.device)
                 conf_loss = zero
                 bbox_loss = zero
            return {'conf_loss': conf_loss, 'bbox_loss': bbox_loss}

        gt_boxes = torch.tensor(gt_boxes, device=pred_boxes_conf.device)
        ignore_boxes = torch.tensor(ignore_boxes, device=pred_boxes_conf.device) if ignore_boxes else None

        pred_boxes = pred_boxes_conf[:, :4]
        pred_confs = pred_boxes_conf[:, 4]

        if pred_boxes.numel() == 0:
            conf_loss = torch.tensor(0.0, device=pred_boxes_conf.device)
            bbox_loss = torch.tensor(0.0, device=pred_boxes_conf.device)
        else:
            ious_gt = compute_iou(pred_boxes, gt_boxes)
            best_iou_gt, best_gt_idx = ious_gt.max(dim=1)

            if ignore_boxes is not None and ignore_boxes.numel() > 0:
                 ious_ignore = compute_iou(pred_boxes, ignore_boxes)
                 max_iou_ignore, _ = ious_ignore.max(dim=1)
                 ignore_mask = max_iou_ignore > 0.5
            else:
                 ignore_mask = torch.zeros_like(best_iou_gt, dtype=torch.bool)

            conf_target = (best_iou_gt > 0.5).float()
            conf_loss = F.mse_loss(pred_confs[~ignore_mask], conf_target[~ignore_mask], reduction='sum')

            match_mask = (best_iou_gt > 0.5) & (~ignore_mask)
            matched_pred = pred_boxes[match_mask]
            matched_gt = gt_boxes[best_gt_idx[match_mask]]

            if len(matched_pred) > 0:
                 bbox_loss = F.smooth_l1_loss(matched_pred, matched_gt, reduction='sum')
            else:
                 bbox_loss = torch.tensor(0.0, device=pred_boxes_conf.device)

        return {
            'conf_loss': conf_loss,
            'bbox_loss': bbox_loss
        }

    def compute_batch_loss(self, preds, targets, ignore_list, img_size):
        pred_boxes_confs_list = decode_preds(preds, img_size)

        batch_conf_loss = 0.0
        batch_bbox_loss = 0.0
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
            batch_conf_loss += loss_dict['conf_loss']
            batch_bbox_loss += loss_dict['bbox_loss']

        total_batch_loss = batch_conf_loss + batch_bbox_loss

        avg_conf_loss = batch_conf_loss / num_images_in_batch
        avg_bbox_loss = batch_bbox_loss / num_images_in_batch
        avg_total_loss = total_batch_loss / num_images_in_batch

        log_dict = {
            'total_loss': avg_total_loss,
            'conf_loss': avg_conf_loss.detach(),
            'bbox_loss': avg_bbox_loss.detach()
        }

        return {'total_loss': avg_total_loss, **log_dict}


    def forward_loss(self, dehaze_imgs, targets, ignore_list=None):
        B, _, H_img, W_img = dehaze_imgs.shape

        if ignore_list is None:
            ignore_list = [[] for _ in range(B)]

        preds = self(dehaze_imgs)

        loss_dict = self.compute_batch_loss(
            preds,
            targets,
            ignore_list,
            (W_img, H_img)
        )

        self.reset_memory()

        return loss_dict

    @torch.no_grad()
    def predict(self, high_res_images, conf_thresh=0.95, iou_thresh=0.45):
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

        self.reset_memory()

        return results

    def reset_memory(self):
        self._hist_f = None
        self._hist_p = None

    def get_decay_weights(self, length, gamma=0.8, device='cpu'):
        weights = torch.tensor([gamma ** i for i in reversed(range(length))], dtype=torch.float32)
        return weights.to(device) 