import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import ops
import timm
from einops import rearrange

# Relative import for utils
from .utils import decode_preds, compute_iou, load_config

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * hidden_dim,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_candidate = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                        out_channels=hidden_dim,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        bias=self.bias)

    def forward(self, input_tensor, h_cur):
        combined_input = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.conv_gates(combined_input)
        reset_gate, update_gate = torch.split(gates, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        combined_candidate = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_candidate))
        h_next = (1 - update_gate) * h_cur + update_gate * candidate
        return h_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)


class EfficientNetLite0Backbone(nn.Module):
    def __init__(self, output_stage_idx=3):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_lite0', pretrained=False, features_only=True, out_indices=(output_stage_idx,))
        self.output_channels = self.model.feature_info.channels()[0]

    def forward(self, x):
        features = self.model(x)
        return features[0]


class AttentionGuidedStem(nn.Module):
    def __init__(self, in_channels, out_channels, strides=None):
        super().__init__()
        if strides is None:
            strides = [1, 1, 1]

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

# --- TDN Model ---
class TDN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_cfg_dict = {}
        try:
             model_cfg_dict = load_config('models/detector/TDN/TDN_config.yaml')['model']
             print("Loaded config from models/detector/TDN/TDN_config.yaml")
        except Exception as e:
             print(f"Warning: Could not load config file 'models/detector/TDN/TDN_config.yaml'. Using default values. Error: {e}")
             model_cfg_dict = {
                 'history_length': 4,
                 'nums_p_channels': 48,
             }

        self.backbone = EfficientNetLite0Backbone(output_stage_idx=3)
        nums_f_channels = self.backbone.output_channels
        print(f"Using EfficientNet-lite0 backbone. Feature channels: {nums_f_channels}")

        nums_p_channels = model_cfg_dict.get('nums_p_channels', 48)
        self.pos_enc = PositionalEncoder(3, nums_p_channels)
        print(f"Positional encoder output channels: {nums_p_channels}")

        self.convgru_f_hidden_dim = nums_f_channels
        self.convgru_p_hidden_dim = nums_p_channels
        self.convgru_f = ConvGRUCell(input_dim=nums_f_channels, hidden_dim=self.convgru_f_hidden_dim, kernel_size=(3, 3), bias=True)
        self.convgru_p = ConvGRUCell(input_dim=nums_p_channels, hidden_dim=self.convgru_p_hidden_dim, kernel_size=(3, 3), bias=True)
        self._hidden_state_f = None
        self._hidden_state_p = None

        self.mask_create_out_channels = self.convgru_p_hidden_dim
        self.mask_create = AttentionGuidedStem(self.convgru_p_hidden_dim, self.mask_create_out_channels, strides=[1, 1, 1])

        combined_channels = self.convgru_f_hidden_dim + self.mask_create_out_channels

        self.attn_gen = ECALayer(channels=combined_channels)

        self.final_proj_channels = 96
        self.final_conv = nn.Conv2d(combined_channels, self.final_proj_channels, kernel_size=1)

        self.head = DetectionHead(in_channels=self.final_proj_channels)


    def forward(self, x):
        # --- This part remains the same ---
        B, _, H_img, W_img = x.shape
        device = x.device

        f_t = self.backbone(x)
        _, C_f, H_f, W_f = f_t.shape

        p_t_full_res = self.pos_enc(x)
        p_t = F.interpolate(p_t_full_res, size=(H_f, W_f), mode='bilinear', align_corners=False)
        _, C_p, _, _ = p_t.shape

        if self._hidden_state_f is None or self._hidden_state_f.shape[0] != B:
            self._hidden_state_f = self.convgru_f.init_hidden(B, (H_f, W_f), device)
            self._hidden_state_p = self.convgru_p.init_hidden(B, (H_f, W_f), device)
        elif self._hidden_state_f.shape[2:] != (H_f, W_f):
             self._hidden_state_f = F.interpolate(self._hidden_state_f, size=(H_f, W_f), mode='bilinear', align_corners=False)
             self._hidden_state_p = F.interpolate(self._hidden_state_p, size=(H_f, W_f), mode='bilinear', align_corners=False)

        # h_f_cur = self._hidden_state_f.detach()
        # h_p_cur = self._hidden_state_p.detach()
        h_f_cur = self._hidden_state_f
        h_p_cur = self._hidden_state_p

        self._hidden_state_f = self.convgru_f(f_t, h_f_cur)
        self._hidden_state_p = self.convgru_p(p_t, h_p_cur)

        processed_p = self.mask_create(self._hidden_state_p)
        combined_features = torch.cat([processed_p, self._hidden_state_f], dim=1)
        attn_features = self.attn_gen(combined_features)
        final_features = self.final_conv(attn_features)
        out = self.head(final_features)

        return out

    def reset_memory(self):
        self._hidden_state_f = None
        self._hidden_state_p = None

    def _compute_single_loss(self, pred_boxes_conf, target_tensor, ignore_tensor):
        """
        Computes loss for a single image.
        Args:
            pred_boxes_conf (Tensor): Predicted boxes [N_pred, 5] (x1, y1, x2, y2, conf).
            target_tensor (Tensor): Ground truth tensor [N_gt, 9]. Format assumes cols 2,3,4,5 are (left, top, w, h) and col 6 is out-of-view flag.
            ignore_tensor (Tensor): Ignore tensor [N_ignore, 9]. Format assumes cols 2,3,4,5 are (left, top, w, h).
        """
        device = pred_boxes_conf.device

        # --- Filter GT targets: Keep only 'in-view' targets ---
        # !!! Assumption: target_tensor[:, 6] == 0 means the object is IN VIEW !!!
        # !!! Adjust the condition if your out-of-view flag logic is different !!!
        if target_tensor is not None and target_tensor.numel() > 0:
            in_view_mask = (target_tensor[:, 6] == 0)
            target_tensor_in_view = target_tensor[in_view_mask]
        else:
            target_tensor_in_view = torch.empty((0, target_tensor.shape[1] if target_tensor is not None else 9), device=device, dtype=pred_boxes_conf.dtype) # Maintain column count if possible

        # --- Convert IN-VIEW GT boxes from (left, top, w, h) to (x1, y1, x2, y2) ---
        if target_tensor_in_view.numel() > 0:
            # Extract columns based on the label format
            x1_gt = target_tensor_in_view[:, 2]
            y1_gt = target_tensor_in_view[:, 3]
            w_gt  = target_tensor_in_view[:, 4]
            h_gt  = target_tensor_in_view[:, 5]
            # Calculate bottom-right coordinates
            x2_gt = x1_gt + w_gt
            y2_gt = y1_gt + h_gt
            gt_boxes = torch.stack((x1_gt, y1_gt, x2_gt, y2_gt), dim=-1)
            num_gt = gt_boxes.shape[0]
        else:
            gt_boxes = torch.empty((0, 4), device=device, dtype=pred_boxes_conf.dtype)
            num_gt = 0
        # --- End GT Conversion ---

        # --- Convert IGNORE boxes from (left, top, w, h) to (x1, y1, x2, y2) ---
        # We assume the ignore_tensor already contains the boxes intended to be ignored.
        # We don't filter ignore_tensor based on its own out-of-view flag here,
        # unless specified otherwise.
        if ignore_tensor is not None and ignore_tensor.numel() > 0:
            x1_ign = ignore_tensor[:, 2]
            y1_ign = ignore_tensor[:, 3]
            w_ign  = ignore_tensor[:, 4]
            h_ign  = ignore_tensor[:, 5]
            x2_ign = x1_ign + w_ign
            y2_ign = y1_ign + h_ign
            ignore_boxes = torch.stack((x1_ign, y1_ign, x2_ign, y2_ign), dim=-1)
        else:
            ignore_boxes = torch.empty((0, 4), device=device, dtype=pred_boxes_conf.dtype)
        # --- End Ignore Conversion ---


        # --- Loss Calculation (rest remains similar) ---
        if num_gt == 0: # No valid (in-view) ground truth boxes for this image
            if pred_boxes_conf.numel() > 0:
                 conf_target = torch.zeros_like(pred_boxes_conf[:, 4])
                 conf_loss = F.mse_loss(pred_boxes_conf[:, 4], conf_target, reduction='sum')
                 bbox_loss = torch.tensor(0.0, device=device)
            else:
                 conf_loss = torch.tensor(0.0, device=device)
                 bbox_loss = torch.tensor(0.0, device=device)
            return {'conf_loss': conf_loss, 'bbox_loss': bbox_loss, 'num_gt': 0.0}

        # Predictions are already decoded to (x1, y1, x2, y2, conf)
        pred_boxes = pred_boxes_conf[:, :4]
        pred_confs = pred_boxes_conf[:, 4]

        if pred_boxes.numel() == 0: # No predictions left after potential filtering in decode
            conf_loss = torch.tensor(0.0, device=device)
            bbox_loss = torch.tensor(0.0, device=device)
        else:
            # --- IoU Calculations ---
            ious_gt = compute_iou(pred_boxes, gt_boxes) # [N_pred, N_gt]
            best_iou_gt, best_gt_idx = ious_gt.max(dim=1) # [N_pred]

            if ignore_boxes.numel() > 0:
                 ious_ignore = compute_iou(pred_boxes, ignore_boxes) # [N_pred, N_ignore]
                 max_iou_ignore, _ = ious_ignore.max(dim=1) # [N_pred]
                 # Ignore predictions overlapping significantly with ignore areas (IoU > 0.5)
                 # Also ignore predictions overlapping with GT boxes that were filtered out (out-of-view)?
                 # For simplicity, we only use the provided ignore_boxes for now.
                 ignore_mask = max_iou_ignore > 0.5
            else:
                 ignore_mask = torch.zeros_like(best_iou_gt, dtype=torch.bool)
            # --- End IoU ---

            # --- Confidence Loss ---
            conf_thresh = 0.5 # IoU threshold for positive match
            conf_target = (best_iou_gt > conf_thresh).float()
            # Only compute loss on predictions not masked by ignore regions
            valid_preds_mask = ~ignore_mask
            valid_conf_preds = pred_confs[valid_preds_mask]
            valid_conf_targets = conf_target[valid_preds_mask]

            if valid_conf_preds.numel() > 0:
                conf_loss = F.mse_loss(valid_conf_preds, valid_conf_targets, reduction='sum')
            else:
                conf_loss = torch.tensor(0.0, device=device)
            # --- End Confidence Loss ---

            # --- Bbox Regression Loss ---
            # Match predictions to GT based on IoU threshold, excluding ignored predictions
            match_mask = (best_iou_gt > conf_thresh) & valid_preds_mask
            matched_pred_boxes = pred_boxes[match_mask]
            # Ensure we only index gt_boxes with valid indices from best_gt_idx
            if match_mask.any(): # Check if there are any matches
                matched_gt_indices = best_gt_idx[match_mask]
                # Clamp indices just in case, although max should return valid indices if ious_gt is not empty
                # matched_gt_indices = torch.clamp(matched_gt_indices, 0, num_gt - 1) # Might hide errors
                matched_gt_boxes = gt_boxes[matched_gt_indices]
            else:
                matched_gt_boxes = torch.empty_like(matched_pred_boxes) # Create empty tensor if no matches


            if matched_pred_boxes.numel() > 0:
                 bbox_loss = F.smooth_l1_loss(matched_pred_boxes, matched_gt_boxes, reduction='sum', beta=1.0)
            else:
                 bbox_loss = torch.tensor(0.0, device=device)
            # --- End Bbox Loss ---

        return {
            'conf_loss': conf_loss,
            'bbox_loss': bbox_loss,
            'num_gt': float(num_gt) # Return num_gt (in-view targets) for potential normalization later
        }


    def compute_batch_loss(self, preds, targets, ignore_list, img_size):
        """
        Computes loss for the entire batch.
        Args:
            preds (Tensor): Raw model output [B, 5, Hf, Wf].
            targets (list[Tensor]): List of ground truth tensors for each image.
            ignore_list (list[Tensor]): List of ignore tensors for each image.
            img_size (tuple): Original image size (W, H).
        """
        pred_boxes_confs_list = decode_preds(preds, img_size) # List of [N_pred_i, 5] tensors

        batch_conf_loss = 0.0
        batch_bbox_loss = 0.0
        total_num_gt = 0.0
        num_images_in_batch = len(pred_boxes_confs_list)
        device = preds.device

        if num_images_in_batch == 0:
             return {'total_loss': torch.tensor(0.0, device=device),
                    'conf_loss': torch.tensor(0.0, device=device),
                    'bbox_loss': torch.tensor(0.0, device=device)}

        for i in range(num_images_in_batch):
            single_image_preds = pred_boxes_confs_list[i]
            # Get the corresponding tensors from the input lists
            single_image_targets = targets[i] if targets is not None and i < len(targets) else None
            single_image_ignore = ignore_list[i] if ignore_list is not None and i < len(ignore_list) else None

            # Ensure tensors are on the correct device
            if single_image_targets is not None:
                single_image_targets = single_image_targets.to(device)
            if single_image_ignore is not None:
                single_image_ignore = single_image_ignore.to(device)

            loss_dict = self._compute_single_loss(
                single_image_preds,
                single_image_targets,
                single_image_ignore
            )
            batch_conf_loss += loss_dict['conf_loss']
            batch_bbox_loss += loss_dict['bbox_loss']
            total_num_gt += loss_dict.get('num_gt', 0.0)

        # Normalize loss (e.g., by batch size)
        # Avoid division by zero if batch size is somehow zero (though checked earlier)
        norm_factor = float(num_images_in_batch) if num_images_in_batch > 0 else 1.0
        avg_conf_loss = batch_conf_loss / norm_factor
        avg_bbox_loss = batch_bbox_loss / norm_factor



        total_batch_loss = avg_conf_loss + avg_bbox_loss  # 带有计算图


        log_dict = {
            'log_total_loss': total_batch_loss.detach(),
            'conf_loss': avg_conf_loss.detach(),
            'bbox_loss': avg_bbox_loss.detach()
        }


        return {'total_loss': total_batch_loss,
                'conf_loss': log_dict['conf_loss'],
                'bbox_loss': log_dict['bbox_loss']}


    def forward_loss(self, dehaze_imgs, targets, ignore_list=None):

        B, _, H_img, W_img = dehaze_imgs.shape
        device = dehaze_imgs.device

        if ignore_list is None:
             ignore_list = [torch.empty((0, 1), device=device) for _ in range(B)]
        elif not isinstance(ignore_list, list) or not all(isinstance(el, torch.Tensor) for el in ignore_list):
             raise ValueError("ignore_list must be a list of Tensors.")

        if targets is None or not isinstance(targets, list) or not all(isinstance(el, torch.Tensor) for el in targets):
            raise ValueError("targets must be a list of Tensors for training.")

        self.reset_memory()

        preds = self(dehaze_imgs)

        # Compute loss
        loss_dict = self.compute_batch_loss(
            preds,
            targets,
            ignore_list,
            (W_img, H_img)
        )

        return loss_dict

    @torch.no_grad()
    def predict(self, high_res_images, conf_thresh=0.5, iou_thresh=0.45):
        self.eval()
        B, _, H_img, W_img = high_res_images.shape
        device = high_res_images.device

        output = self(high_res_images)
        decoded_preds_list = decode_preds(output, (W_img, H_img))

        results = []
        for preds_i in decoded_preds_list:
            if not isinstance(preds_i, torch.Tensor) or preds_i.numel() == 0:
                results.append(torch.empty((0, 5), device=device))
                continue

            if preds_i.ndim == 1:
                 preds_i = preds_i.unsqueeze(0)
            if preds_i.shape[1] != 5:
                 print(f"Warning: Unexpected prediction shape {preds_i.shape} after decode, skipping NMS.")
                 results.append(torch.empty((0, 5), device=device))
                 continue

            boxes = preds_i[:, :4]
            scores = preds_i[:, 4]

            keep = scores >= conf_thresh
            boxes = boxes[keep]
            scores = scores[keep]

            if boxes.numel() == 0:
                results.append(torch.empty((0, 5), device=device))
                continue

            nms_indices = ops.nms(boxes, scores, iou_thresh)
            boxes = boxes[nms_indices]
            scores = scores[nms_indices]

            filtered_preds = torch.cat([boxes, scores.unsqueeze(1)], dim=1) # [N_filtered, 5]
            results.append(filtered_preds)

        return results
