from collections import deque

import numpy as np
# import timm # Not used in the provided code snippet

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import ops # Not used in the provided code snippet
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
    # Ensure boxes are float and on the same device
    boxes1 = boxes1.float().to(boxes2.device)
    boxes2 = boxes2.float()
    # Handle empty tensors gracefully
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.empty((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=torch.float32)
    return box_iou(boxes1, boxes2)

def focal_loss(inputs, targets, alpha=0.5, gamma=1.0, reduction='sum'):
    """
    Args:
        inputs (Tensor): A float tensor of arbitrary shape. The predictions (probabilities).
        targets (Tensor): A float tensor with the same shape as inputs. The binary targets (0 or 1).
        alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples.
        gamma (float): Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
        reduction (string): 'none' | 'mean' | 'sum'
            'none': No reduction will be applied to the output.
            'mean': The output will be averaged.
            'sum': The output will be summed.
    Returns:
        Tensor: Loss scalar or tensor with the same shape of inputs depending on reduction mode.
    """
    # Ensure inputs are probabilities
    # BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    # Using logits directly with BCEWithLogitsLoss is more numerically stable
    # inputs here are assumed to be sigmoid(raw_conf), so they are probabilities already.
    # If using raw logits, change this to F.binary_cross_entropy_with_logits
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

    pt = torch.where(targets == 1, inputs, 1 - inputs) # p_t is the probability of the target class
    # pt = inputs * targets + (1 - inputs) * (1 - targets) # Alternative way to calculate pt

    # Add a small epsilon to prevent log(0) in the modulating factor (1-pt)
    # This is usually handled within the BCE loss itself if using BCEWithLogitsLoss
    # But since we are using F.binary_cross_entropy on probabilities, add epsilon to be safe
    pt = torch.clamp(pt, min=1e-6, max=1.0 - 1e-6)


    loss = alpha * (1 - pt) ** gamma * BCE_loss

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        # Note: Original implementation often averages over the number of positive samples
        # or the number of samples assigned to a ground truth box, not just all non-ignored.
        # For simplicity here, we average over the number of non-ignored samples,
        # consistent with how conf loss is computed in the original code structure.
        return loss.mean()
    else:
        return loss


def decode_preds(preds, img_size):
    B, C, H, W = preds.shape
    assert C == 5, f"Expected input channels 5, but got {C}"

    stride_y = img_size[0] / H
    stride_x = img_size[1] / W

    # Use torch.arange and broadcasting instead of meshgrid for potentially better performance/memory
    grid_y = torch.arange(H, device=preds.device).view(1, H, 1, 1).repeat(B, 1, W, 1)
    grid_x = torch.arange(W, device=preds.device).view(1, 1, W, 1).repeat(B, H, 1, 1)

    # These are already activated values from DetectionHead (sigmoid for dx, dy, conf; exp for dw, dh)
    # Clamp dx, dy to be within [0, 1] just in case sigmoid output is slightly outside due to precision
    dx = torch.clamp(preds[:, 0, :, :].unsqueeze(-1), 0., 1.)
    dy = torch.clamp(preds[:, 1, :, :].unsqueeze(-1), 0., 1.)
    dw = preds[:, 2, :, :].unsqueeze(-1) # These are exp(raw_dw)
    dh = preds[:, 3, :, :].unsqueeze(-1) # These are exp(raw_dh)
    conf = preds[:, 4, :, :].unsqueeze(-1) # These are sigmoid(raw_conf)

    # Calculate center coordinates (cx, cy) relative to image top-left
    # dx, dy are fractional offsets within the grid cell (0-1)
    cx = (grid_x.float() + dx) * stride_x
    cy = (grid_y.float() + dy) * stride_y

    # Calculate box width and height (bw, bh)
    # dw, dh are scale factors applied to stride
    bw = dw * stride_x
    bh = dh * stride_y

    # Convert center/width/height to x1, y1, x2, y2
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # Stack and reshape
    boxes_conf = torch.cat([x1, y1, x2, y2, conf], dim=-1) # [B, H, W, 5]
    boxes_conf = boxes_conf.view(B, H * W, 5) # [B, H*W, 5]

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
        # Should be B, C, H, W compatible, so size [1, out_channels, 1, 1] is correct
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(1, out_channels, 1, 1)) for _ in range(num_levels)
        ])

    def forward(self, features):
        # features is a list of tensors from backbone, e.g., [C2, C3, C4, C5]
        reduced = [conv(f) for conv, f in zip(self.reduce_convs, features)] # [P2_reduced, P3_reduced, P4_reduced, P5_reduced]

        # Get the size of the smallest feature map (P5)
        target_size = reduced[-1].shape[-2:]

        # Downsample/Resize all features to the smallest scale (P5 size)
        # Using bilinear for resizing might be better than avg_pool for features
        # resized = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in reduced]
        # Or keep avg_pool as in original code if that was intended
        resized = [F.adaptive_avg_pool2d(f, target_size) for f in reduced]

        # Sum the resized features
        fused = torch.stack(resized, dim=0).sum(dim=0) # [B, out_channels, H_p5, W_p5]

        # Upsample the fused feature map back to original reduced sizes and add position embeddings
        results = []
        for i in range(self.num_levels):
            # Interpolate fused feature map to the size of the i-th reduced feature map
            upsampled = F.interpolate(fused, size=reduced[i].shape[-2:], mode='nearest') # mode='nearest' is often used in FPN for simplicity
            results.append(upsampled + self.pos_embeds[i]) # Add position embedding

        # results is a list of tensors [P2', P3', P4', P5']
        return results


class ResNet50LiteFPNBackbone(nn.Module):
    def __init__(self, out_channels=128, pretrained=True):
        super().__init__()
        resnet = resnet50(pretrained=pretrained)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool, resnet.layer1)  # Output stride 4, channels 256
        self.layer2 = resnet.layer2  # Output stride 8, channels 512
        self.layer3 = resnet.layer3  # Output stride 16, channels 1024
        self.layer4 = resnet.layer4  # Output stride 32, channels 2048

        # FPN takes [C2, C3, C4, C5]
        self.fpn = LiteFPN(in_channels_list=[256, 512, 1024, 2048], out_channels=out_channels)

    def forward(self, x):
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4) # Corrected input to layer4
        # FPN returns a list of features [P2', P3', P4', P5']
        return self.fpn([c2, c3, c4, c5])




class DeformableAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, num_points=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.in_channels = in_channels
        # Channels per head
        self.channels_per_head = in_channels // num_heads
        assert self.channels_per_head * num_heads == in_channels, "in_channels must be divisible by num_heads"

        # Projects input to get offset, attention weights, and value
        self.offset_proj = nn.Conv2d(in_channels, num_heads * num_points * 2, kernel_size=1)
        self.attn_weight_proj = nn.Conv2d(in_channels, num_heads * num_points, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Initialize offset projection to zeros to start with identity mapping
        nn.init.constant_(self.offset_proj.weight, 0.)
        nn.init.constant_(self.offset_proj.bias, 0.)

        # Initialize attention weights projection to allow identity mapping initially
        # with weights around 1/num_points
        nn.init.constant_(self.attn_weight_proj.weight, 0.)
        nn.init.constant_(self.attn_weight_proj.bias, np.log(1.0 / self.num_points))

        # Initialize value and output projections
        nn.init.kaiming_uniform_(self.value_proj.weight, a=1)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.kaiming_uniform_(self.output_proj.weight, a=1)
        nn.init.constant_(self.output_proj.bias, 0.)


    # 移除 training=False 参数，因为在这个模块中不使用
    def forward(self, x):
        # x: Input tensor [B, C, H, W] - 这是 Attention 层要处理的特征图
        B, C, H, W = x.size()
        # Ensure input channels match
        assert C == self.in_channels, f"Input channels {C} must match layer channels {self.in_channels}"

        # Project to get value, offset, and attention weights
        value = self.value_proj(x) # [B, C, H, W]
        offsets = self.offset_proj(x) # [B, num_heads * num_points * 2, H, W]
        attn_weights = self.attn_weight_proj(x) # [B, num_heads * num_points, H, W]

        # Reshape and apply softmax to attention weights
        attn_weights = attn_weights.view(B, self.num_heads, self.num_points, H, W) # [B, num_heads, num_points, H, W]
        attn_weights = F.softmax(attn_weights, dim=2) # Softmax over sampling points per head per location

        # Prepare base grid for sampling (normalized coordinates [-1, 1])
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        # Stack grid_x and grid_y and reshape for broadcasting
        # base_grid shape: [1, 1, 1, H, W, 2] (B, num_heads, num_points, H, W, coords)
        base_grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, 1, H, W, 2)

        # Reshape offsets and add to base grid to get sampling locations
        # offsets shape: [B, num_heads * num_points * 2, H, W]
        # Reshape offsets to [B, num_heads, num_points, 2, H, W]
        offset = offsets.view(B, self.num_heads, self.num_points, 2, H, W)
        # Permute to [B, num_heads, num_points, H, W, 2] to match base_grid
        offset = offset.permute(0, 1, 2, 4, 5, 3)
        # Add offset to base grid (broadcasting happens)
        sampling_grid = base_grid + offset # [B, num_heads, num_points, H, W, 2]

        # Reshape sampling_grid for F.grid_sample
        # F.grid_sample expects [N, H_out, W_out, 2]
        # Here, N = B * num_heads * num_points
        sampling_grid = sampling_grid.view(B * self.num_heads * self.num_points, H, W, 2)

        # Prepare value for F.grid_sample
        # value shape: [B, C, H, W]
        # Reshape value to [B, num_heads, channels_per_head, H, W]
        value_for_sample = value.view(B, self.num_heads, self.channels_per_head, H, W)
        # Repeat value for each point: [B, num_heads, 1, channels_per_head, H, W] -> [B, num_heads, num_points, channels_per_head, H, W]
        value_for_sample = value_for_sample.unsqueeze(2).expand(-1, -1, self.num_points, -1, -1, -1)
        # Reshape for F.grid_sample: [B * num_heads * num_points, channels_per_head, H, W]
        value_for_sample = value_for_sample.reshape(B * self.num_heads * self.num_points, self.channels_per_head, H, W)

        # Perform sampling using grid_sample
        # sampled shape: [B * num_heads * num_points, channels_per_head, H, W]
        # sampling_grid coordinates are normalized [-1, 1]
        sampled = F.grid_sample(value_for_sample, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # Reshape sampled back to [B, num_heads, num_points, channels_per_head, H, W]
        sampled = sampled.view(B, self.num_heads, self.num_points, self.channels_per_head, H, W)

        # Apply attention weights: [B, num_heads, num_points, 1, H, W] * [B, num_heads, num_points, channels_per_head, H, W]
        attn_weights = attn_weights.unsqueeze(3) # Add channel dimension for broadcasting
        weighted = (sampled * attn_weights).sum(dim=2) # Sum over points: [B, num_heads, channels_per_head, H, W]

        # Combine heads: [B, num_heads * channels_per_head, H, W] = [B, in_channels, H, W]
        weighted = weighted.view(B, self.in_channels, H, W)

        # Output projection and add residual connection
        return self.output_proj(weighted) + x # 返回处理后的特征图


class ConvGRUCell(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.channels = channels
        # Input + previous hidden state -> 2 * channels input
        self.conv_zr = nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding)
        # Input + (reset * previous hidden) -> 2 * channels input
        self.conv_h = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        # x: current input tensor [B, C, H, W]
        # h_prev: previous hidden state [B, C, H, W]

        # Concatenate input and previous hidden state along channel dimension
        combined = torch.cat([x, h_prev], dim=1) # [B, 2C, H, W]

        # Compute update gate (z) and reset gate (r)
        zr = self.conv_zr(combined) # [B, 2C, H, W]
        z, r = torch.split(zr, self.channels, dim=1) # z: [B, C, H, W], r: [B, C, H, W]
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        # Compute candidate hidden state (h_hat)
        combined_reset = torch.cat([x, r * h_prev], dim=1) # [B, 2C, H, W]
        h_hat = torch.tanh(self.conv_h(combined_reset)) # [B, C, H, W]

        # Compute new hidden state (h)
        h = (1 - z) * h_prev + z * h_hat # [B, C, H, W]
        return h

class MultiFPNConvGRU(nn.Module):
    def __init__(self, channels=256, kernel_size=3):
        super().__init__()
        # Assuming the FPN outputs 4 levels (P2', P3', P4', P5')
        self.fpn_count = 4
        # Create a separate ConvGRUCell for each FPN level
        self.gru_cells = nn.ModuleList([
            ConvGRUCell(channels, kernel_size) for _ in range(self.fpn_count)
        ])

    def forward(self, hist_states, current_tensor_list):
        # hist_states: List[Tensor] with shape [B, C, H, W] representing historical states for each level, or None at t=0
        # current_tensor_list: List[Tensor] with shape [B, C, H, W] representing current features from FPN/Attention for each level
        updated_hist_states = []
        for i in range(self.fpn_count):
            # Get the previous state for the i-th level
            h_prev = hist_states[i]
            # Get the current input for the i-th level
            x_current = current_tensor_list[i]

            # If it's the first step (h_prev is None), initialize with zeros
            if h_prev is None:
                 h_prev = torch.zeros_like(x_current)

            # Compute the new hidden state using the GRU cell for this level
            h_new = self.gru_cells[i](x_current, h_prev)
            updated_hist_states.append(h_new)

        # Return the list of updated historical states for all levels
        return updated_hist_states


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        # The head predicts 5 values per grid cell per class: dx, dy, dw, dh, conf
        # For num_classes=1, this is 5 channels total.
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes * 5, kernel_size=1)
        )

        # Initialize the bias of the confidence prediction layer for stable training start
        # This encourages the model to predict low confidence initially
        pi = 0.01
        bias_value = -torch.log(torch.tensor((1 - pi) / pi))

        # Apply bias initialization only to the confidence part of the output layer
        with torch.no_grad():
            # Assuming output channels are ordered as [dx1, dy1, dw1, dh1, conf1, dx2, dy2, ...]
            # For num_classes=1, channels are [dx, dy, dw, dh, conf]
            if num_classes > 0:
                 # The confidence bias is at index 4, 9, 14, ...
                 self.head[-1].bias[4::5].fill_(bias_value)

    def forward(self, feats):
        # feats is a list of feature maps [P2', P3', P4', P5'] from FPN
        raw_preds_list = [] # List of raw outputs (logits for conf)
        activated_preds_list = [] # List of activated outputs (sigmoid for dx, dy, conf; exp for dw, dh)

        for x in feats: # Process each feature map level
            raw_preds = self.head(x) # [B, num_classes * 5, H, W]
            B, C, H, W = raw_preds.shape
            assert C == self.num_classes * 5, f"Expected {self.num_classes * 5} channels, but got {C} for input shape {raw_preds.shape}"

            # Reshape raw predictions to [B, num_classes, 5, H, W]
            raw_preds_reshaped = raw_preds.view(B, self.num_classes, 5, H, W)

            # Apply activations
            # dx, dy: sigmoid to constrain offsets to (0, 1) relative to cell top-left
            # dw, dh: exp to predict scales, typically unbounded raw values
            # conf: sigmoid for confidence score (0, 1)
            dx_dy = torch.sigmoid(raw_preds_reshaped[:, :, 0:2, :, :]) # [B, num_classes, 2, H, W]
            # Clamp exp(dw), exp(dh) to prevent numerical issues and excessively large boxes
            # A common clamp value is log(max_ratio), e.g., log(4) or log(1000/img_size)
            # The original code uses max=4, which means exp(raw_dw) <= exp(4) approx 54.6
            # This feels small for object detection. A more typical clamp might be higher,
            # or applied to the log space before exp. Let's keep the original clamp for now.
            # Consider clamping the raw log values before exp for better stability
            # dw_dh = torch.exp(torch.clamp(raw_preds_reshaped[:, :, 2:4, :, :], max=4)) # [B, num_classes, 2, H, W]
            # Let's try clamping the raw values to a slightly larger range, e.g., log(1000/32) approx log(31.25) approx 3.4, or log(100) approx 4.6
            # Using 6 seems reasonable for larger objects on smaller strides.
            dw_dh = torch.exp(torch.clamp(raw_preds_reshaped[:, :, 2:4, :, :], min=-6, max=6)) # Clamp log(scale) to [-6, 6] -> scale in [e^-6, e^6] approx [0.0025, 403]
            conf = torch.sigmoid(raw_preds_reshaped[:, :, 4:5, :, :]) # [B, num_classes, 1, H, W]

            # Concatenate activated predictions
            activated_preds_reshaped = torch.cat([dx_dy, dw_dh, conf], dim=2) # [B, num_classes, 5, H, W]
            # Reshape back to [B, num_classes * 5, H, W] for consistency with raw_preds_list
            activated_preds = activated_preds_reshaped.view(B, self.num_classes * 5, H, W)

            raw_preds_list.append(raw_preds)
            activated_preds_list.append(activated_preds)

        # Return lists of raw and activated predictions for each FPN level
        return raw_preds_list, activated_preds_list

class DRIFT_NET(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # IoU threshold for marking predictions as 'ignore'
        self.ignore_iou_threshold = 0.3 # Common value, adjust as needed

        # Flag to potentially use MSE loss for confidence instead of Focal Loss
        self._use_mse = False # Keep as False for standard detection training
        self._evaluating = False
        # Load model-specific config
        model_cfg_dict = {}
        try:
             # Assuming the config path is correct relative to where the script is run
             model_cfg_dict = load_config('models/detector/DRIFT_NET/DRIFT_NET_config.yaml')['model']
        except FileNotFoundError:
             print("Warning: DRIFT_NET_config.yaml not found. Using default model parameters.")
             model_cfg_dict = {
                 'backbone_channel': 192,
                 'step': 4
             }
        except Exception as e:
             print(f"Warning: Error loading DRIFT_NET_config.yaml: {e}. Using default model parameters.")
             model_cfg_dict = {
                 'backbone_channel': 192,
                 'step': 4
             }

        # Number of time steps processed together during training
        self.step = model_cfg_dict.get('step', 4)
        # Output channel dimension for FPN and subsequent layers
        self.backbone_channel = model_cfg_dict.get('backbone_channel', 192)

        # Backbone + LiteFPN
        self.backbone = ResNet50LiteFPNBackbone(out_channels=self.backbone_channel, pretrained=True)

        # Deformable Attention layer for each FPN level
        # Assuming 4 FPN levels (P2', P3', P4', P5')
        self.attn = nn.ModuleList([DeformableAttention(in_channels=self.backbone_channel) for _ in range(4)])

        # ConvGRU for temporal feature fusion across FPN levels
        self.ConvGRU = MultiFPNConvGRU(channels=self.backbone_channel)

        # Detection head applied to the output of the ConvGRU for each level
        # Assuming 1 class for detection (e.g., person, vehicle)
        self.detection = DetectionHead(in_channels=self.backbone_channel, num_classes=1)

        # Internal state for temporal memory during inference
        # Initialized to None, will be updated by ConvGRU
        # List of size 4 for the 4 FPN levels
        self._hist_f = [None] * 4

    def forward(self, x, training=False):
        # x: Input tensor [B, C, H, W]

        # Extract spatial features using backbone and FPN
        # f_t: List of features for current time step [P2', P3', P4', P5']
        f_t = self.backbone(x) # List of [B, C, H_f, W_f]

        # Apply Deformable Attention to spatial features
        attn_f = [self.attn[i](x_i) for i, x_i in enumerate(f_t)] # List of [B, C, H_f, W_f]

        # Number of FPN levels
        FPN_levels = len(attn_f)
        B, C = attn_f[0].shape[:2]

        if training:
            # During training, process sequences of length 'step'
            # The batch size B must be a multiple of self.step
            assert B % self.step == 0, f"Batch size ({B}) must be a multiple of self.step ({self.step}) during training."

            num_sequences = B // self.step
            # Use the original batch size B here for reshaping reference
            B_original = B

            # Reshape each feature map in attn_f from [B_original, C, H, W] to [num_sequences, step, C, H, W]
            # This prepares the data for processing step-by-step for each sequence.
            attn_f_reshaped_sequences = [
                f_level.view(num_sequences, self.step, C, f_level.shape[2], f_level.shape[3])
                for f_level in attn_f
            ] # List of [num_sequences, step, C, H_f, W_f]

            # Initialize history for each FPN level.
            # history_states will be a list of tensors, one for each FPN level.
            # Each tensor will have shape [num_sequences, C, H_f, W_f] and will track the history
            # for all sequences at that specific FPN level. Initially, they are None.
            history_states = [None] * FPN_levels

            # Process step by step (t=0 to step-1)
            for t in range(self.step):
                # Get the features for the current time step 't' across all sequences
                # This is a list of [num_sequences, C, H_f, W_f] tensors, one for each FPN level
                current_frame_features_t = [
                    f_level_reshaped[:, t, :, :, :] # Select the t-th frame from all sequences for this level
                    for f_level_reshaped in attn_f_reshaped_sequences
                ] # List of [num_sequences, C, H_f, W_f]

                # Apply ConvGRU.
                # ConvGRU takes:
                # 1. hist_states: List of [num_sequences, C, H_f, W_f] tensors (previous states per level, across sequences)
                # 2. current_tensor_list: List of [num_sequences, C, H_f, W_f] tensors (current inputs per level, across sequences)
                # It returns a list of updated history tensors [num_sequences, C, H_f, W_f].
                history_states = self.ConvGRU(history_states, current_frame_features_t)

            # After the loop, history_states contains the final hidden states
            # for the last time step (t = step - 1) of each sequence, for each FPN level.
            # history_states is a list of [num_sequences, C, H_f, W_f] tensors.

            # Apply detection head to these final states.
            # The detection head expects a list of feature maps [B', C, H_f, W_f].
            # Here, B' is num_sequences (the number of independent sequences processed).
            raw_preds_list, activated_preds_list = self.detection(history_states)

        else: # Inference mode
            # Process one frame at a time, maintaining the history _hist_f
            # attn_f is already a list of [B, C, H_f, W_f] where B is the batch size (usually 1)
            # If B > 1, it processes them as a batch, sharing the same history initialization.
            # This might not be ideal for independent sequences in a batch.
            # For typical video inference (B=1), this is correct.
            if self._hist_f[0] is None or self._hist_f[0].shape[0] != B:
                 # Initialize history if batch size changes or it's the first frame
                 # Note: This initializes history independently for each item in the batch dimension.
                 # If B > 1 represents independent sequences, this is okay.
                 # If B > 1 represents consecutive frames, this logic might need adjustment
                 # to maintain history across the batch dimension correctly, or process B=1.
                 # Assuming B=1 is the primary inference case, or B>1 are independent sequences.
                 self._hist_f = [torch.zeros_like(f) for f in attn_f]

            # Apply ConvGRU, updating _hist_f in place (or reassigning)
            # ConvGRU takes List[Tensor] for hist_states and current_tensor_list
            # In inference, the batch size for GRU is B.
            self._hist_f = self.ConvGRU(self._hist_f, attn_f) # _hist_f is now List of [B, C, H_f, W_f]

            # Apply detection head to the updated history states
            raw_preds_list, activated_preds_list = self.detection(self._hist_f)

        # Return raw and activated predictions
        # In training, this is for the final frame of each sequence in the batch (batch size B // step)
        # In inference, this is for the current frame(s) (batch size B)
        return raw_preds_list, activated_preds_list

    def _compute_single_loss(self, raw_preds_flat, activated_preds_flat, targets, ignore_list, img_size, feature_map_size):
        """
        Computes losses for a single image on a single feature map level using grid-based matching.

        Args:
            raw_preds_flat (Tensor): Raw predictions (logits for conf, raw dx, dy, dw, dh) flattened to [H*W, 5].
            activated_preds_flat (Tensor): Activated predictions (sigmoid dx, dy, exp dw, dh, sigmoid conf) flattened to [H*W, 5].
            targets (list): List of GT annotations for the image [class_id, track_id, x, y, w, h].
            ignore_list (list): List of ignore annotations for the image [class_id, track_id, x, y, w, h].
            img_size (tuple): (H_img, W_img).
            feature_map_size (tuple): (H_f, W_f).

        Returns:
            dict: {'conf_loss', 'bbox_loss', 'num_matched', 'num_non_ignored'} for this level/image.
        """
        H_f, W_f = feature_map_size
        img_H, img_W = img_size
        stride_y = img_H / H_f
        stride_x = img_W / W_f
        device = raw_preds_flat.device

        N_preds = raw_preds_flat.shape[0] # Number of predictions (H_f * W_f)

        # Create grid coordinates (for bbox target encoding and positive sample matching)
        grid_y_coords, grid_x_coords = torch.meshgrid(torch.arange(H_f, device=device),
                                                      torch.arange(W_f, device=device),
                                                      indexing='ij')
        # Flatten grid coordinates [H_f*W_f]
        grid_x_coords_flat = grid_x_coords.flatten().float()
        grid_y_coords_flat = grid_y_coords.flatten().float()


        # --- Initialize Targets and Masks ---
        # Confidence target: 1 for positive, 0 for negative
        conf_target = torch.zeros(N_preds, dtype=torch.float32, device=device)
        # Bbox target: Encoded dx, dy, dw, dh for positive samples
        bbox_target_raw = torch.zeros(N_preds, 4, dtype=torch.float32, device=device)
        # Masks
        positive_mask = torch.zeros(N_preds, dtype=torch.bool, device=device)
        ignore_mask = torch.zeros(N_preds, dtype=torch.bool, device=device)


        # --- Determine Ignore Samples (Based on Predicted Boxes IoU with Ignore Regions) ---
        # We still use predicted boxes here to avoid penalizing predictions that fall into ignore regions
        pred_boxes_decoded = self.decode_activated_preds_to_boxes(activated_preds_flat, img_size, feature_map_size) # [N_preds, 4]

        ignore_boxes = []
        if ignore_list is not None:
             ignore_boxes = [[float(ann[2]), float(ann[3]),
                              float(ann[2]) + float(ann[4]),
                              float(ann[3]) + float(ann[5])] for ann in ignore_list]
        ignore_boxes_tensor = torch.tensor(ignore_boxes, device=device, dtype=torch.float32) if ignore_boxes else None

        if ignore_boxes_tensor is not None and ignore_boxes_tensor.numel() > 0 and pred_boxes_decoded.numel() > 0:
            # Compute IoU between all predictions and ignore boxes
            # ious_ignore shape: [N_preds, N_ignore_boxes]
            ious_ignore = compute_iou(pred_boxes_decoded, ignore_boxes_tensor)
            # Find the maximum IoU for each prediction with any ignore box
            max_iou_ignore, _ = ious_ignore.max(dim=1) # [N_preds]
            # Predictions with max IoU > ignore_iou_threshold with any ignore box are ignored
            ignore_mask = max_iou_ignore > self.ignore_iou_threshold


        # --- Determine Positive Samples and Bbox Targets (Based on GT Box Centers) ---
        gt_boxes = []
        if targets is not None:
            gt_boxes = [[float(ann[2]), float(ann[3]),
                         float(ann[2]) + float(ann[4]),
                         float(ann[3]) + float(ann[5])] for ann in targets]
        gt_boxes_tensor = torch.tensor(gt_boxes, device=device, dtype=torch.float32) if gt_boxes else None

        if gt_boxes_tensor is not None and gt_boxes_tensor.numel() > 0:
            # Iterate through each ground truth box
            for gt_box in gt_boxes_tensor:
                # Calculate GT box center
                gt_cx = (gt_box[0] + gt_box[2]) / 2.0
                gt_cy = (gt_box[1] + gt_box[3]) / 2.0
                gt_w = gt_box[2] - gt_box[0]
                gt_h = gt_box[3] - gt_box[1]

                # Find the grid cell index corresponding to the GT center
                grid_x_gt = int(gt_cx / stride_x)
                grid_y_gt = int(gt_cy / stride_y)

                # Check if the grid cell is within the feature map boundaries
                if 0 <= grid_y_gt < H_f and 0 <= grid_x_gt < W_f:
                    # Calculate the flattened index for this grid cell
                    pred_idx = grid_y_gt * W_f + grid_x_gt

                    # Mark this grid cell as a positive sample
                    positive_mask[pred_idx] = True
                    # Set confidence target to 1.0 for this positive sample
                    conf_target[pred_idx] = 1.0

                    # Calculate the raw bbox targets (dx, dy, dw, dh) for this cell
                    # These targets are relative to the grid cell's top-left corner (grid_x_gt, grid_y_gt)
                    # based on the decoding formula: cx = (grid_x + dx) * stride_x
                    target_dx_raw = (gt_cx / stride_x) - grid_x_gt
                    target_dy_raw = (gt_cy / stride_y) - grid_y_gt
                    # dw, dh targets are log(GT_dim / stride)
                    target_dw_raw = torch.log(gt_w / stride_x + 1e-6) # Add epsilon for numerical stability
                    target_dh_raw = torch.log(gt_h / stride_y + 1e-6) # Add epsilon

                    # Store the calculated targets
                    bbox_target_raw[pred_idx, :] = torch.tensor([target_dx_raw, target_dy_raw, target_dw_raw, target_dh_raw], device=device)

        non_ignored_mask = ~ignore_mask
        num_non_ignored = non_ignored_mask.sum().item()


        # --- Calculate Losses ---
        num_matched = positive_mask.sum().item() # Count of positive samples

        # Bbox Loss: Only calculated for positive samples
        bbox_loss = torch.tensor(0.0, device=device) # Initialize bbox loss to 0
        if num_matched > 0:
            # Select raw bbox predictions and targets for positive samples
            matched_pred_raw_bbox = raw_preds_flat[positive_mask][:, :4]
            matched_bbox_target_raw = bbox_target_raw[positive_mask]
            # print('matched_pred_raw_bbox',matched_pred_raw_bbox)
            # print('matched_bbox_target_raw',matched_bbox_target_raw)
            # Calculate Smooth L1 loss between raw bbox predictions and raw targets
            bbox_loss = F.smooth_l1_loss(matched_pred_raw_bbox, matched_bbox_target_raw, reduction='sum') # Sum over matched samples


        # Confidence Loss: Calculated for all non-ignored samples
        conf_loss = torch.tensor(0.0, device=device) # Initialize conf loss to 0
        if num_non_ignored > 0:
            # Select raw confidence logits and targets for non-ignored samples
            pred_raw_conf_non_ignored = raw_preds_flat[non_ignored_mask][:, 4] # Get logits
            conf_target_non_ignored = conf_target[non_ignored_mask] # Get target (0 or 1)

            # Apply sigmoid to get probabilities for Focal Loss
            pred_conf_activated_non_ignored = torch.sigmoid(pred_raw_conf_non_ignored)

            if getattr(self, "_use_mse", False):
                conf_loss = F.mse_loss(pred_conf_activated_non_ignored, conf_target_non_ignored, reduction='sum') # Sum over non-ignored samples
            else:
                conf_loss = focal_loss(pred_conf_activated_non_ignored, conf_target_non_ignored, alpha=0.25, gamma=2.0, reduction='sum') # Sum over non-ignored samples


        return {
            'conf_loss': conf_loss, # Sum of conf loss over non-ignored samples
            'bbox_loss': bbox_loss, # Sum of bbox loss over positive samples
            'num_matched': num_matched, # Count of positive samples
            'num_non_ignored': num_non_ignored # Count of non-ignored samples
        }

    def compute_batch_loss(self, raw_preds_list, activated_preds_list, targets, ignore_list, img_size):
        """
        Computes total loss for a batch across all FPN levels.

        Args:
            raw_preds_list (list): List of raw prediction tensors [B', 5, H_f, W_f] for each level.
                                  B' is the number of sequences processed (B // step).
            activated_preds_list (list): List of activated prediction tensors [B', 5, H_f, W_f] for each level.
            targets (list): List of list of GT annotations for each of the B' images.
            ignore_list (list, optional): List of list of ignore annotations for each of the B' images. Defaults to None.
            img_size (tuple): (H_img, W_img).

        Returns:
            dict: Total loss and logging metrics.
        """
        # B' is the number of images in the batch processed by the detection head (B // step)
        num_images_in_batch = raw_preds_list[0].shape[0]
        img_H, img_W = img_size

        total_conf_loss_sum = 0.0 # Sum of conf loss over all non-ignored samples in the batch/levels
        total_bbox_loss_sum = 0.0 # Sum of bbox loss over all positive samples in the batch/levels
        total_matched_in_batch = 0 # Total count of positive samples in the batch/levels
        total_non_ignored_in_batch = 0 # Total count of non-ignored samples in the batch/levels

        num_levels = len(raw_preds_list)

        # Iterate through each image in the batch
        for i in range(num_images_in_batch):
            single_image_targets = targets[i]
            single_image_ignore = ignore_list[i] if ignore_list and i < len(ignore_list) and ignore_list[i] is not None else [] # Handle potential missing ignore list/entry

            # Iterate through each FPN level
            for level_idx in range(num_levels):
                raw_preds_level = raw_preds_list[level_idx] # [B', 5, H_f, W_f]
                activated_preds_level = activated_preds_list[level_idx] # [B', 5, H_f, W_f]

                H_f, W_f = raw_preds_level.shape[2:]

                # Flatten predictions for the current image and level
                raw_preds_flat = raw_preds_level[i].permute(1, 2, 0).contiguous().view(-1, 5) # [H_f*W_f, 5]
                activated_preds_flat = activated_preds_level[i].permute(1, 2, 0).contiguous().view(-1, 5) # [H_f*W_f, 5]

                # Compute loss for this single image and level using the new matching strategy
                loss_dict = self._compute_single_loss(
                    raw_preds_flat,
                    activated_preds_flat,
                    single_image_targets,
                    single_image_ignore,
                    img_size,
                    (H_f, W_f)
                )

                # Accumulate sums
                total_conf_loss_sum += loss_dict['conf_loss']
                total_bbox_loss_sum += loss_dict['bbox_loss']
                total_matched_in_batch += loss_dict['num_matched']
                total_non_ignored_in_batch += loss_dict['num_non_ignored']

        # Calculate average losses
        # Average confidence loss over all non-ignored samples across the batch and levels
        final_avg_conf_loss = total_conf_loss_sum / max(1, total_non_ignored_in_batch)
        # Average bbox loss over all positive samples across the batch and levels
        final_avg_bbox_loss = total_bbox_loss_sum / max(1, total_matched_in_batch)

        # Total loss is a weighted sum of average losses
        # Use weights (e.g., 2.0 for conf, 1.0 for bbox) - adjust as needed during tuning
        final_avg_total_loss = 2.0 * final_avg_conf_loss + 1.0 * final_avg_bbox_loss

        # Create log_dict for logging purposes, detaching values here
        log_dict = {
            'conf_loss': final_avg_conf_loss.detach(),
            'bbox_loss': final_avg_bbox_loss.detach(),
            # Log total counts for the batch and levels
            'total_matched': total_matched_in_batch,
            'total_non_ignored': total_non_ignored_in_batch
            # Could also log average counts per image if needed:
            # 'avg_matched_per_image': total_matched_in_batch / max(1, num_images_in_batch),
            # 'avg_non_ignored_per_image': total_non_ignored_in_batch / max(1, num_images_in_batch)
        }

        # Return the original final_avg_total_loss tensor under the key 'total_loss'
        # Use **log_dict to include other logging metrics (which are correctly detached)
        return {'total_loss': final_avg_total_loss, **log_dict}


    def decode_activated_preds_to_boxes(self, activated_preds_flat, img_size, feature_map_size):
         """
         Decodes flattened activated predictions [N, 5] into bounding boxes [N, 4].
         Used internally for IoU calculation during loss computation (specifically for ignore regions).
         This must match the logic in decode_preds but handles flattened input.
         """
         N = activated_preds_flat.shape[0] # Number of predictions (H_f * W_f)
         H_img, W_img = img_size
         H_f, W_f = feature_map_size
         stride_y = H_img / H_f
         stride_x = W_img / W_f # Corrected typo W_W_img -> W_img
         device = activated_preds_flat.device

         # Create grid coordinates (flattened)
         grid_y, grid_x = torch.meshgrid(torch.arange(H_f, device=device),
                                         torch.arange(W_f, device=device),
                                         indexing='ij')
         # Flatten grid coordinates [H_f*W_f]
         grid_x = grid_x.flatten().float()
         grid_y = grid_y.flatten().float()

         # These are activated values from DetectionHead
         # dx, dy are sigmoid outputs (0-1)
         # dw, dh are exp outputs
         # Clamp dx, dy to be within [0, 1] for robustness
         dx = torch.clamp(activated_preds_flat[:, 0], 0., 1.) # [N]
         dy = torch.clamp(activated_preds_flat[:, 1], 0., 1.) # [N]
         dw = activated_preds_flat[:, 2] # [N]
         dh = activated_preds_flat[:, 3] # [N]

         # Calculate center coordinates (cx, cy) and width/height (bw, bh)
         # cx = (grid_x + dx) * stride_x  -- dx, dy are 0-1 based on sigmoid. This assumes dx, dy are offsets relative to top-left corner.
         cx = (grid_x + dx) * stride_x
         cy = (grid_y + dy) * stride_y
         # dw, dh from exp are scale factors applied to stride.
         bw = dw * stride_x
         bh = dh * stride_y

         # Convert center/width/height to x1, y1, x2, y2
         x1 = cx - bw / 2
         y1 = cy - bh / 2
         x2 = cx + bw / 2
         y2 = cy + bh / 2

         # Clamp box coordinates to image boundaries [0, W_img], [0, H_img] for robustness
         x1 = torch.clamp(x1, 0., float(W_img))
         y1 = torch.clamp(y1, 0., float(H_img))
         x2 = torch.clamp(x2, 0., float(W_img))
         y2 = torch.clamp(y2, 0., float(H_img))

         # Ensure x1 <= x2 and y1 <= y2 (handle potential flipped boxes due to large dx/dy or small dw/dh)
         # This can happen if cx - bw/2 > cx + bw/2, i.e., -bw/2 > bw/2, bw < 0. bw = exp(dw)*stride, so bw is always >=0.
         # Flipped boxes are more likely if center prediction is far off or width/height is near zero.
         # A simpler way is to ensure width/height are positive after prediction.
         # Or just swap if needed.
         # Let's ensure width/height are positive by clamping exp(dw), exp(dh) in DetectionHead.
         # And ensure x1 <= x2, y1 <= y2 just in case.
         boxes = torch.stack([x1, y1, x2, y2], dim=1)
         # Swap if x1 > x2 or y1 > y2
         boxes[:, [0, 2]] = torch.sort(boxes[:, [0, 2]], dim=1)[0]
         boxes[:, [1, 3]] = torch.sort(boxes[:, [1, 3]], dim=1)[0]

         return boxes # [N, 4]


    def forward_loss(self, dehaze_imgs, targets, ignore_list=None):

        B, _, H_img, W_img = dehaze_imgs.shape

        # --- Determine mode based on custom flag ---
        if not self._evaluating: # If not evaluating, assume training mode

            # Ensure batch size is a multiple of self.step for training
            # Truncate batch if needed
            if B % self.step != 0:
                 print(f"Warning: Batch size {B} not divisible by step {self.step} in training mode. Truncating batch to {B - (B % self.step)}.")
                 dehaze_imgs = dehaze_imgs[:B - (B % self.step)]
                 targets = targets[:B - (B % self.step)]
                 if ignore_list is not None:
                     ignore_list = ignore_list[:B - (B % self.step)]
                 B, _, H_img, W_img = dehaze_imgs.shape # Update B after truncation

            final_frame_indices = torch.arange(self.step - 1, B, self.step).tolist()
            targets_for_loss = [targets[i] for i in final_frame_indices]
            ignore_list_for_loss = [ignore_list[i] for i in final_frame_indices] if ignore_list is not None else None

            # Perform the forward pass in training mode.
            # Explicitly pass training=True to the forward method.
            raw_preds_list, activated_preds_list = self(dehaze_imgs, training=True)


        else:
            raw_preds_list = []

            activated_preds_list = []

            targets_for_loss = []

            ignore_list_for_loss = []


            for i in range(B):

                single_img = dehaze_imgs[i].unsqueeze(0)

                single_targets = targets[i]

                single_ignore = ignore_list[i] if ignore_list is not None else None

                raw_preds_single_list, activated_preds_single_list = self(single_img,
                                                                          training=False)

                if i == 0:

                    raw_preds_list = [[] for _ in range(len(raw_preds_single_list))]

                    activated_preds_list = [[] for _ in range(len(activated_preds_single_list))]

                for level_idx in range(len(raw_preds_single_list)):
                    raw_preds_list[level_idx].append(raw_preds_single_list[level_idx])

                    activated_preds_list[level_idx].append(
                        activated_preds_single_list[level_idx])

                targets_for_loss.append(single_targets)

                ignore_list_for_loss.append(single_ignore)

            raw_preds_list = [torch.cat(preds_level_list, dim=0) for preds_level_list in
                              raw_preds_list]

            activated_preds_list = [torch.cat(preds_level_list, dim=0) for preds_level_list in
                                    activated_preds_list]

        loss_dict = self.compute_batch_loss(
            raw_preds_list,
            activated_preds_list,
            targets_for_loss,
            ignore_list_for_loss,
            (H_img, W_img),
        )
        self.reset_memory()
        return loss_dict


    @torch.no_grad()
    def predict(self, high_res_images, conf_thresh=0.5, iou_thresh=0.45):
        """
        Performs inference on a batch of images. Assumes images are processed sequentially
        in time for the temporal model.

        Args:
            high_res_images (Tensor): Input batch of images [B, C, H, W].
                                      For standard inference, B should be 1.
                                      If B > 1, it processes them as a single batch,
                                      maintaining one history state for the whole batch.
                                      For processing *multiple independent sequences*,
                                      you would need to loop over sequences externally
                                      and call predict with batch size 1, resetting memory
                                      between sequences.
            conf_thresh (float): Confidence threshold for filtering predictions.
            iou_thresh (float): NMS IoU threshold.

        Returns:
            list: A list of tensors, where each tensor contains the final
                  detections [x1, y1, x2, y2, conf] for one image after NMS.
                  Note: The current implementation processes the batch as one step
                  and returns results for the *last* image in the batch if B > 1
                  due to the list comprehension [boxes_conf[i] for i in range(B)]
                  in decode_preds and then taking results[0] at the end.
                  This predict function needs refinement for batch inference if B>1
                  sequences are independent. Let's assume B=1 for typical inference.
        """
        self.eval() # Set model to evaluation mode
        B, _, H_img, W_img = high_res_images.shape
        device = high_res_images.device

        # Perform forward pass in inference mode.
        # This updates and uses the internal history _hist_f.
        # activated_preds_list will contain predictions for the current frame(s).
        _, activated_preds_list = self(high_res_images, training=False) # List of [B, 5, H_f, W_f]

        batch_results = [] # List to store results for each image in the batch

        # Process results for each image in the batch
        for b in range(B):
            all_preds_for_image = [] # Collect predictions from all FPN levels for image 'b'

            # Decode predictions for image 'b' from each FPN level
            for preds_level in activated_preds_list: # preds_level: [B, 5, H_f, W_f]
                # decode_preds returns a list of [H_f*W_f, 5] tensors, one for each image in the batch
                decoded_level_b = decode_preds(preds_level, (H_img, W_img))[b] # [H_f*W_f, 5]
                all_preds_for_image.append(decoded_level_b)

            # Concatenate predictions from all levels for image 'b'
            preds_i = torch.cat(all_preds_for_image, dim=0) # [Total_predictions, 5]

            # If there are no predictions, add empty tensor and continue
            if preds_i.numel() == 0:
                batch_results.append(torch.empty((0, 5), device=device))
                continue

            # Separate boxes and scores
            boxes = preds_i[:, :4] # [Total_predictions, 4]
            scores = preds_i[:, 4] # [Total_predictions]

            # Apply confidence threshold
            keep = scores > conf_thresh
            boxes, scores = boxes[keep], scores[keep]

            # If no boxes remain after thresholding, add empty tensor and continue
            if boxes.numel() == 0:
                batch_results.append(torch.empty((0, 5), device=device))
                continue

            # Apply Non-Maximum Suppression (NMS)
            # nms returns indices of boxes to keep
            nms_indices = nms(boxes, scores, iou_thresh)

            # Filter boxes and scores using NMS indices and combine them
            filtered_preds = torch.cat([boxes[nms_indices], scores[nms_indices].unsqueeze(1)], dim=1) # [Num_after_NMS, 5]

            batch_results.append(filtered_preds)

        # print(batch_results[0])
        return batch_results[0] # Return list of [Num_after_NMS, 5] tensors, one tensor per image in batch

    def reset_memory(self):
        """Resets the internal ConvGRU hidden state for inference."""
        # print("Resetting DRIFT_NET memory state.")
        self._hist_f = [None] * 4 # Reset history for 4 FPN levels

