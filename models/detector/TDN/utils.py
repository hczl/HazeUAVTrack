import torch
import yaml
from torchvision.ops import box_iou

def load_config(path):
    """
    Loads configuration from a YAML file.
    """
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def compute_iou(boxes1, boxes2):
    """
    Computes IoU between two sets of bounding boxes.
    boxes1, boxes2: [N, 4] and [M, 4] in format (x1, y1, x2, y2)
    returns IoU matrix of shape [N, M]
    """
    return box_iou(boxes1, boxes2)


def decode_preds(preds, img_size):
    """
    Converts raw predictions from [B, 5, H, W] to a list of [N_i, 5] tensors
    for each image in the batch, where N_i is H*W.
    Each tensor contains (x1, y1, x2, y2, conf) in original image coordinates.
    Keeps gradient tracking.
    """
    B, C, H, W = preds.shape
    assert C == 5, "Predictions must have 5 channels (dx, dy, dw, dh, conf)"

    stride_x = img_size[0] / W
    stride_y = img_size[1] / H

    # Create grid coordinates (j, i) for each cell in the feature map
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=preds.device),
                                    torch.arange(W, device=preds.device),
                                    indexing='ij')

    # Add batch dimension and repeat for broadcasting
    grid_x = grid_x.float().view(1, H, W).repeat(B, 1, 1)
    grid_y = grid_y.float().view(1, H, W).repeat(B, 1, 1)

    # Extract prediction components
    dx = preds[:, 0, :, :] # (B, H, W)
    dy = preds[:, 1, :, :] # (B, H, W)
    dw = preds[:, 2, :, :] # (B, H, W)
    dh = preds[:, 3, :, :] # (B, H, W)
    conf = preds[:, 4, :, :] # (B, H, W)

    # Calculate center coordinates in original image scale
    cx = (grid_x + dx) * stride_x # (B, H, W)
    cy = (grid_y + dy) * stride_y # (B, H, W)

    # Calculate width and height in original image scale
    # Assuming dw, dh are relative scales to image size as in original code
    bw = dw * img_size[0] # (B, H, W)
    bh = dh * img_size[1] # (B, H, W)

    # Calculate box corners (x1, y1, x2, y2)
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # Stack box coordinates and confidence
    # Result shape (B, 5, H, W) -> Permute to (B, H, W, 5) -> Reshape to (B, H*W, 5)
    boxes_conf = torch.stack([x1, y1, x2, y2, conf], dim=1)
    boxes_conf = boxes_conf.permute(0, 2, 3, 1).contiguous()
    boxes_conf = boxes_conf.view(B, H * W, 5) # (B, N_preds_per_image, 5)

    # Split the batch dimension into a list of tensors
    results_list = [boxes_conf[i] for i in range(B)]

    return results_list

