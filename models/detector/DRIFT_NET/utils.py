import torch
import yaml
from torchvision.ops import box_iou
import torch.nn.functional as F
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
    Converts raw predictions from [B, 5, H, W] to a list of [N_i, 5] tensors.
    Each tensor contains (x1, y1, x2, y2, conf) in original image coordinates.
    """
    B, C, H, W = preds.shape
    assert C == 5, "Predictions must have 5 channels (dx, dy, dw, dh, conf)"

    # img_size = (H_img, W_img), need stride = W_img / W, H_img / H
    stride_y = img_size[0] / H
    stride_x = img_size[1] / W

    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=preds.device),
                                    torch.arange(W, device=preds.device),
                                    indexing='ij')

    grid_x = grid_x.float().view(1, H, W).repeat(B, 1, 1)
    grid_y = grid_y.float().view(1, H, W).repeat(B, 1, 1)

    dx = preds[:, 0, :, :]
    dy = preds[:, 1, :, :]
    dw = preds[:, 2, :, :]
    dh = preds[:, 3, :, :]
    conf = preds[:, 4, :, :]

    # Compute center coordinates
    cx = (grid_x + dx) * stride_x
    cy = (grid_y + dy) * stride_y

    # Compute box width/height
    bw = dw * stride_x  # already exp-ed
    bh = dh * stride_y

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    boxes_conf = torch.stack([x1, y1, x2, y2, conf], dim=1)
    boxes_conf = boxes_conf.permute(0, 2, 3, 1).contiguous()
    boxes_conf = boxes_conf.view(B, H * W, 5)

    return [boxes_conf[i] for i in range(B)]


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='sum'):
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.where(targets == 1, inputs, 1 - inputs)
    loss = alpha * (1 - pt) ** gamma * BCE_loss
    return loss.sum() if reduction == 'sum' else loss.mean()
