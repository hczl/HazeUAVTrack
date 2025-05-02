import torch
import torch.nn.functional as F

def make_dark_channel_tensor(img_tensor, window_size=15):
    """
    Args:
        img_tensor (torch.Tensor): [B, 3, H, W], normalized (0~1), float32 or float64
        window_size (int): size of the patch for min pooling

    Returns:
        dark_channel (torch.Tensor): [B, 1, H, W], values in (0 ~ 1)
    """
    if not isinstance(img_tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if img_tensor.ndim != 4 or img_tensor.shape[1] != 3:
        raise ValueError("Input must have shape [B, 3, H, W]")

    # Per-pixel min across RGB channels => shape [B, 1, H, W]
    min_rgb, _ = img_tensor.min(dim=1, keepdim=True)

    # Use min-pooling to get local minimum over a patch (dark channel)
    pad = window_size // 2
    dark_channel = -F.max_pool2d(-min_rgb, kernel_size=window_size, stride=1, padding=pad)

    return dark_channel  # shape: [B, 1, H, W]