import torch
from torch import nn


class NONE(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        return img

    def forward_loss(self, haze_img, clean_img):
        return {
            'total_loss': torch.zeros(1, device=haze_img.device, requires_grad=True)
        }