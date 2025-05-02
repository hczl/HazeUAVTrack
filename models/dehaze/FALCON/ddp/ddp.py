import sys

import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfms

sys.path.append('')
from .dcp_orig import DarkChannel as darkchannel


class DDP(nn.Module):
    def __init__(self, kernel_size=3):
        super(DDP, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, I):
        I_inv = -I

        I_inv_transposed = I_inv.permute(0, 2, 1, 3).contiguous()
        dark_channel_inv_transposed = F.max_pool2d(I_inv_transposed, kernel_size=(I_inv.size(1), 1), stride=1, padding=0)
        dark_channel_inv = dark_channel_inv_transposed.permute(0, 2, 1, 3)

        eroded_dark_channel_inv = F.max_pool2d(dark_channel_inv, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)
        eroded_dark_channel = -eroded_dark_channel_inv

        return eroded_dark_channel


class DDP_input(nn.Module):
    def __init__(self, kernel_size=3):
        super(DDP_input, self).__init__()

    @torch.no_grad()
    def forward(self, I, k):
        I_inv = -I

        I_inv_transposed = I_inv.permute(0, 2, 1, 3).contiguous()
        dark_channel_inv_transposed = F.max_pool2d(I_inv_transposed, kernel_size=(I_inv.size(1), 1), stride=1, padding=0)
        dark_channel_inv = dark_channel_inv_transposed.permute(0, 2, 1, 3)

        eroded_dark_channel_inv = F.max_pool2d(dark_channel_inv, kernel_size=k, stride=1, padding=k//2)
        eroded_dark_channel = -eroded_dark_channel_inv

        return eroded_dark_channel
