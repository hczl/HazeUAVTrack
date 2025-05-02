from torch import nn


class NONE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
    def forward(self, img):
        return img
