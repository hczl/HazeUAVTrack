# LiteDetNet: Single-Class Minimal Object Detection Model (PyTorch)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ghost Convolution Block
class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GhostConv, self).__init__()
        self.primary_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.cheap_operation = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1, groups=out_channels // 2)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

# LiteFire Block (Highly simplified)
class LiteFireBlock(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(LiteFireBlock, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1 = GhostConv(squeeze_channels, expand_channels)

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        return self.expand1x1(x)

# Backbone
class LiteDetBackbone(nn.Module):
    def __init__(self):
        super(LiteDetBackbone, self).__init__()
        self.stage1 = GhostConv(3, 8)
        self.block1 = LiteFireBlock(8, 4, 8)
        self.block2 = LiteFireBlock(8, 4, 8)

    def forward(self, x):
        x = self.stage1(x)
        x = self.block1(x)
        x = self.block2(x)
        return x

# Neck (Simplified Feature Map Upscale)
class LitePAN(nn.Module):
    def __init__(self):
        super(LitePAN, self).__init__()
        self.conv = GhostConv(8, 8)

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x

# Head: Single-class detection head
class SingleClassHead(nn.Module):
    def __init__(self, in_channels):
        super(SingleClassHead, self).__init__()
        self.obj = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.reg = nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, x):
        obj_out = self.obj(x)  # objectness heatmap
        reg_out = self.reg(x)  # box regression
        return obj_out, reg_out

# Complete Model
class LITNET(nn.Module):
    def __init__(self):
        super(LITNET, self).__init__()
        self.backbone = LiteDetBackbone()
        self.neck = LitePAN()
        self.head = SingleClassHead(8)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        obj_out, reg_out = self.head(x)
        return obj_out, reg_out

# Example usage
if __name__ == "__main__":
    model = LITNET()
    dummy_input = torch.randn(1, 3, 224, 224)
    obj_out, reg_out = model(dummy_input)
    print(obj_out.shape, reg_out.shape)
