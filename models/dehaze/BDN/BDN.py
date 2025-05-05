import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

from .utils import total_variation_loss, PerceptualLoss


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAMBlock(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out + x  # 残差连接


class BDN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        backbone = mobilenet_v3_small(pretrained=True).features
        self.encoder = nn.Sequential(*list(backbone.children())[:6])  # 使用前6层
        encoder_out_channels = list(backbone.children())[5].out_channels  # 第6层的输出通道数

        self.cbam = CBAMBlock(encoder_out_channels)

        # 透射图估计分支
        self.trans_branch = nn.Sequential(
            nn.Conv2d(encoder_out_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 大气光估计分支（与输入图像通道数匹配）
        self.atm_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out_channels, 16),
            nn.ReLU(),
            nn.Linear(16, in_channels),
            nn.Sigmoid()
        )

        self.PerceptualLoss = PerceptualLoss()

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.cbam(feat)

        t = self.trans_branch(feat)
        if t.shape[2:] != x.shape[2:]:
            t = F.interpolate(t, size=x.shape[2:], mode='bilinear', align_corners=False)

        A = self.atm_fc(feat).view(x.size(0), self.in_channels, 1, 1)

        # 去雾公式
        t = t.clamp(min=0.05)
        J = (x - A) / t + A
        return J.clamp(0, 1)

    def forward_loss(self, haze_img, clean_img):
        dehaze_img = self(haze_img)
        l1 = F.l1_loss(dehaze_img, clean_img)
        perceptual = self.PerceptualLoss(dehaze_img, clean_img)
        tv = total_variation_loss(dehaze_img)

        loss_dict = {
            'l1_loss': l1,
            'perceptual_loss': perceptual,
            'tv_loss': tv,
            'total_loss': l1 + 0.5 * perceptual + 0.1 * tv
        }
        return loss_dict
