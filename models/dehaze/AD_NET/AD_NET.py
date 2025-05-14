import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class ChannelAttention(nn.Module):

    def __init__(self, in_channels, ratio=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(self.gap(x)))


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        return self.sigmoid(self.conv(x))


class LightweightConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, expansion_ratio=2):
        super().__init__()
        hidden = int(in_channels * expansion_ratio)
        self.use_res = stride == 1 and in_channels == out_channels

        layers = [
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        if self.use_res:
             out = out + identity
        return out


class EfficientCSAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.mix = nn.Conv2d(channels, channels, 1)


    def forward(self, x):
        att = self.ca(x) * self.sa(x)
        return self.mix(x * att)


class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        vgg = vgg16(weights='VGG16_Weights.DEFAULT').features[:10]
        self.vgg = nn.Sequential(*vgg).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        return F.l1_loss(self.vgg(pred), self.vgg(target))


def total_variation_loss(image):
    if image.size(2) < 2 or image.size(3) < 2:
        return torch.tensor(0.0, device=image.device)

    diff_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    diff_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    return (diff_h.mean() + diff_w.mean())


class AD_NET_Core(nn.Module):

    def __init__(self, in_channels, out_channels, base_channels=32, num_blocks=[2, 2, 2]):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.encoder = nn.ModuleList([
            self._make_stage(base_channels, base_channels, num_blocks[0], stride=2),
            self._make_stage(base_channels, base_channels * 2, num_blocks[1], stride=2),
            self._make_stage(base_channels * 2, base_channels * 4, num_blocks[2], stride=1)
        ])

        self.attentions = nn.ModuleList([
            EfficientCSAttention(base_channels),
            EfficientCSAttention(base_channels * 2),
            EfficientCSAttention(base_channels * 4)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(base_channels, base_channels // 2, 1),
            nn.Conv2d(base_channels * 2, base_channels, 1),
            nn.Conv2d(base_channels * 4, base_channels * 2, 1)
        ])

        self.adjust_s2_conv = nn.Conv2d(base_channels, 256, kernel_size=1)

        self.decoder0 = self._make_decoder_upsample(base_channels * 4, base_channels * 2)
        self.decoder1 = self._make_decoder_upsample(base_channels * 2 + 256, base_channels)
        self.decoder2 = self._make_decoder_noup(base_channels + base_channels // 2, base_channels // 2)

        self.output = nn.Sequential(
            nn.Conv2d(base_channels // 2, out_channels, 3, padding=1),
            # Sigmoid is added in the outer U-Net
        )


    def _make_stage(self, in_c, out_c, num_blocks, stride):
        layers = [LightweightConvBlock(in_c, out_c, stride)]
        for _ in range(1, num_blocks):
            layers.append(LightweightConvBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def _make_decoder_upsample(self, in_c, out_c):
         return nn.Sequential(
             nn.Conv2d(in_c, in_c * 4, 3, padding=1, bias=False),
             nn.BatchNorm2d(in_c * 4),
             nn.ReLU6(inplace=True),
             nn.PixelShuffle(2),
             LightweightConvBlock(in_c, out_c)
         )

    def _make_decoder_noup(self, in_c, out_c):
        return nn.Sequential(
             LightweightConvBlock(in_c, out_c, stride=1)
        )


    def forward(self, x):
        x0 = self.init_conv(x)

        s1 = self.encoder[0](x0)
        s1 = self.attentions[0](s1)

        s2 = self.encoder[1](s1)
        s2 = self.attentions[1](s2)

        s3 = self.encoder[2](s2)
        s3 = self.attentions[2](s3)

        d3 = self.decoder0(s3)

        skip_s2 = self.skip_convs[1](s2)
        skip_s2_adjusted = self.adjust_s2_conv(skip_s2)
        upsampled_skip_s2_adjusted = F.interpolate(skip_s2_adjusted, size=d3.shape[-2:], mode='nearest')

        d3_input_to_decoder1 = torch.cat([d3, upsampled_skip_s2_adjusted], dim=1)

        d2 = self.decoder1(d3_input_to_decoder1)

        skip_s1 = self.skip_convs[0](s1)
        upsampled_skip_s1 = F.interpolate(skip_s1, size=d2.shape[-2:], mode='nearest')

        d2_input_to_decoder2 = torch.cat([d2, upsampled_skip_s1], dim=1)

        d1 = self.decoder2(d2_input_to_decoder2)

        return self.output(d1)


class AD_NET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, num_blocks=[2, 2, 2], external_channels=[64, 128, 256]):
        super().__init__()

        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, external_channels[0], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.enc1 = nn.Sequential(
            nn.Conv2d(external_channels[0], external_channels[1], 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(external_channels[1], external_channels[2], 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.ad_net_core = AD_NET_Core(
            in_channels=external_channels[2],
            out_channels=external_channels[2],
            base_channels=base_channels, # Use the provided base_channels for the core
            num_blocks=num_blocks
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(external_channels[2], external_channels[1], kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.dec1_conv = nn.Sequential(
            nn.Conv2d(external_channels[1] + external_channels[1], external_channels[0], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(external_channels[0], external_channels[0], kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.dec2_conv = nn.Sequential(
            nn.Conv2d(external_channels[0] + external_channels[0], external_channels[0], 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(external_channels[0], out_channels, 1)
        self.sigmoid = nn.Sigmoid()

        self.perceptual_loss_fn = PerceptualLoss()

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)

        ad_out = self.ad_net_core(e2)

        d1 = self.dec1(ad_out)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1_conv(d1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e0], dim=1)
        d2 = self.dec2_conv(d2)

        out = self.final_conv(d2)
        out = self.sigmoid(out)

        return out

    def forward_loss(self, haze_img, clean_img):
        dehaze = self(haze_img)

        l1 = F.l1_loss(dehaze, clean_img)
        perceptual = self.perceptual_loss_fn(dehaze, clean_img)
        tv = total_variation_loss(dehaze)

        return {
            'l1_loss': l1,
            'perceptual_loss': perceptual,
            'tv_loss': tv,
            'total_loss': l1 + 0.5 * perceptual + 0.1 * tv
        }
