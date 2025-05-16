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
        ca_att = self.ca(x)
        sa_att = self.sa(x)
        att = ca_att * sa_att
        attended_x = x * att
        out = self.mix(attended_x)
        return out


class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        vgg = vgg16(weights='VGG16_Weights.DEFAULT').features[:10]
        self.vgg = nn.Sequential(*vgg).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def preprocess(self, x):
        x = (x - self.mean) / self.std
        return x

    def forward(self, pred, target):
        pred_vgg = self.vgg(self.preprocess(pred))
        target_vgg = self.vgg(self.preprocess(target))
        return F.l1_loss(pred_vgg, target_vgg)


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
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.encoder0 = self._make_stage(base_channels, base_channels, num_blocks[0], stride=1)
        self.encoder1 = self._make_stage(base_channels, base_channels * 2, num_blocks[1], stride=2)
        self.encoder2 = self._make_stage(base_channels * 2, base_channels * 4, num_blocks[2], stride=2)

        self.attentions = nn.ModuleList([
            EfficientCSAttention(base_channels),
            EfficientCSAttention(base_channels * 2),
            EfficientCSAttention(base_channels * 4)
        ])

        self.skip_convs = nn.ModuleList([
             nn.Conv2d(base_channels, base_channels // 2, 1, bias=False),
             nn.Conv2d(base_channels * 2, base_channels, 1, bias=False),
        ])

        self.decoder0 = self._make_decoder_upsample(base_channels * 4, base_channels * 2)
        self.decoder1 = self._make_decoder_upsample(base_channels * 2 + base_channels, base_channels)
        self.decoder2 = self._make_decoder_noup(base_channels + base_channels // 2, base_channels // 2)

        self.output = nn.Sequential(
            nn.Conv2d(base_channels // 2, out_channels, 3, padding=1),
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

        s0 = self.encoder0(x0)
        s0_att = self.attentions[0](s0)

        s1 = self.encoder1(s0_att)
        s1_att = self.attentions[1](s1)

        s2 = self.encoder2(s1_att)
        s2_att = self.attentions[2](s2)

        d0 = self.decoder0(s2_att)

        skip_s1 = self.skip_convs[1](s1_att)
        d1_input = torch.cat([d0, skip_s1], dim=1)
        d1 = self.decoder1(d1_input)

        skip_s0 = self.skip_convs[0](s0_att)
        d2_input = torch.cat([d1, skip_s0], dim=1)
        d2 = self.decoder2(d2_input)

        out = self.output(d2)

        return out


class AD_NET(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, num_blocks=[2, 2, 2], external_channels=[64, 128, 256]):
        super().__init__()

        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, external_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(external_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.enc1 = nn.Sequential(
            nn.Conv2d(external_channels[0], external_channels[1], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(external_channels[1]),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(external_channels[1], external_channels[2], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(external_channels[2]),
            nn.ReLU(inplace=True)
        )

        self.ad_net_core = AD_NET_Core(
            in_channels=external_channels[2],
            out_channels=external_channels[2],
            base_channels=base_channels,
            num_blocks=num_blocks
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(external_channels[2], external_channels[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(external_channels[1]),
            nn.ReLU(inplace=True)
        )
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(external_channels[1] + external_channels[1], external_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(external_channels[0]),
            nn.ReLU(inplace=True)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(external_channels[0], external_channels[0], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(external_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(external_channels[0] + external_channels[0], external_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(external_channels[0]),
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

        mse = F.mse_loss(dehaze, clean_img)
        perceptual = self.perceptual_loss_fn(dehaze, clean_img)
        tv = total_variation_loss(dehaze)

        mse_weight = 1.0
        perceptual_weight = 0.05
        tv_weight = 0.001

        total_loss = mse_weight * mse + perceptual_weight * perceptual + tv_weight * tv

        return {
            'mse_loss': mse,
            'perceptual_loss': perceptual,
            'tv_loss': tv,
            'total_loss': total_loss
        }
