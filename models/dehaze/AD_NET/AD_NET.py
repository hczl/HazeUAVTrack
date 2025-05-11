import torch
import torch.nn as nn
import torch.nn.functional as F
# 考虑使用 torchvision 中实际的 MobileNetV3 块或自行实现
# from torchvision.models import mobilenet_v3_small # 这里没有直接使用，但思路相似
import numpy as np # 保留以备潜在的实用功能

# 假设这些模块位于相对于此脚本的 'utils.py' 文件中
from .utils import total_variation_loss, PerceptualLoss

# --- 辅助模块 (为了清晰起见，定义在主类外部，但在主类内部使用) ---

# 重用通道注意力模块和空间注意力模块 (来自 CBAM) - 它们很轻量
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        # 自适应平均池化到 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应最大池化到 1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享的多层感知机 (MLP)，使用 1x1 卷积实现
        self.fc = nn.Sequential(
            # 降维卷积
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            # 升维卷积
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        # Sigmoid 激活，生成注意力权重 (0 到 1 之间)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化后通过 MLP
        avg_out = self.fc(self.avg_pool(x))
        # 最大池化后通过 MLP
        max_out = self.fc(self.max_pool(x))
        # 结合平均池化和最大池化的结果
        out = avg_out + max_out
        # 应用 Sigmoid 得到通道注意力权重
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用较小的卷积核 (例如 5x5) 以提高潜在的速度，或者保留 7x7
        # 输入通道是平均池化和最大池化拼接后的结果 (2个通道)
        self.conv = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False) # 改为 5x5
        # Sigmoid 激活，生成空间注意力权重 (0 到 1 之间)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上计算平均值，保留维度
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 在通道维度上计算最大值，保留维度
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均值和最大值结果沿通道维度拼接
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # 通过卷积和 Sigmoid 得到空间注意力权重
        return self.sigmoid(self.conv(x_cat))

# 轻量级卷积块 (受 MobileNetV3 的倒残差结构启发)
class LightweightConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expansion_factor=3):
        super().__init__()
        # 隐藏层通道数，是输入通道数的 expansion_factor 倍
        hidden_channels = in_channels * expansion_factor
        # 判断是否使用残差连接：只有当步长为 1 且输入通道等于输出通道时才使用
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        # 逐点扩展 (1x1 卷积) - 增加通道数
        layers.append(nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))

        # 深度可分离卷积 (kxkx) - 在每个通道上独立进行卷积
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_channels, bias=False))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))

        # 逐点投影 (1x1 卷积) - 减少通道数到输出通道数
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.conv = nn.Sequential(*layers)

    # LightweightConvBlock 类内部
    def forward(self, x):
        out = self.conv(x)
        # 检查是否打算使用残差连接 AND 空间维度是否匹配
        # self.use_res_connect 在 __init__ 中已根据初始步长和通道数设置
        # 确保输出和输入的 H, W 尺寸相同才能进行残差连接
        if self.use_res_connect and x.shape[2:] == out.shape[2:]:
            return x + out
        else:
            return out


# 高效通道-空间注意力块 (自定义，轻量级)
class EfficientCSAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        # 通道注意力模块
        self.channel_att = ChannelAttention(in_channels, ratio)
        # 空间注意力模块
        self.spatial_att = SpatialAttention()
        # 可选：在注意力后添加一个 1x1 卷积用于特征混合
        self.conv_mix = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 依次应用通道注意力和空间注意力
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)

        # 在注意力后混合特征
        out = self.conv_mix(out)
        # 添加残差连接
        return out + x


# --- 新的 AD_NET 模型 (快速且端到端) ---

class AD_NET(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, num_blocks=[2, 3, 3, 2]):
        """
        一个快速、端到端的去雾网络，使用轻量级 CNN 块和高效注意力。
        它不依赖于大气散射模型公式。

        Args:
            in_channels (int): 输入图像的通道数 (RGB 为 3)。
            base_channels (int): 第一个编码器阶段的基础通道数。
                                 后续阶段的通道数将是它的倍数。
                                 值越低模型越快。
            num_blocks (list): 整数列表，指定每个编码器/瓶颈阶段中 LightweightConvBlock 的数量。
        """
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks

        # 初始卷积，增加通道数
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # --- 编码器阶段 ---
        # 使用带有步长的 LightweightConvBlock 进行下采样
        # 在每个阶段后添加 EfficientCSAttention 进行特征细化
        # H/2 尺寸
        self.encoder_stage1 = self._make_layer(base_channels, base_channels, num_blocks[0], stride=2)
        self.attn1 = EfficientCSAttention(base_channels)

        # H/4 尺寸
        self.encoder_stage2 = self._make_layer(base_channels, base_channels * 2, num_blocks[1], stride=2)
        self.attn2 = EfficientCSAttention(base_channels * 2)

        # H/8 尺寸
        self.encoder_stage3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks[2], stride=2)
        self.attn3 = EfficientCSAttention(base_channels * 4)

        # 瓶颈层 (可以只包含几个额外的块或注意力)
        # H/16 尺寸
        self.bottleneck = self._make_layer(base_channels * 4, base_channels * 8, num_blocks[3], stride=2)
        self.attn4 = EfficientCSAttention(base_channels * 8)


        # --- 解码器阶段 ---
        # 使用 ConvTranspose2d 进行上采样并与跳跃连接结合
        # 在特征结合后添加 EfficientCSAttention
        # 从 H/16 上采样到 H/8
        self.decoder_stage3 = self._make_decoder_layer(base_channels * 8, base_channels * 4, base_channels * 4)
        self.attn5 = EfficientCSAttention(base_channels * 4)

        # 从 H/8 上采样到 H/4
        self.decoder_stage2 = self._make_decoder_layer(base_channels * 4, base_channels * 2, base_channels * 2)
        self.attn6 = EfficientCSAttention(base_channels * 2)

        # 从 H/4 上采样到 H/2
        self.decoder_stage1 = self._make_decoder_layer(base_channels * 2, base_channels, base_channels)
        self.attn7 = EfficientCSAttention(base_channels)


        # 最终输出层
        self.output_conv = nn.Sequential(
            # H/2 上采样到 H
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # 输出 3 个通道
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            # Sigmoid 激活，将输出值限制在 0 到 1 之间
            nn.Sigmoid()
        )

        # --- 损失函数 (可以重用原始 AD_NET 的) ---
        # 假设 PerceptualLoss 在 .utils 中已定义
        self.PerceptualLoss = PerceptualLoss()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """ 辅助函数，用于创建编码器/瓶颈阶段 """
        layers = []
        # 第一个块处理潜在的步长和通道变化
        layers.append(LightweightConvBlock(in_channels, out_channels, stride=stride))
        # 剩余的块步长为 1，且通道数不变
        for _ in range(1, num_blocks):
            layers.append(LightweightConvBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _make_decoder_layer(self, in_channels, skip_channels, out_channels):
        """ 辅助函数，用于创建解码器阶段 """
        # in_channels 来自前一个解码器阶段
        # skip_channels 来自编码器的跳跃连接
        # out_channels 是当前解码器阶段的输出通道数
        return nn.Sequential(
            # 上采样
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # 与跳跃连接结合 - 需要处理通道维度
            # 下一个层将以 (out_channels + skip_channels) 作为输入
            # 结合并细化
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        """
        去雾网络的前向传播。

        Args:
            x (torch.Tensor): 输入的雾霾图像张量 (B, C, H, W)，范围 [0, 1]。

        Returns:
            torch.Tensor: 去雾后的图像张量 (B, C, H, W)，范围 [0, 1]。
        """
        # x 是雾霾图像 (B, C, H, W) - 假设已经归一化到 [0, 1]

        # 初始卷积 (尺寸 H)
        x0 = self.initial_conv(x)

        # 编码器
        # 阶段 1 (尺寸 H/2)
        enc1 = self.encoder_stage1(x0)
        enc1_att = self.attn1(enc1)

        # 阶段 2 (尺寸 H/4)
        enc2 = self.encoder_stage2(enc1_att)
        enc2_att = self.attn2(enc2)

        # 阶段 3 (尺寸 H/8)
        enc3 = self.encoder_stage3(enc2_att)
        enc3_att = self.attn3(enc3)

        # 瓶颈层 (尺寸 H/16)
        enc4 = self.bottleneck(enc3_att)
        enc4_att = self.attn4(enc4)

        # 解码器
        # 上采样 enc4_att，并准备与 enc3_att 拼接
        # 从 H/16 上采样到 H/8
        dec3_up = self.decoder_stage3[0](enc4_att)
        # 确保拼接前尺寸匹配
        # 如果输入 H,W 是偶数，ConvTranspose2d 应该能处理。
        # 如果不是，插值可能更安全，或者在 ConvTranspose2d 中处理 padding
        # 这里我们使用 interpolate 作为鲁棒的备选方案
        if dec3_up.shape[2:] != enc3_att.shape[2:]:
             dec3_up = F.interpolate(dec3_up, size=enc3_att.shape[2:], mode='bilinear', align_corners=False)

        # 与跳跃连接结合，并应用剩余的解码器层 + 注意力
        # 跳跃连接
        dec3_combined = torch.cat([dec3_up, enc3_att], dim=1)
        # 应用剩余的卷积层
        dec3 = self.decoder_stage3[1:](dec3_combined)
        # 应用注意力
        dec3_att = self.attn5(dec3)


        # 上采样 dec3_att，并准备与 enc2_att 拼接
        # 从 H/8 上采样到 H/4
        dec2_up = self.decoder_stage2[0](dec3_att)
        if dec2_up.shape[2:] != enc2_att.shape[2:]:
             dec2_up = F.interpolate(dec2_up, size=enc2_att.shape[2:], mode='bilinear', align_corners=False)
        # 跳跃连接
        dec2_combined = torch.cat([dec2_up, enc2_att], dim=1)
        # 应用剩余的卷积层
        dec2 = self.decoder_stage2[1:](dec2_combined)
        # 应用注意力
        dec2_att = self.attn6(dec2)

        # 上采样 dec2_att，并准备与 enc1_att 拼接
        # 从 H/4 上采样到 H/2
        dec1_up = self.decoder_stage1[0](dec2_att)
        if dec1_up.shape[2:] != enc1_att.shape[2:]:
             dec1_up = F.interpolate(dec1_up, size=enc1_att.shape[2:], mode='bilinear', align_corners=False)
        # 跳跃连接
        dec1_combined = torch.cat([dec1_up, enc1_att], dim=1)
        # 应用剩余的卷积层
        dec1 = self.decoder_stage1[1:](dec1_combined)
        # 应用注意力
        dec1_att = self.attn7(dec1)

        # 最终输出
        # 从 H/2 上采样到 H
        dehaze_img = self.output_conv(dec1_att)

        # 确保输出在有效范围 [0, 1]
        return dehaze_img.clamp(0, 1)

    def forward_loss(self, haze_img, clean_img):
        """
        计算训练所需的损失。

        Args:
            haze_img (torch.Tensor): 输入的雾霾图像张量 (B, C, H, W)，范围 [0, 1]。
            clean_img (torch.Tensor): 地面真实无雾图像张量 (B, C, H, W)，范围 [0, 1]。

        Returns:
            dict: 包含不同损失分量和总损失的字典。
        """
        # 获取去雾后的输出
        dehaze_img = self(haze_img)

        # 计算损失 (与原始 AD_NET 相同)
        l1 = F.l1_loss(dehaze_img, clean_img)
        perceptual = self.PerceptualLoss(dehaze_img, clean_img)
        tv = total_variation_loss(dehaze_img)

        # 根据需要调整损失权重
        # 示例权重：L1 是主要的，Perceptual 帮助改善视觉质量，TV 使输出平滑
        loss_dict = {
            'l1_loss': l1,
            'perceptual_loss': perceptual,
            'tv_loss': tv,
            # 示例权重
            'total_loss': l1 + 0.5 * perceptual + 0.1 * tv
        }
        return loss_dict
