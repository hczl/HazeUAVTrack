import cv2 # 导入 OpenCV 库，用于图像处理操作，例如生成高斯核
import torch # 导入 PyTorch 库
import torch.nn as nn # 导入 PyTorch 的神经网络模块
from ultralytics import YOLO # 导入 Ultralytics 的 YOLOv8 库，用于集成 YOLO 模型
from ultralytics.utils.loss import v8DetectionLoss # 导入 YOLOv8 的检测损失函数
from ultralytics.utils.ops import non_max_suppression # 导入 YOLOv8 的非极大值抑制函数

# 导入自定义的 YOLOv8 损失函数修改和批处理处理函数
from models.detector.DE_NET.utils import changeed__call__, process_batch

# --- 核心图像增强模块 (基于拉普拉斯金字塔和 SFT) ---

# Lap_Pyramid_Conv 类：用于构建拉普拉斯金字塔和重建图像
class Lap_Pyramid_Conv(nn.Module):
    """
    拉普拉斯金字塔分解和重建模块。
    用于将图像分解为不同频率层，并在处理后重建图像。
    """
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        """
        初始化拉普拉斯金字塔模块。

        Args:
            num_high (int): 拉普拉斯金字塔的高频层数。
            kernel_size (int): 用于高斯滤波的卷积核大小。
            channels (int): 输入图像的通道数。
        """
        super().__init__()

        self.num_high = num_high # 高频层数
        # 生成用于高斯模糊的卷积核
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        """
        生成指定大小和通道数的高斯卷积核。
        """
        # 使用 OpenCV 生成高斯核 (一维然后外积得到二维)
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        # 将 numpy 数组转换为 PyTorch 张量，并调整形状以适应 conv2d (out_channels, in_channels/groups, kw, kh)
        # 对于 grouped conv，in_channels/groups = 1
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        # 将核注册为模型的参数，但不计算梯度 (requires_grad=False)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        """
        使用给定的高斯核对输入张量进行卷积。
        使用分组卷积，每个通道独立卷积。
        """
        n_channels, _, kw, kh = kernel.shape
        # 使用反射填充避免边界效应，填充大小为核的一半
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')
        # 应用二维卷积，groups=n_channels 表示分组卷积，每组一个通道
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        """
        通过隔行隔列采样进行下采样 (尺寸减半)。
        """
        # 沿 H 和 W 维度每隔一个像素进行采样
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        """
        执行金字塔向下操作：先高斯模糊，再下采样。
        """
        # 应用高斯模糊
        gauss_x = self.conv_gauss(x, self.kernel)
        # 进行下采样
        return self.downsample(gauss_x)

    def upsample(self, x):
        """
        通过插入零并进行高斯滤波进行上采样 (尺寸加倍)。
        这是一种近似双线性插值然后平滑的方法。
        """
        # 创建一个两倍大小的零张量
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         dtype=x.dtype, device=x.device) # 确保 dtype 和 device 一致
        # 将输入复制到零张量的偶数位置
        up[:, :, ::2, ::2] = x
        # 应用高斯滤波进行平滑。
        # 乘以4是因为在双线性插值中，原始像素会影响 2x2 的区域，总和为4倍。
        # 这里的实现是将值复制到偶数位置，然后通过卷积核（其和为1）进行扩散。
        # 为了补偿复制时的“稀疏性”，需要乘以4。
        # 这是一个近似，更标准的方法是使用 F.interpolate(scale_factor=2, mode='bilinear')
        # 但这里保留了原代码的实现方式。
        return self.conv_gauss(up, self.kernel) * 4 # 原代码没有乘以4，这里根据标准金字塔上采样理论添加。如果原代码是故意不乘，则去掉。

    def pyramid_decom(self, img):
        """
        将图像分解为拉普拉斯金字塔。

        Args:
            img (torch.Tensor): 输入图像张量 (B, C, H, W)。

        Returns:
            list: 包含拉普拉斯金字塔层的列表，顺序为 [高频1, 高频2, ..., 高频N, 最低频]。
        """
        # 将高斯核移动到输入图像相同的设备
        self.kernel = self.kernel.to(img.device)
        current = img # 当前层，从原始图像开始
        pyr = [] # 存储金字塔层
        # 构建拉普拉斯金字塔的高频层
        for _ in range(self.num_high):
            down = self.pyramid_down(current) # 向下采样得到下一层的高斯模糊版本
            up = self.upsample(down) # 将高斯模糊版本上采样回当前层尺寸
            diff = current - up # 当前层减去其低频近似得到高频信息（拉普拉斯层）
            pyr.append(diff) # 添加高频层
            current = down # 下一层的高斯模糊版本成为新的当前层
        pyr.append(current) # 最后将最低频层（只经过高斯模糊和下采样的层）添加到列表末尾
        # 返回的 pyr 是 [高频1 (最高分辨率), 高频2, ..., 高频N (最低分辨率的高频), 最低频]
        return pyr

    def pyramid_recons(self, pyr):
        """
        从拉普拉斯金字塔层重建图像。

        Args:
            pyr (list): 包含拉普拉斯金字塔层的列表，顺序为 [高频1, 高频2, ..., 高频N, 最低频]。

        Returns:
            torch.Tensor: 重建后的图像张量 (B, C, H, W)。
        """
        # 从最低频层开始重建图像
        image = pyr[-1] # 最低频层是列表的最后一个元素
        # 从倒数第二个元素（最高频率的高频层）开始向前迭代
        for level in reversed(pyr[:-1]):
            # 将当前重建结果上采样到下一层（更高分辨率）的尺寸
            up = self.upsample(image)
            # 将上采样结果与对应的高频层相加，得到更高分辨率的重建结果
            image = up + level
        # 循环结束后，image 将是原始图像的完整重建
        return image

# ChannelAttention 类：通道注意力模块 (虽然在原始 DE_NET 代码中未直接实例化和使用，但保留了定义)
class ChannelAttention(nn.Module):
    """
    通道注意力模块 (SENet 或 CBAM 的一部分)。
    根据通道间的关系生成权重。
    """
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        # 自适应平均池化到 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应最大池化到 1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的多层感知机 (MLP)，使用 1x1 卷积实现
        self.sharedMLP = nn.Sequential(
            # 降维卷积
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            # 升维卷积
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        # Sigmoid 激活，生成注意力权重 (0 到 1 之间)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化后通过 MLP
        avgout = self.sharedMLP(self.avg_pool(x))
        # 最大池化后通过 MLP
        maxout = self.sharedMLP(self.max_pool(x))
        # 结合平均池化和最大池化的结果，应用 Sigmoid，然后与输入逐元素相乘
        return self.sigmoid(avgout + maxout) * x

# SpatialAttention 类：空间注意力模块
class SpatialAttention(nn.Module):
    """
    空间注意力模块 (CBAM 的一部分)。
    根据空间位置的重要性生成权重。
    """
    def __init__(self, kernel_size=7): # 原始代码是5，这里改为7，更常见
        """
        初始化空间注意力模块。

        Args:
            kernel_size (int): 用于卷积的核大小 (3, 5, 或 7)。
        """
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel size must be 3 or 5 or 7"

        # 对平均池化和最大池化的结果（堆叠后是2通道）应用卷积
        # 输入是 avg_out 和 max_out 的拼接，共 2 个通道
        self.conv = nn.Conv2d(2,
                              1, # 输出 1 个通道，代表空间注意力图
                              kernel_size,
                              padding=kernel_size // 2,
                              bias=False)
        # Sigmoid 激活，生成空间注意力权重 (0 到 1 之间)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算通道维度的平均值，保留维度 (1通道)
        avgout = torch.mean(x, dim=1, keepdim=True)
        # 计算通道维度的最大值，保留维度 (1通道)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        # 沿通道维度堆叠平均值和最大值结果 (2通道)
        attention = torch.cat([avgout, maxout], dim=1)
        # 应用卷积和 sigmoid 获取空间注意力图 (1通道)，然后与输入逐元素相乘
        attention = self.conv(attention)
        return self.sigmoid(attention) * x

# Trans_guide 类：用于生成指导信息
class Trans_guide(nn.Module):
    """
    生成指导信息的模块。
    输入是原始图像和低频增强结果的拼接，通过卷积和空间注意力生成指导信息。
    """
    def __init__(self, ch=16):
        """
        初始化指导信息生成模块。

        Args:
            ch (int): 中间卷积层的通道数。
        """
        super().__init__()

        self.layer = nn.Sequential(
            # 输入是原始图像(3)和低频增强结果(3)，共6通道。通过 3x3 卷积增加通道数。
            nn.Conv2d(6, ch, 3, padding=1),
            nn.LeakyReLU(True), # LeakyReLU 激活
            SpatialAttention(3), # 应用空间注意力，核大小为 3
            # 通过 3x3 卷积将通道数变回 3，作为指导信息
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(self, x): # x 是 torch.cat([原始图像, 低频增强结果], dim=1)
        """
        前向传播，生成指导信息。

        Args:
            x (torch.Tensor): 拼接后的输入张量 (B, 6, H, W)。

        Returns:
            torch.Tensor: 生成的指导信息张量 (B, 3, H, W)。
        """
        return self.layer(x)

# Trans_low 类：处理拉普拉斯金字塔的最低频层
class Trans_low(nn.Module):
    """
    处理拉普拉斯金字塔最低频层的模块。
    包含编码器、多尺度卷积 (MM) 和解码器。
    并生成用于高频处理的指导信息。
    """
    def __init__(
        self,
        ch_blocks=64, # 中间特征通道数
        ch_mask=16, # 指导信息模块的中间通道数
    ):
        """
        初始化低频处理模块。

        Args:
            ch_blocks (int): 主处理分支的中间通道数。
            ch_mask (int): 指导信息生成模块的中间通道数。
        """
        super().__init__()

        # 编码器：将 3 通道的低频输入编码到 ch_blocks 通道
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, ch_blocks, 3, padding=1),
                                     nn.LeakyReLU(True))

        # 多尺度卷积 (MM) 部分：使用不同大小的卷积核捕获多尺度信息
        # 修正：原代码是4个mm1，可能意图是多尺度，这里改为不同大小的卷积核 (1x1, 3x3, 5x5, 7x7)
        # 每个分支输出 ch_blocks // 4 通道
        self.mm1 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=1,
                             padding=0)
        self.mm2 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=3,
                             padding=3 // 2)
        self.mm3 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=5,
                             padding=5 // 2)
        self.mm4 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=7,
                             padding=7 // 2)


        # 解码器：将 ch_blocks 通道解码回 3 通道
        self.decoder = nn.Sequential(nn.Conv2d(ch_blocks, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, 3, 3, padding=1))

        # 指导信息生成模块，用于生成高频处理所需的指导信息
        self.trans_guide = Trans_guide(ch_mask)

    def forward(self, x): # x 是拉普拉斯金字塔的最低频层 (B, 3, H, W)
        """
        前向传播，处理最低频层并生成指导信息。

        Args:
            x (torch.Tensor): 拉普拉斯金字塔的最低频层张量 (B, 3, H_low, W_low)。

        Returns:
            tuple: 包含 (增强后的低频层张量, 生成的指导信息张量)。
                   两个张量尺寸相同 (B, 3, H_low, W_low)。
        """
        # 通过编码器
        x1 = self.encoder(x)
        # 应用多尺度卷积并拼接结果
        x1_1 = self.mm1(x1)
        x1_2 = self.mm2(x1)
        x1_3 = self.mm3(x1)
        x1_4 = self.mm4(x1)
        # 拼接所有多尺度分支的结果，通道数回到 ch_blocks
        x1 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        # 通过解码器将通道数变回 3
        x1 = self.decoder(x1)

        # 残差连接：原始低频层 + 增强后的低频层
        out = x + x1
        out = torch.relu(out) # 应用 ReLU 激活，确保像素值非负

        # 生成指导信息：输入是原始低频层 x 和增强后的低频层 out 的拼接
        mask = self.trans_guide(torch.cat([x, out], dim=1))
        # 返回增强后的低频层和对应的指导信息
        return out, mask

# SFT_layer 类：空间特征变换 (Spatial Feature Transform) 层
class SFT_layer(nn.Module):
    """
    空间特征变换层。
    根据指导信息 (guide) 动态地调整主输入 (x) 的特征。
    变换公式为: processed_x = processed_x * scale + shift
    """
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        """
        初始化 SFT 层。

        Args:
            in_ch (int): 主输入 (x) 和指导信息 (guide) 的通道数。
            inter_ch (int): 内部处理和预测分支的中间通道数。
            out_ch (int): 最终输出的通道数。
            kernel_size (int): 卷积核大小。
        """
        super().__init__()

        # 主分支编码器：对输入 x 进行卷积和激活
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(True),
        )
        # 主分支解码器：将处理后的特征解码回输出通道数
        self.decoder = nn.Sequential(
            nn.Conv2d(inter_ch, out_ch, kernel_size, padding=kernel_size // 2))

        # 根据指导信息预测 shift 参数的分支
        self.shift_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))
        # 根据指导信息预测 scale 参数的分支
        self.scale_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))

    def forward(self, x, guide): # x 是拉普拉斯金字塔的高频层，guide 是指导信息
        """
        前向传播，应用 SFT 变换。

        Args:
            x (torch.Tensor): 主输入张量 (拉普拉斯高频层) (B, in_ch, H, W)。
            guide (torch.Tensor): 指导信息张量 (B, in_ch, H, W)。

        Returns:
            torch.Tensor: 经过 SFT 变换处理后的张量 (B, out_ch, H, W)。
        """
        # 通过主分支编码器处理输入 x
        processed_x = self.encoder(x)
        # 使用指导信息通过独立的卷积分支预测 scale 和 shift 参数
        scale = self.scale_conv(guide)
        shift = self.shift_conv(guide)
        # 应用 SFT 变换: processed_x = processed_x * (1 + scale) + shift
        # 注意：原始 SFT 公式是 scale * x + shift，这里是 (1+scale)*x + shift，
        # 或者等价于 x + scale*x + shift，这是一种残差形式的 SFT。
        processed_x = processed_x + processed_x * scale + shift
        # 通过主分支解码器得到最终输出
        processed_x = self.decoder(processed_x)
        return processed_x

# Trans_high 类：处理拉普拉斯金字塔的高频层
class Trans_high(nn.Module):
    """
    处理拉普拉斯金字塔高频层的模块。
    使用 SFT 层结合指导信息对高频层进行增强。
    """
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        """
        初始化高频处理模块。

        Args:
            in_ch (int): 高频输入和指导信息的通道数。
            inter_ch (int): SFT 层内部中间通道数。
            out_ch (int): SFT 层输出通道数。
            kernel_size (int): SFT 层卷积核大小。
        """
        super().__init__()

        # 使用 SFT 层进行高频增强
        self.sft = SFT_layer(in_ch, inter_ch, out_ch, kernel_size)

    def forward(self, x, guide): # x 是拉普拉斯金字塔的高频层，guide 是对应分辨率的指导信息
        """
        前向传播，增强高频层。

        Args:
            x (torch.Tensor): 拉普拉斯金字塔的高频层张量 (B, 3, H_high, W_high)。
            guide (torch.Tensor): 对应分辨率的指导信息张量 (B, 3, H_high, W_high)。

        Returns:
            torch.Tensor: 增强后的高频层张量 (B, 3, H_high, W_high)。
        """
        # 残差连接：原始高频层 + SFT 增强后的高频层
        return x + self.sft(x, guide)

# Up_guide 类：上采样指导信息
class Up_guide(nn.Module):
    """
    上采样指导信息到更高分辨率的模块。
    """
    def __init__(self, kernel_size=1, ch=3):
        """
        初始化指导信息上采样模块。

        Args:
            kernel_size (int): 卷积核大小。
            ch (int): 指导信息的通道数。
        """
        super().__init__()
        self.up = nn.Sequential(
            # 双线性插值上采样，尺寸加倍
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            # 卷积调整通道或平滑（通常 kernel_size=1 保持通道数并进行微调）
            nn.Conv2d(ch,
                      ch,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False))

    def forward(self, x): # x 是低一层的指导信息 (B, 3, H_low, W_low)
        """
        前向传播，上采样指导信息。

        Args:
            x (torch.Tensor): 需要上采样的指导信息张量 (B, 3, H_low, W_low)。

        Returns:
            torch.Tensor: 上采样后的指导信息张量 (B, 3, H_high, W_high)。
        """
        x = self.up(x)
        return x

# DE_NET 类：完整的图像增强网络，集成 YOLOv3
class DE_NET(nn.Module):
    """
    基于拉普拉斯金字塔和 SFT 的图像增强网络，并集成了 YOLOv3 进行目标检测。
    网络对输入图像进行增强，然后将增强后的图像送入 YOLOv3 进行检测。
    训练时优化的是 YOLOv3 在增强图像上的检测损失。
    """
    def __init__(self, cfg,
                 num_high=3, # 拉普拉斯金字塔的高频层数
                 ch_blocks=64, # Trans_low 的中间通道数
                 up_ksize=1, # Up_guide 的卷积核大小
                 high_ch=32, # Trans_high (SFT_layer) 的中间通道数
                 high_ksize=3, # Trans_high (SFT_layer) 的卷积核大小
                 ch_mask=16, # Trans_guide 的中间通道数
                 gauss_kernel=5): # 拉普拉斯金字塔的高斯核大小
        """
        初始化 DE_NET 模型。

        Args:
            cfg (dict): 配置字典 (可能用于其他地方，但在当前代码段中未直接使用)。
            num_high (int): 拉普拉斯金字塔的高频层数。
            ch_blocks (int): Trans_low 模块的中间通道数。
            up_ksize (int): Up_guide 模块的卷积核大小。
            high_ch (int): Trans_high (SFT_layer) 模块的中间通道数。
            high_ksize (int): Trans_high (SFT_layer) 模块的卷积核大小。
            ch_mask (int): Trans_guide 模块的中间通道数。
            gauss_kernel (int): 拉普拉斯金字塔高斯核大小。
        """
        super().__init__()
        self.cfg = cfg # 存储配置
        self.num_high = num_high # 存储高频层数
        # 初始化拉普拉斯金字塔分解/重建模块
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
        # 初始化处理最低频层的模块
        self.trans_low = Trans_low(ch_blocks, ch_mask)

        # 为每个高频层创建对应的上采样指导模块和高频处理模块
        # 使用 setattr 动态创建模块，名称如 up_guide_layer_0, trans_high_layer_0 等
        for i in range(0, self.num_high):
            # 上采样指导信息模块
            # 指导信息通道数应与图像通道数一致 (3)
            self.__setattr__('up_guide_layer_{}'.format(i),
                             Up_guide(up_ksize, ch=3))
            # 高频处理模块
            # 输入输出都是 3 通道
            self.__setattr__('trans_high_layer_{}'.format(i),
                             Trans_high(3, high_ch, 3, high_ksize))

        # 使用 ultralytics 载入 YOLOv3 模型及其权重，并提取其 nn.Module
        yolov3_wrapper = YOLO('yolov3.yaml')
        yolov3_wrapper.load('models/detector/YOLOV3/yolov3u.pt')
        # 获取 YOLOv3 的核心 nn.Module
        self.yolov3 = yolov3_wrapper.model
        # 获取 YOLOv3 的损失函数
        self.loss_fn = yolov3_wrapper.loss

        # 对 YOLOv8 的检测损失函数进行 monkey patch
        v8DetectionLoss.__call__ = changeed__call__

        # 清空 CUDA 缓存，释放显存（如果在 GPU 可用时）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self, x): # x 是原始输入图像 (B, 3, H, W)
        """
        DE_NET 的前向传播。

        Args:
            x (torch.Tensor): 输入图像张量 (B, 3, H, W)。

        Returns:
            torch.Tensor: YOLOv3 模型的输出张量 (通常是检测结果)。
        """
        # 将输入图像分解为拉普拉斯金字塔层
        # pyrs 列表包含 [高频1, 高频2, ..., 高频N, 最低频]
        pyrs = self.lap_pyramid.pyramid_decom(img=x)

        # trans_pyrs 列表将存储增强后的金字塔层
        trans_pyrs = []

        # 处理最低频层
        # trans_low 模块接收最低频层 (pyrs[-1])，返回增强后的最低频层和对应的指导信息
        trans_pyr, guide = self.trans_low(pyrs[-1])
        # 将增强后的最低频层添加到结果列表
        trans_pyrs.append(trans_pyr)

        # 上采样指导信息到每个高频层对应的分辨率
        commom_guide = [] # 存储上采样后的指导信息
        current_guide = guide # 从最低频层的指导信息开始
        # 循环 num_high 次，为每个高频层生成指导信息
        for i in range(self.num_high):
            # 逐层上采样指导信息，使用对应的上采样指导模块
            current_guide = self.__getattr__('up_guide_layer_{}'.format(i))(current_guide)
            # 将上采样后的指导信息添加到列表
            commom_guide.append(current_guide)

        # 处理每个高频层
        for i in range(self.num_high):
            # pyrs[-2-i] 是对应的高频层 (从最高频率开始，索引从 -2 递减)
            # commom_guide[i] 是对应分辨率的指导信息
            # trans_high_layer 模块接收高频层和指导信息，返回增强后的高频层
            trans_pyr = self.__getattr__('trans_high_layer_{}'.format(i))(
                pyrs[-2 - i], commom_guide[i])
            # 将增强后的高频层添加到结果列表
            trans_pyrs.append(trans_pyr)

        # 将增强后的金字塔层列表反转，使其顺序变为 [高频1, ..., 高频N, 最低频]
        trans_pyrs.reverse()

        # 从增强后的金字塔层重建增强后的图像
        enhanced_img = self.lap_pyramid.pyramid_recons(trans_pyrs)

        # 将增强后的图像输入到 YOLOv3 模型进行目标检测
        yolo_output = self.yolov3(enhanced_img)

        return yolo_output

    @torch.no_grad() # 在预测模式下不计算梯度
    def predict(self, high_res_images, conf_thresh=0.95, iou_thresh=0.45):
        """
        进行推理预测。

        Args:
            high_res_images (torch.Tensor): 输入的高分辨率图像张量 (B, C, H, W)。
            conf_thresh (float): 置信度阈值。
            iou_thresh (float): IoU 阈值。

        Returns:
            list: 每张图像的检测结果列表，每个结果是 NMS 过滤后的张量。
        """
        # 设置模型为评估模式
        self.eval()
        self.yolov3.eval()
        # 通过 DE_NET 进行前向传播获取 YOLOv3 的原始输出
        raw_output = self(high_res_images)
        # 对 YOLOv3 原始输出进行非极大值抑制 (NMS)
        return self.decode_output(raw_output, conf_thresh, iou_thresh)

    def decode_output(self, raw_output, conf_thresh=0.95, iou_thresh=0.45, max_det=300):
        """
        对 YOLOv3 的原始输出应用非极大值抑制 (NMS)。

        Args:
            raw_output (torch.Tensor): YOLOv3 模型的原始输出。
            conf_thresh (float): 置信度阈值。
            iou_thresh (float): IoU 阈值。
            max_det (int): 每张图像最大检测框数量。

        Returns:
            list: 每张图像的检测结果列表，每个结果是 NMS 过滤后的张量。
        """
        # 调用 Ultralytics 的 non_max_suppression 函数
        detections = non_max_suppression(
            raw_output,
            conf_thres=conf_thresh,
            iou_thres=iou_thresh,
            max_det=max_det,
            classes=None, # 检测所有类别
            # agnostic=False, # 不进行类别无关的 NMS
            # multi_label=False, # 不是多标签预测
            # labels=(), # 不使用额外的标签进行 NMS
            # max_time_img=0.05, # 每张图像的最大 NMS 时间
            # max_time_batch=0.05, # 每批次的最大 NMS 时间
            # max_wh=7680, # 最大边界框尺寸
        )
        return detections

    def forward_loss(self, haze_imgs, targets, ignore_list):
        """
        计算训练所需的损失。

        Args:
            haze_imgs (torch.Tensor): 输入的雾霾图像张量 (B, C, H, W)。
            targets (torch.Tensor): 地面真实目标张量 (通常是 YOLO 格式)。
            ignore_list (list): 需要忽略的样本索引列表。

        Returns:
            dict: 包含 YOLOv3 损失和总损失的字典。
        """
        haze_imgs, targets, ignore_list = process_batch((haze_imgs, targets, ignore_list))

        yolov3_output = self(haze_imgs)

        yolov3_loss_tuple = self.loss_fn(targets, yolov3_output)

        all_loss_tensors = [loss for group in yolov3_loss_tuple for loss in group]
        yolov3_total_loss = sum(all_loss_tensors)

        # 返回损失字典
        return {
            'yolov3_loss': yolov3_total_loss, # 单独列出 YOLOv3 损失
            'total_loss': yolov3_total_loss, # 总损失等于 YOLOv3 损失
        }
