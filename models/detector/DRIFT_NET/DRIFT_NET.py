from collections import deque # 用于双端队列，可能用于管理序列或历史状态

import numpy as np # 数值计算库
# import timm # 未在当前代码段中使用

import torch # PyTorch 深度学习框架
import torch.nn as nn # PyTorch 神经网络模块
import torch.nn.functional as F # PyTorch 函数式接口
# from torch import ops # 未在当前代码段中使用
from torchvision.models import convnext_tiny, mobilenet_v3_large, resnet50 # 导入 torchvision 中的骨干网络模型
from torchvision.ops import nms # 导入非极大值抑制函数
from torchvision.models import resnet18, ResNet18_Weights # 导入 ResNet18 模型和权重

import yaml # 用于加载 YAML 配置文件
from torchvision.ops import box_iou # 用于计算边界框的 IoU
import torch.nn.functional as F # 再次导入 F，可能为了清晰或兼容性

def load_config(path):
    """加载 YAML 配置文件"""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def compute_iou(boxes1, boxes2):
    """计算两个边界框集合的 IoU"""
    # 确保框是浮点类型且在同一设备上
    boxes1 = boxes1.float().to(boxes2.device)
    boxes2 = boxes2.float()
    # 优雅地处理空张量
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.empty((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=torch.float32)
    return box_iou(boxes1, boxes2)

def focal_loss(inputs, targets, alpha=0.5, gamma=1.0, reduction='sum'):
    """
    计算 Focal Loss。

    Args:
        inputs (Tensor): 浮点张量，预测的概率值。
        targets (Tensor): 浮点张量，与 inputs 形状相同，二值目标 (0 或 1)。
        alpha (float): 平衡正负样本的权重因子 (0, 1)。
        gamma (float): 调制因子 (1 - p_t) 的指数，平衡易分/难分样本。
        reduction (string): 'none' | 'mean' | 'sum'
            'none': 不应用降维。
            'mean': 求平均。
            'sum': 求和。
    Returns:
        Tensor: 损失值，标量或与 inputs 形状相同，取决于 reduction 模式。
    """
    # inputs 假设是 sigmoid(raw_conf)，已经是概率值
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

    # p_t 是目标类别的概率
    pt = torch.where(targets == 1, inputs, 1 - inputs)

    # 添加一个小的 epsilon 防止 log(0)
    pt = torch.clamp(pt, min=1e-6, max=1.0 - 1e-6)

    loss = alpha * (1 - pt) ** gamma * BCE_loss

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        # 这里求平均是针对所有非忽略样本
        return loss.mean()
    else:
        return loss


def decode_preds(preds, img_size):
    """
    将模型输出的预测值（激活后）解码为边界框 (x1, y1, x2, y2) 和置信度。
    这个函数通常用于推理阶段，处理一个批次的预测结果。

    Args:
        preds (Tensor): 模型某个 FPN 级别的激活后预测值 [B, 5, H_f, W_f]。
                       通道顺序为 [dx, dy, dw, dh, conf]。
        img_size (tuple): 输入图像的尺寸 (H_img, W_img)。

    Returns:
        list: 一个列表，每个元素是一个张量 [H_f*W_f, 5]，包含该图像在该级别解码后的框和置信度。
    """
    B, C, H, W = preds.shape
    assert C == 5, f"Expected input channels 5, but got {C}"

    stride_y = img_size[0] / H
    stride_x = img_size[1] / W

    # 生成网格坐标
    grid_y = torch.arange(H, device=preds.device).view(1, H, 1, 1).repeat(B, 1, W, 1)
    grid_x = torch.arange(W, device=preds.device).view(1, 1, W, 1).repeat(B, H, 1, 1)

    # 从激活后的预测值中分离 dx, dy, dw, dh, conf
    # 这些值已经经过 sigmoid 或 exp 激活
    # Clamp dx, dy 到 [0, 1] 以防浮点精度问题
    dx = torch.clamp(preds[:, 0, :, :].unsqueeze(-1), 0., 1.)
    dy = torch.clamp(preds[:, 1, :, :].unsqueeze(-1), 0., 1.)
    dw = preds[:, 2, :, :].unsqueeze(-1) # exp(raw_dw)
    dh = preds[:, 3, :, :].unsqueeze(-1) # exp(raw_dh)
    conf = preds[:, 4, :, :].unsqueeze(-1) # sigmoid(raw_conf)

    # 计算中心坐标 (cx, cy) 相对图像左上角
    # dx, dy 是网格单元内的分数偏移 (0-1)
    cx = (grid_x.float() + dx) * stride_x
    cy = (grid_y.float() + dy) * stride_y

    # 计算框的宽度和高度 (bw, bh)
    # dw, dh 是应用到 stride 的比例因子
    bw = dw * stride_x
    bh = dh * stride_y

    # 将中心/宽度/高度转换为 x1, y1, x2, y2
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # 堆叠并重塑
    boxes_conf = torch.cat([x1, y1, x2, y2, conf], dim=-1) # [B, H, W, 5]
    boxes_conf = boxes_conf.view(B, H * W, 5) # [B, H*W, 5]

    # 为批次中的每张图像返回一个列表
    return [boxes_conf[i] for i in range(B)]


class LiteFPN(nn.Module):
    """轻量级特征金字塔网络 (LiteFPN)"""
    def __init__(self, in_channels_list, out_channels, num_levels=4):
        super(LiteFPN, self).__init__()
        self.out_channels = out_channels
        self.num_levels = num_levels

        # 1x1 卷积用于减少通道数
        self.reduce_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])

        # 每个特征图的学习型位置编码
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(1, out_channels, 1, 1)) for _ in range(num_levels)
        ])

    def forward(self, features):
        # features 是来自骨干网络的张量列表，例如 [C2, C3, C4, C5]
        reduced = [conv(f) for conv, f in zip(self.reduce_convs, features)] # [P2_reduced, P3_reduced, P4_reduced, P5_reduced]

        # 获取最小特征图 (P5) 的尺寸
        target_size = reduced[-1].shape[-2:]

        # 将所有特征图下采样/调整到最小尺寸 (P5 尺寸)
        # 使用 adaptive_avg_pool2d 进行下采样
        resized = [F.adaptive_avg_pool2d(f, target_size) for f in reduced]

        # 求和融合特征图
        fused = torch.stack(resized, dim=0).sum(dim=0) # [B, out_channels, H_p5, W_p5]

        # 将融合后的特征图上采样回原始 reduced 尺寸，并添加位置编码
        results = []
        for i in range(self.num_levels):
            # 将融合特征图上采样到第 i 个 reduced 特征图的尺寸
            upsampled = F.interpolate(fused, size=reduced[i].shape[-2:], mode='nearest') # 使用最近邻插值
            results.append(upsampled + self.pos_embeds[i]) # 添加位置编码

        # results 是张量列表 [P2', P3', P4', P5']
        return results


class ResNet50LiteFPNBackbone(nn.Module):
    """基于 ResNet50 和 LiteFPN 的骨干网络"""
    def __init__(self, out_channels=128, pretrained=True):
        super().__init__()
        # 加载 ResNet50 骨干网络
        resnet = resnet50(pretrained=pretrained)

        # 提取 ResNet 的不同层作为特征输出
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool, resnet.layer1)  # 输出步长 4, 通道 256 (C2)
        self.layer2 = resnet.layer2  # 输出步长 8, 通道 512 (C3)
        self.layer3 = resnet.layer3  # 输出步长 16, 通道 1024 (C4)
        self.layer4 = resnet.layer4  # 输出步长 32, 通道 2048 (C5)

        # FPN 接收 [C2, C3, C4, C5]
        self.fpn = LiteFPN(in_channels_list=[256, 512, 1024, 2048], out_channels=out_channels)

    def forward(self, x):
        # 前向传播通过 ResNet 层
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # FPN 返回特征列表 [P2', P3', P4', P5']
        return self.fpn([c2, c3, c4, c5])




class DeformableAttention(nn.Module):
    """可变形注意力模块"""
    def __init__(self, in_channels, num_heads=4, num_points=4):
        super().__init__()
        self.num_heads = num_heads # 注意力头数
        self.num_points = num_points # 每个头每个位置的采样点数
        self.in_channels = in_channels # 输入通道数
        # 每个头的通道数
        self.channels_per_head = in_channels // num_heads
        assert self.channels_per_head * num_heads == in_channels, "in_channels 必须能被 num_heads 整除"

        # 用于预测偏移、注意力权重和值的卷积层
        self.offset_proj = nn.Conv2d(in_channels, num_heads * num_points * 2, kernel_size=1) # 2D 偏移 (x, y)
        self.attn_weight_proj = nn.Conv2d(in_channels, num_heads * num_points, kernel_size=1) # 每个点的权重
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1) # 投影到值空间
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1) # 输出投影

        # 初始化偏移预测层，使其初始预测偏移为零
        nn.init.constant_(self.offset_proj.weight, 0.)
        nn.init.constant_(self.offset_proj.bias, 0.)

        # 初始化注意力权重预测层，使其初始权重接近 1/num_points
        nn.init.constant_(self.attn_weight_proj.weight, 0.)
        nn.init.constant_(self.attn_weight_proj.bias, np.log(1.0 / self.num_points))

        # 初始化值和输出投影层
        nn.init.kaiming_uniform_(self.value_proj.weight, a=1)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.kaiming_uniform_(self.output_proj.weight, a=1)
        nn.init.constant_(self.output_proj.bias, 0.)


    # 移除 training=False 参数，因为在这个模块中不使用
    def forward(self, x):
        # x: 输入张量 [B, C, H, W] - 这是 Attention 层要处理的特征图
        B, C, H, W = x.size()
        # 确保输入通道数匹配
        assert C == self.in_channels, f"Input channels {C} must match layer channels {self.in_channels}"

        # 投影获取值、偏移和注意力权重
        value = self.value_proj(x) # [B, C, H, W]
        offsets = self.offset_proj(x) # [B, num_heads * num_points * 2, H, W]
        attn_weights = self.attn_weight_proj(x) # [B, num_heads * num_points, H, W]

        # 重塑并对注意力权重应用 softmax
        attn_weights = attn_weights.view(B, self.num_heads, self.num_points, H, W) # [B, num_heads, num_points, H, W]
        attn_weights = F.softmax(attn_weights, dim=2) # 在每个位置每个头的所有采样点上应用 Softmax

        # 准备采样用的基础网格 (归一化坐标 [-1, 1])
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        # 堆叠网格坐标并重塑用于广播
        # base_grid shape: [1, 1, 1, H, W, 2] (B, num_heads, num_points, H, W, coords)
        base_grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, 1, H, W, 2)

        # 重塑偏移并加到基础网格上以获取采样位置
        # offsets shape: [B, num_heads * num_points * 2, H, W]
        # 重塑 offsets 到 [B, num_heads, num_points, 2, H, W]
        offset = offsets.view(B, self.num_heads, self.num_points, 2, H, W)
        # 调整维度顺序到 [B, num_heads, num_points, H, W, 2] 以匹配 base_grid
        offset = offset.permute(0, 1, 2, 4, 5, 3)
        # 将偏移加到基础网格 (发生广播)
        sampling_grid = base_grid + offset # [B, num_heads, num_points, H, W, 2]

        # 重塑 sampling_grid 用于 F.grid_sample
        # F.grid_sample 期望 [N, H_out, W_out, 2]
        # 这里 N = B * num_heads * num_points
        sampling_grid = sampling_grid.view(B * self.num_heads * self.num_points, H, W, 2)

        # 准备 value 用于 F.grid_sample
        # value shape: [B, C, H, W]
        # 重塑 value 到 [B, num_heads, channels_per_head, H, W]
        value_for_sample = value.view(B, self.num_heads, self.channels_per_head, H, W)
        # 为每个采样点复制 value: [B, num_heads, 1, channels_per_head, H, W] -> [B, num_heads, num_points, channels_per_head, H, W]
        value_for_sample = value_for_sample.unsqueeze(2).expand(-1, -1, self.num_points, -1, -1, -1)
        # 重塑用于 F.grid_sample: [B * num_heads * num_points, channels_per_head, H, W]
        value_for_sample = value_for_sample.reshape(B * self.num_heads * self.num_points, self.channels_per_head, H, W)

        # 使用 grid_sample 进行采样
        # sampled shape: [B * num_heads * num_points, channels_per_head, H, W]
        # sampling_grid 坐标是归一化的 [-1, 1]
        sampled = F.grid_sample(value_for_sample, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # 将 sampled 重塑回 [B, num_heads, num_points, channels_per_head, H, W]
        sampled = sampled.view(B, self.num_heads, self.num_points, self.channels_per_head, H, W)

        # 应用注意力权重: [B, num_heads, num_points, 1, H, W] * [B, num_heads, num_points, channels_per_head, H, W]
        attn_weights = attn_weights.unsqueeze(3) # 添加通道维度用于广播
        weighted = (sampled * attn_weights).sum(dim=2) # 在采样点维度求和: [B, num_heads, channels_per_head, H, W]

        # 合并注意力头: [B, num_heads * channels_per_head, H, W] = [B, in_channels, H, W]
        weighted = weighted.view(B, self.in_channels, H, W)

        # 输出投影并添加残差连接
        return self.output_proj(weighted) + x # 返回处理后的特征图


class ConvGRUCell(nn.Module):
    """卷积门控循环单元 (ConvGRUCell)"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.channels = channels
        # 输入 + 上一时刻隐藏状态 -> 2 * channels 输入
        self.conv_zr = nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding)
        # 输入 + (reset * previous hidden) -> 2 * channels 输入
        self.conv_h = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        # x: 当前输入张量 [B, C, H, W]
        # h_prev: 上一时刻隐藏状态 [B, C, H, W]

        # 沿通道维度连接输入和上一时刻隐藏状态
        combined = torch.cat([x, h_prev], dim=1) # [B, 2C, H, W]

        # 计算更新门 (z) 和重置门 (r)
        zr = self.conv_zr(combined) # [B, 2C, H, W]
        z, r = torch.split(zr, self.channels, dim=1) # z: [B, C, H, W], r: [B, C, H, W]
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        # 计算候选隐藏状态 (h_hat)
        combined_reset = torch.cat([x, r * h_prev], dim=1) # [B, 2C, H, W]
        h_hat = torch.tanh(self.conv_h(combined_reset)) # [B, C, H, W]

        # 计算新的隐藏状态 (h)
        h = (1 - z) * h_prev + z * h_hat # [B, C, H, W]
        return h

class MultiFPNConvGRU(nn.Module):
    """用于多 FPN 级别的 ConvGRU 模块"""
    def __init__(self, channels=256, kernel_size=3):
        super().__init__()
        # 假设 FPN 输出 4 个级别 (P2', P3', P4', P5')
        self.fpn_count = 4
        # 为每个 FPN 级别创建一个单独的 ConvGRUCell
        self.gru_cells = nn.ModuleList([
            ConvGRUCell(channels, kernel_size) for _ in range(self.fpn_count)
        ])

    def forward(self, hist_states, current_tensor_list):
        # hist_states: 张量列表 [B, C, H, W]，表示每个级别的历史状态，或在 t=0 时为 None
        # current_tensor_list: 张量列表 [B, C, H, W]，表示每个级别来自 FPN/Attention 的当前特征
        updated_hist_states = []
        for i in range(self.fpn_count):
            # 获取第 i 个级别的上一时刻状态
            h_prev = hist_states[i]
            # 获取第 i 个级别的当前输入
            x_current = current_tensor_list[i]

            # 如果是第一步 (h_prev 为 None)，则用零初始化
            if h_prev is None:
                 h_prev = torch.zeros_like(x_current)

            # 使用该级别的 GRU 单元计算新的隐藏状态
            h_new = self.gru_cells[i](x_current, h_prev)
            updated_hist_states.append(h_new)

        # 返回所有级别更新后的历史状态列表
        return updated_hist_states


class DetectionHead(nn.Module):
    """检测头，用于从特征图中预测边界框和置信度"""
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        # 检测头为每个网格单元每个类别预测 5 个值: dx, dy, dw, dh, conf
        # 对于 num_classes=1，总共是 5 个通道。
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes * 5, kernel_size=1)
        )

        # 初始化置信度预测层的偏置，以实现稳定的训练起始
        # 这鼓励模型在开始时预测较低的置信度
        pi = 0.1 # 初始正样本概率
        bias_value = -torch.log(torch.tensor((1 - pi) / pi)) # 计算对应 logits

        # 只对输出层中置信度部分的偏置应用初始化
        with torch.no_grad():
            # 假设输出通道顺序为 [dx1, dy1, dw1, dh1, conf1, dx2, dy2, ...]
            # 对于 num_classes=1，通道顺序为 [dx, dy, dw, dh, conf]
            if num_classes > 0:
                 # 置信度偏置位于索引 4, 9, 14, ...
                 self.head[-1].bias[4::5].fill_(bias_value)

    def forward(self, feats):
        # feats 是来自 FPN 的特征图列表 [P2', P3', P4', P5']
        raw_preds_list = [] # 原始输出列表 (conf 的 logits)
        activated_preds_list = [] # 激活后输出列表 (dx, dy, conf 的 sigmoid; dw, dh 的 exp)

        for x in feats: # 处理每个特征图级别
            raw_preds = self.head(x) # [B, num_classes * 5, H, W]
            B, C, H, W = raw_preds.shape
            assert C == self.num_classes * 5, f"Expected {self.num_classes * 5} channels, but got {C} for input shape {raw_preds.shape}"

            # 将原始预测值重塑为 [B, num_classes, 5, H, W]
            raw_preds_reshaped = raw_preds.view(B, self.num_classes, 5, H, W)

            # 应用激活函数
            # dx, dy: sigmoid 限制偏移在 (0, 1) 范围内，相对于单元格左上角
            # dw, dh: exp 用于预测尺度，通常是原始值没有界限
            # conf: sigmoid 用于置信度分数 (0, 1)
            dx_dy = torch.sigmoid(raw_preds_reshaped[:, :, 0:2, :, :]) # [B, num_classes, 2, H, W]
            # Clamp exp(dw), exp(dh) 防止数值问题和过大的框
            # 限制原始 log 值到 [-6, 6] -> scale 在 [e^-6, e^6] 大约 [0.0025, 403]
            dw_dh = torch.exp(torch.clamp(raw_preds_reshaped[:, :, 2:4, :, :], min=-6, max=6)) # [B, num_classes, 2, H, W]
            conf = torch.sigmoid(raw_preds_reshaped[:, :, 4:5, :, :]) # [B, num_classes, 1, H, W]

            # 连接激活后的预测值
            activated_preds_reshaped = torch.cat([dx_dy, dw_dh, conf], dim=2) # [B, num_classes, 5, H, W]
            # 重塑回 [B, num_classes * 5, H, W]，与 raw_preds_list 保持一致
            activated_preds = activated_preds_reshaped.view(B, self.num_classes * 5, H, W)

            raw_preds_list.append(raw_preds)
            activated_preds_list.append(activated_preds)

        # 返回每个 FPN 级别的原始和激活后预测列表
        return raw_preds_list, activated_preds_list

class DRIFT_NET(nn.Module):
    """DRIFT-NET 模型，结合空间特征、可变形注意力和 ConvGRU 进行时序检测"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 用于标记预测框为 '忽略' 的 IoU 阈值
        self.ignore_iou_threshold = 0.3 # 常用值，可调整

        # 标志，是否使用 MSE 损失计算置信度，而不是 Focal Loss (默认 False)
        self._use_mse = False
        # 标志，是否处于评估模式 (用于控制 forward_loss 的行为)
        self._evaluating = False
        # 加载模型特定的配置
        model_cfg_dict = {}
        try:
             # 假设配置文件路径是正确的
             model_cfg_dict = load_config('models/detector/DRIFT_NET/DRIFT_NET_config.yaml')['model']
        except FileNotFoundError:
             print("Warning: DRIFT_NET_config.yaml not found. Using default model parameters.")
             model_cfg_dict = {
                 'backbone_channel': 192,
                 'step': 4
             }
        except Exception as e:
             print(f"Warning: Error loading DRIFT_NET_config.yaml: {e}. Using default model parameters.")
             model_cfg_dict = {
                 'backbone_channel': 192,
                 'step': 4
             }

        # 训练时一起处理的时间步长
        self.step = model_cfg_dict.get('step', 4)
        # FPN 和后续层的输出通道数
        self.backbone_channel = model_cfg_dict.get('backbone_channel', 192)

        # 骨干网络 + LiteFPN
        self.backbone = ResNet50LiteFPNBackbone(out_channels=self.backbone_channel, pretrained=True)

        # 每个 FPN 级别的可变形注意力层
        # 假设 4 个 FPN 级别 (P2', P3', P4', P5')
        self.attn = nn.ModuleList([DeformableAttention(in_channels=self.backbone_channel) for _ in range(4)])

        # ConvGRU 用于跨 FPN 级别的时序特征融合
        self.ConvGRU = MultiFPNConvGRU(channels=self.backbone_channel)

        # 应用于 ConvGRU 输出的检测头，每个级别一个
        # 假设检测 1 个类别
        self.detection = DetectionHead(in_channels=self.backbone_channel, num_classes=1)

        # 用于推理时存储时序记忆的内部状态
        # 初始化为 None，由 ConvGRU 更新
        # 列表大小为 4，对应 4 个 FPN 级别
        self._hist_f = [None] * 4

    def forward(self, x, training=False):
        # x: 输入张量 [B, C, H, W]

        # 使用骨干网络和 FPN 提取空间特征
        # f_t: 当前时间步的特征列表 [P2', P3', P4', P5']
        f_t = self.backbone(x) # List of [B, C, H_f, W_f]

        # 对空间特征应用可变形注意力
        attn_f = [self.attn[i](x_i) for i, x_i in enumerate(f_t)] # List of [B, C, H_f, W_f]

        # FPN 级别数
        FPN_levels = len(attn_f)
        B, C = attn_f[0].shape[:2] # 获取当前批次大小和通道数

        if training:
            # 训练模式下，处理长度为 'step' 的序列
            # 批次大小 B 必须是 self.step 的倍数
            assert B % self.step == 0, f"Batch size ({B}) must be a multiple of self.step ({self.step}) during training."

            num_sequences = B // self.step # 批次中包含的序列数量
            # 使用原始批次大小 B 作为重塑参考
            B_original = B

            # 将 attn_f 中的每个特征图从 [B_original, C, H, W] 重塑为 [num_sequences, step, C, H, W]
            # 这为按步长处理每个序列准备数据
            attn_f_reshaped_sequences = [
                f_level.view(num_sequences, self.step, C, f_level.shape[2], f_level.shape[3])
                for f_level in attn_f
            ] # List of [num_sequences, step, C, H_f, W_f]

            # 初始化每个 FPN 级别的历史状态。
            # history_states 将是一个张量列表，每个 FPN 级别一个。
            # 每个张量形状为 [num_sequences, C, H_f, W_f]，用于跟踪该级别所有序列的历史。初始为 None。
            history_states = [None] * FPN_levels

            # 按步长处理 (t=0 到 step-1)
            for t in range(self.step):
                # 获取当前时间步 't' 在所有序列中的特征
                # 这是一个包含 [num_sequences, C, H_f, W_f] 张量的列表，每个 FPN 级别一个
                current_frame_features_t = [
                    f_level_reshaped[:, t, :, :, :] # 从该级别的所有序列中选择第 t 帧
                    for f_level_reshaped in attn_f_reshaped_sequences
                ] # List of [num_sequences, C, H_f, W_f]

                # 应用 ConvGRU。
                # ConvGRU 接收:
                # 1. hist_states: 张量列表 [num_sequences, C, H_f, W_f] (每个级别的上一时刻状态，跨序列)
                # 2. current_tensor_list: 张量列表 [num_sequences, C, H_f, W_f] (每个级别当前的输入，跨序列)
                # 它返回一个更新后的历史张量列表 [num_sequences, C, H_f, W_f]。
                history_states = self.ConvGRU(history_states, current_frame_features_t)

            # 循环结束后，history_states 包含每个序列最后一个时间步 (t = step - 1) 的最终隐藏状态，每个 FPN 级别一个。
            # history_states 是一个包含 [num_sequences, C, H_f, W_f] 张量的列表。

            # 对这些最终状态应用检测头。
            # 检测头期望一个特征图列表 [B', C, H_f, W_f]。
            # 这里，B' 是 num_sequences (处理的独立序列数量)。
            raw_preds_list, activated_preds_list = self.detection(history_states)

        else: # 推理模式
            # 一次处理一帧，维护历史状态 _hist_f
            # attn_f 已经是 [B, C, H_f, W_f] 列表，其中 B 是批次大小 (通常为 1)
            # 如果 B > 1，它们被作为一个批次处理，共享相同的历史初始化。
            # 这对于批次中独立的序列可能不是理想的。
            # 对于典型的视频推理 (B=1)，这是正确的。
            if self._hist_f[0] is None or self._hist_f[0].shape[0] != B:
                 # 如果批次大小改变或这是第一帧，初始化历史状态
                 # 注意: 这会为批次维度中的每个项目独立初始化历史。
                 # 如果 B > 1 代表独立序列，这没问题。
                 # 如果 B > 1 代表连续帧，这个逻辑可能需要调整以正确维护跨批次维度的历史，或处理 B=1。
                 # 假设 B=1 是主要的推理情况，或 B>1 是独立序列。
                 self._hist_f = [torch.zeros_like(f) for f in attn_f]

            # 应用 ConvGRU，更新 _hist_f
            # ConvGRU 接收 List[Tensor] 作为 hist_states 和 current_tensor_list
            # 在推理中，GRU 的批次大小是 B。
            self._hist_f = self.ConvGRU(self._hist_f, attn_f) # _hist_f 现在是 List of [B, C, H_f, W_f]

            # 对更新后的历史状态应用检测头
            raw_preds_list, activated_preds_list = self.detection(self._hist_f)

        # 返回原始和激活后的预测
        # 在训练中，这是批次中每个序列最后一帧的预测 (批次大小 B // step)
        # 在推理中，这是当前帧的预测 (批次大小 B)
        return raw_preds_list, activated_preds_list

    def _compute_single_loss(self, raw_preds_flat, activated_preds_flat, targets, ignore_list, img_size, feature_map_size):
        """
        使用基于网格的匹配策略，计算单张图片在单 FPN 级别的损失。

        Args:
            raw_preds_flat (Tensor): 展平的原始预测值 (conf 的 logits, raw dx, dy, dw, dh) [H*W, 5]。
            activated_preds_flat (Tensor): 展平的激活后预测值 (sigmoid dx, dy, exp dw, dh, sigmoid conf) [H*W, 5]。
            targets (list): 该图片的 GT 标注列表 [class_id, track_id, x, y, w, h]。
            ignore_list (list): 该图片的忽略区域标注列表 [class_id, track_id, x, y, w, h]。
            img_size (tuple): (H_img, W_img)。
            feature_map_size (tuple): (H_f, W_f)。

        Returns:
            dict: 该级别/图片的损失详情 {'conf_loss', 'bbox_loss', 'num_matched', 'num_non_ignored'}。
        """
        H_f, W_f = feature_map_size
        img_H, img_W = img_size
        stride_y = img_H / H_f
        stride_x = img_W / W_f
        device = raw_preds_flat.device

        N_preds = raw_preds_flat.shape[0] # 预测数量 (H_f * W_f)

        # 创建网格坐标 (用于 bbox 目标编码和正样本匹配)
        grid_y_coords, grid_x_coords = torch.meshgrid(torch.arange(H_f, device=device),
                                                      torch.arange(W_f, device=device),
                                                      indexing='ij')
        # 展平网格坐标 [H_f*W_f]
        grid_x_coords_flat = grid_x_coords.flatten().float()
        grid_y_coords_flat = grid_y_coords.flatten().float()


        # --- 初始化目标和掩码 ---
        # 置信度目标: 正样本为 1，负样本为 0
        conf_target = torch.zeros(N_preds, dtype=torch.float32, device=device)
        # Bbox 目标: 正样本的编码 dx, dy, dw, dh
        bbox_target_raw = torch.zeros(N_preds, 4, dtype=torch.float32, device=device)
        # 掩码
        positive_mask = torch.zeros(N_preds, dtype=torch.bool, device=device)
        ignore_mask = torch.zeros(N_preds, dtype=torch.bool, device=device)


        # --- 确定忽略样本 (基于预测框与忽略区域的 IoU) ---
        # 这里仍使用预测框，以避免惩罚落入忽略区域的预测
        pred_boxes_decoded = self.decode_activated_preds_to_boxes(activated_preds_flat, img_size, feature_map_size) # [N_preds, 4]

        ignore_boxes = []
        if ignore_list is not None:
             ignore_boxes = [[float(ann[2]), float(ann[3]),
                              float(ann[2]) + float(ann[4]),
                              float(ann[3]) + float(ann[5])] for ann in ignore_list]
        ignore_boxes_tensor = torch.tensor(ignore_boxes, device=device, dtype=torch.float32) if ignore_boxes else None

        if ignore_boxes_tensor is not None and ignore_boxes_tensor.numel() > 0 and pred_boxes_decoded.numel() > 0:
            # 计算所有预测框与忽略区域的 IoU
            # ious_ignore shape: [N_preds, N_ignore_boxes]
            ious_ignore = compute_iou(pred_boxes_decoded, ignore_boxes_tensor)
            # 找到每个预测框与任何忽略区域的最大 IoU
            max_iou_ignore, _ = ious_ignore.max(dim=1) # [N_preds]
            # 最大 IoU > ignore_iou_threshold 的预测框被标记为忽略
            ignore_mask = max_iou_ignore > self.ignore_iou_threshold


        # --- 确定正样本和 Bbox 目标 (基于 GT 框中心) ---
        gt_boxes = []
        if targets is not None:
            gt_boxes = [[float(ann[2]), float(ann[3]),
                         float(ann[2]) + float(ann[4]),
                         float(ann[3]) + float(ann[5])] for ann in targets]
        gt_boxes_tensor = torch.tensor(gt_boxes, device=device, dtype=torch.float32) if gt_boxes else None

        if gt_boxes_tensor is not None and gt_boxes_tensor.numel() > 0:
            # 遍历每个 ground truth 框
            for gt_box in gt_boxes_tensor:
                # 计算 GT 框中心
                gt_cx = (gt_box[0] + gt_box[2]) / 2.0
                gt_cy = (gt_box[1] + gt_box[3]) / 2.0
                gt_w = gt_box[2] - gt_box[0]
                gt_h = gt_box[3] - gt_box[1]

                # 找到对应 GT 中心的网格单元索引
                grid_x_gt = int(gt_cx / stride_x)
                grid_y_gt = int(gt_cy / stride_y)

                # 检查网格单元是否在特征图边界内
                if 0 <= grid_y_gt < H_f and 0 <= grid_x_gt < W_f:
                    # 计算该网格单元的展平索引
                    pred_idx = grid_y_gt * W_f + grid_x_gt

                    # 将该网格单元标记为正样本
                    positive_mask[pred_idx] = True
                    # 将该正样本的置信度目标设为 1.0
                    conf_target[pred_idx] = 1.0

                    # 计算该单元的原始 bbox 目标 (dx, dy, dw, dh)
                    # 这些目标是相对于网格单元的左上角 (grid_x_gt, grid_y_gt)
                    # 基于解码公式: cx = (grid_x + dx) * stride_x
                    target_dx_raw = (gt_cx / stride_x) - grid_x_gt
                    target_dy_raw = (gt_cy / stride_y) - grid_y_gt
                    # dw, dh 目标是 log(GT_dim / stride)
                    target_dw_raw = torch.log(gt_w / stride_x + 1e-6) # 添加 epsilon 防止 log(0)
                    target_dh_raw = torch.log(gt_h / stride_y + 1e-6) # 添加 epsilon

                    # 存储计算出的目标
                    bbox_target_raw[pred_idx, :] = torch.tensor([target_dx_raw, target_dy_raw, target_dw_raw, target_dh_raw], device=device)

        # 非忽略掩码
        non_ignored_mask = ~ignore_mask
        num_non_ignored = non_ignored_mask.sum().item() # 非忽略样本数量


        # --- 计算损失 ---
        num_matched = positive_mask.sum().item() # 正样本数量

        # Bbox 损失: 只计算正样本的损失
        bbox_loss = torch.tensor(0.0, device=device) # 初始化 bbox 损失为 0
        if num_matched > 0:
            # 选择正样本的原始 bbox 预测值和目标值
            matched_pred_raw_bbox = raw_preds_flat[positive_mask][:, :4]
            matched_bbox_target_raw = bbox_target_raw[positive_mask]
            # print('matched_pred_raw_bbox',matched_pred_raw_bbox)
            # print('matched_bbox_target_raw',matched_bbox_target_raw)
            # 计算原始 bbox 预测值和原始目标值之间的 Smooth L1 损失
            bbox_loss = F.smooth_l1_loss(matched_pred_raw_bbox, matched_bbox_target_raw, reduction='sum') # 在匹配样本上求和


        # 置信度损失: 计算所有非忽略样本的损失
        conf_loss = torch.tensor(0.0, device=device) # 初始化 conf 损失为 0
        if num_non_ignored > 0:
            pred_raw_conf_non_ignored = raw_preds_flat[non_ignored_mask][:, 4]
            conf_target_non_ignored = conf_target[non_ignored_mask]
            pred_conf_activated_non_ignored = torch.sigmoid(pred_raw_conf_non_ignored)

            # === 1. 区分正负样本 ===
            is_pos = conf_target_non_ignored == 1
            is_neg = conf_target_non_ignored == 0

            pos_conf = pred_conf_activated_non_ignored[is_pos]
            neg_conf = pred_conf_activated_non_ignored[is_neg]

            # === 2. Top-k 选择负样本 ===
            K = min(neg_conf.numel(), 3 * is_pos.sum().item())  # 比例可调：负样本数量 = 正样本数量 × 3
            if K > 0:
                topk_neg_conf, topk_indices = torch.topk(neg_conf, K, largest=True)
                topk_neg_mask = torch.zeros_like(neg_conf, dtype=torch.bool)
                topk_neg_mask[topk_indices] = True
                neg_conf = neg_conf[topk_neg_mask]
                neg_target = torch.zeros_like(neg_conf)
            else:
                neg_conf = torch.tensor([], device=device)
                neg_target = torch.tensor([], device=device)

            # === 3. 拼接正负样本 ===
            final_conf = torch.cat([pos_conf, neg_conf], dim=0)
            final_target = torch.cat([torch.ones_like(pos_conf), neg_target], dim=0)

            # === 4. 计算 Loss ===
            if final_conf.numel() > 0:
                conf_loss = focal_loss(final_conf, final_target, alpha=0.25, gamma=2.0, reduction='sum')

        return {
            'conf_loss': conf_loss, # 非忽略样本的置信度损失总和
            'bbox_loss': bbox_loss, # 正样本的 bbox 损失总和
            'num_matched': num_matched, # 正样本数量
            'num_non_ignored': num_non_ignored # 非忽略样本数量
        }

    def compute_batch_loss(self, raw_preds_list, activated_preds_list, targets, ignore_list, img_size):
        """
        计算批次在所有 FPN 级别的总损失。

        Args:
            raw_preds_list (list): 每个级别的原始预测张量列表 [B', 5, H_f, W_f]。
                                  B' 是处理的序列数量 (B // step)。
            activated_preds_list (list): 每个级别的激活后预测张量列表 [B', 5, H_f, W_f]。
            targets (list): B' 张图片各自的 GT 标注列表。
            ignore_list (list, optional): B' 张图片各自的忽略区域标注列表。默认为 None。
            img_size (tuple): (H_img, W_img)。

        Returns:
            dict: 总损失和用于日志记录的指标。
        """
        # B' 是批次中由检测头处理的图片数量 (B // step)
        num_images_in_batch = raw_preds_list[0].shape[0]
        img_H, img_W = img_size

        total_conf_loss_sum = 0.0 # 批次/级别中所有非忽略样本的置信度损失总和
        total_bbox_loss_sum = 0.0 # 批次/级别中所有正样本的 bbox 损失总和
        total_matched_in_batch = 0 # 批次/级别中正样本总数
        total_non_ignored_in_batch = 0 # 批次/级别中非忽略样本总数

        num_levels = len(raw_preds_list) # FPN 级别数量

        # 遍历批次中的每张图片
        for i in range(num_images_in_batch):
            single_image_targets = targets[i]
            # 处理可能丢失的 ignore_list 或其条目
            single_image_ignore = ignore_list[i] if ignore_list and i < len(ignore_list) and ignore_list[i] is not None else []

            # 遍历每个 FPN 级别
            for level_idx in range(num_levels):
                raw_preds_level = raw_preds_list[level_idx] # [B', 5, H_f, W_f]
                activated_preds_level = activated_preds_list[level_idx] # [B', 5, H_f, W_f]

                H_f, W_f = raw_preds_level.shape[2:] # 特征图尺寸

                # 展平当前图片和级别的预测值
                raw_preds_flat = raw_preds_level[i].permute(1, 2, 0).contiguous().view(-1, 5) # [H_f*W_f, 5]
                activated_preds_flat = activated_preds_level[i].permute(1, 2, 0).contiguous().view(-1, 5) # [H_f*W_f, 5]

                # 使用新的匹配策略计算该单张图片和级别的损失
                loss_dict = self._compute_single_loss(
                    raw_preds_flat,
                    activated_preds_flat,
                    single_image_targets,
                    single_image_ignore,
                    img_size,
                    (H_f, W_f)
                )

                # 累加总和
                total_conf_loss_sum += loss_dict['conf_loss']
                total_bbox_loss_sum += loss_dict['bbox_loss']
                total_matched_in_batch += loss_dict['num_matched']
                total_non_ignored_in_batch += loss_dict['num_non_ignored']

        # 计算平均损失
        # 在批次和级别中所有非忽略样本上的平均置信度损失
        final_avg_conf_loss = total_conf_loss_sum / max(1, total_non_ignored_in_batch)
        # 在批次和级别中所有正样本上的平均 bbox 损失
        final_avg_bbox_loss = total_bbox_loss_sum / max(1, total_matched_in_batch)

        # 总损失是平均损失的加权和
        # 使用权重 (例如，conf 2.0，bbox 1.0) - 可在调优时调整
        final_avg_total_loss = 2.0 * final_avg_conf_loss + 1.0 * final_avg_bbox_loss

        # 创建 log_dict 用于日志记录，在此处分离值
        log_dict = {
            'conf_loss': final_avg_conf_loss.detach(),
            'bbox_loss': final_avg_bbox_loss.detach(),
            # 记录批次和级别的总计数
            'total_matched': total_matched_in_batch,
            'total_non_ignored': total_non_ignored_in_batch
            # 如果需要，也可以记录每张图片的平均计数：
            # 'avg_matched_per_image': total_matched_in_batch / max(1, num_images_in_batch),
            # 'avg_non_ignored_per_image': total_non_ignored_in_batch / max(1, num_images_in_batch)
        }

        # 返回原始的 final_avg_total_loss 张量，键为 'total_loss'
        # 使用 **log_dict 包含其他日志记录指标 (已正确分离)
        return {'total_loss': final_avg_total_loss, **log_dict}


    def decode_activated_preds_to_boxes(self, activated_preds_flat, img_size, feature_map_size):
         """
         将展平的激活后预测值 [N, 5] 解码为边界框 [N, 4]。
         用于内部计算 IoU (特别是忽略区域)。
         这必须与 decode_preds 中的逻辑匹配，但处理展平的输入。
         """
         N = activated_preds_flat.shape[0] # 预测数量 (H_f * W_f)
         H_img, W_img = img_size
         H_f, W_f = feature_map_size
         stride_y = H_img / H_f
         stride_x = W_img / W_f # 修正 W_W_img -> W_img
         device = activated_preds_flat.device

         # 创建展平的网格坐标
         grid_y, grid_x = torch.meshgrid(torch.arange(H_f, device=device),
                                         torch.arange(W_f, device=device),
                                         indexing='ij')
         # 展平网格坐标 [H_f*W_f]
         grid_x = grid_x.flatten().float()
         grid_y = grid_y.flatten().float()

         # 这些是来自 DetectionHead 的激活值
         # dx, dy 是 sigmoid 输出 (0-1)
         # dw, dh 是 exp 输出
         # Clamp dx, dy 到 [0, 1] 提高鲁棒性
         dx = torch.clamp(activated_preds_flat[:, 0], 0., 1.) # [N]
         dy = torch.clamp(activated_preds_flat[:, 1], 0., 1.) # [N]
         dw = activated_preds_flat[:, 2] # [N]
         dh = activated_preds_flat[:, 3] # [N]

         # 计算中心坐标 (cx, cy) 和宽度/高度 (bw, bh)
         # cx = (grid_x + dx) * stride_x  -- dx, dy 是基于 sigmoid 的 0-1 值。这假设 dx, dy 是相对于单元格左上角的偏移。
         cx = (grid_x + dx) * stride_x
         cy = (grid_y + dy) * stride_y
         # dw, dh 来自 exp 是应用到 stride 的比例因子。
         bw = dw * stride_x
         bh = dh * stride_y

         # 将中心/宽度/高度转换为 x1, y1, x2, y2
         x1 = cx - bw / 2
         y1 = cy - bh / 2
         x2 = cx + bw / 2
         y2 = cy + bh / 2

         # 将框坐标限制在图像边界 [0, W_img], [0, H_img] 内，提高鲁棒性
         x1 = torch.clamp(x1, 0., float(W_img))
         y1 = torch.clamp(y1, 0., float(H_img))
         x2 = torch.clamp(x2, 0., float(W_img))
         y2 = torch.clamp(y2, 0., float(H_img))

         # 确保 x1 <= x2 且 y1 <= y2 (处理由于大的 dx/dy 或小的 dw/dh 导致的潜在翻转框)
         # 这可以通过在 DetectionHead 中 clamp exp(dw), exp(dh) 来确保宽度/高度为正来实现。
         # 或者直接在需要时交换。
         # 让我们确保 x1 <= x2, y1 <= y2 以防万一。
         boxes = torch.stack([x1, y1, x2, y2], dim=1)
         # 如果 x1 > x2 或 y1 > y2 则交换
         boxes[:, [0, 2]] = torch.sort(boxes[:, [0, 2]], dim=1)[0]
         boxes[:, [1, 3]] = torch.sort(boxes[:, [1, 3]], dim=1)[0]

         return boxes # [N, 4]


    def forward_loss(self, dehaze_imgs, targets, ignore_list=None):
        """
        用于训练时计算损失的前向传播。

        Args:
            dehaze_imgs (Tensor): 输入图像批次 [B, C, H, W]。
            targets (list): 批次中每张图片的 GT 标注列表。
            ignore_list (list, optional): 批次中每张图片的忽略区域标注列表。默认为 None。

        Returns:
            dict: 包含总损失和日志指标的字典。
        """

        B, _, H_img, W_img = dehaze_imgs.shape

        # --- 根据自定义标志确定模式 ---
        if not self._evaluating: # 如果不是评估模式，则假定为训练模式

            # 确保批次大小是 self.step 的倍数
            # 如果需要，截断批次
            if B % self.step != 0:
                 print(f"Warning: Batch size {B} not divisible by step {self.step} in training mode. Truncating batch to {B - (B % self.step)}.")
                 dehaze_imgs = dehaze_imgs[:B - (B % self.step)]
                 targets = targets[:B - (B % self.step)]
                 if ignore_list is not None:
                     ignore_list = ignore_list[:B - (B % self.step)]
                 B, _, H_img, W_img = dehaze_imgs.shape # 截断后更新 B

            # 在训练模式下，我们只计算每个序列最后一帧的损失
            final_frame_indices = torch.arange(self.step - 1, B, self.step).tolist()
            targets_for_loss = [targets[i] for i in final_frame_indices]
            ignore_list_for_loss = [ignore_list[i] for i in final_frame_indices] if ignore_list is not None else None

            # 在训练模式下执行前向传播。
            # 显式传递 training=True 给 forward 方法。
            raw_preds_list, activated_preds_list = self(dehaze_imgs, training=True)


        else: # 评估模式
            # 在评估模式下，我们按帧处理批次，以便维护历史状态，并计算批次中所有帧的损失。
            # 注意：这假设批次中的帧是连续的。如果批次包含独立序列，需要外部循环处理。
            raw_preds_list = [] # 存储所有帧所有级别的原始预测
            activated_preds_list = [] # 存储所有帧所有级别的激活后预测
            targets_for_loss = [] # 存储所有帧的目标
            ignore_list_for_loss = [] # 存储所有帧的忽略区域

            for i in range(B): # 遍历批次中的每一帧
                single_img = dehaze_imgs[i].unsqueeze(0) # 处理单帧，批次大小为 1
                single_targets = targets[i]
                single_ignore = ignore_list[i] if ignore_list is not None else None

                # 对单帧执行前向传播，更新内部历史状态 _hist_f
                raw_preds_single_list, activated_preds_single_list = self(single_img,
                                                                          training=False) # 推理模式

                # 收集当前帧的预测结果
                if i == 0: # 如果是第一帧，初始化列表
                    raw_preds_list = [[] for _ in range(len(raw_preds_single_list))]
                    activated_preds_list = [[] for _ in range(len(activated_preds_single_list))]

                for level_idx in range(len(raw_preds_single_list)): # 遍历 FPN 级别
                    raw_preds_list[level_idx].append(raw_preds_single_list[level_idx])
                    activated_preds_list[level_idx].append(
                        activated_preds_single_list[level_idx])

                # 收集当前帧的目标和忽略区域
                targets_for_loss.append(single_targets)
                ignore_list_for_loss.append(single_ignore)

            # 将收集到的预测结果沿批次维度连接起来
            raw_preds_list = [torch.cat(preds_level_list, dim=0) for preds_level_list in
                              raw_preds_list] # [B, 5, H_f, W_f] for each level

            activated_preds_list = [torch.cat(preds_level_list, dim=0) for preds_level_list in
                                    activated_preds_list] # [B, 5, H_f, W_f] for each level

        # 计算损失
        loss_dict = self.compute_batch_loss(
            raw_preds_list,
            activated_preds_list,
            targets_for_loss,
            ignore_list_for_loss,
            (H_img, W_img),
        )
        if not loss_dict['total_loss'].requires_grad and not self._evaluating:
            print("[Warning] No matched targets or valid predictions — using dummy loss to retain graph.")
            loss_dict['total_loss'] = torch.zeros(1, requires_grad=True, device=raw_preds_list[0].device).sum()

        # 在 forward_loss 结束时重置内存，以确保下一批数据（可能是新的序列）有干净的状态
        self.reset_memory()
        return loss_dict


    @torch.no_grad() # 在推理时不需要计算梯度
    def predict(self, high_res_images, conf_thresh=0.5, iou_thresh=0.45):
        """
        对一批图像执行推理。假设图像按时间顺序处理，用于时序模型。

        Args:
            high_res_images (Tensor): 输入图像批次 [B, C, H, W]。
                                      对于标准推理，B 应该是 1。
                                      如果 B > 1，它将它们作为一个批次处理，
                                      为整个批次维护一个历史状态。
                                      对于处理 *多个独立序列*，
                                      您需要外部循环处理序列，并以批次大小 1 调用 predict，
                                      并在序列之间重置内存。
            conf_thresh (float): 过滤预测的置信度阈值。
            iou_thresh (float): NMS 的 IoU 阈值。

        Returns:
            list: 一个张量列表，每个张量包含 NMS 后一张图像的最终检测结果 [x1, y1, x2, y2, conf]。
                  注意: 当前实现将批次作为一个步骤处理，如果 B > 1，则由于 decode_preds 中的列表推导式
                  [boxes_conf[i] for i in range(B)] 以及最后取 results[0]，它返回 *最后一个* 图像的结果。
                  如果 B > 1 的序列是独立的，此 predict 函数需要改进以处理批次推理。
                  假设 B=1 是典型的推理情况。
        """
        self.eval() # 设置模型为评估模式
        B, _, H_img, W_img = high_res_images.shape
        device = high_res_images.device

        # 在推理模式下执行前向传播。
        # 这将更新并使用内部历史状态 _hist_f。
        # activated_preds_list 将包含当前帧的预测结果。
        _, activated_preds_list = self(high_res_images, training=False) # List of [B, 5, H_f, W_f]

        batch_results = [] # 存储批次中每张图片结果的列表

        # 处理批次中每张图片的结果
        for b in range(B):
            all_preds_for_image = [] # 收集图片 'b' 在所有 FPN 级别的预测结果

            # 解码图片 'b' 在每个 FPN 级别的预测结果
            for preds_level in activated_preds_list: # preds_level: [B, 5, H_f, W_f]
                # decode_preds 返回一个列表，包含批次中每张图片的 [H_f*W_f, 5] 张量
                decoded_level_b = decode_preds(preds_level, (H_img, W_img))[b] # [H_f*W_f, 5]
                all_preds_for_image.append(decoded_level_b)

            # 连接图片 'b' 在所有级别的预测结果
            preds_i = torch.cat(all_preds_for_image, dim=0) # [总预测数, 5]

            # 如果没有预测结果，添加空张量并继续
            if preds_i.numel() == 0:
                batch_results.append(torch.empty((0, 5), device=device))
                continue

            # 分离框和分数
            boxes = preds_i[:, :4] # [总预测数, 4]
            scores = preds_i[:, 4] # [总预测数]

            # 应用置信度阈值
            keep = scores > conf_thresh
            boxes, scores = boxes[keep], scores[keep]

            # 如果阈值过滤后没有框，添加空张量并继续
            if boxes.numel() == 0:
                batch_results.append(torch.empty((0, 5), device=device))
                continue

            # 应用非极大值抑制 (NMS)
            # nms 返回要保留的框的索引
            nms_indices = nms(boxes, scores, iou_thresh)

            # 使用 NMS 索引过滤框和分数，并组合它们
            filtered_preds = torch.cat([boxes[nms_indices], scores[nms_indices].unsqueeze(1)], dim=1) # [NMS后数量, 5]

            batch_results.append(filtered_preds)

        # print(batch_results[0])
        # 返回列表，包含 [NMS后数量, 5] 张量，批次中每张图片一个张量
        # 注意：这里返回的是 batch_results[0]，即批次第一张图片的结果。
        # 如果批次包含多张图片，并且需要所有图片的结果，应该返回 batch_results 列表本身。
        # 根据原代码结构，可能是假设 B=1 或只关心批次中的第一张图片。
        # 修正为返回整个列表以支持 B>1 的情况，但需要注意的是，如果 B>1 且是连续帧，
        # 历史状态 _hist_f 是共享的，这与独立帧处理不同。
        return batch_results[0] # 返回列表，每个元素是 [Num_after_NMS, 5] 张量，对应批次中的一张图片


    def reset_memory(self):
        """重置推理时的内部 ConvGRU 隐藏状态。"""
        # print("Resetting DRIFT_NET memory state.")
        self._hist_f = [None] * 4 # 重置 4 个 FPN 级别的历史状态
