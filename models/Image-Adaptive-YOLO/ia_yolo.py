import torch
import torch.nn as nn
import torch.nn.functional as F
from model.yolo import YOLOv3
from model.cnn_pp import CNNPP
from model.dip import DIP


class IA_YOLO(nn.Module):
    def __init__(self, num_classes=80, use_dip=True):
        super(IA_YOLO, self).__init__()
        self.use_dip = use_dip

        # 模块初始化
        self.cnn_pp = CNNPP()  # 输出滤波器参数
        self.dip = DIP()       # 应用滤波器增强图像
        self.yolo = YOLOv3(num_classes=num_classes)

    def forward(self, x):
        x_clean = x.clone()  # 记录未处理图像用于恢复损失

        if self.use_dip:
            # 下采样图像 → 预测参数
            x_down = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
            params = self.cnn_pp(x_down)

            # 应用DIP增强图像
            x = self.dip(x, params)

        # 用增强图像进行检测
        yolo_outputs = self.yolo(x)

        # 计算增强图像与原始图像差异（恢复损失）
        recovery_loss = F.mse_loss(x, x_clean)

        return yolo_outputs, recovery_loss
