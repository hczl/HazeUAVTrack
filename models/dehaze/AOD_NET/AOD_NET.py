import torch
import torch.nn as nn
import torch.nn.functional as F


class AOD_NET(nn.Module):
    def __init__(self):
        """
        AOD-Net 模型初始化。
        这是一个基于大气散射模型的端到端去雾网络，
        通过学习一个称为 K 的参数来估计透射率和大气光。
        """
        super(AOD_NET, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)

        # 可学习的参数 b，对应大气散射模型中的大气光 A
        # 在 AOD-Net 公式 J = K * I - K + b 中使用
        self.b = nn.Parameter(torch.tensor(1.0))
        # 使用均方误差损失函数 (MSELoss) 来衡量去雾结果与干净图像之间的差异
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        AOD-Net 的前向传播。

        Args:
            x (torch.Tensor): 输入的雾霾图像张量 (B, C, H, W)。

        Returns:
            torch.Tensor: 去雾后的图像张量 (B, C, H, W)，经过 ReLU 激活。
        """
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))

        cat1 = torch.cat([x1, x2], dim=1)
        x3 = F.relu(self.conv3(cat1))

        cat2 = torch.cat([x2, x3], dim=1)
        x4 = F.relu(self.conv4(cat2))

        cat3 = torch.cat([x1, x2, x3, x4], dim=1)
        # Conv5 应用 ReLU，输出 K 参数
        k = F.relu(self.conv5(cat3))

        # AOD-Net 核心去雾公式: J = K * I - K + b
        # 其中 J 是去雾后的图像，I 是输入的雾霾图像 x
        # k 是 Conv5 的输出 K 参数
        # self.b 是可学习的参数 b
        output = k * x - k + self.b
        # 对最终输出应用 ReLU，确保像素值非负
        return F.relu(output)

    def forward_loss(self, haze_img, clean_img):
        """
        计算训练所需的损失。

        Args:
            haze_img (torch.Tensor): 输入的雾霾图像张量 (B, C, H, W)。
            clean_img (torch.Tensor): 地面真实无雾图像张量 (B, C, H, W)。

        Returns:
            dict: 包含总损失的字典。
        """
        # 通过模型获取去雾后的图像
        dehaze_img = self(haze_img)
        # 计算去雾图像与干净图像之间的 MSE 损失
        loss = self.criterion(dehaze_img, clean_img)
        # 返回包含总损失的字典
        return {
            'total_loss': loss,
        }
