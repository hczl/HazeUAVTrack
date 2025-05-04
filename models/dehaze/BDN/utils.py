import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

# 感知损失辅助模块
class PerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16]  # 取到 relu3_3
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.resize = resize
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return F.l1_loss(self.vgg(x), self.vgg(y))


# 损失函数定义
def loss_forward(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    pred: 去雾后的图像, shape (B, C, H, W)
    target: 无雾真实图像, shape (B, C, H, W)
    return: 包含多种 loss 和 total_loss 的字典
    """
    l1 = F.l1_loss(pred, target)
    ssim = 1 - ssim_loss(pred, target)
    perceptual = PerceptualLoss()(pred, target)
    tv = total_variation_loss(pred)

    # Loss 权重（可以微调）
    loss_dict = {
        'l1_loss': l1,
        'ssim_loss': ssim,
        'perceptual_loss': perceptual,
        'tv_loss': tv
    }

    # 组合总损失
    total = l1 + 0.8 * ssim + 0.5 * perceptual + 0.1 * tv
    loss_dict['total_loss'] = total

    return loss_dict


# TV 损失
def total_variation_loss(x):
    loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return loss
