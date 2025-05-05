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



# TV 损失
def total_variation_loss(x):
    loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return loss
