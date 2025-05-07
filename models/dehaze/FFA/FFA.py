import torch.nn as nn
import torch
from torchvision.models import vgg16

from .PerceptualLoss import LossNetwork
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x
        return res
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [ Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class FFA(nn.Module):
    def __init__(self,conv=default_conv):
        super(FFA, self).__init__()
        self.gps=3
        self.blocks=19
        self.dim=64
        self.lambda_per = 0.04  # 感知损失系数
        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1= Group(conv, self.dim, kernel_size,blocks=self.blocks)
        self.g2= Group(conv, self.dim, kernel_size,blocks=self.blocks)
        self.g3= Group(conv, self.dim, kernel_size,blocks=self.blocks)
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer=PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)
        self.l1_loss = nn.L1Loss()
        self.perceptual = self._init_perceptual_loss()

    def _init_perceptual_loss(self):
        vgg = vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        return LossNetwork(vgg).to(next(self.parameters()).device)

    def forward(self, x1):

        x = self.pre(x1)
        res1=self.g1(x)
        res2=self.g2(res1)
        res3=self.g3(res2)
        w=self.ca(torch.cat([res1,res2,res3],dim=1))
        w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
        out=w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3
        out=self.palayer(out)
        x=self.post(out)
        return x + x1

    def forward_loss(self, haze_img, clean_img):
        new_h = haze_img.shape[2] // 4
        new_w = haze_img.shape[3] // 4
        downsampled_size = (new_h, new_w)
        downsampled_haze = F.interpolate(haze_img, size=downsampled_size, mode='area')
        downsampled_clean = F.interpolate(clean_img, size=downsampled_size, mode='area')

        downsampled_output = self(downsampled_haze)

        # 5. 使用下采样后的输出和目标计算损失
        l1 = self.l1_loss(downsampled_output, downsampled_clean)
        perceptual = self.perceptual(downsampled_output, downsampled_clean) # Perceptual loss 也是全卷积的，可以处理下采样后的特征图

        # 6. 计算总损失
        total = l1 + self.lambda_per * perceptual

        return {
            'total_loss': total,
            'l1_loss': l1,
            'perceptual_loss': perceptual
        }