import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import lrelu, rgb2lum, tanh_range, lerp

class ExposureProcess(nn.Module):
    def __init__(self, exposure_range):
        super().__init__()
        self.exposure_range = exposure_range

    def forward(self, img, param):
        return img * torch.exp(param[:, None, None, :] * np.log(2)) # 参数维度不需要改动

class UsmProcess(nn.Module):
    def __init__(self, usm_range):
        super().__init__()
        self.usm_range = usm_range

    def forward(self, img, param):
        def make_gaussian_2d_kernel(sigma, dtype=torch.float32):
            radius = 12
            x = torch.arange(-radius, radius + 1, dtype=dtype)
            k = torch.exp(-0.5 * (x / sigma)**2)
            k = k / torch.sum(k)
            return k.unsqueeze(1) * k.unsqueeze(0)

        kernel_i = make_gaussian_2d_kernel(torch.tensor(5.0)) # 固定 sigma=5
        kernel_i = kernel_i.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1) # [1, 1, 25, 25]

        pad_w = (25 - 1) // 2
        padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect')
        outputs = []
        for channel_idx in range(img.shape[1]): # 遍历通道维度
            data_c = padded[:, channel_idx:(channel_idx + 1), :, :] # 通道索引调整
            data_c = F.conv2d(data_c, kernel_i, stride=1, padding='valid')
            outputs.append(data_c)

        output = torch.cat(outputs, dim=1)
        img_out = (img - output) * param[:, None, None, :] + img
        return img_out

class UsmProcess_sigma(nn.Module): # sigma 可变的 USM
    def __init__(self, usm_range):
        super().__init__()
        self.usm_range = usm_range

    def forward(self, img, param):
        def make_gaussian_2d_kernel(sigma, dtype=torch.float32):
            radius = 12
            x = torch.arange(-radius, radius + 1, dtype=dtype)
            k = torch.exp(-0.5 * (x / sigma)**2)
            k = k / torch.sum(k)
            return k.unsqueeze(1) * k.unsqueeze(0)

        kernel_i_list = []
        for i in range(param.shape[0]): # 假设param是batch_size
            kernel_i = make_gaussian_2d_kernel(param[i])
            kernel_i_list.append(kernel_i)
        kernel_i = torch.stack(kernel_i_list).unsqueeze(1) # [batch_size, 1, 25, 25]

        pad_w = (25 - 1) // 2
        padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect')
        outputs = []
        for channel_idx in range(img.shape[1]): # 遍历通道维度
            data_c = padded[:, channel_idx:(channel_idx + 1), :, :] # 通道索引调整
            data_c = F.conv2d(data_c, kernel_i, stride=1, padding='valid', groups=img.shape[0]) # groups for batch-wise kernel
            outputs.append(data_c)

        output = torch.cat(outputs, dim=1)
        img_out = (img - output) * param[:, None, None, :] + img
        return img_out


class DefogFilter(nn.Module):
    def __init__(self, defog_range):
        super().__init__()
        self.defog_range = defog_range

    def forward(self, img, param, defog_A, IcA):
        param = tanh_range(*self.defog_range)(param)
        # print(param, param.shape, type(param),type(IcA))
        tx = 1 - param[:, None, None, None]*IcA
        tx_1 = tx.repeat(1, 3, 1, 1) # [B, 1, 1, 1] -> [B, 3, 1, 1]
        defog_A_expand = defog_A[:, :, None, None].repeat(1, 1, img.shape[2], img.shape[3]) # [B, 3] -> [B, 3, H, W]
        return (img - defog_A_expand) / torch.maximum(tx_1, torch.tensor(0.01)) + defog_A_expand

class ImprovedWhiteBalanceFilter(nn.Module):
    def __init__(self):
        super().__init__()

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = torch.tensor([[0, 1, 1]], dtype=torch.float32, device=features.device)
        assert mask.shape == (1, 3)
        features = features * mask
        color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))
        color_scaling *= 1.0 / (
                                       1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
                                       0.06 * color_scaling[:, 2])[:, None]
        return color_scaling
    def forward(self, img, param):
        param = self.filter_param_regressor(param)
        return img * param[:, :, None, None]

class GammaFilter(nn.Module):
    def __init__(self, gamma_range):
        super().__init__()
        self.gamma_range = gamma_range

    def filter_param_regressor(self, features):
        log_gamma_range = np.log(self.gamma_range)
        return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

    def forward(self, img, param):
        param = self.filter_param_regressor(param).view(2, 1)
        # print(param.shape)
        param = param.repeat(1, 3)
        # print(param.shape)
        return torch.pow(torch.clamp(img, min=0.0001), param[:, :, None, None]) # 参数维度不需要改动


class ColorProcess(nn.Module):
    def __init__(self, curve_steps, color_curve_range):
        super().__init__()
        self.curve_steps = curve_steps
        self.color_curve_range = color_curve_range

    def forward(self, img, param, defog_A, IcA): # 忽略 defog_A, IcA 参数
        color_curve = param # shape already (B, 1, 1, 3, curve_steps)
        color_curve_sum = torch.sum(color_curve, dim=4, keepdim=True) + 1e-30
        total_image = img * 0
        for i in range(self.curve_steps):
            clip_min = 1.0 * i / self.curve_steps
            clip_max = 1.0 / self.curve_steps
            clipped_img = torch.clamp(img - clip_min, 0, clip_max)
            total_image += clipped_img * color_curve[:, :, :, :, i] # color_curve[:, :, :, :, i] shape (B, 1, 1, 3, 1)
        total_image *= self.curve_steps / color_curve_sum.squeeze(4) # color_curve_sum.squeeze(4) shape (B, 1, 1, 3)
        return total_image

class ToneFilter(nn.Module):
    def __init__(self, curve_steps,  tone_curve_range):
        super().__init__()
        self.tone_curve_range = tone_curve_range
        self.curve_steps = curve_steps
    def filter_param_regressor(self, features):
        tone_curve = features[:, :, None, None,]
        print(tone_curve.shape)
        tone_curve = tanh_range(*self.tone_curve_range)(tone_curve)
        return tone_curve

    def forward(self, img, param): # 忽略 defog_A, IcA 参数
        tone_curve = self.filter_param_regressor(param)
        tone_curve_sum = torch.sum(tone_curve, dim=1, keepdim=True) + 1e-30
        total_image = img * 0
        for i in range(self.curve_steps):
            clip_min = 1.0 * i / self.curve_steps
            clip_max = 1.0 / self.curve_steps
            clipped_img = torch.clamp(img - clip_min, 0, clip_max)
            total_image += clipped_img * tone_curve[:, :, :, :, i] # tone_curve[:, :, :, :, i] shape (B, 1, 1, 1, 1)
        total_image *= self.curve_steps / tone_curve_sum.squeeze(4) # tone_curve_sum.squeeze(4) shape (B, 1, 1, 1)
        return total_image

class VignetProcess(nn.Module):
    def __init__(self):
        super().__init__()
        pass # VignetFilter 的 process 返回的是 0， 实际效果在 mask 上，这里 process 就简单返回 0 即可

    def forward(self, img, param): # 忽略 param 参数, 因为 VignetFilter 的 process 里没用到
        return img * 0

class ContrastProcess(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, img, param, defog_A, IcA): # 忽略 defog_A, IcA 参数
        luminance = torch.clamp(rgb2lum(img), 0.0, 1.0)
        contrast_lum = -torch.cos(np.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance[:, None, :, :] + 1e-6) * contrast_lum[:, None, :, :] # luminance[:, None, :, :]  为了广播
        return lerp(img, contrast_image, param[:, None, None, None])

class WNBProcess(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, img, param, defog_A, IcA): # 忽略 defog_A, IcA 参数
        luminance = rgb2lum(img)
        luminance_expand = luminance[:, None, :, :].repeat(1, 3, 1, 1) # expand to 3 channels
        return lerp(img, luminance_expand, param[:, None, None, None])

class LevelProcess(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, img, param):
        lower = param[:, 0]
        upper = param[:, 1] + 1
        lower = lower[:, None, None, None]
        upper = upper[:, None, None, None]
        return torch.clamp((img - lower) / (upper - lower + 1e-6), 0.0, 1.0)

class SaturationPlusProcess(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, img, param, defog_A, IcA): # 忽略 defog_A, IcA 参数
        img = torch.clamp(img, max=1.0)
        hsv = F.rgb_to_hsv(img) # 直接使用 F.rgb_to_hsv, 输入 NCHW, 输出 NCHW
        s = hsv[:, 1:2, :, :] # 通道索引调整
        v = hsv[:, 2:3, :, :] # 通道索引调整
        enhanced_s = s + (1 - s) * (0.5 - torch.abs(0.5 - v)) * 0.8
        hsv1 = torch.cat([hsv[:, 0:1, :, :], enhanced_s, hsv[:, 2:, :, :]], dim=1) # 通道索引调整
        full_color = F.hsv_to_rgb(hsv1) # 直接使用 F.hsv_to_rgb, 输入 NCHW, 输出 NCHW

        param = param[:, None, None, None]
        color_param = param
        img_param = 1.0 - param
        return img * img_param + full_color * color_param
