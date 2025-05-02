import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import  rgb2lum, tanh_range, lerp


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
        return (img - defog_A_expand) / torch.maximum(tx_1, torch.tensor(0.01).to(img.device)) + defog_A_expand # 确保常数张量也在同一设备

class ImprovedWhiteBalanceFilter(nn.Module):
    def __init__(self):
        super().__init__()

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = torch.tensor([[0, 1, 1]], dtype=torch.float32, device=features.device)
        assert mask.shape == (1, 3)
        features = features * mask
        color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))

        multiplier = 1.0 / (
                                       1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
                                       0.06 * color_scaling[:, 2])[:, None]
        color_scaling = color_scaling * multiplier

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
        param = self.filter_param_regressor(param).view(param.shape[0], 1)
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
        # Using img * 0 or torch.zeros_like(img) both create a new tensor,
        # so the += operation inside the loop is generally safe.
        # total_image = img * 0
        total_image = torch.zeros_like(img) # More explicit initialization

        for i in range(self.curve_steps):
            clip_min = 1.0 * i / self.curve_steps
            clip_max = 1.0 / self.curve_steps
            clipped_img = torch.clamp(img - clip_min, 0, clip_max)
            total_image += clipped_img * color_curve[:, :, :, :, i] # color_curve[:, :, :, :, i] shape (B, 1, 1, 3, 1)

        # --- FIX: Replace inplace *= with out-of-place * ---
        multiplier = self.curve_steps / color_curve_sum.squeeze(4)
        total_image = total_image * multiplier # Create a new tensor
        # --- END FIX ---

        return total_image

class ToneFilter(nn.Module):
    def __init__(self, curve_steps,  tone_curve_range):
        super().__init__()
        self.tone_curve_range = tone_curve_range
        self.curve_steps = curve_steps
    def filter_param_regressor(self, features):
        tone_curve =  torch.reshape(features, shape=(-1, 1, self.curve_steps))[:, None, None, :]
        # print(tone_curve.shape)
        tone_curve = tanh_range(*self.tone_curve_range)(tone_curve)
        return tone_curve

    def forward(self, img, param): # 忽略 defog_A, IcA 参数
        param = self.filter_param_regressor(param)
        # print(param.shape)
        tone_curve = torch.reshape(param, shape=(-1, self.curve_steps))[:, :, None, None, None]

        # print(tone_curve.shape)
        tone_curve_sum = torch.sum(tone_curve.squeeze(4), dim=1, keepdim=True) + 1e-30
        # Using img * 0 or torch.zeros_like(img) both create a new tensor,
        # so the += operation inside the loop is generally safe.
        # total_image = img * 0
        total_image = torch.zeros_like(img) # More explicit initialization

        for i in range(self.curve_steps):
            clip_min = 1.0 * i / self.curve_steps
            clip_max = 1.0 / self.curve_steps
            clipped_img = torch.clamp(img - clip_min, 0, clip_max)
            total_image += clipped_img * tone_curve[:, i, :, :, :] # tone_curve[:, :, :, :, i] shape (B, 1, 1, 1, 1)

        multiplier = self.curve_steps / tone_curve_sum
        total_image = total_image * multiplier 

        return total_image


class ContrastFilter(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def filter_param_regressor(self, features):
        return torch.tanh(features)

    def forward(self, img, param):
        param = self.filter_param_regressor(param)
        luminance = torch.clamp(rgb2lum(img), 0.0, 1.0)
        contrast_lum = -torch.cos(np.pi * luminance) * 0.5 + 0.5
        # Note: The division by luminance can result in NaN/Inf if luminance is 0.
        # Adding a small epsilon (1e-6) helps, but consider if this calculation is numerically stable.
        contrast_image = img / (luminance[:, None, :, :] + 1e-6) * contrast_lum[:, None, :, :]
        # print(contrast_image.shape,param[:, None, None, None].shape)
        return lerp(img, contrast_image, param[:, None, None, None])


class UsmFilter(nn.Module):
    def __init__(self, usm_range):
        super().__init__()
        self.usm_range = usm_range

    def filter_param_regressor(self, features):
        return tanh_range(*self.usm_range)(features)

    def forward(self, img, param):
        param = self.filter_param_regressor(param)

        def make_gaussian_2d_kernel(sigma, dtype=torch.float32):
            radius = 12
            x = torch.arange(-radius, radius + 1, dtype=dtype)
            k = torch.exp(-0.5 * (x / sigma) ** 2)
            k = k / torch.sum(k)
            return k.unsqueeze(1) * k.unsqueeze(0)

        kernel_i = make_gaussian_2d_kernel(torch.tensor(5.0))  # 固定 sigma=5
        kernel_i = kernel_i.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)  # [1, 1, 25, 25]

        kernel_i = kernel_i.to(img.device) # Ensure kernel is on the correct device

        pad_w = (25 - 1) // 2
        padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect')
        outputs = []
        # Perform convolution channel by channel as the kernel is not batched
        # Alternatively, you could use groups=img.shape[0] if kernel_i was [B, 1, H, W]
        for channel_idx in range(img.shape[1]):
            data_c = padded[:, channel_idx:(channel_idx + 1), :, :]
            data_c = F.conv2d(data_c, kernel_i, stride=1, padding='valid')
            outputs.append(data_c)

        output = torch.cat(outputs, dim=1)
        # This operation creates a new tensor, so it's not inplace
        img_out = (img - output) * param[:, None, None, None] + img
        return torch.clamp(img_out, 0.0, 1.0)

