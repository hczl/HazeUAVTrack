import torch
import math
import torch.nn.functional as F

import yaml
def DarkChannel(im):
    # print(im.shape)
    if im.ndim == 4: # 改为 (N, C, H, W)
        if im.shape[1] > 3:
            im = im.permute(0, 3, 1, 2)
        b, g, r = torch.unbind(im, dim=0)
    elif im.ndim == 3:  # 改为(C, H, W)
        if im.shape[0] > 3:
            im = im.permute(2, 0, 1)
        b, g, r = torch.unbind(im, dim=0)
    else:
        raise ValueError("Input tensor should have 3 or 4 dimensions (HWC or CHW or NHWC or NCHW)")

    dc = torch.minimum(torch.minimum(r, g), b)
    return dc

def AtmLight(im, dark):
    if im.ndim == 4:  # 改为 (N, C, H, W)
        if im.shape[1] > 3:
            im = im.permute(0, 3, 1, 2)
        h, w = im.shape[1:3]
    elif im.ndim == 3:  # 改为(C, H, W)
        if im.shape[0] > 3:
            im = im.permute(2, 0, 1)
        h, w = im.shape[:2]
    else:
        raise ValueError("Input tensor should have 3 or 4 dimensions (HWC or CHW or NHWC or NCHW)")

    imsz = h * w
    numpx = max(math.floor(imsz / 1000), 1)

    darkvec = dark.reshape(-1)
    imvec = im.reshape(-1, im.shape[0]) # 保持通道数动态

    indices = torch.argsort(darkvec, descending=True) # 获取降序索引
    indices = indices[:numpx] # 取前 numpx 个最亮像素的索引
    # print(im.shape)
    atmsum = torch.zeros(im.shape[0], dtype=im.dtype, device=im.device) # 初始化为 0，通道数动态
    # print(atmsum.shape, imvec[0].shape)
    for ind in indices:
        atmsum += imvec[ind]

    A = atmsum / numpx
    # print(A.shape)
    return A.unsqueeze(0) # 保持形状为 (1, C)

def DarkIcA(im, A):
    if im.ndim == 4: # 改为 (N, C, H, W)
        if im.shape[1] > 3:
            im = im.permute(0, 3, 1, 2)
    elif im.ndim == 3:  # 改为(C, H, W)
        if im.shape[0] > 3:
            im = im.permute(2, 0, 1)
    else:
        raise ValueError("Input tensor should have 3 or 4 dimensions (HWC or CHW or NHWC or NCHW)")

    im3 = torch.zeros(im.shape, dtype=im.dtype, device=im.device)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    return DarkChannel(im3)


def lrelu(x, leak=0.2):
    return F.leaky_relu(x, negative_slope=leak)

def tanh01(x):
  return F.tanh(x) * 0.5 + 0.5


def tanh_range(l, r, initial=None):

  def get_activation(left, right, initial):

    def activation(x):
      if initial is not None:
        bias = math.atanh(2 * (initial - left) / (right - left) - 1)
      else:
        bias = 0
      return tanh01(x + bias) * (right - left) + left

    return activation

  return get_activation(l, r, initial)

def lerp(a, b, t):
    return a + (b - a) * t

def rgb2lum(img):
    return 0.27 * img[:, 0, :, :] + 0.67 * img[:, 1, :, :] + 0.06 * img[:, 2, :, :]
