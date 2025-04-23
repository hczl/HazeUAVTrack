import torch
import torch.nn as nn

from models.IA_YOLOV3.filters import DefogFilter, ImprovedWhiteBalanceFilter, GammaFilter, ToneFilter, ContrastFilter, \
    UsmFilter


class DIP_Module(nn.Module):
    def __init__(self, cfg):
        super(DIP_Module, self).__init__()
        # Placeholder operations - these should be replaced with actual DIP implementations
        self.curve_steps = cfg.curve_steps
        self.defog_filter = DefogFilter(cfg.defog_range)
        self.improved_white_balance_filter = ImprovedWhiteBalanceFilter()
        self.gamma_filter = GammaFilter(cfg.gamma_range)
        self.tone_filter = ToneFilter(cfg.curve_steps,cfg.tone_curve_range)
        self.contrast_filter = ContrastFilter()
        self.usm_filter = UsmFilter(cfg.usm_range)


    def forward(self, inputs, filter_features, defog_A, IcA):
        x = self.defog_filter(inputs, filter_features[:,0], defog_A, IcA)
        x = self.improved_white_balance_filter(x,filter_features[:,1:4])
        x = self.gamma_filter(x,filter_features[:,4])
        x = self.tone_filter(x,filter_features[:,5:5+self.curve_steps])
        x = self.contrast_filter(x,filter_features[:,5+self.curve_steps])
        x = self.usm_filter(x,filter_features[:,6+self.curve_steps])
        # print(x.shape)
        return x