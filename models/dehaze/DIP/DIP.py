import torch
import torch.nn as nn
import yaml
import torch.nn.functional as F
from .CNN_PP import CNN_PP
from .filters import DefogFilter, ImprovedWhiteBalanceFilter, GammaFilter, ToneFilter, ContrastFilter, \
    UsmFilter
from .utils import DarkChannel, AtmLight, DarkIcA


class DIP_Module(nn.Module):
    def __init__(self, cfg):
        super(DIP_Module, self).__init__()
        # Placeholder operations - these should be replaced with actual DIP implementations
        self.curve_steps = cfg['curve_steps']
        self.defog_filter = DefogFilter(cfg['defog_range'])
        self.improved_white_balance_filter = ImprovedWhiteBalanceFilter()
        self.gamma_filter = GammaFilter(cfg['gamma_range'])
        self.tone_filter = ToneFilter(cfg['curve_steps'],cfg['tone_curve_range'])
        self.contrast_filter = ContrastFilter()
        self.usm_filter = UsmFilter(cfg['usm_range'])


    def forward(self, inputs, filter_features, defog_A, IcA):
        x = self.defog_filter(inputs, filter_features[:,0], defog_A, IcA)
        x = self.improved_white_balance_filter(x,filter_features[:,1:4])
        x = self.gamma_filter(x,filter_features[:,4])
        x = self.tone_filter(x,filter_features[:,5:5+self.curve_steps])
        x = self.contrast_filter(x,filter_features[:,5+self.curve_steps])
        x = self.usm_filter(x,filter_features[:,6+self.curve_steps])
        # print(x.shape)
        return x

class DIP(nn.Module):
    def __init__(self):
        super().__init__()
        with open('models/dehaze/DIP/dip_config.yaml', 'r') as f:
            self.dip_config = yaml.safe_load(f)
        self.DIP_module = DIP_Module(self.dip_config)
        self.cnn_pp = CNN_PP(7 + self.dip_config['curve_steps'])
        # loss
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        n, c, _, _ = inputs.shape
        resized_inputs = F.interpolate(inputs, size=(256, 256), mode='bilinear', align_corners=False)
        dark = torch.zeros((inputs.shape[0], inputs.shape[2], inputs.shape[3]),
                           dtype=torch.float32, device=inputs.device)
        defog_A = torch.zeros((inputs.shape[0], inputs.shape[1]), dtype=torch.float32, device=inputs.device)
        IcA = torch.zeros((inputs.shape[0], inputs.shape[2], inputs.shape[3]), dtype=torch.float32,
                          device=inputs.device)

        for i in range(inputs.shape[0]):
            train_data_i = inputs[i]
            dark_i = DarkChannel(train_data_i)
            defog_A_i = AtmLight(train_data_i, dark_i)
            IcA_i = DarkIcA(train_data_i, defog_A_i)
            dark[i, ...] = dark_i
            defog_A[i, ...] = defog_A_i
            IcA[i, ...] = IcA_i
        IcA = torch.unsqueeze(IcA, dim=1)

        filter_features = self.cnn_pp(resized_inputs)
        dip_output = self.DIP_module(inputs, filter_features, defog_A, IcA)
        return dip_output

    def forward_loss(self, haze_img, clean_img):
        dehaze_img = self(haze_img)
        print(dehaze_img.device)
        loss = self.criterion(dehaze_img, clean_img)
        return {
            'total_loss': loss,
        }

