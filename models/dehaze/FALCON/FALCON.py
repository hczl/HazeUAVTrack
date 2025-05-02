import torch.nn.init as init
import yaml
from torch.nn.modules.module import T

from .ddp.ddp import DDP, DDP_input
from .ffc import ConcatTupleLayer, FFCResnetBlock
from .perceptual import PerceptualNet

from .unet_parts import *
from .utils import make_dark_channel_tensor


class FALCON(nn.Module):
    def __init__(self, bilinear=False):
        super(FALCON, self).__init__()
        with open('models/dehaze/FALCON/falcon_config.yaml', 'r') as f:
            self.falcon_config = yaml.safe_load(f)

        self.input_kernel = self.falcon_config['train']['input_kernel']

        self.n_channels = len(self.input_kernel) + 3
        self.out_channels = 3
        self.bilinear = bilinear
        self.config_ffc = self.falcon_config['ffc']
        self.inc = (DoubleConv(self.n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.middle_part = FFCResnetBlock(512, padding_type='reflect', activation_layer=nn.ReLU,
                                          norm_layer=nn.BatchNorm2d, config_ffc=self.config_ffc)
        self.middle_concat = ConcatTupleLayer()

        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, self.out_channels))
        self.ddp = DDP(kernel_size=15)
        self.ddp_input = DDP_input()
        self._initialize_weights(self.falcon_config['model_config']['init'])
        self.mse = nn.MSELoss()
        self.perc_loss_network = PerceptualNet(
            net=self.falcon_config['train']['perceptual']['net'],
            style_layers=self.falcon_config['train']['perceptual']['style'],
            content_layers=self.falcon_config['train']['perceptual']['content'],
            device=self.falcon_config['device']
        )
    def _init_weights(self, m, name):
        if name == 'he_u':
            init_fn = init.kaiming_uniform_
        if name == 'he_n':
            init_fn = init.kaiming_normal_
        if name == 'xavier':
            init_fn = init.xavier_uniform_

        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init_fn(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init_fn(m.weight)
            init.constant_(m.bias, 0)

    def _initialize_weights(self, name):
        for m in self.modules():
            self._init_weights(m, name)
        print(f"Model is initialized with {name}")

    def forward(self, x, w = 1, trainable=True):
        x_ = []
        for b in range(x.size(0)):
            for k in self.input_kernel:
                x_.append(self.ddp_input(x[b:b + 1, ...], k))
        x_ = torch.cat(x_).view(x.size(0), -1, x.size(2), x.size(3)).contiguous()
        x = torch.cat((x, x_), dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        for i in range(self.config_ffc['loop']):
            x4 = self.middle_part(x4)
        x4 = self.middle_concat(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if trainable:
            return logits
        t_haze = w * self.ddp(logits)
        return logits, t_haze

    def loss(self, haze_img, clean_img):
        dehaze_img, t_map = self(haze_img,trainable=False)
        t_gt = make_dark_channel_tensor(clean_img)
        loss_img = self.mse(dehaze_img, clean_img)
        loss_map = self.mse(t_map, t_gt)
        loss_perc = self.perc_loss_network(dehaze_img, clean_img) \
            if self.falcon_config['train']['perceptual']['net'] else torch.tensor(0.).to(self.config['device'])

        total_loss = (self.falcon_config['train']['loss_ratio'][0] * loss_img +
                      self.falcon_config['train']['loss_ratio'][1] * loss_map +
                      self.falcon_config['train']['loss_ratio'][2] * loss_perc )

        return {
            'loss_img': loss_img,
            'loss_map': loss_map,
            'loss_perc': loss_perc,
            'total_loss': total_loss,
        }