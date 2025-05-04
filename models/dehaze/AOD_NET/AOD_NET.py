import torch
import torch.nn as nn
import torch.nn.functional as F


class AOD_NET(nn.Module):
    def __init__(self):
        super(AOD_NET, self).__init__()

        # Conv layers according to original AOD-Net configuration
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1)  # output: C=3
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)  # output: C=3
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)  # input: conv1 + conv2
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)  # input: conv2 + conv3
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)  # input: conv1 + conv2 + conv3 + conv4

        self.b = nn.Parameter(torch.tensor(1.0))
        self.criterion = nn.MSELoss()
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat([x1, x2], dim=1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat([x2, x3], dim=1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat([x1, x2, x3, x4], dim=1)
        k = F.relu(self.conv5(cat3))

        # AOD-Net公式: J = K * I - K + b
        output = k * x - k + self.b
        return F.relu(output)

    def forward_loss(self, haze_img, clean_img):
        dehaze_img = self(haze_img)
        loss = self.criterion(dehaze_img, clean_img)
        return {
            'total_loss': loss,
        }
