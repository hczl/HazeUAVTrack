import torch
from torch import nn, optim
import torch.nn.functional as F

class CNN_PP(nn.Module):
    def __init__(self, out_dim):
        super(CNN_PP, self).__init__()
        channels = 16 # as defined in extract_parameters_2
        self.conv0 = nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(channels, 2*channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(2*channels, 2*channels, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2*channels, 2*channels, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(2*channels, 2*channels, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(2048, 64)
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, x):

        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, 2048)
        features = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        filter_features = self.fc2(features)

        return filter_features

if __name__ == '__main__':
    # Assuming cfg.num_filter_parameters is defined somewhere, let's use 10 for example
    output_dim = 10  # cfg.num_filter_parameters
    model = CNN_PP(15)

    # Example input (assuming input image size is 256x256 to get to 8x8 after 5 downsamplings, then 8x8x32 = 2048)
    input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, 256x256 image

    output = model(input_tensor)
    print(output.shape)  # Should be torch.Size([1, 10]) if output_dim = 10
