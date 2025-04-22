import torch
from torch import nn


class PlaceholderYOLOv3(nn.Module): # Reusing placeholder YOLOv3
    def __init__(self, num_classes):
        super(PlaceholderYOLOv3, self).__init__()
        self.linear = nn.Linear(1000, num_classes)

    def forward(self, x):
        return self.linear(x)

class PlaceholderYOLOv3Loss(nn.Module): # Reusing placeholder Loss
    def __init__(self):
        super(PlaceholderYOLOv3Loss, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(predictions - targets)**2

class yolov3(nn.Module):
    def __init__(self):
        super().__init__()
        pass
