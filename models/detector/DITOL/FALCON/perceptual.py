import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights, VGG19_Weights

cnn_normalization_mean = torch.Tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.Tensor([0.229, 0.224, 0.225])
WEIGHTS={
    'vgg16':"VGG16_Weights",
    'vgg19':"VGG19_Weights"
}
class Normalization(nn.Module):
    def __init__(self, device, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1).to(device)
        self.std = std.view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

class PerceptualNet(nn.Module):
    def __init__(self, net, device, style_layers=[], content_layers=['3','8','15']):
        super(PerceptualNet, self).__init__()
        self.normalization = Normalization(device, cnn_normalization_mean, cnn_normalization_std)
        if net :

            vgg_model = getattr(models, net)(weights=getattr(models, WEIGHTS[net.lower()]).IMAGENET1K_V1).features.to(device)
            for param in vgg_model.parameters():
                param.requires_grad = False
            self.vgg_layers = vgg_model
            self.style_layers = style_layers if None not in style_layers else []
            self.content_layers = content_layers if None not in content_layers else []
            print(f"VGG MODEL : {net}, sty : {style_layers}, cont : {content_layers}")
        else:
            print(f"Not computing Perceptual Loss")

    def forward(self, x, y):
        x = (x*0.5 + 0.5)
        y = (y*0.5 + 0.5)
        
        x = self.normalization(x)
        y = self.normalization(y)
        
        style_loss = 0.0
        content_loss = 0.0
        
        x_features = self.get_features(x, set(self.style_layers + self.content_layers))
        y_features = self.get_features(y, set(self.style_layers + self.content_layers))
        
        for layer in self.style_layers:
            G_x = self.gram_matrix(x_features[layer])
            G_y = self.gram_matrix(y_features[layer])
            style_loss += F.mse_loss(G_x, G_y)
        for layer in self.content_layers:
            content_loss += F.mse_loss(x_features[layer], y_features[layer])

        return style_loss + content_loss
    
    def get_features(self, x, selected_layers):
        features = {}
        last_layer = max(list(map(int, selected_layers)))
        for name, layer in self.vgg_layers._modules.items():
            x = layer(x)
            if name in selected_layers:
                features[name] = x
            if last_layer <= int(name):
                break
        return features

    def gram_matrix(self, input):
        a, b, c, d = input.size() 
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)
