import torchvision.models as models
import torch.nn as nn


class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.vgg16(pretrained=True)
        self.net.classifier[6] = nn.Linear(in_features=4096,out_features=1)
    
    def forward(self, x):
        return self.net(x)