from torch.nn import Module
import torch.nn as nn
from torchvision import models

class SignNet(Module):
    def __init__(self, num_classes):
        super(SignNet, self).__init__()

        self.model_ft = models.resnet18(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, image):
        return self.model_ft(image)

if __name__ == '__main__':
    pass