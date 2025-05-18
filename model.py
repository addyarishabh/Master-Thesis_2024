import torch
import torch.nn as nn
from torchvision import models
from kan import KANLinear  # Ensure the `kan` library is installed

class ConvNeXtKAN(nn.Module):
    def __init__(self, num_classes=5):
        super(ConvNeXtKAN, self).__init__()
        self.convnext = models.convnext_tiny(pretrained=True)

        # Modify ConvNeXt for feature extraction
        num_features = self.convnext.classifier[2].in_features
        self.convnext.classifier = nn.Identity()

        # Custom KAN-based classifier
        self.kan1 = KANLinear(num_features, 256)
        self.kan2 = KANLinear(256, num_classes)

    def forward(self, x):
        x = self.convnext(x)
        x = self.kan1(x)
        x = self.kan2(x)
        return x

