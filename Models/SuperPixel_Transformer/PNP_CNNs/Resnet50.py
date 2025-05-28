import torch
import torch.nn as nn
import torchvision.models as models
from Models.SuperPixel_Transformer.config import match_size


class ResNet50(nn.Module):
    def __init__(self, num_classes=512, pretrained=True):
        super(ResNet50, self).__init__()
        # Load pre-trained ResNet50 Models
        resnet = models.resnet50(pretrained=pretrained)

        # Remove the fully connected layers (last layers)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Exclude FC layer and avgpool

        # Add a convolutional layer to reduce channels to 512
        self.conv_reduce = nn.Conv2d(2048, 512, kernel_size=1)  # Reduce channels from 2048 to 512

        # Add a convolutional layer to adjust output channels if needed
        self.conv_out = nn.Conv2d(512, num_classes, kernel_size=1)  # Adjust channels to 512 (or num_classes)

        # Upsample the output to match the input size (224x224)
        self.upsample = nn.Upsample(size=match_size, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.features(x)       # Extract feature maps from ResNet
        x = self.conv_reduce(x)    # Reduce channel dimensions to 512
        x = self.conv_out(x)       # Adjust output channels if needed
        x = self.upsample(x)       # Upsample to 224x224
        return x

